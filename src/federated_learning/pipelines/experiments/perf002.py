"""FL-PERF-002: monitoring + scalability benchmarks (virtual clients on one 2-core machine, CPU only)."""
from __future__ import annotations

import io
import time
import numpy as np
from federated_learning.privacy.homomorphic import CKKSAggregator
from federated_learning.privacy.secure_aggregation import SecAggAggregator
from federated_learning.privacy.smpc import SMPCAggregator
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset
from federated_learning.data_partitioning.iid import partition_iid
from federated_learning.aggregators.mean import WeightedMean
from federated_learning.training.config import FLConfig
from federated_learning.training.centralized import centralized_baseline
from federated_learning.servers.orchestrator import run_fl
from federated_learning.monitoring.resources import CommCounter, ResourceMonitor, accuracy_loss_pct, gpu_snapshot, overhead_pct
from federated_learning.monitoring.tracking import log_mlflow
from federated_learning.pipelines.experiments.common import out_dir, provenance, save_csv, save_json, savefig, scale, plt, load_json

STORY = "fl_perf_002"

PRICE = dict(vcpu_hour_usd=0.04, egress_gb_usd=0.09, note="ASSUMPTIONS, not verified list prices")

def _aggregator(name: str, seed: int):
    if name == "plain":
        return WeightedMean()
    if name == "secagg_complete":
        return SecAggAggregator(seed=seed)
    if name == "secagg_sparse14":
        return SecAggAggregator(degree=14, seed=seed)
    if name == "he_ckks":
        return CKKSAggregator()
    if name == "smpc_3party":
        return SMPCAggregator(3)
    raise ValueError(name)

def _security_cfg(name: str) -> dict:
    return dict(plain=dict(crypto="none", security="none"), secagg_complete=dict(crypto="pairwise-masking+Shamir", security="semi-honest server"),
                secagg_sparse14=dict(crypto="pairwise-masking (k=14 graph)+Shamir", security="semi-honest server"),
                he_ckks=dict(crypto="CKKS N=8192", security="semi-honest server, trusted key holder"),
                smpc_3party=dict(crypto="SPDZ-style additive+MAC, m=3", security="dishonest majority among 3 servers, abort on cheating"))[name]

def run_config(name: str, n_clients: int, rounds: int, seed: int, data, mlflow_dir: str | None = None) -> dict:
    parts = partition_iid(data.y_train, n_clients, seed)
    agg = _aggregator(name, seed)
    cfg = FLConfig(rounds=rounds, local_epochs=1, seed=seed, eval_every=rounds)
    status, err = "ok", None
    try:
        with ResourceMonitor(interval=0.05) as mon:
            r = run_fl(cfg, data, parts, aggregator=agg)
        res = mon.summary()
        row = dict(config=name, n_clients=n_clients, seed=seed, rounds=rounds, status=status,
                   acc=r.final["acc"], f1=r.final["f1"], train_wall_s=r.totals["train_wall_s"], client_time_s=r.totals["client_time_s"],
                   agg_time_s=r.totals["agg_time_s"], mean_round_s=r.totals["mean_round_s"], up_mb=r.totals["up_bytes"] / 2 ** 20,
                   down_mb=r.totals["down_bytes"] / 2 ** 20, comm_mb_per_round=r.totals["comm_mb"] / rounds,
                   throughput_samples_per_s=r.totals["throughput_samples_per_s"], cpu_time_s=res["cpu_time_s"],
                   cpu_util_mean_pct=res["cpu_util_mean_pct"], cpu_util_peak_pct=res["cpu_util_peak_pct"],
                   rss_mean_mb=res["rss_mean_mb"], rss_peak_mb=res["rss_peak_mb"], gpu_available=res["gpu_available"],
                   max_client_round_s=max(h["client_time_max_s"] for h in r.history))
    except Exception as e:  # failure behaviour is itself a result
        row = dict(config=name, n_clients=n_clients, seed=seed, rounds=rounds, status="failed", error=repr(e)[:200])
    if mlflow_dir and row["status"] == "ok":
        params = dict(n_clients=n_clients, config=name, algo="fedavg", rounds=rounds, seed=seed, **_security_cfg(name))
        row["mlflow_run_id"] = log_mlflow(f"{name}-n{n_clients}-s{seed}", params, {k: v for k, v in row.items() if isinstance(v, (int, float)) and k not in ("seed",)},
                                          mlflow_dir, experiment="fl_perf_002", tags=dict(clients="virtual", hardware="2-core CPU, no GPU"))
    return row

def task_2_1_monitoring() -> dict:
    cfg = dict(busy_s=1.0, alloc_mb=300)
    prov = provenance("FL-PERF-002.1", cfg)
    out = {}
    # CPU accounting check: a single-thread busy loop of ~1 s must show cpu_time ~ wall
    with ResourceMonitor() as m:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < cfg["busy_s"]:
            sum(i * i for i in range(2000))
    s = m.summary()
    out["cpu_check"] = dict(wall_s=s["wall_s"], cpu_time_s=s["cpu_time_s"], cpu_util_mean_pct=s["cpu_util_mean_pct"],
                            relative_gap=abs(s["cpu_time_s"] - s["wall_s"]) / s["wall_s"], expectation="cpu_time ~ wall, util ~ 100% for 1 busy thread")
    # memory accounting check
    with ResourceMonitor(0.02) as m2:
        base = m2.rss[0]
        a = np.ones(cfg["alloc_mb"] * 2 ** 20 // 8)
        time.sleep(0.2)
        del a
    s2 = m2.summary()
    out["memory_check"] = dict(allocated_mb=cfg["alloc_mb"], rss_peak_delta_mb=s2["rss_peak_mb"] - base, rss_mean_mb=s2["rss_mean_mb"],
                               peak_vs_mean_distinct=s2["rss_peak_mb"] > s2["rss_mean_mb"], expectation="peak delta ~ allocation")
    # communication counter vs real serialisation
    v = np.random.default_rng(0).normal(size=1442).astype(np.float32)
    buf = io.BytesIO(); np.save(buf, v)
    cc = CommCounter(); cc.add_up(v.nbytes)
    out["comm_check"] = dict(counted_payload_bytes=cc.up_bytes, npy_serialised_bytes=len(buf.getvalue()),
                             framing_overhead_bytes=len(buf.getvalue()) - cc.up_bytes, note="counter = payload only; protocol framing (TLS/gRPC) is not counted")
    # monitor overhead: same FL job with / without monitor
    data = make_dataset(seed=0); parts = partition_iid(data.y_train, 10, 0)
    cf = FLConfig(rounds=scale(8, 2), local_epochs=1, seed=0, eval_every=100)
    ts_off, ts_on = [], []
    for _ in range(scale(3, 1)):
        t = time.perf_counter(); run_fl(cf, data, parts); ts_off.append(time.perf_counter() - t)
        with ResourceMonitor():
            t = time.perf_counter(); run_fl(cf, data, parts); ts_on.append(time.perf_counter() - t)
    out["monitor_overhead"] = dict(wall_without_s=float(np.mean(ts_off)), wall_with_s=float(np.mean(ts_on)),
                                   overhead_pct=overhead_pct(float(np.mean(ts_on)), float(np.mean(ts_off))))
    # per-round timeline from one real run
    r = run_fl(FLConfig(rounds=scale(10, 3), local_epochs=1, seed=0), data, parts)
    out["round_timeline"] = [dict(round=h["round"], round_s=h["round_time_s"], client_s=h["client_time_s"], agg_s=h["agg_time_s"],
                                  up_bytes=h["up_bytes"], down_bytes=h["down_bytes"]) for h in r.history]
    out["gpu"] = ResourceMonitor().summary() if False else dict(gpu_available=gpu_snapshot()["gpu_available"],
                                                                 note="no NVIDIA tooling: GPU metrics unavailable (not estimated)")
    save_json(STORY, "task_2_1_monitoring", dict(provenance=prov, **out))
    save_csv(STORY, "task_2_1_round_timeline", out["round_timeline"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    tl = out["round_timeline"]
    ax[0].stackplot([t["round"] for t in tl], [t["client_s"] for t in tl], [t["agg_s"] for t in tl], labels=["clients (all, sequential)", "server aggregation"])
    ax[0].set_xlabel("round"); ax[0].set_ylabel("seconds"); ax[0].set_title("Per-round timeline (10 virtual clients)"); ax[0].legend(fontsize=7)
    ax[1].bar(["busy-loop CPU time", "busy-loop wall"], [out["cpu_check"]["cpu_time_s"], out["cpu_check"]["wall_s"]])
    ax[1].set_ylabel("seconds"); ax[1].set_title("Monitor validation: CPU accounting")
    savefig(fig, STORY, "task_2_1_monitoring", prov)
    return out

def _bench(task: str, n_clients: int, configs: list[str], seeds: int, rounds: int, central: bool = False) -> dict:
    cfg = dict(n_clients=n_clients, configs=configs, seeds=seeds, rounds=rounds, local_epochs=1, hidden=32, partition="iid", timing_design="seed-outer, randomised config order, warm-up excluded")
    prov = provenance(task, cfg)
    data = make_dataset(seed=0)
    mldir = str(out_dir(STORY) / "mlruns")
    rows = []
    if central:
        with ResourceMonitor() as m:
            c = centralized_baseline(data, epochs=rounds, lr=0.05, bs=32, seed=0)
        s = m.summary()
        rows.append(dict(config="centralized", n_clients=1, seed=0, rounds=rounds, status="ok", acc=c["acc"], f1=c["f1"], train_wall_s=c["train_wall_s"],
                         cpu_time_s=s["cpu_time_s"], rss_peak_mb=s["rss_peak_mb"], up_mb=0.0, note="pooled data, same model/optimiser, epochs=rounds"))
    run_config(configs[0], n_clients, 2, 0, data, mldir)          # warm-up run, excluded from the results
    for sd in range(seeds):
        # randomised configuration order within each seed: timing must not depend on run order (drift, caches, thermal state)
        for name in [str(n) for n in np.random.default_rng(3000 + sd).permutation(configs)]:
            row = run_config(name, n_clients, rounds, sd, data, mldir)
            rows.append(row)
            print(f"  {task} {name} n={n_clients} seed={sd}: {row['status']} wall={row.get('train_wall_s', float('nan')):.1f}s", flush=True)
    base = {r["seed"]: r for r in rows if r["config"] == "plain" and r["status"] == "ok"}
    for r in rows:
        if r["status"] == "ok" and r["config"] not in ("plain", "centralized") and r["seed"] in base:
            b = base[r["seed"]]
            r["train_overhead_pct"] = overhead_pct(r["train_wall_s"], b["train_wall_s"])
            r["comm_overhead_pct"] = overhead_pct(r["up_mb"] + r["down_mb"], b["up_mb"] + b["down_mb"])
            r["uplink_overhead_pct"] = overhead_pct(r["up_mb"], b["up_mb"])
            r["acc_loss_pp"] = accuracy_loss_pct(r["acc"], b["acc"])
    save_csv(STORY, f"{task.replace('-', '_').lower()}_runs", rows)
    save_json(STORY, f"{task.replace('-', '_').lower()}", dict(provenance=prov, rows=rows, mlflow_dir=mldir, ui_command=f"mlflow ui --backend-store-uri sqlite:///{mldir}/mlflow.db"))
    return dict(provenance=prov, rows=rows)

def _fig_bench(rows, name, prov):
    ok = [r for r in rows if r["status"] == "ok"]
    cfgs = list(dict.fromkeys(r["config"] for r in ok))
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.3))
    for a_, key, yl in zip(ax, ("train_wall_s", "up_mb", "cpu_time_s", "rss_peak_mb"),
                           ("training wall time [s]", "uplink MiB (whole run)", "process CPU time [s]", "peak RSS [MiB]")):
        vals = [np.mean([r[key] for r in ok if r["config"] == c and key in r]) for c in cfgs]
        a_.bar(cfgs, vals); a_.set_ylabel(yl); a_.tick_params(axis="x", rotation=70, labelsize=7); a_.set_yscale("log" if key in ("train_wall_s", "up_mb") else "linear")
    savefig(fig, STORY, name, prov, "virtual clients; 2-core CPU; no GPU")

def task_2_2_benchmark_10() -> dict:
    r = _bench("FL-PERF-002.2", 10, ["plain", "secagg_complete", "he_ckks", "smpc_3party"], scale(10, 1), scale(10, 2), central=True)
    _fig_bench(r["rows"], "task_2_2_benchmark_10", r["provenance"]); return r

def task_2_3_benchmark_50() -> dict:
    r = _bench("FL-PERF-002.3", 50, ["plain", "secagg_complete", "he_ckks", "smpc_3party"], scale(5, 1), scale(10, 2))
    _fig_bench(r["rows"], "task_2_3_benchmark_50", r["provenance"]); return r

def task_2_4_benchmark_100_500() -> dict:
    out = {}
    r100 = _bench("FL-PERF-002.4a", 100, ["plain", "secagg_complete", "secagg_sparse14", "he_ckks", "smpc_3party"], scale(5, 1), scale(10, 2))
    r500 = _bench("FL-PERF-002.4b", 500, ["plain", "secagg_sparse14", "he_ckks", "smpc_3party"], scale(3, 1), scale(10, 2))
    notrun = [dict(config="secagg_complete", n_clients=500, status="NOT_RUN",
                   reason="O(n^2)=250k pairwise key agreements/AEAD/PRG per round + O(n^2 t) Shamir work on a 2-core machine; the sparse-graph variant is the scalable protocol")]
    save_json(STORY, "task_2_4_summary", dict(provenance=r100["provenance"], rows=r100["rows"] + r500["rows"], not_run=notrun,
                                              honesty="all 'clients' are virtual clients in one process sharing 2 CPU cores; no network, no real hospitals"))
    _fig_bench(r500["rows"], "task_2_4_benchmark_500", r500["provenance"]); _fig_bench(r100["rows"], "task_2_4_benchmark_100", r100["provenance"])
    return dict(r100=r100["rows"], r500=r500["rows"], not_run=notrun)

def task_2_5_scalability_analysis() -> dict:
    prov = provenance("FL-PERF-002.5", dict(sources=["task_2_2", "task_2_3", "task_2_4a", "task_2_4b"]))
    rows = []
    for nm in ("fl_perf_002.2", "fl_perf_002.3", "fl_perf_002.4a", "fl_perf_002.4b"):
        d = load_json(STORY, nm)
        if d:
            rows += [r for r in d["rows"] if r["config"] != "centralized"]
    if not rows:
        raise RuntimeError("run tasks 2.2-2.4 first")
    cfgs = list(dict.fromkeys(r["config"] for r in rows))
    agg = []
    for c in cfgs:
        for n in sorted({r["n_clients"] for r in rows if r["config"] == c}):
            rr = [r for r in rows if r["config"] == c and r["n_clients"] == n and r["status"] == "ok"]
            if not rr:
                continue
            row = dict(config=c, n_clients=n, runs=len(rr))
            for k in ("train_wall_s", "mean_round_s", "client_time_s", "agg_time_s", "up_mb", "down_mb", "comm_mb_per_round", "cpu_time_s", "cpu_util_mean_pct",
                      "rss_peak_mb", "throughput_samples_per_s", "acc", "train_overhead_pct", "comm_overhead_pct", "acc_loss_pp"):
                v = [r[k] for r in rr if k in r]
                row[k] = float(np.mean(v)) if v else None
            row["agg_share_of_round"] = row["agg_time_s"] / row["train_wall_s"] if row["train_wall_s"] else None
            row["est_cost_usd_per_run"] = row["cpu_time_s"] / 3600 * PRICE["vcpu_hour_usd"] + (row["up_mb"] + row["down_mb"]) / 1024 * PRICE["egress_gb_usd"]
            agg.append(row)
    # scaling exponent: log(time per round) ~ a + b log n
    expo = []
    for c in cfgs:
        pts = [(a["n_clients"], a["mean_round_s"]) for a in agg if a["config"] == c and a["n_clients"] >= 10]
        if len(pts) >= 3:
            b = np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0]
            expo.append(dict(config=c, loglog_slope=float(b), points=len(pts), note="slope~1 linear in n; slope~2 quadratic (empirical, 3-4 points)"))
    kpi = []
    for a in agg:
        if a["config"] != "plain" and a.get("train_overhead_pct") is not None:
            kpi.append(dict(config=a["config"], n_clients=a["n_clients"], training_overhead_pct=a["train_overhead_pct"], target_lt_30=a["train_overhead_pct"] < 30,
                            comm_overhead_pct=a["comm_overhead_pct"], target_lt_40=a["comm_overhead_pct"] < 40,
                            accuracy_loss_pp=a["acc_loss_pp"], target_lt_1=a["acc_loss_pp"] < 1.0))
    fails = [r for r in rows if r["status"] != "ok"]
    save_csv(STORY, "task_2_5_scalability_table", agg); save_csv(STORY, "task_2_5_kpi_check", kpi); save_csv(STORY, "task_2_5_scaling_exponents", expo)
    save_json(STORY, "task_2_5_scalability_analysis", dict(provenance=prov, table=agg, exponents=expo, kpi=kpi, failures=fails, price_assumptions=PRICE,
                                                          max_tested_clients=max(a["n_clients"] for a in agg), scale_caveat="virtual clients on one machine"))
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    items = (("mean_round_s", "mean round time [s]"), ("comm_mb_per_round", "communication per round [MiB]"), ("cpu_time_s", "process CPU time per run [s]"),
             ("rss_peak_mb", "peak RSS [MiB]"), ("throughput_samples_per_s", "client throughput [samples/s]"), ("agg_share_of_round", "aggregation share of wall time"))
    for a_, (k, yl) in zip(ax.ravel(), items):
        for c in cfgs:
            pts = [(a["n_clients"], a[k]) for a in agg if a["config"] == c and a[k] is not None]
            if pts:
                a_.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=c)
        a_.set_xscale("log"); a_.set_xlabel("virtual clients"); a_.set_ylabel(yl)
        if k in ("mean_round_s", "comm_mb_per_round", "cpu_time_s"):
            a_.set_yscale("log")
    ax[0, 0].legend(fontsize=7)
    savefig(fig, STORY, "task_2_5_scalability_curves", prov, "plain vs secure FL; 10-500 virtual clients; seeds per point see table")
    return dict(provenance=prov, table=agg, exponents=expo, kpi=kpi)
