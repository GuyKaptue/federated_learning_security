"""FL-PUB-005: reproducibility audit and ablation study."""
from __future__ import annotations

import hashlib
import json
import subprocess  # nosec B404
import sys
import numpy as np
from federated_learning import attacks as A
from federated_learning.privacy.homomorphic import CKKSAggregator
from federated_learning.privacy.secure_aggregation import SecAggAggregator
from federated_learning.privacy.smpc import SMPCAggregator
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset
from federated_learning.data_partitioning.iid import partition_iid
from federated_learning.privacy.differential_privacy import ClientSideDP
from federated_learning.aggregators.defenses import ClippedMean
from federated_learning.aggregators.robust import CoordinateMedian
from federated_learning.aggregators.mean import WeightedMean
from federated_learning.training.config import FLConfig
from federated_learning.evaluation.metrics import rounds_to_target
from federated_learning.servers.orchestrator import run_fl
from federated_learning.pipelines.experiments.attack003 import BACKDOOR_LEVELS, K, TRIGGER_CANDIDATES, _median_norm, _run_backdoor
from federated_learning.pipelines.experiments.common import OUT, ROOT, load_json, mean_ci, out_dir, plt, provenance, save_csv, save_json, savefig, scale

STORY = "fl_pub_005"

def task_5_1_reproducibility() -> dict:
    prov = provenance("FL-PUB-005.1", dict(checks="determinism, re-execution, provenance, metric consistency, artifact hashes, notebooks"))
    checks = []

    def chk(name, ok, detail=""):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))

    data = make_dataset(seed=0); parts = partition_iid(data.y_train, 10, 0)
    cfg = FLConfig(rounds=scale(5, 2), local_epochs=1, seed=4)
    r1, r2 = run_fl(cfg, data, parts), run_fl(cfg, data, parts)
    chk("same seed => bit-identical global model", np.array_equal(r1.w, r2.w), f"max|diff|={np.abs(r1.w - r2.w).max()}")
    r3 = run_fl(FLConfig(rounds=cfg.rounds, local_epochs=1, seed=5), data, parts)
    chk("different seed => different model", not np.array_equal(r1.w, r3.w))
    d2 = make_dataset(seed=0)
    chk("dataset generation deterministic", np.array_equal(data.X_train, d2.X_train) and np.array_equal(data.y_test, d2.y_test))
    # re-execution of a stored benchmark run and comparison with the stored value
    import pandas as pd
    p = out_dir("fl_bench_004") / "task_4_5_runs.csv"
    if p.exists():
        from federated_learning.pipelines.experiments.bench004 import exec_run
        df = pd.read_csv(p); row = df[(df.condition == "dir0.3") & (df.algo == "fedprox") & (df.seed == 0)].iloc[0]
        again = exec_run(data, "fedprox", "dir0.3", 10, 0)
        chk("stored FedProx run re-executes to the stored accuracy", abs(again["acc"] - row["acc"]) < 1e-9, f"stored={row['acc']:.6f} re-run={again['acc']:.6f}")
    else:
        chk("stored FedProx run re-executes to the stored accuracy", False, "task_4_5_runs.csv missing (run FL-BENCH-004.5)")
    # provenance of every JSON result
    prov_rows, missing = [], []
    for jf in sorted(OUT.glob("*/*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception as e:
            missing.append((str(jf.relative_to(ROOT)), f"unreadable: {e}")); continue
        pv = d.get("provenance") if isinstance(d, dict) else None
        if pv is None:
            continue
        recomputed = hashlib.sha256(json.dumps(pv["config"], sort_keys=True, default=str).encode()).hexdigest() if "config" in pv else None
        ok = recomputed == pv.get("config_sha256") and pv.get("experiment_id", "").endswith(pv["config_sha256"][:10])
        prov_rows.append(dict(file=str(jf.relative_to(ROOT)), experiment_id=pv.get("experiment_id"), config_sha256=pv.get("config_sha256", "")[:12], consistent=ok,
                              git_commit=pv.get("git_commit"), python=pv["environment"]["python"], dataset=pv.get("dataset_id")))
    chk("provenance present and config hash recomputes", bool(prov_rows) and all(r["consistent"] for r in prov_rows), f"{len(prov_rows)} result files; inconsistent={[r['file'] for r in prov_rows if not r['consistent']]}")
    chk("no result file unreadable", not missing, str(missing))
    chk("git commit identifier available", any(r["git_commit"] for r in prov_rows), "workspace is not a git repository => commit recorded as null; SHA-256 manifest used instead")
    # metric consistency: summary CSV vs raw runs
    cons = []
    for story, runs, summ, key, grp in (("fl_sec_001", "task_1_5_comparison_runs", "task_1_5_comparison_summary", "acc", ["config"]),
                                         ("fl_bench_004", "task_4_5_runs", "task_4_5_summary", "f1", ["condition", "algo"])):
        pr, ps = out_dir(story) / f"{runs}.csv", out_dir(story) / f"{summ}.csv"
        if pr.exists() and ps.exists():
            a = pd.read_csv(pr).groupby(grp)[key].mean().reset_index(); b = pd.read_csv(ps)[grp + [key]]
            m = a.merge(b, on=grp, suffixes=("_raw", "_summary")); gap = float((m[key + "_raw"] - m[key + "_summary"]).abs().max())
            cons.append(dict(story=story, metric=key, max_abs_gap=gap)); chk(f"{story}: summary mean == mean of raw runs ({key})", gap < 1e-9, f"gap={gap:.2e}")
    # artifact hashes
    manifest = []
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and "mlruns" not in f.parts and f.suffix in (".json", ".csv", ".png"):
            manifest.append(dict(path=str(f.relative_to(ROOT)), sha256=hashlib.sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size))
    save_csv(STORY, "task_5_1_artifact_hashes", manifest)
    # notebooks
    import json as _json
    nbs = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    def _has_outputs(path):
        try:
            return any(c.get("outputs") for c in _json.loads(path.read_text()).get("cells", []) if c.get("cell_type") == "code")
        except Exception:
            return False
    ex = [n for n in nbs if _has_outputs(n)]
    chk("notebooks present", len(nbs) >= 1, f"{len(nbs)} notebooks, {len(ex)} with stored outputs")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout  # nosec B603
    (out_dir(STORY) / "pip_freeze.txt").write_text(freeze)
    chk("environment frozen", len(freeze.splitlines()) > 10, f"{len(freeze.splitlines())} packages in pip_freeze.txt")
    res = dict(provenance=prov, checks=checks, provenance_table=prov_rows, consistency=cons, n_artifacts_hashed=len(manifest),
               passed=sum(c["passed"] for c in checks), failed=[c for c in checks if not c["passed"]],
               scope_note="Reproducibility here = same machine & software versions. Cross-platform bitwise identity is NOT claimed (BLAS/OpenSSL builds differ).")
    save_json(STORY, "task_5_1_reproducibility", res)
    return res

def task_5_2_ablation() -> dict:
    cfg = dict(K=K, rounds=scale(20, 3), seeds=scale(10, 1), sigma=0.5, target=0.85, byz_frac=0.2, backdoor="medium", timing_design="seed-outer, randomised config order, warm-up excluded")
    prov = provenance("FL-PUB-005.2", cfg)
    data = make_dataset(seed=0)
    med = _median_norm(data)
    bd = load_json("fl_attack_003", "task_3_1_backdoor")
    trig = (tuple(bd["trigger"]["idx"]), bd["trigger"]["value"]) if bd else TRIGGER_CANDIDATES["f0-3=+4"]
    f_mal = int(round(cfg["byz_frac"] * K)); mal = set(range(f_mal))
    confs = {
        "plain FL": (lambda s: WeightedMean(), "-"),
        "+HE (CKKS)": (lambda s: CKKSAggregator(), "HE"),
        "+SecAgg": (lambda s: SecAggAggregator(seed=s), "SA"),
        "+SMPC (3-party)": (lambda s: SMPCAggregator(3), "SMPC"),
        "+DP (central, sigma=0.5)": (lambda s: ClippedMean(med, cfg["sigma"], s), "DP"),
        "+robust (median)": (lambda s: CoordinateMedian(), "ROB"),
        "SecAgg + client-DP": (lambda s: ClientSideDP(SecAggAggregator(seed=s), med, cfg["sigma"], s), "SA+DP"),
        "HE + client-DP": (lambda s: ClientSideDP(CKKSAggregator(), med, cfg["sigma"], s), "HE+DP"),
        "SMPC + norm-bound": (lambda s: SMPCAggregator(3, norm_bound=2 * med), "SMPC+NB"),
    }
    incompatible = [dict(combination="SecAgg + median/Krum/trimmed-mean", reason="robust rules need per-client updates; masked updates are uniformly random individually", status="NOT_RUN (incompatible by construction)"),
                    dict(combination="HE (CKKS) + Krum/median", reason="needs encrypted comparisons/sorting (deep circuits; not provided by additive CKKS aggregation used here)", status="NOT_RUN"),
                    dict(combination="plain + DP + robust (median)", reason="composition would need a separate privacy analysis for median sensitivity", status="NOT_RUN")]
    rows = []
    run_fl(FLConfig(rounds=2, local_epochs=2, seed=0), data, partition_iid(data.y_train, K, 0))   # warm-up run, excluded
    for s in range(cfg["seeds"]):
        # randomised configuration order within each seed (timing must not depend on run order)
        for name in [str(n) for n in np.random.default_rng(2000 + s).permutation(list(confs))]:
            mk, tag = confs[name]
            parts = partition_iid(data.y_train, K, s)
            a = mk(s)
            clean = run_fl(FLConfig(rounds=cfg["rounds"], local_epochs=2, seed=s), data, parts, aggregator=a)
            byz = run_fl(FLConfig(rounds=cfg["rounds"], local_epochs=2, seed=s), data, parts, aggregator=mk(s), client_hook=A.byzantine_hook("sign_flip", mal, 5.0, seed=s))
            bdr = _run_backdoor(data, s, BACKDOOR_LEVELS[cfg["backdoor"]], trig, aggregator=mk(s), rounds=cfg["rounds"])
            rows.append(dict(config=name, tag=tag, seed=s, acc=clean.final["acc"], f1=clean.final["f1"], train_wall_s=clean.totals["train_wall_s"], up_mb=clean.totals["up_bytes"] / 2 ** 20,
                             rounds_to_target=rounds_to_target(clean.history, "acc", cfg["target"]) or cfg["rounds"] + 1,
                             acc_under_signflip=byz.final["acc"], backdoor_asr=bdr.final["asr"], backdoor_clean_acc=bdr.final["clean_acc"]))
            print(f"  ablation seed {s} {name} done", flush=True)
    base = {r["seed"]: r for r in rows if r["config"] == "plain FL"}
    for r in rows:
        b = base[r["seed"]]
        r["train_overhead_pct"] = 100 * (r["train_wall_s"] - b["train_wall_s"]) / b["train_wall_s"]
        r["uplink_overhead_pct"] = 100 * (r["up_mb"] - b["up_mb"]) / b["up_mb"]
        r["acc_loss_pp"] = 100 * (b["acc"] - r["acc"])
    gl = load_json("fl_attack_003", "task_3_4_gradient_leakage"); mia = load_json("fl_attack_003", "task_3_3_membership_inference")
    gl_plain = next((r["success_rate"] for r in gl["rows"] if r["factor"].startswith("batch") and r["hidden"] == 32 and r["batch"] == 2), None) if gl else None
    gl_sec = next((r["success_rate"] for r in gl["rows"] if r["factor"].startswith("secure") and r["aggregated_clients"] == max(x["aggregated_clients"] for x in gl["rows"] if x["factor"].startswith("secure"))), None) if gl else None
    mia_none = next((s["auc"] for s in mia["summary"] if s["config"] == "none"), None) if mia else None
    summ = []
    for name, (_, tag) in confs.items():
        rr = [r for r in rows if r["config"] == name]; row = dict(config=name)
        for k in ("acc", "f1", "rounds_to_target", "train_overhead_pct", "uplink_overhead_pct", "acc_loss_pp", "acc_under_signflip", "backdoor_asr"):
            m, lo, hi = mean_ci([r[k] for r in rr]); row[k] = m; row[k + "_ci"] = f"[{lo:.3g}, {hi:.3g}]"
        hidden = tag in ("HE", "SA", "SMPC", "SA+DP", "HE+DP", "SMPC+NB")
        row["server_sees_individual_updates"] = not hidden
        row["gradient_leakage_success_B2"] = (gl_sec if hidden else gl_plain)
        row["gl_source"] = "task 3.4 (measured there): secure-aggregation K=max sum" if hidden else "task 3.4 (measured there): single client B=2"
        row["mia_auc_undefended_model"] = mia_none if "DP" not in tag else "see task 3.3 (DP configs)"
        summ.append(row)
    save_csv(STORY, "task_5_2_runs", rows); save_csv(STORY, "task_5_2_summary", summ); save_csv(STORY, "task_5_2_incompatible", incompatible)
    save_json(STORY, "task_5_2_ablation", dict(provenance=prov, summary=summ, incompatible=incompatible, median_honest_norm=med,
              notes=["MIA AUC of HE/SecAgg/SMPC global models equals the undefended model up to numerical error (aggregate identical) -> THEORETICAL carry-over, not re-measured here",
                     "DP rows report sigma only; epsilon is in task 3.3", "gradient-leakage numbers are copied from task 3.4, not re-measured in this ablation"]))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    names = [s["config"] for s in summ]
    ax[0].barh(names, [s["train_overhead_pct"] for s in summ]); ax[0].set_xlabel("training time overhead vs plain [%]"); ax[0].set_xscale("symlog", linthresh=10)
    ax[1].barh(names, [s["acc_under_signflip"] for s in summ]); ax[1].set_xlabel("accuracy under 20% sign-flip clients"); ax[1].set_xlim(0, 1)
    ax[2].barh(names, [s["backdoor_asr"] for s in summ]); ax[2].set_xlabel("backdoor ASR (medium intensity)")
    for a_ in ax[1:]:
        a_.set_yticklabels([])
    savefig(fig, STORY, "task_5_2_ablation", prov, f"{K} virtual clients, {cfg['rounds']} rounds, {cfg['seeds']} seeds")
    return dict(provenance=prov, summary=summ)
