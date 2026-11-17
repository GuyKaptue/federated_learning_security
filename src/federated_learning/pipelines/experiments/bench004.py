"""FL-BENCH-004: FedAvg / FedProx / SCAFFOLD / FedNova benchmark on synthetic non-IID data."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from federated_learning.evaluation import statistics as S
from federated_learning.attacks.byzantine import byzantine_hook
from federated_learning.privacy.homomorphic import CKKSAggregator
from federated_learning.data_partitioning.metrics import label_skew_stat
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset
from federated_learning.data_partitioning.dirichlet import partition_dirichlet
from federated_learning.data_partitioning.iid import partition_iid
from federated_learning.aggregators.mean import WeightedMean
from federated_learning.training.config import ALGOS, FLConfig
from federated_learning.training.centralized import centralized_baseline
from federated_learning.evaluation.metrics import rounds_to_target
from federated_learning.clients.local_training import local_train
from federated_learning.servers.orchestrator import run_fl
from federated_learning.training.models import build_model, get_flat, set_flat
from federated_learning.monitoring.resources import ResourceMonitor
from federated_learning.pipelines.experiments.common import load_json, mean_ci, plt, provenance, save_csv, save_json, savefig, scale

STORY = "fl_bench_004"

ROUNDS = scale(30, 4)

TARGET_ACC = 0.85

MU_DEFAULT = 0.05

LABEL = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fednova": "FedNova", "proposed": "Proposed framework (FedAvg + CKKS HE aggregation)"}

def _parts(data, kind, K, seed):
    if kind == "iid":
        return partition_iid(data.y_train, K, seed)
    return partition_dirichlet(data.y_train, K, float(kind.replace("dir", "")), seed)

def _extra(model, w, data):
    set_flat(model, w); model.eval()
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(data.X_test)), 1)[:, 1].numpy()
    y = data.y_test
    bins = np.linspace(0, 1, 11); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return dict(pr_auc=float(average_precision_score(y, p)), ece=float(ece))

def exec_run(data, algo, kind, K, seed, E=2, part=1.0, mu=MU_DEFAULT, hetero=False, aggregator=None, hook=None, rounds=ROUNDS) -> dict:
    parts = _parts(data, kind, K, seed)
    cfg = FLConfig(algo=algo, rounds=rounds, local_epochs=E, participation=part, mu=mu if algo == "fedprox" else 0.0, hetero_epochs=hetero, seed=seed)
    with ResourceMonitor(0.1) as mon:
        r = run_fl(cfg, data, parts, aggregator=aggregator, client_hook=hook)
    res = mon.summary()
    row = dict(algo=algo, condition=kind, clients=K, seed=seed, local_epochs=E, participation=part, mu=cfg.mu, rounds=rounds, **{k: r.final[k] for k in ("acc", "precision", "recall", "f1", "auc", "loss", "specificity")},
               rounds_to_target=rounds_to_target(r.history, "acc", TARGET_ACC), comm_mb=r.totals["comm_mb"], comm_mb_per_round=r.totals["comm_mb"] / rounds,
               train_wall_s=r.totals["train_wall_s"], cpu_time_s=res["cpu_time_s"], rss_peak_mb=res["rss_peak_mb"], skew=label_skew_stat(data.y_train, parts))
    row.update(_extra(r.extras["model"], r.w, data))
    row["_curve"] = [h["acc"] for h in r.history]
    row["_w"] = r.w
    return row

def _clean(rows):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

def _curves_fig(curves: dict, title, story_name, prov, note=""):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for k, v in curves.items():
        a = np.array(v); m = a.mean(0); sd = a.std(0)
        ax.plot(range(1, len(m) + 1), m, label=k); ax.fill_between(range(1, len(m) + 1), m - sd, m + sd, alpha=0.15)
    ax.set_xlabel("communication round"); ax.set_ylabel("test accuracy"); ax.set_title(title); ax.legend(fontsize=7)
    return savefig(fig, STORY, story_name, prov, note)

def task_4_1_fedavg() -> dict:
    cfg = dict(rounds=ROUNDS, seeds=scale(5, 2), conditions=["iid", "dir0.3"], participation=[0.2, 0.5, 1.0], epochs=[1, 2, 5], K=10)
    prov = provenance("FL-BENCH-004.1", cfg)
    data = make_dataset(seed=0)
    # validation 1: weighted aggregation identity
    rng = np.random.default_rng(0); D = rng.normal(size=(7, 50)).astype(np.float32); W = rng.integers(5, 90, 7).astype(float)
    manual = sum(W[i] / W.sum() * D[i] for i in range(7))
    v1 = float(np.abs(WeightedMean().aggregate(D, W) - manual).max())
    # validation 2: one client, one round == plain local SGD on that client's data (FedAvg degenerates to local training)
    parts1 = [np.arange(len(data.y_train))]
    r = run_fl(FLConfig(rounds=1, local_epochs=2, seed=3, eval_every=1), data, parts1)
    m = build_model(data.n_features, 32, seed=3); gen = torch.Generator().manual_seed(3 * 1_000_003 + 1 * 1009 + 0)
    w_ref, _ = local_train(m, get_flat(m), torch.from_numpy(data.X_train), torch.from_numpy(data.y_train), 2, 0.05, 32, gen)
    v2 = float(np.abs(r.w - w_ref).max())
    rows, curves = [], {}
    for kind in cfg["conditions"]:
        for s in range(cfg["seeds"]):
            x = exec_run(data, "fedavg", kind, cfg["K"], s); rows.append(x); curves.setdefault(kind, []).append(x["_curve"])
    sweep = []
    for part in cfg["participation"]:
        for s in range(min(3, cfg["seeds"])):
            x = exec_run(data, "fedavg", "dir0.3", cfg["K"], s, part=part); sweep.append(dict(factor="participation", value=part, **{k: x[k] for k in ("seed", "acc", "f1", "rounds_to_target", "comm_mb")}))
    for E in cfg["epochs"]:
        for s in range(min(3, cfg["seeds"])):
            x = exec_run(data, "fedavg", "dir0.3", cfg["K"], s, E=E); sweep.append(dict(factor="local_epochs", value=E, **{k: x[k] for k in ("seed", "acc", "f1", "rounds_to_target", "comm_mb")}))
    cen = centralized_baseline(data, epochs=ROUNDS)
    summ = []
    for kind in cfg["conditions"]:
        rr = [r for r in rows if r["condition"] == kind]
        row = dict(condition=kind, skew=float(np.mean([r["skew"] for r in rr])))
        for k in ("acc", "f1", "loss", "comm_mb"):
            m_, lo, hi = mean_ci([r[k] for r in rr]); row[k] = m_; row[k + "_ci"] = f"[{lo:.4f}, {hi:.4f}]"
        summ.append(row)
    save_csv(STORY, "task_4_1_runs", _clean(rows)); save_csv(STORY, "task_4_1_sweeps", sweep); save_csv(STORY, "task_4_1_summary", summ)
    save_json(STORY, "task_4_1_fedavg", dict(provenance=prov, validation=dict(weighted_aggregation_max_abs_diff=v1, single_client_equals_local_sgd_max_abs_diff=v2),
                                           summary=summ, sweeps=sweep, centralized_reference=cen, curves={k: np.mean(v, 0).tolist() for k, v in curves.items()}))
    _curves_fig({k: v for k, v in curves.items()}, "FedAvg: IID vs label-skew (mean ± sd)", "task_4_1_fedavg", prov, f"{cfg['K']} clients, E=2, {cfg['seeds']} seeds; centralized acc {cen['acc']:.3f}")
    return dict(provenance=prov, summary=summ, v1=v1, v2=v2)

def task_4_2_fedprox() -> dict:
    cfg = dict(mus=[0.0, 0.001, 0.01, 0.1, 1.0], seeds=scale(5, 2), kind="dir0.1", K=10, hetero_epochs=True, E=5)
    prov = provenance("FL-BENCH-004.2", cfg)
    data = make_dataset(seed=0)
    # validation: mu=0 must equal FedAvg bit-for-bit; proximal gradient matches finite differences; drift shrinks with mu
    a = exec_run(data, "fedavg", cfg["kind"], 10, 0, E=2); b = exec_run(data, "fedprox", cfg["kind"], 10, 0, E=2, mu=0.0)
    v_eq = float(np.abs(a["_w"] - b["_w"]).max())
    rng = np.random.default_rng(1); w0 = rng.normal(size=20); wg = rng.normal(size=20); mu = 0.7
    f = lambda w: 0.5 * mu * np.sum((w - wg) ** 2)
    num = np.array([(f(w0 + 1e-6 * e) - f(w0 - 1e-6 * e)) / 2e-6 for e in np.eye(20)])
    v_grad = float(np.abs(num - mu * (w0 - wg)).max())
    parts = _parts(data, cfg["kind"], 10, 0); m = build_model(data.n_features, 32, seed=0); w_g = get_flat(m)
    X, y = torch.from_numpy(data.X_train[parts[0]]), torch.from_numpy(data.y_train[parts[0]])
    drift = []
    for mu_ in (0.0, 0.01, 0.1, 1.0, 10.0):
        wl, _ = local_train(m, w_g, X, y, 5, 0.05, 32, torch.Generator().manual_seed(0), mu=mu_)
        drift.append(dict(mu=mu_, local_drift_norm=float(np.linalg.norm(wl - w_g))))
    rows, curves = [], {}
    for mu_ in cfg["mus"]:
        for s in range(cfg["seeds"]):
            x = exec_run(data, "fedprox", cfg["kind"], cfg["K"], s, E=cfg["E"], mu=mu_, hetero=True); rows.append(x); curves.setdefault(f"mu={mu_}", []).append(x["_curve"])
    summ = []
    for mu_ in cfg["mus"]:
        rr = [r for r in rows if r["mu"] == mu_]; row = dict(mu=mu_)
        for k in ("acc", "f1", "loss"):
            m_, lo, hi = mean_ci([r[k] for r in rr]); row[k] = m_; row[k + "_ci"] = f"[{lo:.4f}, {hi:.4f}]"
        row["rounds_to_target"] = float(np.mean([r["rounds_to_target"] if r["rounds_to_target"] else ROUNDS + 1 for r in rr])); summ.append(row)
    save_csv(STORY, "task_4_2_runs", _clean(rows)); save_csv(STORY, "task_4_2_summary", summ); save_csv(STORY, "task_4_2_local_drift", drift)
    save_json(STORY, "task_4_2_fedprox", dict(provenance=prov, validation=dict(mu0_equals_fedavg_max_abs_w_diff=v_eq, prox_gradient_fd_max_err=v_grad), local_drift=drift, summary=summ,
                                            curves={k: np.mean(v, 0).tolist() for k, v in curves.items()}))
    _curves_fig(curves, "FedProx: effect of mu (label skew 0.1, heterogeneous epochs)", "task_4_2_fedprox", prov, f"{cfg['seeds']} seeds")
    return dict(provenance=prov, summary=summ)

def task_4_3_scaffold() -> dict:
    cfg = dict(seeds=scale(5, 2), kind="dir0.1", K=10, E=5)
    prov = provenance("FL-BENCH-004.3", cfg)
    data = make_dataset(seed=0); parts = _parts(data, cfg["kind"], 10, 0)
    # validation: full participation => server control variate c equals the mean of client control variates
    r = run_fl(FLConfig(algo="scaffold", rounds=5, local_epochs=cfg["E"], seed=0), data, parts)
    c_loc = np.stack([r.extras["c_local"][k] for k in range(10)])
    v_mean = float(np.abs(r.extras["c_global"] - c_loc.mean(0)).max())
    v_nonzero = float(np.linalg.norm(r.extras["c_global"]))
    # drift measurement via callback
    def drift_run(algo, s):
        out = []
        cb = lambda rnd, w, ids, D, t: out.append(float(np.linalg.norm(D - D.mean(0), axis=1).mean()))
        x = run_fl(FLConfig(algo=algo, rounds=ROUNDS, local_epochs=cfg["E"], seed=s), data, _parts(data, cfg["kind"], 10, s), round_callback=cb)
        return out, x
    drift = {}
    for algo in ("fedavg", "scaffold"):
        drift[algo] = np.mean([drift_run(algo, s)[0] for s in range(min(3, cfg["seeds"]))], axis=0).tolist()
    rows, curves = [], {}
    for algo in ("fedavg", "scaffold"):
        for s in range(cfg["seeds"]):
            x = exec_run(data, algo, cfg["kind"], 10, s, E=cfg["E"]); rows.append(x); curves.setdefault(algo, []).append(x["_curve"])
    part_rows = []
    for algo in ("fedavg", "scaffold"):
        for s in range(min(3, cfg["seeds"])):
            x = exec_run(data, algo, cfg["kind"], 10, s, E=cfg["E"], part=0.3); part_rows.append(dict(algo=algo, participation=0.3, seed=s, acc=x["acc"], f1=x["f1"]))
    summ = []
    for algo in ("fedavg", "scaffold"):
        rr = [r for r in rows if r["algo"] == algo]; row = dict(algo=algo)
        for k in ("acc", "f1", "comm_mb", "train_wall_s"):
            m_, lo, hi = mean_ci([r[k] for r in rr]); row[k] = m_; row[k + "_ci"] = f"[{lo:.4g}, {hi:.4g}]"
        summ.append(row)
    ratio = summ[1]["comm_mb"] / summ[0]["comm_mb"]
    save_csv(STORY, "task_4_3_runs", _clean(rows)); save_csv(STORY, "task_4_3_summary", summ); save_csv(STORY, "task_4_3_partial_participation", part_rows)
    save_json(STORY, "task_4_3_scaffold", dict(provenance=prov, validation=dict(c_equals_mean_ci_full_participation_max_abs_diff=v_mean, c_global_norm=v_nonzero, comm_ratio_scaffold_over_fedavg=ratio),
                                             drift_per_round=drift, summary=summ, partial=part_rows, curves={k: np.mean(v, 0).tolist() for k, v in curves.items()}))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, v in curves.items():
        ax[0].plot(range(1, ROUNDS + 1), np.mean(v, 0), label=k)
    ax[0].set_xlabel("round"); ax[0].set_ylabel("test accuracy"); ax[0].set_title("SCAFFOLD vs FedAvg (label skew 0.1)"); ax[0].legend()
    for k, v in drift.items():
        ax[1].plot(range(1, len(v) + 1), v, label=k)
    ax[1].set_xlabel("round"); ax[1].set_ylabel("mean ||delta_k - mean delta|| (client drift proxy)"); ax[1].set_title("Update dispersion"); ax[1].legend()
    savefig(fig, STORY, "task_4_3_scaffold", prov, f"{cfg['seeds']} seeds, E={cfg['E']}")
    return dict(provenance=prov, summary=summ)

def _quadratic_toy(algo: str, taus, a, b, p, lr=0.01, rounds=3000):
    """f_k(w) = 0.5 a_k ||w - b_k||^2, deterministic local GD for tau_k steps. Returns final global w."""
    w = np.zeros_like(b[0])
    for _ in range(rounds):
        deltas = []
        for k in range(len(a)):
            wk = w.copy()
            for _ in range(taus[k]):
                wk = wk - lr * a[k] * (wk - b[k])
            deltas.append(wk - w)
        deltas = np.stack(deltas); taus_ = np.asarray(taus, float)
        if algo == "fedavg":
            w = w + (p[:, None] * deltas).sum(0)
        else:
            w = w + (p * taus_).sum() * (p[:, None] * deltas / taus_[:, None]).sum(0)
    return w

def task_4_4_fednova() -> dict:
    cfg = dict(seeds=scale(5, 2), kind="dir0.3", K=10, E=5)
    prov = provenance("FL-BENCH-004.4", cfg)
    # validation A (closed-form): objective inconsistency with unequal local steps
    rng = np.random.default_rng(0)
    a = np.array([1.0, 2.0, 0.5]); b = [rng.normal(size=4) for _ in range(3)]; p = np.array([0.4, 0.3, 0.3])
    w_star = sum(p[k] * a[k] * b[k] for k in range(3)) / sum(p[k] * a[k] for k in range(3))
    toy = []
    for taus in ([5, 5, 5], [2, 10, 20], [1, 5, 40]):
        for algo in ("fedavg", "fednova"):
            w = _quadratic_toy(algo, taus, a, b, p)
            toy.append(dict(local_steps=str(taus), algo=algo, dist_to_global_optimum=float(np.linalg.norm(w - w_star))))
    # validation B: equal tau => FedNova == FedAvg exactly (equal-size IID partition)
    data = make_dataset(seed=0)
    ra = exec_run(data, "fedavg", "iid", 10, 0, E=2, rounds=3); rb = exec_run(data, "fednova", "iid", 10, 0, E=2, rounds=3)
    v_eq = float(np.abs(ra["_w"] - rb["_w"]).max())
    rows, curves = [], {}
    for hetero in (False, True):
        for algo in ("fedavg", "fednova"):
            for s in range(cfg["seeds"]):
                x = exec_run(data, algo, cfg["kind"], cfg["K"], s, E=cfg["E"], hetero=hetero); x["hetero_epochs"] = hetero; rows.append(x)
                curves.setdefault(f"{algo}|hetero={hetero}", []).append(x["_curve"])
    summ = []
    for hetero in (False, True):
        for algo in ("fedavg", "fednova"):
            rr = [r for r in rows if r["algo"] == algo and r["hetero_epochs"] == hetero]; row = dict(algo=algo, hetero_epochs=hetero)
            for k in ("acc", "f1", "loss"):
                m_, lo, hi = mean_ci([r[k] for r in rr]); row[k] = m_; row[k + "_ci"] = f"[{lo:.4f}, {hi:.4f}]"
            summ.append(row)
    save_csv(STORY, "task_4_4_runs", _clean(rows)); save_csv(STORY, "task_4_4_summary", summ); save_csv(STORY, "task_4_4_quadratic_toy", toy)
    save_json(STORY, "task_4_4_fednova", dict(provenance=prov, validation=dict(equal_tau_fednova_equals_fedavg_max_abs_w_diff=v_eq), quadratic_toy=toy, summary=summ,
                                            curves={k: np.mean(v, 0).tolist() for k, v in curves.items()}))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    xs = np.arange(3); w_ = 0.35
    for i, algo in enumerate(("fedavg", "fednova")):
        ax[0].bar(xs + (i - 0.5) * w_, [t["dist_to_global_optimum"] for t in toy if t["algo"] == algo], w_, label=algo)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([str(t) for t in ([5, 5, 5], [2, 10, 20], [1, 5, 40])]); ax[0].set_xlabel("local steps per client"); ax[0].set_ylabel("||w - w*||")
    ax[0].set_yscale("symlog", linthresh=1e-6); ax[0].set_title("Quadratic toy: stationary-point bias (closed form)"); ax[0].legend()
    for k, v in curves.items():
        ax[1].plot(range(1, ROUNDS + 1), np.mean(v, 0), label=k)
    ax[1].set_xlabel("round"); ax[1].set_ylabel("test accuracy"); ax[1].set_title("Real task, label skew 0.3"); ax[1].legend(fontsize=6)
    savefig(fig, STORY, "task_4_4_fednova", prov, f"{cfg['seeds']} seeds")
    return dict(provenance=prov, summary=summ, toy=toy)

CONDITIONS = [dict(name="iid", kind="iid", K=10, E=2, part=1.0, seeds=scale(10, 3)), dict(name="dir0.3", kind="dir0.3", K=10, E=2, part=1.0, seeds=scale(10, 3)),
              dict(name="dir0.1", kind="dir0.1", K=10, E=2, part=1.0, seeds=scale(5, 2)), dict(name="dir0.3_50clients", kind="dir0.3", K=50, E=2, part=1.0, seeds=scale(5, 2)),
              dict(name="dir0.3_E5", kind="dir0.3", K=10, E=5, part=1.0, seeds=scale(5, 2)), dict(name="dir0.3_part0.3", kind="dir0.3", K=10, E=2, part=0.3, seeds=scale(5, 2)),
              dict(name="dir0.3_signflip10pct", kind="dir0.3", K=10, E=2, part=1.0, seeds=scale(5, 2), attack=True)]

def task_4_5_comparison() -> dict:
    cfg = dict(conditions=CONDITIONS, algos=list(ALGOS), mu=MU_DEFAULT, rounds=ROUNDS, target_acc=TARGET_ACC, lr=0.05)
    prov = provenance("FL-BENCH-004.5", cfg)
    data = make_dataset(seed=0)
    cen = centralized_baseline(data, epochs=ROUNDS)
    rows, curves = [], {}
    for c in CONDITIONS:
        for s in range(c["seeds"]):
            for algo in list(ALGOS) + (["proposed"] if c["name"] in ("iid", "dir0.3") else []):
                hook = byzantine_hook("sign_flip", {0}, 5.0, seed=s) if c.get("attack") else None
                x = exec_run(data, "fedavg" if algo == "proposed" else algo, c["kind"], c["K"], s, E=c["E"], part=c["part"], hook=hook,
                             aggregator=CKKSAggregator() if algo == "proposed" else None)
                x["algo"] = algo; x["condition"] = c["name"]
                rows.append(x); curves.setdefault(f"{c['name']}|{algo}", []).append(x["_curve"])
        print(f"  comparison condition {c['name']} done", flush=True)
    save_csv(STORY, "task_4_5_runs", _clean(rows))
    summ = []
    for c in CONDITIONS:
        for algo in list(ALGOS) + ["proposed"]:
            rr = [r for r in rows if r["condition"] == c["name"] and r["algo"] == algo]
            if not rr:
                continue
            row = dict(condition=c["name"], algo=algo, n_seeds=len(rr))
            for k in ("acc", "precision", "recall", "f1", "auc", "pr_auc", "specificity", "ece", "loss", "comm_mb", "train_wall_s", "cpu_time_s"):
                m_, lo, hi = mean_ci([r[k] for r in rr]); row[k] = m_; row[k + "_ci"] = f"[{lo:.4f}, {hi:.4f}]"
            rt = [r["rounds_to_target"] if r["rounds_to_target"] else ROUNDS + 1 for r in rr]
            row["rounds_to_target_mean"] = float(np.mean(rt)); row["target_reached_frac"] = float(np.mean([r["rounds_to_target"] is not None for r in rr]))
            summ.append(row)
    save_csv(STORY, "task_4_5_summary", summ)
    save_json(STORY, "task_4_5_comparison", dict(provenance=prov, summary=summ, centralized_reference=cen, curves={k: np.mean(v, 0).tolist() for k, v in curves.items()},
                                               proposed_definition="FedAvg local training + CKKS homomorphic aggregation (chapter module 3); chapter modules 1,2,4 (TLS/auth/IDPS) NOT implemented",
                                               note="non-IID = Dirichlet label skew; clients virtual"))
    for cond in ("iid", "dir0.3"):
        _curves_fig({LABEL[a]: curves[f"{cond}|{a}"] for a in list(ALGOS) + ["proposed"] if f"{cond}|{a}" in curves}, f"Convergence — {cond}", f"task_4_5_convergence_{cond}", prov, f"10 clients, E=2, {CONDITIONS[0]['seeds']} seeds, {ROUNDS} rounds")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    conds = [c["name"] for c in CONDITIONS]; w_ = 0.2
    for i, algo in enumerate(ALGOS):
        ax[0].bar(np.arange(len(conds)) + (i - 1.5) * w_, [next((s["f1"] for s in summ if s["condition"] == c and s["algo"] == algo), 0) for c in conds], w_, label=LABEL[algo])
        ax[1].bar(np.arange(len(conds)) + (i - 1.5) * w_, [next((s["comm_mb"] for s in summ if s["condition"] == c and s["algo"] == algo), 0) for c in conds], w_)
        ax[2].bar(np.arange(len(conds)) + (i - 1.5) * w_, [next((s["train_wall_s"] for s in summ if s["condition"] == c and s["algo"] == algo), 0) for c in conds], w_)
    for a_, yl in zip(ax, ("F1 (mean of seeds)", "communication [MiB, whole run]", "training wall time [s]")):
        a_.set_xticks(range(len(conds))); a_.set_xticklabels(conds, rotation=60, fontsize=7); a_.set_ylabel(yl)
    ax[0].set_ylim(0.5, 1); ax[0].legend(fontsize=7)
    savefig(fig, STORY, "task_4_5_summary_bars", prov, "all conditions; see CSV for seeds per condition")
    return dict(provenance=prov, summary=summ)

def task_4_6_statistics() -> dict:
    prov = provenance("FL-BENCH-004.6", dict(source="task_4_5_runs", tests=["Friedman", "Wilcoxon signed-rank", "paired t", "Holm", "Cohen dz", "bootstrap CI"]))
    import pandas as pd
    from federated_learning.pipelines.experiments.common import out_dir
    df = pd.read_csv(out_dir(STORY) / "task_4_5_runs.csv")
    df["rt"] = df["rounds_to_target"].fillna(ROUNDS + 1)
    out = dict(provenance=prov, conditions={})
    tables = []
    for cond in ("iid", "dir0.3"):
        sub = df[(df.condition == cond) & (df.algo.isin(ALGOS))]
        for metric in ("f1", "acc", "rt", "comm_mb"):
            samples = {a: sub[sub.algo == a].sort_values("seed")[metric].to_numpy() for a in ALGOS}
            n = len(next(iter(samples.values())))
            if n < 3:
                continue
            fr = S.friedman(samples); pw = S.pairwise_table(samples)
            cis = {a: dict(zip(("mean", "t_lo", "t_hi"), S.t_ci(v))) | dict(zip(("boot_mean", "boot_lo", "boot_hi"), S.bootstrap_ci(v))) for a, v in samples.items()}
            out["conditions"].setdefault(cond, {})[metric] = dict(n=n, friedman=fr, pairwise=pw, ci=cis, min_attainable_wilcoxon_p=S.min_wilcoxon_p(n))
            for r in pw:
                tables.append({"condition": cond, "metric": metric, "a": r["a"], "b": r["b"], "n": r["n"], "mean_diff": r["mean_diff"], "cohen_dz": r["dz"],
                               "wilcoxon_p": r["wilcoxon_p"], "ttest_p": r["ttest_p"], "holm_p_wilcoxon": r["holm_p"], "significant_holm_0.05": bool(r["holm_p"] < 0.05)})
    save_csv(STORY, "task_4_6_pairwise_tests", tables)
    out["unit_of_replication"] = "seed (independent data split + initialisation); rounds are NOT treated as replicates"
    out["pairing_justification"] = "paired by seed: identical partition and initialisation across algorithms"
    save_json(STORY, "task_4_6_statistics", out)
    f1 = {c: out["conditions"][c]["f1"]["ci"] for c in out["conditions"] if "f1" in out["conditions"][c]}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for a_, cond in zip(ax, ("iid", "dir0.3")):
        ci = f1.get(cond, {})
        for i, (alg, v) in enumerate(ci.items()):
            a_.errorbar(i, v["mean"], yerr=[[v["mean"] - v["t_lo"]], [v["t_hi"] - v["mean"]]], fmt="o", capsize=4)
        a_.set_xticks(range(len(ci))); a_.set_xticklabels([LABEL[k] for k in ci], rotation=20, fontsize=8); a_.set_title(f"F1, 95% t-CI — {cond}")
    ax[0].set_ylabel("F1 (test)")
    savefig(fig, STORY, "task_4_6_confidence_intervals", prov, "10 seeds per algorithm")
    return out

def task_4_7_leaderboard() -> dict:
    prov = provenance("FL-BENCH-004.7", dict(source="task_4_5_summary", ranking="mean rank across 5 dimensions, equal weights, 4 algorithms"))
    summ = load_json(STORY, "task_4_5_comparison")["summary"]
    boards = {}
    attack = {s["algo"]: s for s in summ if s["condition"] == "dir0.3_signflip10pct"}
    for cond in ("iid", "dir0.3"):
        rows = []
        for algo in list(ALGOS) + ["proposed"]:
            s = next((x for x in summ if x["condition"] == cond and x["algo"] == algo), None)
            if not s:
                continue
            comm_to_target = s["comm_mb"] / ROUNDS * s["rounds_to_target_mean"]
            rows.append(dict(algorithm=LABEL[algo], key=algo, utility_f1=s["f1"], convergence_rounds=s["rounds_to_target_mean"], communication_mb_to_target=comm_to_target,
                             robustness_acc_under_signflip=(attack[algo]["acc"] if algo in attack else None), training_cost_s=s["train_wall_s"]))
        core = [r for r in rows if r["key"] in ALGOS]
        for dim, hi in (("utility_f1", True), ("convergence_rounds", False), ("communication_mb_to_target", False), ("robustness_acc_under_signflip", True), ("training_cost_s", False)):
            order = sorted(core, key=lambda r: r[dim], reverse=hi)
            for rk, r in enumerate(order, 1):
                r["rank_" + dim] = rk
        for r in core:
            r["mean_rank"] = float(np.mean([r[k] for k in r if k.startswith("rank_")]))
        for r in rows:
            if r["key"] == "proposed":
                r["mean_rank"] = None
                r["note"] = "not ranked: robustness dimension NOT_MEASURED for this row; compare dimension-wise (it is FedAvg + CKKS aggregation)"
        boards[cond] = sorted(rows, key=lambda r: (r["mean_rank"] is None, r["mean_rank"]))
        save_csv(STORY, f"task_4_7_leaderboard_{cond}", boards[cond])
    save_json(STORY, "task_4_7_leaderboard", dict(provenance=prov, boards=boards, methodology=["each dimension ranked over the 4 core algorithms (1=best)", "overall = unweighted mean rank (a different weighting can change the order)",
                                                  "convergence = mean rounds to reach 85% test accuracy (censored at rounds+1)", "communication = per-round MiB x rounds-to-target",
                                                  "robustness = accuracy with 1/10 sign-flip attacker (scale 5), label-skew 0.3", "no statistical significance implied by rank; see task 4.6"]))
    return dict(provenance=prov, boards=boards)
