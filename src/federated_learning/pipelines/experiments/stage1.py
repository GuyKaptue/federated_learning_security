"""Stage-1 experiments: federated preprocessing, calibration, and robustness / heterogeneity ablation grids.

All statistics use the *seed* (or split) as the unit of replication; confidence intervals are Student-t over seeds
(``common.mean_ci``), effect sizes are paired Cohen's d_z and Cliff's delta, p-values are exact Wilcoxon with Holm correction
inside each stated family. Every result is MEASURED on SYN-IDS-v1 (completely synthetic) or on the public four-site UCI
heart-disease cohorts; clients are virtual (one machine). No credentialed clinical repository (e.g. PhysioNet) is used.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from federated_learning import attacks as A
from federated_learning.aggregators.mean import WeightedMean
from federated_learning.aggregators.robust import CoordinateMedian, Krum, TrimmedMean
from federated_learning.data_partitioning.iid import partition_iid
from federated_learning.evaluation import statistics as S
from federated_learning.evaluation.calibration import calibration_report, predict_proba, reliability_curve
from federated_learning.pipelines.experiments import ext as E_
from federated_learning.pipelines.experiments.common import mean_ci, plt, provenance, save_csv, save_json, savefig, scale
from federated_learning.preprocessing.dataset import Dataset
from federated_learning.preprocessing.federated_scaler import FederatedScaler, pooled_stats, prepare_sites
from federated_learning.preprocessing.multisite_heart import load_sites
from federated_learning.servers.orchestrator import run_fl
from federated_learning.synthetic_data.generators.syn_ids_v1 import SynIDSConfig, generate_raw, make_dataset
from federated_learning.training.config import ALGOS, FLConfig

STORY = "fl_stage1"
FINAL = ("acc", "f1", "auc")


def _summ(rows: list[dict], keys: tuple, metrics: tuple) -> list[dict]:
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rr in groups.items():
        row = dict(zip(keys, g), n=len(rr))
        for m in metrics:
            mm, lo, hi = mean_ci([r[m] for r in rr])
            row[m] = mm; row[m + "_lo"] = lo; row[m + "_hi"] = hi
        out.append(row)
    return out


def _contrasts(rows: list[dict], family_keys: tuple, arm_key: str, ref: str, metric: str, seed_key: str = "seed") -> list[dict]:
    """Paired contrast of every arm against `ref` within each family cell; Holm over the arms of one cell."""
    out = []
    cells: dict = {}
    for r in rows:
        cells.setdefault(tuple(r[k] for k in family_keys), []).append(r)
    for cell, rr in cells.items():
        arms = sorted({r[arm_key] for r in rr} - {ref}, key=str)
        by = lambda a: np.array([x[metric] for x in sorted((x for x in rr if x[arm_key] == a), key=lambda x: x[seed_key])])  # noqa: E731
        base = by(ref)
        rec = []
        for a in arms:
            v = by(a)
            if len(v) != len(base) or len(v) < 3:
                continue
            pt = S.paired_tests(v, base)
            rec.append(dict(zip(family_keys, cell), arm=a, ref=ref, metric=metric, n=len(v), mean_diff=pt["mean_diff"], dz=pt["dz"],
                            cliffs_delta=S.cliffs_delta(v, base), wilcoxon_p=pt["wilcoxon_p"]))
        for r_, p in zip(rec, S.holm([r_["wilcoxon_p"] for r_ in rec])):
            r_["holm_p"] = p
        out += rec
    return out


# ------------------------------------------------------------------ 1. federated preprocessing
def _shifted_federation(seed: int, K: int, n_samples: int, shift: float = 0.4):
    """SYN-IDS-v1 raw features with a per-client affine measurement shift x_k = a_k * x + b_k (different 'sensors')."""
    X, y = generate_raw(SynIDSConfig(n_samples=n_samples, seed=0))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    rng = np.random.default_rng(seed + 31)
    sd = Xtr.std(0)
    a = np.exp(rng.normal(0, shift, size=(K, X.shape[1]))); b = rng.normal(0, 1.0, size=(K, X.shape[1])) * sd * shift * 2
    ptr = partition_iid(ytr, K, seed); pte = partition_iid(yte, K, seed + 1)
    tr = [(a[k] * Xtr[ptr[k]] + b[k], ytr[ptr[k]]) for k in range(K)]
    te = [(a[k] * Xte[pte[k]] + b[k], yte[pte[k]]) for k in range(K)]
    return tr, te


def _dataset_from_sites(Xtr, ytr, Xte, yte) -> tuple[Dataset, list]:
    sizes = np.cumsum([0] + [len(y) for y in ytr])
    parts = [np.arange(sizes[i], sizes[i + 1]) for i in range(len(ytr))]
    return Dataset(np.vstack(Xtr).astype(np.float32), np.concatenate(ytr).astype(np.int64), np.vstack(Xte).astype(np.float32),
                   np.concatenate(yte).astype(np.int64), {}), parts


PREP_MODES = ("pooled_mean", "federated", "local", "none")


def task_s1_1_federated_preprocessing() -> dict:
    n_seeds = scale(10, 2); rounds = scale(20, 3); K = 10; n_samples = scale(12000, 2500)
    heart_modes = ("pooled_median", "pooled_mean", "federated", "local", "none")
    cfg = dict(K_synthetic=K, rounds=rounds, seeds=n_seeds, n_samples=n_samples, modes=PREP_MODES, heart_modes=heart_modes,
               secagg=dict(threshold_frac=0.67, scale_bits=24, rounds=2), algo="fedavg", local_epochs=2,
               shift="per-client affine x_k = a_k*x + b_k, a_k=exp(N(0,0.4)), b_k=N(0,1)*sd*0.8")
    prov = provenance("FL-S1-001", cfg)
    rows, err_rows = [], []
    for s in range(n_seeds):
        tr, te = _shifted_federation(s, K, n_samples)
        Xtr = [t[0] for t in tr]; Xte = [t[0] for t in te]; ytr = [t[1] for t in tr]; yte = [t[1] for t in te]
        mu, sd = pooled_stats(Xtr)
        fs = FederatedScaler.fit(Xtr, graph_seed=s)
        err_rows.append(dict(dataset="synthetic-shifted", seed=s, max_abs_err_mean=float(np.abs(fs.mean_ - mu).max()),
                             max_abs_err_std=float(np.abs(fs.scale_ - sd - 1e-8).max()),
                             up_bytes=fs.stats["up_bytes"], down_bytes=fs.stats["down_bytes"], time_s=fs.stats["time_s"],
                             mean_range=float(np.ptp(mu)), n_features=len(mu)))
        for mode in PREP_MODES:
            ztr, zte = prepare_sites(Xtr, Xte, mode, **(dict(graph_seed=s) if mode == "federated" else {}))
            data, parts = _dataset_from_sites(ztr, ytr, zte, yte)
            r = run_fl(FLConfig(rounds=rounds, local_epochs=2, seed=s), data, parts)
            rows.append(dict(dataset="synthetic-shifted", seed=s, mode=mode, **{k: r.final[k] for k in FINAL}, loss=r.final["loss"]))
        print(f"  s1.1 synthetic seed {s} done", flush=True)
    for s in range(n_seeds):                                         # natural 4-site heart cohorts, one split per seed
        pm = load_sites(s, prep="pooled_median"); raw_err = None
        from federated_learning.preprocessing.multisite_heart import SITE_NAMES, _site_raw
        trs = []
        for n in SITE_NAMES:
            X, y = _site_raw(n)
            a, _b, c, _d = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
            trs.append(a)
        mu, sd = pooled_stats(trs); fs = FederatedScaler.fit(trs, graph_seed=s)
        raw_err = dict(dataset="heart-4sites", seed=s, max_abs_err_mean=float(np.abs(fs.mean_ - mu).max()), max_abs_err_std=float(np.abs(fs.scale_ - sd - 1e-8).max()),
                       up_bytes=fs.stats["up_bytes"], down_bytes=fs.stats["down_bytes"], time_s=fs.stats["time_s"], mean_range=float(np.ptp(mu)), n_features=len(mu))
        err_rows.append(raw_err)
        for mode in heart_modes:
            sd_ = pm if mode == "pooled_median" else load_sites(s, prep=mode)
            r = run_fl(FLConfig(rounds=rounds, local_epochs=2, seed=s), sd_, sd_.parts)
            rows.append(dict(dataset="heart-4sites", seed=s, mode=mode, **{k: r.final[k] for k in FINAL}, loss=r.final["loss"]))
        print(f"  s1.1 heart split {s} done", flush=True)
    summ = _summ(rows, ("dataset", "mode"), FINAL)
    tests = []
    for ref in ("pooled_mean", "local"):
        for m in FINAL:
            tests += [dict(t, family=f"vs {ref}") for t in _contrasts(rows, ("dataset",), "mode", ref, m)]
    save_csv(STORY, "s1_1_runs", rows); save_csv(STORY, "s1_1_stat_error", err_rows)
    save_csv(STORY, "s1_1_summary", summ); save_csv(STORY, "s1_1_tests", tests)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.3))
    for ax, ds in zip(axs, ("synthetic-shifted", "heart-4sites")):
        ms = [m for m in (heart_modes if ds == "heart-4sites" else PREP_MODES)]
        vals = [next(x for x in summ if x["dataset"] == ds and x["mode"] == m) for m in ms]
        ax.bar(range(len(ms)), [v["f1"] for v in vals], yerr=[[v["f1"] - v["f1_lo"] for v in vals], [v["f1_hi"] - v["f1"] for v in vals]], capsize=3)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=20, fontsize=7); ax.set_ylabel("test F1 (95% CI over seeds)"); ax.set_title(ds)
        ax.set_ylim(max(0, min(v["f1_lo"] for v in vals) - 0.05), 1)
    savefig(fig, STORY, "s1_1_preprocessing", prov, "FedAvg; preprocessing statistics from train rows only")
    out = dict(provenance=prov, summary=summ, stat_error=err_rows, tests=tests,
               caveat="the federated scaler reveals the aggregate (n, mean, variance) to the server; no DP; malicious clients could lie about their sums; "
                      "median imputation is not a sum, so 'federated' imputes with the global mean (compare pooled_mean, not pooled_median, for exactness)")
    save_json(STORY, "s1_1_federated_preprocessing", out)
    return out


# ------------------------------------------------------------------ 2. calibration
def task_s1_2_calibration() -> dict:
    n_seeds = scale(10, 2); rounds = scale(30, 3); K = 10
    cfg = dict(K=K, rounds=rounds, seeds=n_seeds, algos=ALGOS, datasets=["synthetic dirichlet(0.3)", "heart-4sites"],
               ece="15 equal-width bins on P(class 1)", brier="mean squared error of P(class 1)", lr=0.05, mu=0.05, local_epochs=2)
    prov = provenance("FL-S1-002", cfg)
    rows, curves = [], {}
    data = make_dataset(seed=0, n_samples=scale(12000, 2500))
    pooled: dict = {}
    for s in range(n_seeds):
        parts = E_._parts(data.y_train, K, "0.3", s)
        sd = load_sites(s)
        for a in ALGOS:
            for ds, d, pt in (("synthetic", data, parts), ("heart-4sites", sd, sd.parts)):
                r = run_fl(FLConfig(algo=a, rounds=rounds, seed=s, mu=0.05 if a == "fedprox" else 0.0, local_epochs=2), d, pt)
                cal = calibration_report(r.extras["model"], r.w, d.X_test, d.y_test)
                rows.append(dict(dataset=ds, algo=a, seed=s, **{k: r.final[k] for k in FINAL}, loss=r.final["loss"], **cal))
                p = predict_proba(r.extras["model"], r.w, d.X_test)
                pooled.setdefault((ds, a), []).append((d.y_test, p))
        print(f"  s1.2 seed {s} done", flush=True)
    for (ds, a), lst in pooled.items():
        curves[f"{ds}|{a}"] = reliability_curve(np.concatenate([y for y, _ in lst]), np.concatenate([p for _, p in lst]), 10)
    summ = _summ(rows, ("dataset", "algo"), ("acc", "f1", "auc", "brier", "ece"))
    tests = []
    for m in ("brier", "ece"):
        tests += _contrasts(rows, ("dataset",), "algo", "fedavg", m)
    save_csv(STORY, "s1_2_runs", rows); save_csv(STORY, "s1_2_summary", summ); save_csv(STORY, "s1_2_tests", tests)
    rel = [dict(dataset=k.split("|")[0], algo=k.split("|")[1], bin=i, count=c["count"][i], mean_p=c["mean_p"][i], pos_rate=c["pos_rate"][i])
           for k, c in curves.items() for i in range(len(c["count"]))]
    save_csv(STORY, "s1_2_reliability", rel)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, ds in zip(axs, ("synthetic", "heart-4sites")):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        for a in ALGOS:
            c = curves[f"{ds}|{a}"]; ok = np.asarray(c["count"]) >= 5
            ax.plot(np.asarray(c["mean_p"])[ok], np.asarray(c["pos_rate"])[ok], "o-", ms=3, label=a)
        ax.set_xlabel("predicted P(class 1)"); ax.set_ylabel("observed positive rate"); ax.set_title(f"{ds}: reliability (pooled over seeds)"); ax.legend(fontsize=7)
    savefig(fig, STORY, "s1_2_reliability", prov, "bins with >= 5 predictions")
    out = dict(provenance=prov, summary=summ, tests=tests,
               caveat="calibration is measured at the final round on the held-out test set without any post-hoc recalibration; the heart test set has ~280 rows, so ECE is noisy there")
    save_json(STORY, "s1_2_calibration", out)
    return out


# ------------------------------------------------------------------ 3. ablation grids
K_AB = 20


def _agg(name: str, frac: float):
    f = max(1, int(round(frac * K_AB)))
    return {"mean": lambda: WeightedMean(), "median": lambda: CoordinateMedian(),
            "trimmed_mean": lambda: TrimmedMean(min(0.45, frac + 0.05)), "krum": lambda: Krum(f, 1)}[name]()


def _fl(data, parts, seed, rounds, aggregator=None, hook=None, **kw):
    return run_fl(FLConfig(rounds=rounds, local_epochs=kw.pop("local_epochs", 1), seed=seed, **kw), data, parts, aggregator=aggregator, client_hook=hook)


def task_s1_3_ablation_grids() -> dict:
    n_seeds = scale(10, 3); rounds = scale(20, 3)
    fracs = scale([0.0, 0.05, 0.10, 0.20, 0.30, 0.40], [0.0, 0.2]); alphas = scale([0.1, 0.3, 0.5, 1.0], [0.3, 1.0])
    parts_ = scale([0.1, 0.25, 0.5, 1.0], [0.25, 1.0]); hiddens = scale([8, 16, 32, 64, 128], [8, 32])
    cfg = dict(rounds=rounds, seeds=n_seeds, K_malicious_axis=K_AB, malicious_fractions=fracs, attack="sign_flip scale 5 (model-update level)",
               aggregators=["mean", "median", "trimmed_mean", "krum"], dirichlet_alphas=alphas, K_alpha_axis=10, participation=parts_, K_participation_axis=K_AB,
               hidden=hiddens, algo_axes=dict(alpha=list(ALGOS)))
    prov = provenance("FL-S1-003", cfg)
    data = make_dataset(seed=0, n_samples=scale(12000, 2500))
    rows = []
    for s in range(n_seeds):
        # (a) malicious fraction x aggregator
        iid = partition_iid(data.y_train, K_AB, s)
        for fr in fracs:
            mal = set(range(int(round(fr * K_AB))))
            for an in cfg["aggregators"]:
                hook = A.byzantine_hook("sign_flip", mal, 5.0, seed=s) if mal else None
                r = _fl(data, iid, s, rounds, _agg(an, fr), hook)
                rows.append(dict(axis="malicious_fraction", level=fr, arm=an, seed=s, **{k: r.final[k] for k in FINAL}))
        # (b) heterogeneity (Dirichlet alpha) x strategy, and x robust aggregator (clean)
        for al in alphas:
            pa = E_._parts(data.y_train, 10, str(al), s)
            for a in ALGOS:
                r = _fl(data, pa, s, rounds, algo=a, mu=0.05 if a == "fedprox" else 0.0, local_epochs=2)
                rows.append(dict(axis="dirichlet_alpha", level=al, arm=a, seed=s, **{k: r.final[k] for k in FINAL}))
            for an in ("median", "trimmed_mean"):
                r = _fl(data, pa, s, rounds, _agg(an, 0.1), local_epochs=2)
                rows.append(dict(axis="dirichlet_alpha", level=al, arm=f"fedavg+{an}", seed=s, **{k: r.final[k] for k in FINAL}))
        # (c) participation: clean mean, and 20% sign-flip with median
        for pr in parts_:
            r = _fl(data, iid, s, rounds, participation=pr)
            rows.append(dict(axis="participation", level=pr, arm="mean_clean", seed=s, **{k: r.final[k] for k in FINAL}))
            mal = set(range(4))
            r = _fl(data, iid, s, rounds, _agg("median", 0.2), A.byzantine_hook("sign_flip", mal, 5.0, seed=s), participation=pr)
            rows.append(dict(axis="participation", level=pr, arm="median_20pct_signflip", seed=s, **{k: r.final[k] for k in FINAL}))
        # (d) model size
        for h in hiddens:
            r = _fl(data, iid, s, rounds, hidden=h)
            rows.append(dict(axis="hidden_units", level=h, arm="fedavg", seed=s, **{k: r.final[k] for k in FINAL},
                             n_params=r.totals["n_params"], comm_mb=r.totals["comm_mb"], train_s=r.totals["train_wall_s"]))
        print(f"  s1.3 seed {s} done", flush=True)
    summ = _summ(rows, ("axis", "level", "arm"), FINAL)
    tests = []
    for m in ("acc",):
        tests += [dict(t, family="aggregator vs mean") for t in _contrasts([r for r in rows if r["axis"] == "malicious_fraction"], ("axis", "level"), "arm", "mean", m)]
        tests += [dict(t, family="strategy vs fedavg") for t in _contrasts([r for r in rows if r["axis"] == "dirichlet_alpha"], ("axis", "level"), "arm", "fedavg", m)]
        tests += [dict(t, family="attacked median vs clean mean") for t in _contrasts([r for r in rows if r["axis"] == "participation"], ("axis", "level"), "arm", "mean_clean", m)]
    save_csv(STORY, "s1_3_runs", rows); save_csv(STORY, "s1_3_summary", summ); save_csv(STORY, "s1_3_tests", tests)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.3))
    for ax, axis, xl in zip(axs, ("malicious_fraction", "dirichlet_alpha", "participation", "hidden_units"),
                            ("malicious fraction", "Dirichlet alpha", "participation rate", "hidden units")):
        for arm in dict.fromkeys(x["arm"] for x in summ if x["axis"] == axis):
            pts = sorted((x for x in summ if x["axis"] == axis and x["arm"] == arm), key=lambda x: x["level"])
            ax.errorbar([p["level"] for p in pts], [p["acc"] for p in pts], yerr=[[p["acc"] - p["acc_lo"] for p in pts], [p["acc_hi"] - p["acc"] for p in pts]],
                        marker="o", ms=3, capsize=2, label=arm)
        ax.set_xlabel(xl); ax.legend(fontsize=6)
        if axis == "dirichlet_alpha":
            ax.set_xscale("log")
    axs[0].set_ylabel("test accuracy (95% CI over seeds)")
    savefig(fig, STORY, "s1_3_ablation_grids", prov, f"{n_seeds} seeds")
    out = dict(provenance=prov, summary=summ, tests=tests,
               caveat=f"{n_seeds} seeds per cell (exact Wilcoxon cannot reach p<0.05 with fewer than 6 pairs, so conclusions rest on effect sizes and intervals as well); Krum requires n >= 2f+3 (K=20, f<=8 satisfied); attack is a single model-update behaviour (sign flip, scale 5); "
                      "participation < 1 makes the per-round malicious share random")
    save_json(STORY, "s1_3_ablation_grids", out)
    return out
