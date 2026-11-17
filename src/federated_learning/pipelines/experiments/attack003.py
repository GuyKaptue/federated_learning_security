"""FL-ATTACK-003: offensive security benchmark on SYNTHETIC data only (controlled, local, simulated clients)."""
from __future__ import annotations

import time
import numpy as np
from federated_learning import attacks as A
from federated_learning.privacy.homomorphic import CKKSAggregator
from federated_learning.privacy.secure_aggregation import SecAggAggregator
from federated_learning.privacy.smpc import SMPCAggregator
from federated_learning.preprocessing.dataset import Dataset
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset
from federated_learning.data_partitioning.iid import partition_iid
from federated_learning.aggregators.defenses import AnomalyFiltered, ClippedMean
from federated_learning.aggregators.robust import CoordinateMedian, Krum, TrimmedMean
from federated_learning.aggregators.mean import WeightedMean
from federated_learning.training.config import FLConfig
from federated_learning.servers.orchestrator import run_fl
from federated_learning.training.models import build_model
from federated_learning.pipelines.experiments.common import load_json, mean_ci, plt, provenance, save_csv, save_json, savefig, scale

STORY = "fl_attack_003"

K, ROUNDS = 20, scale(20, 3)

SEEDS = scale(3, 1)

BACKDOOR_LEVELS = {"low": dict(mal_frac=0.10, poison_frac=0.2, boost=1.0), "medium": dict(mal_frac=0.20, poison_frac=0.5, boost=3.0),
                   "high": dict(mal_frac=0.30, poison_frac=0.8, boost=5.0)}

TRIGGER_CANDIDATES = {"f0-3=+4": ((0, 1, 2, 3), 4.0), "f0-3=-4": ((0, 1, 2, 3), -4.0), "f0-7=+2.5": (tuple(range(8)), 2.5), "f0-1=+6": ((0, 1), 6.0)}

def _bd_setup(data, seed, level, trig):
    parts = partition_iid(data.y_train, K, seed)
    n_mal = max(1, int(round(level["mal_frac"] * K)))
    mal = set(range(n_mal))
    rng = np.random.default_rng(seed + 99)
    cd = {}
    for k in mal:
        X, y, _ = A.poison_client_data(data.X_train[parts[k]], data.y_train[parts[k]], level["poison_frac"], 0, rng, *trig)
        cd[k] = (X, y)
    return parts, mal, cd

def _run_backdoor(data, seed, level, trig, aggregator=None, rounds=ROUNDS):
    parts, mal, cd = _bd_setup(data, seed, level, trig)
    ev = lambda m, w: A.backdoor_metrics(m, w, data.X_test, data.y_test, 0, *trig)
    hook = A.boost_hook(mal, level["boost"]) if level["boost"] != 1.0 else None
    return run_fl(FLConfig(rounds=rounds, local_epochs=1, seed=seed), data, parts, aggregator=aggregator, client_hook=hook, client_data=cd, eval_fn=ev)

def _clean_run(data, seed, trig, aggregator=None, rounds=ROUNDS, hook=None):
    parts = partition_iid(data.y_train, K, seed)
    ev = lambda m, w: A.backdoor_metrics(m, w, data.X_test, data.y_test, 0, *trig)
    return run_fl(FLConfig(rounds=rounds, local_epochs=1, seed=seed), data, parts, aggregator=aggregator, client_hook=hook, eval_fn=ev)

def task_3_1_backdoor() -> dict:
    cfg = dict(K=K, rounds=ROUNDS, seeds=SEEDS, levels=BACKDOOR_LEVELS, triggers=list(TRIGGER_CANDIDATES))
    prov = provenance("FL-ATTACK-003.1", cfg)
    data = make_dataset(seed=0)
    # trigger selection: attacker prefers a trigger the CLEAN model does not already obey (low natural ASR)
    sel = []
    for name, trig in TRIGGER_CANDIDATES.items():
        r = _clean_run(data, 0, trig)
        sel.append(dict(trigger=name, clean_model_asr=r.final["asr"], clean_acc=r.final["clean_acc"]))
    best = min(sel, key=lambda s: s["clean_model_asr"])["trigger"]
    trig = TRIGGER_CANDIDATES[best]
    rows, curves = [], {}
    for s in range(SEEDS):
        c = _clean_run(data, s, trig)
        rows.append(dict(level="none(clean FL)", seed=s, clean_acc=c.final["clean_acc"], asr=c.final["asr"], asr_untriggered=c.final["asr_untriggered"]))
        for lv, L in BACKDOOR_LEVELS.items():
            r = _run_backdoor(data, s, L, trig)
            rows.append(dict(level=lv, seed=s, **L, clean_acc=r.final["clean_acc"], asr=r.final["asr"], asr_untriggered=r.final["asr_untriggered"]))
            if s == 0:
                curves[lv] = [h["asr"] for h in r.history]
        if s == 0:
            curves["none"] = [h["asr"] for h in c.history]
    grid = []
    for mf in (0.05, 0.1, 0.2, 0.3):
        for pf in (0.2, 0.5, 0.8):
            r = _run_backdoor(data, 0, dict(mal_frac=mf, poison_frac=pf, boost=1.0), trig)
            grid.append(dict(mal_client_frac=mf, poison_sample_frac=pf, asr=r.final["asr"], clean_acc=r.final["clean_acc"]))
    summ = []
    for lv in ["none(clean FL)"] + list(BACKDOOR_LEVELS):
        rr = [r for r in rows if r["level"] == lv]
        a, alo, ahi = mean_ci([r["asr"] for r in rr]); c_, clo, chi = mean_ci([r["clean_acc"] for r in rr])
        summ.append(dict(level=lv, asr=a, asr_ci=f"[{alo:.3f}, {ahi:.3f}]", clean_acc=c_, clean_acc_ci=f"[{clo:.3f}, {chi:.3f}]", kpi_asr_lt_5pct=bool(a < 0.05)))
    save_csv(STORY, "task_3_1_runs", rows); save_csv(STORY, "task_3_1_summary", summ); save_csv(STORY, "task_3_1_grid", grid); save_csv(STORY, "task_3_1_trigger_selection", sel)
    save_json(STORY, "task_3_1_backdoor", dict(provenance=prov, trigger=dict(name=best, idx=trig[0], value=trig[1]), selection=sel, summary=summ, grid=grid, curves=curves))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for k_, v in curves.items():
        ax[0].plot(range(1, len(v) + 1), v, label=k_)
    ax[0].set_xlabel("round"); ax[0].set_ylabel("attack success rate (ASR)"); ax[0].set_title("Backdoor ASR over rounds (seed 0)"); ax[0].legend()
    ax[1].bar([s["level"] for s in summ], [s["asr"] for s in summ]); ax[1].axhline(0.05, color="r", ls="--", label="KPI target 5%"); ax[1].set_ylabel("final ASR (mean of seeds)")
    ax[1].tick_params(axis="x", rotation=30); ax[1].set_title("ASR by intensity"); ax[1].legend()
    M = np.array([[g["asr"] for g in grid if g["mal_client_frac"] == mf] for mf in (0.05, 0.1, 0.2, 0.3)])
    im = ax[2].imshow(M, vmin=0, vmax=1, cmap="viridis", origin="lower"); ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["0.2", "0.5", "0.8"])
    ax[2].set_yticks(range(4)); ax[2].set_yticklabels(["5%", "10%", "20%", "30%"]); ax[2].set_xlabel("poisoned sample fraction"); ax[2].set_ylabel("malicious clients")
    ax[2].set_title("ASR grid (no boosting)"); fig.colorbar(im, ax=ax[2]); ax[2].grid(False)
    savefig(fig, STORY, "task_3_1_backdoor", prov, f"{K} virtual clients, {ROUNDS} rounds, {SEEDS} seeds; trigger {best}")
    return dict(provenance=prov, summary=summ)

def _byz_aggs(f: int, frac: float):
    return {"mean": lambda: WeightedMean(), "median": lambda: CoordinateMedian(),
            "trimmed_mean": lambda: TrimmedMean(min(0.45, frac + 0.05)), "krum": lambda: Krum(f, 1), "multi_krum": lambda: Krum(f, K - f)}

def task_3_2_byzantine() -> dict:
    behaviors, fracs = ["random", "sign_flip", "scale", "drift"], [0.1, 0.2, 0.3]
    cfg = dict(K=K, rounds=ROUNDS, seeds=SEEDS, behaviors=behaviors, fracs=fracs, scale=5.0)
    prov = provenance("FL-ATTACK-003.2", cfg)
    data = make_dataset(seed=0)
    rows = []
    for s in range(SEEDS):
        parts = partition_iid(data.y_train, K, s)
        for frac in [0.0] + fracs:
            f = int(round(frac * K)); mal = set(range(f))
            for an, mk in _byz_aggs(max(f, 1), frac).items():
                for bh in (["none"] if frac == 0 else behaviors):
                    hook = None if bh == "none" else A.byzantine_hook(bh, mal, 5.0, seed=s)
                    agg = mk()
                    try:
                        r = run_fl(FLConfig(rounds=ROUNDS, local_epochs=1, seed=s), data, parts, aggregator=agg, client_hook=hook)
                        row = dict(seed=s, mal_frac=frac, behavior=bh, aggregator=an, acc=r.final["acc"], loss=r.final["loss"], f1=r.final["f1"],
                                   agg_time_ms_per_round=1e3 * r.totals["agg_time_s"] / ROUNDS, status="ok")
                        if bh != "none" and s == 0:
                            row["curve"] = [h["acc"] for h in r.history]
                    except Exception as e:
                        row = dict(seed=s, mal_frac=frac, behavior=bh, aggregator=an, status="failed", error=repr(e)[:120])
                    rows.append(row)
        print(f"  byzantine seed {s} done", flush=True)
    # client contribution diagnostics (mean aggregator, sign-flip 20%)
    parts = partition_iid(data.y_train, K, 0); mal = set(range(4)); diag = {}
    def cb(rnd, w, ids, D, taus):
        if rnd == ROUNDS:
            honest = [i for i, k in enumerate(ids) if k not in mal]; bad = [i for i, k in enumerate(ids) if k in mal]
            hm = D[honest].mean(0)
            cs = lambda v: float(v @ hm / (np.linalg.norm(v) * np.linalg.norm(hm) + 1e-12))
            diag.update(honest_norm=float(np.linalg.norm(D[honest], axis=1).mean()), malicious_norm=float(np.linalg.norm(D[bad], axis=1).mean()),
                        honest_cos_to_honest_mean=float(np.mean([cs(D[i]) for i in honest])), malicious_cos_to_honest_mean=float(np.mean([cs(D[i]) for i in bad])))
    run_fl(FLConfig(rounds=ROUNDS, local_epochs=1, seed=0), data, parts, client_hook=A.byzantine_hook("sign_flip", mal, 5.0), round_callback=cb)
    ok = [r for r in rows if r["status"] == "ok"]
    summ = []
    for frac in fracs:
        for bh in behaviors:
            for an in _byz_aggs(1, 0.1):
                rr = [r for r in ok if r["mal_frac"] == frac and r["behavior"] == bh and r["aggregator"] == an]
                if rr:
                    m, lo, hi = mean_ci([r["acc"] for r in rr])
                    summ.append(dict(mal_frac=frac, behavior=bh, aggregator=an, acc=m, acc_ci=f"[{lo:.3f}, {hi:.3f}]", loss=float(np.mean([r["loss"] for r in rr])),
                                     agg_ms=float(np.mean([r["agg_time_ms_per_round"] for r in rr]))))
    clean = [dict(aggregator=an, clean_acc=float(np.mean([r["acc"] for r in ok if r["aggregator"] == an and r["behavior"] == "none"]))) for an in _byz_aggs(1, 0.1)]
    fails = [r for r in rows if r["status"] != "ok"]
    save_csv(STORY, "task_3_2_runs", [{k: v for k, v in r.items() if k != "curve"} for r in rows]); save_csv(STORY, "task_3_2_summary", summ); save_csv(STORY, "task_3_2_clean_cost", clean)
    curves = {f"{r['aggregator']}|{r['behavior']}|{r['mal_frac']}": r["curve"] for r in rows if r.get("curve")}
    save_json(STORY, "task_3_2_byzantine", dict(provenance=prov, summary=summ, clean=clean, diagnostics=diag, failures=fails, curves=curves,
                                                krum_assumption="Krum/Multi-Krum require n >= 2f+3; here n=20, f<=6 satisfied"))
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    for a_, bh in zip(axs, behaviors):
        for an in _byz_aggs(1, 0.1):
            pts = [(s_["mal_frac"], s_["acc"]) for s_ in summ if s_["behavior"] == bh and s_["aggregator"] == an]
            a_.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=an)
        a_.set_title(f"{bh}"); a_.set_xlabel("malicious client fraction")
    axs[0].set_ylabel("final test accuracy"); axs[0].legend(fontsize=7)
    savefig(fig, STORY, "task_3_2_byzantine", prov, f"{K} virtual clients, {ROUNDS} rounds, {SEEDS} seeds, attack scale 5")
    return dict(provenance=prov, summary=summ, diagnostics=diag)

def _mia_run(seed: int, aggregator_factory, rounds: int, n_mem: int, hidden: int, base: Dataset):
    rng = np.random.default_rng(seed)
    ti = rng.permutation(len(base.y_train))[: 2 * n_mem]
    te = rng.permutation(len(base.y_test))
    pool_out, pool_sh = te[: n_mem], te[n_mem: 3 * n_mem]
    def build(idx_train):
        return Dataset(base.X_train[idx_train], base.y_train[idx_train], base.X_test, base.y_test, base.meta)
    tgt_idx, sh_idx_in = ti[:n_mem], pool_sh[:n_mem]
    sh_idx_out = pool_sh[n_mem: 2 * n_mem]
    tgt = Dataset(base.X_train[tgt_idx], base.y_train[tgt_idx], base.X_test, base.y_test, base.meta)
    shd = Dataset(base.X_test[sh_idx_in], base.y_test[sh_idx_in], base.X_test, base.y_test, base.meta)
    cfg = FLConfig(rounds=rounds, local_epochs=5, hidden=hidden, seed=seed, eval_every=rounds)
    rt = run_fl(cfg, tgt, partition_iid(tgt.y_train, 10, seed), aggregator=aggregator_factory(seed))
    rs = run_fl(cfg, shd, partition_iid(shd.y_train, 10, seed), aggregator=aggregator_factory(seed + 1000))
    m = build_model(base.n_features, hidden)
    F = lambda w, X, y: A.per_sample_features(m, w, X, y)
    f_mem, f_out = F(rt.w, tgt.X_train, tgt.y_train), F(rt.w, base.X_test[pool_out], base.y_test[pool_out])
    s_in, s_out = F(rs.w, shd.X_train, shd.y_train), F(rs.w, base.X_test[sh_idx_out], base.y_test[sh_idx_out])
    sc = A.shadow_attack_scores(s_in, s_out, np.r_[f_mem, f_out], seed)
    sh_scores = A.shadow_attack_scores(s_in, s_out, np.r_[s_in, s_out], seed)
    sh_m = A.mia_metrics(sh_scores[: len(s_in)], sh_scores[len(s_in):])
    n1 = len(f_mem)
    met = A.mia_metrics(sc[:n1], sc[n1:], threshold=sh_m["threshold"])               # threshold fixed on SHADOW data
    base_m = A.mia_metrics(-f_mem[:, 0], -f_out[:, 0])
    train_acc = float(f_mem[:, 4].mean()); test_acc = float(F(rt.w, base.X_test, base.y_test)[:, 4].mean())
    return dict(shadow_attack=met, loss_baseline=base_m, train_acc=train_acc, test_acc=test_acc, gap=train_acc - test_acc, utility=rt.final["acc"])

def task_3_3_membership_inference() -> dict:
    from opacus.accountants import RDPAccountant
    cfg = dict(n_members=scale(1000, 200), hidden=128, rounds=scale(40, 4), clients=10, local_epochs=5, clip=0.5, sigmas=[0.1, 0.3, 1.0], seeds=SEEDS, delta=1e-5)
    prov = provenance("FL-ATTACK-003.3", cfg)
    base = make_dataset(seed=0)
    confs = {"none": lambda s: WeightedMean(), "clip_only(C=0.5)": lambda s: ClippedMean(0.5, 0.0, s)}
    for sg in cfg["sigmas"]:
        confs[f"dp_sigma={sg}"] = (lambda sg_: (lambda s: ClippedMean(0.5, sg_, s)))(sg)
    rows, rocs = [], {}
    for name, mk in confs.items():
        eps = None
        if name.startswith("dp_sigma"):
            acc = RDPAccountant(); sg = float(name.split("=")[1])
            for _ in range(cfg["rounds"]):
                acc.step(noise_multiplier=sg, sample_rate=1.0)
            eps = float(acc.get_epsilon(cfg["delta"]))
        for s in range(cfg["seeds"]):
            r = _mia_run(s, mk, cfg["rounds"], cfg["n_members"], cfg["hidden"], base)
            sa, lb = r["shadow_attack"], r["loss_baseline"]
            rows.append(dict(config=name, seed=s, eps=eps, auc=sa["auc"], auc_loss_baseline=lb["auc"], tpr_at_fpr_1pct=sa["tpr_at_fpr_1pct"], advantage=sa["max_advantage"],
                             precision=sa["precision"], recall=sa["recall"], tpr=sa["tpr"], fpr=sa["fpr"], train_acc=r["train_acc"], test_acc=r["test_acc"], gap=r["gap"], utility_acc=r["utility"]))
            if s == 0:
                rocs[name] = sa["_roc"]
        print(f"  mia {name} done", flush=True)
    summ = []
    for name in confs:
        rr = [r for r in rows if r["config"] == name]
        row = dict(config=name, eps=rr[0]["eps"])
        for k in ("auc", "auc_loss_baseline", "tpr_at_fpr_1pct", "advantage", "precision", "recall", "gap", "utility_acc"):
            m, lo, hi = mean_ci([r[k] for r in rr]); row[k] = m; row[k + "_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        row["kpi_auc_lt_0.60"] = bool(row["auc"] < 0.60)
        summ.append(row)
    save_csv(STORY, "task_3_3_runs", rows); save_csv(STORY, "task_3_3_summary", summ)
    save_json(STORY, "task_3_3_membership_inference", dict(provenance=prov, summary=summ, rocs=rocs,
              accounting_note="epsilon from Opacus RDPAccountant for the Gaussian mechanism with sample_rate=1.0 (all clients every round), "
                              "user-level, delta=1e-5; the noise implemented is sigma*C/m on the mean of clipped updates; formal guarantee only covers the mechanism as modelled"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, (fpr, tpr) in rocs.items():
        ax[0].plot(fpr, tpr, label=name)
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate"); ax[0].set_title("Membership inference ROC (seed 0)"); ax[0].legend(fontsize=7)
    ax[1].bar([s["config"] for s in summ], [s["auc"] for s in summ]); ax[1].axhline(0.6, color="r", ls="--", label="KPI target 0.60"); ax[1].axhline(0.5, color="gray", ls=":")
    ax[1].set_ylabel("attack AUC (shadow-model attack)"); ax[1].tick_params(axis="x", rotation=30); ax[1].set_title("AUC by privacy configuration"); ax[1].legend()
    savefig(fig, STORY, "task_3_3_membership_inference", prov, f"{cfg['n_members']} members, hidden {cfg['hidden']}, {cfg['rounds']} rounds, {cfg['seeds']} seeds")
    return dict(provenance=prov, summary=summ)

def _gl_trials(model, w, data, B, trials, seed0, K_agg=1, noise=0.0, iters=300):
    rng = np.random.default_rng(seed0)
    res = []
    for t in range(trials):
        Bt = B * K_agg
        idx = rng.choice(len(data.y_test), Bt, replace=False)
        X, y = data.X_test[idx], data.y_test[idx]
        gs = [A.model_grad(model, w, X[i * B:(i + 1) * B], y[i * B:(i + 1) * B]) for i in range(K_agg)]
        g = np.mean(gs, axis=0)
        if noise > 0:
            g = g + rng.normal(0, noise * np.linalg.norm(g) / np.sqrt(g.size), g.shape).astype(np.float32)
        d = X.shape[1]
        Xh = A.dlg_attack(model, w, g, Bt, d, labels=y, iters=iters, seed=seed0 * 31 + t)
        res.append(A.match_and_score(X, Xh))
    return dict(success_rate=float(np.mean([r["success_rate"] for r in res])), mean_rel_err=float(np.mean([r["mean_rel_err"] for r in res])),
                mse=float(np.mean([r["mse"] for r in res])), feat_acc=float(np.mean([r["feat_acc"] for r in res])))

def task_3_4_gradient_leakage() -> dict:
    trials = scale(10, 2)
    cfg = dict(batch_sizes=scale([1, 2, 4, 8, 16], [1, 4]), hiddens=scale([8, 32, 128], [8, 32]), trials=trials, agg_K=scale([1, 2, 5, 10], [1, 2]), noise=scale([0.0, 0.01, 0.1, 0.5, 1.0], [0.0, 0.5]),
               success_rel_err=0.1, iters=300, activation="sigmoid", labels="known to attacker (strong)")
    prov = provenance("FL-ATTACK-003.4", cfg)
    data = make_dataset(seed=0); parts = partition_iid(data.y_train, 10, 0)
    models = {}
    for h in cfg["hiddens"]:
        r = run_fl(FLConfig(rounds=5, local_epochs=1, hidden=h, act="sigmoid", seed=0), data, parts)
        models[h] = (build_model(data.n_features, h, "sigmoid"), r.w, r.final["acc"])
    rows = []
    for h in cfg["hiddens"]:
        m, w, acc = models[h]
        for B in cfg["batch_sizes"]:
            t0 = time.perf_counter()
            r = _gl_trials(m, w, data, B, trials, 7, iters=cfg["iters"])
            rows.append(dict(factor="batch/architecture", hidden=h, n_params=int(w.size), batch=B, aggregated_clients=1, dp_noise_rel=0.0, **r, model_acc=acc, seconds=time.perf_counter() - t0))
        print(f"  gl hidden={h} done", flush=True)
    m, w, acc = models[cfg["hiddens"][1]]
    for Kc in cfg["agg_K"]:
        r = _gl_trials(m, w, data, 2, trials, 11, K_agg=Kc, iters=cfg["iters"])
        rows.append(dict(factor="secure-aggregation(sum of K clients, B=2 each)", hidden=cfg["hiddens"][1], n_params=int(w.size), batch=2 * Kc, aggregated_clients=Kc, dp_noise_rel=0.0, **r, model_acc=acc))
    for nz in cfg["noise"]:
        r = _gl_trials(m, w, data, 2, trials, 13, noise=nz, iters=cfg["iters"])
        rows.append(dict(factor="gaussian-noise(rel. to ||g||), B=2", hidden=cfg["hiddens"][1], n_params=int(w.size), batch=2, aggregated_clients=1, dp_noise_rel=nz, **r, model_acc=acc))
    # exact analytic inversion for B=1 (no optimisation) and under noise
    an = []
    rng = np.random.default_rng(5)
    for nz in cfg["noise"]:
        errs = []
        for t in range(trials * 3):
            i = rng.integers(len(data.y_test)); X, y = data.X_test[i:i + 1], data.y_test[i:i + 1]
            g = A.model_grad(m, w, X, y)
            if nz > 0:
                g = g + rng.normal(0, nz * np.linalg.norm(g) / np.sqrt(g.size), g.shape).astype(np.float32)
            xa = A.analytic_first_layer_inversion(g, X.shape[1], cfg["hiddens"][1])
            errs.append(np.linalg.norm(xa - X[0]) / np.linalg.norm(X[0]))
        an.append(dict(noise_rel=nz, success_rate=float(np.mean(np.array(errs) <= 0.1)), median_rel_err=float(np.median(errs))))
    save_csv(STORY, "task_3_4_runs", rows); save_csv(STORY, "task_3_4_analytic", an)
    save_json(STORY, "task_3_4_gradient_leakage", dict(provenance=prov, rows=rows, analytic=an,
              notes=dict(he="HE/secure aggregation hide individual gradients from the SERVER only; a malicious client/key-holder or an observer of the plaintext global-model update still sees what the protocol outputs",
                         ssim="NOT_APPLICABLE (tabular features, not images)", success_def="relative L2 reconstruction error <= 0.1 after Hungarian matching")))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for h in cfg["hiddens"]:
        rr = [r for r in rows if r["factor"] == "batch/architecture" and r["hidden"] == h]
        ax[0].plot([r["batch"] for r in rr], [r["success_rate"] for r in rr], "o-", label=f"hidden={h} (P={rr[0]['n_params']})")
    ax[0].axhline(0.10, color="r", ls="--"); ax[0].set_xscale("log", base=2); ax[0].set_xlabel("batch size"); ax[0].set_ylabel("reconstruction success rate"); ax[0].set_title("Leakage vs batch / model size"); ax[0].legend(fontsize=7)
    rr = [r for r in rows if r["factor"].startswith("secure")]
    ax[1].plot([r["aggregated_clients"] for r in rr], [r["success_rate"] for r in rr], "o-"); ax[1].axhline(0.10, color="r", ls="--"); ax[1].set_xlabel("clients summed (B=2 each)")
    ax[1].set_ylabel("success rate"); ax[1].set_title("Effect of aggregating gradients")
    rr = [r for r in rows if r["factor"].startswith("gaussian")]
    ax[2].plot([r["dp_noise_rel"] for r in rr], [r["success_rate"] for r in rr], "o-", label="DLG B=2"); ax[2].plot([a["noise_rel"] for a in an], [a["success_rate"] for a in an], "s--", label="analytic B=1")
    ax[2].axhline(0.10, color="r", ls="--"); ax[2].set_xlabel("noise level ||noise||/||g||"); ax[2].set_ylabel("success rate"); ax[2].set_title("Effect of gradient noise"); ax[2].legend()
    savefig(fig, STORY, "task_3_4_gradient_leakage", prov, f"{trials} trials/point, sigmoid MLP, labels known")
    return dict(provenance=prov, rows=rows, analytic=an)

def _median_norm(data, seed=0):
    parts = partition_iid(data.y_train, K, seed)
    out = []
    run_fl(FLConfig(rounds=2, local_epochs=1, seed=seed), data, parts, round_callback=lambda r, w, ids, D, t: out.append(float(np.median(np.linalg.norm(D, axis=1)))))
    return float(np.median(out))

def task_3_5_countermeasures() -> dict:
    cfg = dict(K=K, rounds=ROUNDS, seeds=SEEDS)
    prov = provenance("FL-ATTACK-003.5", cfg)
    data = make_dataset(seed=0)
    med = _median_norm(data)
    bd = load_json(STORY, "task_3_1_backdoor")
    trig = (tuple(bd["trigger"]["idx"]), bd["trigger"]["value"]) if bd else TRIGGER_CANDIDATES["f0-3=+4"]
    rows = []
    # -- backdoor
    defs = {"none(mean)": lambda f: WeightedMean(), "clip(C=median norm)": lambda f: ClippedMean(med), "clip+noise(sigma=0.5)": lambda f: ClippedMean(med, 0.5),
            "median": lambda f: CoordinateMedian(), "trimmed_mean(0.3)": lambda f: TrimmedMean(0.3), "multi-krum": lambda f: Krum(f, K - f),
            "anomaly_filter": lambda f: AnomalyFiltered(), "smpc_norm_bound(2x median)": lambda f: SMPCAggregator(3, norm_bound=2 * med),
            "secagg(mean)": lambda f: SecAggAggregator(), "he_ckks(mean)": lambda f: CKKSAggregator()}
    for lvname in ("medium", "high"):
        L = BACKDOOR_LEVELS[lvname]; f = max(1, int(round(L["mal_frac"] * K)))
        for dn, mk in defs.items():
            for s in range(SEEDS):
                r = _run_backdoor(data, s, L, trig, aggregator=mk(f))
                rows.append(dict(attack=f"backdoor-{lvname}", defense=dn, seed=s, metric="asr", value=r.final["asr"], clean_acc=r.final["clean_acc"]))
        print(f"  countermeasures backdoor-{lvname} done", flush=True)
    # -- byzantine
    for bh in ("sign_flip", "scale", "random"):
        f = 4; mal = set(range(f))
        for dn, mk in defs.items():
            if dn.startswith("clip+noise"):
                continue
            for s in range(SEEDS):
                parts = partition_iid(data.y_train, K, s)
                r = run_fl(FLConfig(rounds=ROUNDS, local_epochs=1, seed=s), data, parts, aggregator=mk(f), client_hook=A.byzantine_hook(bh, mal, 5.0, seed=s))
                rows.append(dict(attack=f"byzantine-{bh}-20%", defense=dn, seed=s, metric="acc", value=r.final["acc"], clean_acc=r.final["acc"]))
        print(f"  countermeasures byzantine-{bh} done", flush=True)
    # -- MIA and gradient leakage: reuse measured results of tasks 3.3 / 3.4
    mia = load_json(STORY, "task_3_3_membership_inference"); gl = load_json(STORY, "task_3_4_gradient_leakage")
    reuse = []
    if mia:
        for s_ in mia["summary"]:
            reuse.append(dict(attack="membership-inference", defense=s_["config"], metric="auc", value=s_["auc"], clean_acc=s_["utility_acc"], eps=s_["eps"]))
    if gl:
        for r in gl["rows"]:
            if r["factor"].startswith("secure"):
                reuse.append(dict(attack="gradient-leakage(B=2/client)", defense=f"secure-aggregation K={r['aggregated_clients']}", metric="success_rate", value=r["success_rate"], clean_acc=r["model_acc"]))
            if r["factor"].startswith("gaussian"):
                reuse.append(dict(attack="gradient-leakage(B=2)", defense=f"gradient-noise {r['dp_noise_rel']}", metric="success_rate", value=r["success_rate"], clean_acc=r["model_acc"]))
    summ = []
    for key in sorted({(r["attack"], r["defense"]) for r in rows}):
        rr = [r for r in rows if (r["attack"], r["defense"]) == key]
        m, lo, hi = mean_ci([r["value"] for r in rr])
        summ.append(dict(attack=key[0], defense=key[1], metric=rr[0]["metric"], value=m, ci=f"[{lo:.3f}, {hi:.3f}]", clean_acc=float(np.mean([r["clean_acc"] for r in rr]))))
    summ += reuse
    save_csv(STORY, "task_3_5_runs", rows); save_csv(STORY, "task_3_5_summary", summ)
    save_json(STORY, "task_3_5_countermeasures", dict(provenance=prov, summary=summ, median_honest_update_norm=med, trigger=trig,
              scope_notes=["secagg/HE return the plain mean: they give NO protection against poisoning/Byzantine clients (the server cannot inspect individual updates)",
                           "robust aggregators need per-update access => incompatible with secagg/HE; SMPC norm-bound is the compatible alternative but only bounds norms",
                           "clipping/noise do not hide gradients from an honest-but-curious server that sees the individual update unless combined with secagg"]))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    for a_, atk, yl in ((axs[0], "backdoor-medium", "ASR (lower is better)"), (axs[1], "byzantine-sign_flip-20%", "test accuracy (higher is better)")):
        ss = [s_ for s_ in summ if s_["attack"] == atk]
        a_.barh([s_["defense"] for s_ in ss], [s_["value"] for s_ in ss]); a_.set_xlabel(yl); a_.set_title(atk); a_.tick_params(axis="y", labelsize=7)
    savefig(fig, STORY, "task_3_5_countermeasures", prov, f"{K} virtual clients, {ROUNDS} rounds, {SEEDS} seeds")
    return dict(provenance=prov, summary=summ)

def task_3_6_risk_analysis() -> dict:
    prov = provenance("FL-ATTACK-003.6", dict(sources="tasks 3.1-3.5 JSON"))
    bd, bz, mia, gl, cm = (load_json(STORY, n) for n in ("task_3_1_backdoor", "task_3_2_byzantine", "task_3_3_membership_inference", "task_3_4_gradient_leakage", "task_3_5_countermeasures"))
    if not all([bd, bz, mia, gl, cm]):
        raise RuntimeError("run tasks 3.1-3.5 first")
    def best(attack_prefix, higher_better):
        ss = [s for s in cm["summary"] if s["attack"].startswith(attack_prefix)]
        return (max if higher_better else min)(ss, key=lambda s: s["value"]) if ss else None
    asr_high = next(s for s in bd["summary"] if s["level"] == "high")["asr"]; asr_none = next(s for s in bd["summary"] if s["level"].startswith("none"))["asr"]
    bd_best = best("backdoor-high", False)
    bz_rows = [s for s in bz["summary"] if s["behavior"] == "sign_flip" and s["mal_frac"] == 0.3]
    bz_mean = next(s["acc"] for s in bz_rows if s["aggregator"] == "mean"); bz_best = max(bz_rows, key=lambda s: s["acc"])
    mia_none = next(s for s in mia["summary"] if s["config"] == "none"); mia_best = min(mia["summary"], key=lambda s: s["auc"])
    gl_b1 = max((r for r in gl["rows"] if r["batch"] == 1 and r["factor"] == "batch/architecture"), key=lambda r: r["success_rate"])
    gl_big = [r for r in gl["rows"] if r["factor"] == "batch/architecture" and r["success_rate"] > 0.1]
    sec_rows = [r for r in gl["rows"] if r["factor"].startswith("secure")]
    matrix = [
        dict(attack="Backdoor (data poisoning + boosting)", attacker="compromised hospital client(s) (20-30%)", objective="trigger-activated misclassification of attacks as benign",
             assumptions="attacker controls local data & update; persistent participation; synthetic trigger", component="global model / aggregation",
             observed_impact=f"ASR {asr_high:.2f} at 'high' vs {asr_none:.2f} for clean FL (trigger-natural ASR)", detection="norm/cosine anomaly filter; SMPC norm bound",
             mitigation=f"best measured: {bd_best['defense']} -> ASR {bd_best['value']:.2f}" if bd_best else "n/a",
             residual_risk="HIGH" if (bd_best and bd_best["value"] > 0.05) else "MEDIUM", evidence="task_3_1_backdoor.json, task_3_5_countermeasures.json"),
        dict(attack="Byzantine update manipulation", attacker="compromised/colluding clients (10-30%)", objective="degrade or steer the global model", assumptions="attacker sends arbitrary vectors; scale 5",
             component="aggregation rule / availability", observed_impact=f"sign-flip 30%: mean aggregator acc {bz_mean:.2f}", detection="median/trimmed mean/Krum, anomaly filter",
             mitigation=f"best at 30% sign-flip: {bz_best['aggregator']} acc {bz_best['acc']:.2f}", residual_risk="MEDIUM (rules rely on honest majority; not valid under secagg/HE)", evidence="task_3_2_byzantine.json"),
        dict(attack="Membership inference (shadow-model)", attacker="black-box observer / honest-but-curious server with the global model", objective="infer a patient record's training membership",
             assumptions="attacker has same-distribution shadow data; overfit regime (small training set, 128 hidden)", component="global model", observed_impact=f"AUC {mia_none['auc']:.3f} undefended (generalisation gap {mia_none['gap']:.2f})",
             detection="none in protocol", mitigation=f"lowest AUC {mia_best['auc']:.3f} with {mia_best['config']} (utility {mia_best['utility_acc']:.2f}; eps={mia_best['eps']})",
             residual_risk="MEDIUM" if mia_none["auc"] < 0.6 else "HIGH", evidence="task_3_3_membership_inference.json"),
        dict(attack="Gradient leakage (DLG / analytic)", attacker="honest-but-curious server seeing individual plaintext updates", objective="reconstruct clinical feature vectors",
             assumptions="small batches, known labels, smooth activation (DLG), first-layer bias (analytic)", component="client->server channel",
             observed_impact=f"B=1 success {gl_b1['success_rate']:.2f}; successful (>10%) in {len(gl_big)} of {len([r for r in gl['rows'] if r['factor']=='batch/architecture'])} batch/arch settings",
             detection="none", mitigation="secure aggregation / HE / SMPC hide individual updates from the server; noise degrades inversion (task 3.4)",
             residual_risk="LOW vs server when secagg/HE/SMPC used; HIGH for plaintext updates", evidence="task_3_4_gradient_leakage.json"),
    ]
    kpi = [dict(kpi="Backdoor success rate < 5%", measured=f"{asr_high:.3f} (high, undefended); best defended {bd_best['value']:.3f}" if bd_best else "n/a", met=bool(bd_best and bd_best["value"] < 0.05),
                note=f"clean-FL ASR of the trigger = {asr_none:.3f}"),
           dict(kpi="Gradient leakage success < 10%", measured=f"B=1 undefended {gl_b1['success_rate']:.2f}; secagg K=10 (B_eff=20) {max(r['success_rate'] for r in sec_rows if r['aggregated_clients']==max(x['aggregated_clients'] for x in sec_rows)):.2f}",
                met=False, note="NOT met for plaintext small-batch updates; met only when updates are aggregated/noised (see task 3.4)"),
           dict(kpi="Membership inference AUC < 0.60", measured=f"undefended {mia_none['auc']:.3f}; lowest {mia_best['auc']:.3f}", met=bool(mia_none["auc"] < 0.6), note="evaluated in an overfit regime on synthetic data")]
    save_csv(STORY, "task_3_6_risk_matrix", matrix); save_csv(STORY, "task_3_6_kpi", kpi)
    save_json(STORY, "task_3_6_risk", dict(provenance=prov, risk_matrix=matrix, kpi=kpi))
    return dict(provenance=prov, risk_matrix=matrix, kpi=kpi)
