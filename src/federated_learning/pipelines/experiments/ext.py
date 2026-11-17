"""Extension tasks closing the open items of the first sprint report:
 FL-BENCH-004.8  validation-based tuning of FedAvg/FedProx/SCAFFOLD/FedNova (synthetic data)
 FL-BENCH-004.9  benchmark on REAL public de-identified medical tables with NESTED tuning, secure-aggregation equivalence, Byzantine check
 FL-ATTACK-003.7 adaptive (omniscient, defence-aware) attacks: ALIE, MIN-MAX, norm-matched backdoor, MIN-MAX backdoor
All clients are virtual; partitions are synthetic (IID / Dirichlet); no hospital federation is involved."""
from __future__ import annotations

import numpy as np
from federated_learning import attacks as A
from federated_learning.attacks.adaptive import AdaptiveAttackAggregator, MedianNormClip
from federated_learning.privacy.homomorphic import CKKSAggregator
from federated_learning.privacy.secure_aggregation import SecAggAggregator
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset
from federated_learning.data_partitioning.dirichlet import partition_dirichlet
from federated_learning.data_partitioning.iid import partition_iid
from federated_learning.aggregators.defenses import AnomalyFiltered
from federated_learning.aggregators.robust import CoordinateMedian, Krum, TrimmedMean
from federated_learning.aggregators.mean import WeightedMean
from federated_learning.training.config import ALGOS, FLConfig
from federated_learning.training.centralized import centralized_baseline
from federated_learning.servers.orchestrator import run_fl
from federated_learning.evaluation import statistics as S
from federated_learning.preprocessing.real_datasets import REAL_NAMES, load_real
from federated_learning.preprocessing.splits import split_validation
from federated_learning.pipelines.experiments.common import mean_ci, plt, provenance, save_csv, save_json, savefig, scale

LR_GRID = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.8]

MU_GRID = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]

DEFAULT = dict(lr=0.05, mu=0.05)          # the untuned setting used in tasks 4.1-4.7

METRICS = ("acc", "f1", "auc")

def _parts(y, K, kind, seed):
    return partition_iid(y, K, seed) if kind == "iid" else partition_dirichlet(y, K, float(kind), seed, min_size=3)

def _run(data, algo, K, kind, seed, rounds, E, lr, mu=0.0, **kw):
    parts = _parts(data.y_train, K, kind, seed)
    cfg = FLConfig(algo=algo, rounds=rounds, local_epochs=E, lr=lr, mu=mu if algo == "fedprox" else 0.0, seed=seed)
    return run_fl(cfg, data, parts, **kw)

def _configs(algo):
    if algo == "fedprox":
        return [dict(lr=l, mu=m) for l in LR_GRID for m in MU_GRID]
    return [dict(lr=l, mu=0.0) for l in LR_GRID]

def tune(val_data, algo, K, kind, rounds, E, seeds):
    """Select the config with best mean validation F1 (ties: lower validation loss). Test data is never touched."""
    table = []
    for c in _configs(algo):
        fs, ls = [], []
        for s in seeds:
            r = _run(val_data, algo, K, kind, s, rounds, E, **c)
            fs.append(r.final["f1"]); ls.append(r.final["loss"])
        table.append(dict(algo=algo, **c, val_f1=float(np.mean(fs)), val_loss=float(np.mean(ls))))
    best = max(table, key=lambda t: (round(t["val_f1"], 6), -t["val_loss"]))
    return dict(lr=best["lr"], mu=best["mu"]), table

def _final(r):
    return {k: r.final[k] for k in METRICS}

def task_4_8_tuning() -> dict:
    STORY = "fl_bench_004"
    K, rounds, E, kind = 10, scale(30, 4), 2, "0.3"
    tune_seeds, eval_seeds = list(range(100, 100 + scale(3, 1))), list(range(scale(10, 3)))
    prov = provenance("FL-BENCH-004.8", dict(K=K, rounds=rounds, E=E, partition=f"Dirichlet {kind}", lr_grid=LR_GRID, mu_grid=MU_GRID, default=DEFAULT,
                                              tune_seeds=tune_seeds, eval_seeds=eval_seeds, selection="mean validation F1 on 25% of TRAIN; test untouched"))
    data = make_dataset(seed=0, n_samples=scale(12000, 2500))
    val = split_validation(data, 0.25, 0)
    tuned, tune_rows = {}, []
    for a in ALGOS:
        tuned[a], tab = tune(val, a, K, kind, rounds, E, tune_seeds); tune_rows += tab
        print(f"  tuned {a}: {tuned[a]}", flush=True)
    rows = []
    for a in ALGOS:
        for s in eval_seeds:
            for tag, c in (("default", dict(lr=DEFAULT["lr"], mu=DEFAULT["mu"])), ("tuned", tuned[a])):
                r = _run(data, a, K, kind, s, rounds, E, **c)
                rows.append(dict(algo=a, setting=tag, seed=s, lr=c["lr"], mu=c["mu"], **_final(r)))
    summ, tests = [], []
    for a in ALGOS:
        d = {t: np.array([r["f1"] for r in rows if r["algo"] == a and r["setting"] == t]) for t in ("default", "tuned")}
        pt = S.paired_tests(d["tuned"], d["default"]); pt.update(algo=a, metric="f1", note="tuned - default, paired by seed")
        tests.append(pt)
        for t in ("default", "tuned"):
            row = dict(algo=a, setting=t, **{k: v for k, v in (tuned[a] if t == "tuned" else DEFAULT).items()})
            for m in METRICS:
                x = [r[m] for r in rows if r["algo"] == a and r["setting"] == t]; mm, lo, hi = mean_ci(x); row[m] = mm; row[m + "_ci"] = f"[{lo:.4f}, {hi:.4f}]"
            summ.append(row)
    tf = {a: np.array([r["f1"] for r in rows if r["algo"] == a and r["setting"] == "tuned"]) for a in ALGOS}
    fr = S.friedman(tf); pw = S.pairwise_table(tf)
    save_csv(STORY, "task_4_8_tuning_grid", tune_rows); save_csv(STORY, "task_4_8_runs", rows); save_csv(STORY, "task_4_8_summary", summ)
    save_csv(STORY, "task_4_8_tuned_vs_default_tests", tests); save_csv(STORY, "task_4_8_tuned_pairwise", pw)
    fig, ax = plt.subplots(figsize=(5.8, 3.4)); w_ = 0.38
    for i, t in enumerate(("default", "tuned")):
        ax.bar(np.arange(4) + (i - .5) * w_, [next(s["f1"] for s in summ if s["algo"] == a and s["setting"] == t) for a in ALGOS], w_, label=t)
    ax.set_xticks(range(4)); ax.set_xticklabels(ALGOS); ax.set_ylabel("test F1 (mean of seeds)"); ax.set_ylim(0.6, 1); ax.legend(); ax.set_title("Untuned vs validation-tuned (synthetic, Dir 0.3)")
    savefig(fig, STORY, "task_4_8_tuning", prov, f"{len(eval_seeds)} seeds; tuned on 25% validation split of train")
    out = dict(provenance=prov, tuned=tuned, summary=summ, tuned_vs_default=tests, friedman_tuned=fr, pairwise_tuned=pw,
               note="Selection by validation F1; same grid size for every algorithm except FedProx (+mu axis). SCAFFOLD/FedNova have no extra hyper-parameter grid.")
    save_json(STORY, "task_4_8_tuning", out); return out

def task_4_9_real_data() -> dict:
    STORY = "fl_bench_004"
    K, rounds, E, kind = 5, scale(40, 4), 2, "0.5"
    outer = list(range(scale(10, 3))); inner = list(range(200, 200 + scale(2, 1)))
    names = REAL_NAMES if not scale(0, 1) else ("wdbc",)
    prov = provenance("FL-BENCH-004.9", dict(datasets=list(names), K=K, rounds=rounds, E=E, partition=f"Dirichlet {kind}", outer_splits=outer, inner_tuning_seeds=inner,
                                              protocol="NESTED: per outer split, hyper-parameters tuned on 25% validation of that split's TRAIN; test untouched",
                                              note="virtual clients; partitions synthetic; datasets small -> wide CIs"))
    rows, tuned_log, central = [], [], []
    for name in names:
        for s in outer:
            d = load_real(name, seed=s); val = split_validation(d, 0.25, s)
            central.append(dict(dataset=name, split=s, **{k: v for k, v in centralized_baseline(d, epochs=rounds).items() if k in METRICS}))
            for a in ALGOS:
                best, _ = tune(val, a, K, kind, rounds, E, [i + s for i in inner]); tuned_log.append(dict(dataset=name, split=s, algo=a, **best))
                for tag, c in (("default", dict(lr=DEFAULT["lr"], mu=DEFAULT["mu"])), ("tuned", best)):
                    r = _run(d, a, K, kind, s, rounds, E, **c); rows.append(dict(dataset=name, split=s, algo=a, setting=tag, lr=c["lr"], mu=c["mu"], **_final(r)))
        print(f"  real data {name} done", flush=True)
    save_csv(STORY, "task_4_9_real_runs", rows); save_csv(STORY, "task_4_9_real_tuned_params", tuned_log); save_csv(STORY, "task_4_9_real_centralized", central)
    summ, tests, rank = [], [], []
    for name in names:
        c = [r for r in central if r["dataset"] == name]
        summ.append(dict(dataset=name, algo="centralized(pooled)", setting="reference", **{m: float(np.mean([r[m] for r in c])) for m in METRICS}))
        for a in ALGOS:
            for t in ("default", "tuned"):
                rr = [r for r in rows if r["dataset"] == name and r["algo"] == a and r["setting"] == t]
                row = dict(dataset=name, algo=a, setting=t, n_splits=len(rr))
                for m in METRICS:
                    mm, lo, hi = mean_ci([r[m] for r in rr]); row[m] = mm; row[m + "_ci"] = f"[{lo:.4f}, {hi:.4f}]"
                summ.append(row)
            f = lambda t: np.array([r["f1"] for r in rows if r["dataset"] == name and r["algo"] == a and r["setting"] == t])
            pt = S.paired_tests(f("tuned"), f("default")); pt.update(dataset=name, algo=a, metric="f1"); tests.append(pt)
        tf = {a: np.array([r["f1"] for r in rows if r["dataset"] == name and r["algo"] == a and r["setting"] == "tuned"]) for a in ALGOS}
        fr = S.friedman(tf); rank.append(dict(dataset=name, friedman_stat=fr.get("statistic"), friedman_p=fr.get("p"), **{f"mean_f1_{a}": float(tf[a].mean()) for a in ALGOS}))
        for pr in S.pairwise_table(tf):
            rank.append(dict(dataset=name, a=pr["a"], b=pr["b"], mean_diff=pr["mean_diff"], wilcoxon_p=pr["wilcoxon_p"], holm_p=pr["holm_p"]))
    save_csv(STORY, "task_4_9_real_summary", summ); save_csv(STORY, "task_4_9_real_tuned_vs_default", tests); save_csv(STORY, "task_4_9_real_ranking", rank)
    # secure aggregation equivalence + Byzantine robustness on real data (tuned FedAvg params, split seeds)
    sec, byz = [], []
    for name in names:
        for s in outer[: scale(5, 2)]:
            d = load_real(name, seed=s); lr_ = next(t["lr"] for t in tuned_log if t["dataset"] == name and t["split"] == s and t["algo"] == "fedavg")
            res = {}
            for tag, agg in (("plain", None), ("ckks", CKKSAggregator()), ("secagg", SecAggAggregator(threshold_frac=0.67, seed=s))):
                r = _run(d, "fedavg", K, kind, s, rounds, E, lr_, aggregator=agg); res[tag] = r
                sec.append(dict(dataset=name, split=s, mechanism=tag, **_final(r)))
            for tag in ("ckks", "secagg"):
                sec[-1 if tag == "secagg" else -2]["max_abs_param_diff_vs_plain"] = float(np.abs(res[tag].w - res["plain"].w).max())
            hook = A.byzantine_hook("sign_flip", {0}, 5.0, seed=s)
            for tag, agg in (("mean", WeightedMean()), ("median", CoordinateMedian()), ("trimmed(0.2)", TrimmedMean(0.2)), ("krum(f=1)", Krum(1, 1))):
                r = _run(d, "fedavg", K, kind, s, rounds, E, lr_, aggregator=agg, client_hook=hook); byz.append(dict(dataset=name, split=s, aggregator=tag, **_final(r)))
    save_csv(STORY, "task_4_9_real_secagg_equivalence", sec); save_csv(STORY, "task_4_9_real_byzantine", byz)
    fig, axs = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.4), squeeze=False)
    for ax, name in zip(axs[0], names):
        for i, t in enumerate(("default", "tuned")):
            ax.bar(np.arange(4) + (i - .5) * 0.38, [next(s["f1"] for s in summ if s["dataset"] == name and s["algo"] == a and s["setting"] == t) for a in ALGOS], 0.38, label=t)
        ax.axhline(next(s["f1"] for s in summ if s["dataset"] == name and s["algo"].startswith("central")), ls="--", c="k", lw=0.8, label="pooled")
        ax.set_xticks(range(4)); ax.set_xticklabels(ALGOS, fontsize=7); ax.set_title(name); ax.set_ylim(0.5, 1)
    axs[0][0].set_ylabel("test F1"); axs[0][0].legend(fontsize=7)
    savefig(fig, STORY, "task_4_9_real_data", prov, f"{len(outer)} outer splits; 5 virtual clients; nested tuning")
    out = dict(provenance=prov, datasets={n: load_real(n, 0).meta for n in names}, summary=summ, tuned_vs_default=tests, ranking=rank,
               caveat="small public datasets (n<=768): test sets of 91-231 samples -> wide CIs; partitions into 5 virtual clients are synthetic; results are NOT evidence about hospital federations")
    save_json(STORY, "task_4_9_real_data", out); return out

DEFENSES = ["mean", "median", "trimmed_mean", "krum", "multi_krum", "clip_median_norm", "anomaly_filter"]

def _defense(name, n, f):
    return {"mean": WeightedMean, "median": CoordinateMedian, "trimmed_mean": lambda: TrimmedMean(min(0.45, f / n + 0.05)), "krum": lambda: Krum(f, 1),
            "multi_krum": lambda: Krum(f, n - f), "clip_median_norm": MedianNormClip, "anomaly_filter": AnomalyFiltered}[name]()

def _adaptive_run(data, K, f, seed, rounds, E, lr, attack, defense, backdoor=None, partition="iid"):
    """backdoor=None -> untargeted attack on accuracy. backdoor=dict(trig, poison_frac, boost) -> data poisoning + (static boost | adaptive wrapper)."""
    parts = _parts(data.y_train, K, partition, seed)
    inner = _defense(defense, K, f)
    mal = set(range(f))
    kw = {}
    if backdoor is not None:
        rng = np.random.default_rng(seed + 99); trig = backdoor["trig"]
        cd = {k: A.poison_client_data(data.X_train[parts[k]], data.y_train[parts[k]], backdoor["poison_frac"], 0, rng, *trig)[:2] for k in mal} if attack != "none_clean" else None
        kw = dict(client_data=cd, eval_fn=lambda m, w: A.backdoor_metrics(m, w, data.X_test, data.y_test, 0, *trig))
        if attack == "static_boost":
            kw["client_hook"] = A.boost_hook(mal, backdoor["boost"]); agg = inner
        elif attack == "none_clean":
            agg = inner
        else:
            agg = AdaptiveAttackAggregator(inner, mal, attack)
    else:
        agg = AdaptiveAttackAggregator(inner, mal, attack)
    try:
        r = run_fl(FLConfig(rounds=rounds, local_epochs=E, lr=lr, seed=seed), data, parts, aggregator=agg, **kw)
        return dict(r.final, diverged=r.final.get("diverged", 0.0))
    except ValueError as e:
        if "non-finite" not in str(e):
            raise
        # the attack drove the model to non-finite weights (model destroyed): convention = constant prediction of class 0
        acc = float((data.y_test == 0).mean())
        return dict(acc=acc, f1=0.0, auc=0.5, asr=1.0, clean_acc=acc, diverged=1.0)

def task_3_7_adaptive_attacks() -> dict:
    STORY = "fl_attack_003"
    seeds = list(range(scale(5, 1)))
    # trigger chosen exactly as in task 3.1 (lowest clean-model ASR), read from the stored selection table
    import pandas as pd
    from federated_learning.pipelines.experiments.attack003 import TRIGGER_CANDIDATES
    sel = pd.read_csv(f"{OUT}/{STORY}/task_3_1_trigger_selection.csv")
    trig = TRIGGER_CANDIDATES[sel.sort_values("clean_model_asr").iloc[0]["trigger"]]
    settings = [dict(name="syn", data=make_dataset(seed=0, n_samples=scale(12000, 2500)), K=20, f=4, rounds=scale(20, 3), E=1, lr=0.05)]
    for n in (REAL_NAMES if not scale(0, 1) else ("wdbc",)):
        settings.append(dict(name=n, data=load_real(n, 0), K=10, f=2, rounds=scale(30, 3), E=2, lr=0.05))
    prov = provenance("FL-ATTACK-003.7", dict(threat_model="omniscient colluding attackers replace their updates after seeing benign updates (upper bound)", settings=[{k: v for k, v in s.items() if k != "data"} for s in settings],
                                              seeds=seeds, defenses=DEFENSES, attacks=["none", "alie", "minmax"], backdoor_variants=["none_clean", "static_boost", "norm_matched", "minmax_backdoor"], trigger=str(trig)))
    unt, bd = [], []
    for st in settings:
        for sd in seeds:
            for dfn in DEFENSES:
                for atk in ("none", "alie", "minmax"):
                    fin = _adaptive_run(st["data"], st["K"], st["f"], sd, st["rounds"], st["E"], st["lr"], atk, dfn)
                    unt.append(dict(dataset=st["name"], seed=sd, defense=dfn, attack=atk, **{k: fin[k] for k in METRICS}))
        print(f"  adaptive untargeted {st['name']} done", flush=True)
    syn = settings[0]
    BD = dict(trig=trig, poison_frac=0.5, boost=3.0)
    for sd in seeds:
        for dfn in DEFENSES:
            for atk in ("none_clean", "static_boost", "norm_matched", "minmax_backdoor"):
                fin = _adaptive_run(syn["data"], syn["K"], syn["f"], sd, syn["rounds"], syn["E"], syn["lr"], atk, dfn, backdoor=BD)
                bd.append(dict(dataset="syn", seed=sd, defense=dfn, attack=atk, asr=fin["asr"], clean_acc=fin["clean_acc"]))
    print("  adaptive backdoor done", flush=True)
    save_csv(STORY, "task_3_7_untargeted_runs", unt); save_csv(STORY, "task_3_7_backdoor_runs", bd)
    su, sb = [], []
    for ds in sorted({r["dataset"] for r in unt}):
        for dfn in DEFENSES:
            for atk in ("none", "alie", "minmax"):
                rr = [r for r in unt if r["dataset"] == ds and r["defense"] == dfn and r["attack"] == atk]
                row = dict(dataset=ds, defense=dfn, attack=atk, n=len(rr))
                for m in METRICS:
                    mm, lo, hi = mean_ci([r[m] for r in rr]); row[m] = mm; row[m + "_ci"] = f"[{lo:.3f}, {hi:.3f}]"
                su.append(row)
    for dfn in DEFENSES:
        for atk in ("none_clean", "static_boost", "norm_matched", "minmax_backdoor"):
            rr = [r for r in bd if r["defense"] == dfn and r["attack"] == atk]
            row = dict(dataset="syn", defense=dfn, attack=atk, n=len(rr))
            for m in ("asr", "clean_acc"):
                mm, lo, hi = mean_ci([r[m] for r in rr]); row[m] = mm; row[m + "_ci"] = f"[{lo:.3f}, {hi:.3f}]"
            sb.append(row)
    save_csv(STORY, "task_3_7_untargeted_summary", su); save_csv(STORY, "task_3_7_backdoor_summary", sb)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    ds0 = "syn"
    for i, atk in enumerate(("none", "alie", "minmax")):
        axs[0].bar(np.arange(len(DEFENSES)) + (i - 1) * 0.27, [next(s["acc"] for s in su if s["dataset"] == ds0 and s["defense"] == d and s["attack"] == atk) for d in DEFENSES], 0.27, label=atk)
    axs[0].set_xticks(range(len(DEFENSES))); axs[0].set_xticklabels(DEFENSES, rotation=40, fontsize=7); axs[0].set_ylabel("test accuracy"); axs[0].set_title("Untargeted adaptive attacks (synthetic, 20% attackers)"); axs[0].legend(fontsize=7)
    for i, atk in enumerate(("none_clean", "static_boost", "norm_matched", "minmax_backdoor")):
        axs[1].bar(np.arange(len(DEFENSES)) + (i - 1.5) * 0.2, [next(s["asr"] for s in sb if s["defense"] == d and s["attack"] == atk) for d in DEFENSES], 0.2, label=atk)
    axs[1].set_xticks(range(len(DEFENSES))); axs[1].set_xticklabels(DEFENSES, rotation=40, fontsize=7); axs[1].set_ylabel("backdoor ASR"); axs[1].set_title("Static vs adaptive backdoor"); axs[1].legend(fontsize=7)
    savefig(fig, STORY, "task_3_7_adaptive", prov, f"{len(seeds)} seeds; omniscient attacker (upper bound)")
    out = dict(provenance=prov, untargeted=su, backdoor=sb, caveat="omniscient attacker = upper bound; attacks are not tuned per defence beyond their built-in constraints; synthetic/small data")
    save_json(STORY, "task_3_7_adaptive_attacks", out); return out
