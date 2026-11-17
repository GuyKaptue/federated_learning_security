"""Second extension (closing the open items of the second report):
 FL-BENCH-004.10  natural multi-site benchmark: UCI heart disease, 4 collection sites = 4 clients (within-site holdout AND leave-one-site-out)
 FL-BENCH-004.11  broader (random-search) tuning over lr, mu, batch size, local epochs, hidden width, server lr vs lr/mu-grid vs untuned
 FL-ATTACK-003.8  stronger adaptive attacks that KNOW the server rule (AGR-tailored, MIN-SUM, Fang, AGR-backdoor)
All clients are virtual (one process). The four sites are natural cohorts, not a live hospital federation."""
from __future__ import annotations

import numpy as np
from federated_learning.evaluation import statistics as S
from federated_learning.privacy.secure_aggregation import SecAggAggregator
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset
from federated_learning.training.config import ALGOS, FLConfig
from federated_learning.training.centralized import centralized_baseline
from federated_learning.evaluation.metrics import evaluate
from federated_learning.servers.orchestrator import run_fl
from federated_learning.preprocessing.real_datasets import load_real
from federated_learning.preprocessing.splits import split_validation
from federated_learning.preprocessing.multisite_heart import SITE_NAMES, SiteData, load_loso, load_sites, site_validation
from federated_learning.preprocessing.dataset import Dataset
from federated_learning.pipelines.experiments.common import OUT, mean_ci, plt, provenance, save_csv, save_json, savefig, scale
from federated_learning.pipelines.experiments import ext as E_

METRICS = ("acc", "f1", "auc", "bacc")

DEFAULT = dict(lr=0.05, mu=0.05)

def _bacc(fin: dict) -> float:
    return 0.5 * (fin["recall"] + fin["specificity"])

def _fin(r) -> dict:
    d = {k: r.final[k] for k in ("acc", "f1", "auc", "loss", "recall", "specificity")}
    d["bacc"] = _bacc(d)
    return d

def _site_run(sd: SiteData, algo: str, seed: int, rounds: int, aggregator=None, **kw):
    kw = dict(kw)
    if algo != "fedprox":
        kw["mu"] = 0.0
    kw.setdefault("local_epochs", 2)
    return run_fl(FLConfig(algo=algo, rounds=rounds, seed=seed, **kw), sd, sd.parts, aggregator=aggregator)

def _tune(val, algo, runner, cfgs, seeds, rounds):
    """Best mean validation F1 (ties: lower validation loss) over `cfgs`; test data never used."""
    table = []
    for c in cfgs:
        fs, ls = [], []
        for s in seeds:
            r = runner(val, algo, s, rounds, **c)
            fs.append(r.final["f1"]); ls.append(r.final["loss"])
        table.append(dict(algo=algo, **c, val_f1=float(np.mean(fs)), val_loss=float(np.mean(ls))))
    best = max(table, key=lambda t: (round(t["val_f1"], 6), -t["val_loss"]))
    return {k: best[k] for k in best if k not in ("algo", "val_f1", "val_loss")}, table

def _summ(rows, keys, metrics=METRICS):
    out = []
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    for g, rr in groups.items():
        row = dict(zip(keys, g), n=len(rr))
        for m in metrics:
            mm, lo, hi = mean_ci([r[m] for r in rr]); row[m] = mm; row[m + "_ci"] = f"[{lo:.4f}, {hi:.4f}]"
        out.append(row)
    return out

def task_4_10_natural_sites() -> dict:
    STORY = "fl_bench_004"
    rounds = scale(40, 4)
    outer = list(range(scale(10, 3))); inner = [200, 201][: scale(2, 1)]
    prov = provenance("FL-BENCH-004.10", dict(dataset="UCI heart disease, 4 natural sites (Cleveland, Hungary, Switzerland, VA Long Beach), 10 features, target num>0",
                                               clients=4, rounds=rounds, local_epochs=2, protocols=["A within-site 70/30 holdout, nested tuning per split", "B leave-one-site-out (external), tuned on 25% validation of the training sites"],
                                               outer_splits=outer, inner_tuning_seeds=inner, grid=dict(lr=E_.LR_GRID, mu=E_.MU_GRID),
                                               preprocessing="median imputation + z-score fitted on pooled TRAIN rows (a centralised step; in a real federation it would need secure aggregation of statistics)"))
    d0 = load_sites(0)
    print("  sites:", d0.meta["site_train_sizes"], "pos rates", np.round(d0.meta["site_positive_rate"], 2), flush=True)
    # ---------------- protocol A
    A_rows, A_site, A_tuned, A_base = [], [], [], []
    for s in outer:
        sd = load_sites(s); val = site_validation(sd, 0.25, s)
        for a in ALGOS:
            best, _ = _tune(val, a, _site_run, E_._configs(a), [i + s for i in inner], rounds); A_tuned.append(dict(split=s, algo=a, **best))
            for tag, c in (("default", dict(lr=DEFAULT["lr"], mu=DEFAULT["mu"])), ("tuned", best)):
                r = _site_run(sd, a, s, rounds, **c)
                A_rows.append(dict(split=s, algo=a, setting=tag, lr=c["lr"], mu=c["mu"], **_fin(r)))
                if tag == "tuned":
                    for si, (X, y) in enumerate(sd.site_tests):
                        e = evaluate(r.extras["model"], r.w, X, y); A_site.append(dict(split=s, algo=a, site=SITE_NAMES[si], n_test=len(y), acc=e["acc"], f1=e["f1"], auc=e["auc"]))
        c = centralized_baseline(sd, epochs=rounds, seed=s); c["bacc"] = _bacc(c); A_base.append(dict(split=s, method="centralized(pooled)", **{k: c[k] for k in METRICS}))
        loc = []
        for si, p in enumerate(sd.parts):
            ld = Dataset(sd.X_train[p], sd.y_train[p], sd.X_test, sd.y_test, {})
            e = centralized_baseline(ld, epochs=rounds, seed=s); e["bacc"] = _bacc(e); loc.append(e)
            X, y = sd.site_tests[si]
            # local model evaluated on its OWN site test set requires the weights; re-train cheaply with returned model is not exposed -> report pooled-test metrics only
        A_base.append(dict(split=s, method="local-only(mean of 4 site models, pooled test)", **{k: float(np.mean([e[k] for e in loc])) for k in METRICS}))
        print(f"  protocol A split {s} done", flush=True)
    save_csv(STORY, "task_4_10_A_runs", A_rows); save_csv(STORY, "task_4_10_A_site_metrics", A_site); save_csv(STORY, "task_4_10_A_tuned_params", A_tuned); save_csv(STORY, "task_4_10_A_baselines", A_base)
    A_sum = _summ(A_rows, ("algo", "setting")); A_site_sum = _summ(A_site, ("algo", "site"), ("acc", "f1", "auc")); A_base_sum = _summ(A_base, ("method",))
    A_tests = []
    for a in ALGOS:
        f = lambda t: np.array([r["f1"] for r in A_rows if r["algo"] == a and r["setting"] == t])
        pt = S.paired_tests(f("tuned"), f("default")); pt.update(algo=a, metric="f1", note="tuned - default, paired by split"); A_tests.append(pt)
    tf = {a: np.array([r["f1"] for r in A_rows if r["algo"] == a and r["setting"] == "tuned"]) for a in ALGOS}
    A_fr = S.friedman(tf); A_pw = S.pairwise_table(tf)
    save_csv(STORY, "task_4_10_A_summary", A_sum); save_csv(STORY, "task_4_10_A_site_summary", A_site_sum); save_csv(STORY, "task_4_10_A_baselines_summary", A_base_sum)
    save_csv(STORY, "task_4_10_A_tuned_vs_default_tests", A_tests); save_csv(STORY, "task_4_10_A_pairwise", A_pw)
    # ---------------- protocol B
    B_rows, B_tuned, B_base = [], [], []
    seedsB = list(range(scale(10, 3)))
    for ho in SITE_NAMES:
        ld = load_loso(ho); val = site_validation(ld, 0.25, 0)
        for a in ALGOS:
            best, _ = _tune(val, a, _site_run, E_._configs(a), inner, rounds); B_tuned.append(dict(held_out=ho, algo=a, **best))
            for tag, c in (("default", dict(lr=DEFAULT["lr"], mu=DEFAULT["mu"])), ("tuned", best)):
                for s in seedsB:
                    r = _site_run(ld, a, s, rounds, **c); B_rows.append(dict(held_out=ho, seed=s, algo=a, setting=tag, **_fin(r)))
        for s in seedsB:
            c = centralized_baseline(ld, epochs=rounds, seed=s); c["bacc"] = _bacc(c); B_base.append(dict(held_out=ho, seed=s, method="centralized(pooled 3 sites)", **{k: c[k] for k in METRICS}))
        print(f"  protocol B held-out {ho} done", flush=True)
    save_csv(STORY, "task_4_10_B_runs", B_rows); save_csv(STORY, "task_4_10_B_tuned_params", B_tuned); save_csv(STORY, "task_4_10_B_baselines", B_base)
    B_sum = _summ(B_rows, ("held_out", "algo", "setting")); B_base_sum = _summ(B_base, ("held_out", "method"))
    macro = []
    for a in ALGOS:
        for t in ("default", "tuned"):
            row = dict(algo=a, setting=t)
            for m in METRICS:
                row[m] = float(np.mean([np.mean([r[m] for r in B_rows if r["held_out"] == ho and r["algo"] == a and r["setting"] == t]) for ho in SITE_NAMES]))
            macro.append(row)
    sitemean = {a: np.array([np.mean([r["auc"] for r in B_rows if r["held_out"] == ho and r["algo"] == a and r["setting"] == "tuned"]) for ho in SITE_NAMES]) for a in ALGOS}
    B_fr = S.friedman(sitemean)
    save_csv(STORY, "task_4_10_B_summary", B_sum); save_csv(STORY, "task_4_10_B_baselines_summary", B_base_sum); save_csv(STORY, "task_4_10_B_macro", macro)
    # ---------------- secure-aggregation equivalence on the 4 natural sites
    sec = []
    for s in outer[: scale(3, 2)]:
        sd = load_sites(s); lr_ = next(t["lr"] for t in A_tuned if t["split"] == s and t["algo"] == "fedavg")
        rp = _site_run(sd, "fedavg", s, rounds, lr=lr_); rs = _site_run(sd, "fedavg", s, rounds, aggregator=SecAggAggregator(threshold_frac=0.67, seed=s), lr=lr_)
        sec.append(dict(split=s, max_abs_param_diff_secagg_vs_plain=float(np.abs(rp.w - rs.w).max()), f1_plain=rp.final["f1"], f1_secagg=rs.final["f1"]))
    save_csv(STORY, "task_4_10_secagg_equivalence", sec)
    # ---------------- figure
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, t in enumerate(("default", "tuned")):
        axs[0].bar(np.arange(4) + (i - .5) * 0.38, [next(s["f1"] for s in A_sum if s["algo"] == a and s["setting"] == t) for a in ALGOS], 0.38, label=t)
    cen = next(s["f1"] for s in A_base_sum if s["method"].startswith("central")); loc = next(s["f1"] for s in A_base_sum if s["method"].startswith("local"))
    axs[0].axhline(cen, ls="--", c="k", lw=0.8, label="pooled"); axs[0].axhline(loc, ls=":", c="r", lw=0.8, label="local-only")
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels(ALGOS); axs[0].set_ylim(0.5, 1); axs[0].set_ylabel("pooled test F1"); axs[0].set_title("A: within-site holdout (4 natural sites)"); axs[0].legend(fontsize=7)
    for i, a in enumerate(ALGOS):
        axs[1].bar(np.arange(4) + (i - 1.5) * 0.2, [next(s["auc"] for s in B_sum if s["held_out"] == ho and s["algo"] == a and s["setting"] == "tuned") for ho in SITE_NAMES], 0.2, label=a)
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(SITE_NAMES, fontsize=8); axs[1].set_ylim(0.4, 1); axs[1].set_ylabel("AUC on held-out site"); axs[1].set_title("B: leave-one-site-out"); axs[1].legend(fontsize=7)
    savefig(fig, STORY, "task_4_10_natural_sites", prov, f"{len(outer)} splits (A), {len(seedsB)} init seeds (B); 4 virtual clients = 4 natural cohorts")
    out = dict(provenance=prov, meta=d0.meta, A=dict(summary=A_sum, site_summary=A_site_sum, baselines=A_base_sum, tuned_vs_default=A_tests, friedman_tuned=A_fr, pairwise_tuned=A_pw),
               B=dict(summary=B_sum, macro=macro, baselines=B_base_sum, friedman_site_means_auc=B_fr), secagg=sec,
               caveat=("n=920 patients from four 1980s public research cohorts; 4 clients only; Switzerland has 8 negatives (positive rate 0.93) so F1/accuracy are inflated there - use AUC/balanced accuracy; "
                       "Protocol B seeds replicate only initialisation/batch order (the data are fixed), so p-values reflect training noise, not uncertainty about new sites (n_sites=4); "
                       "not a live federation; no claim about hospitals or compliance"))
    save_json(STORY, "task_4_10_natural_sites", out); return out

N_RANDOM = 24

LOG = lambda a, b, u: float(10 ** (np.log10(a) + u * (np.log10(b) - np.log10(a))))

def sample_configs(n: int, seed: int = 0) -> list[dict]:
    """Same draws for every algorithm (budget-matched); mu is used only by FedProx, server_lr only by SCAFFOLD."""
    rng = np.random.default_rng(seed)
    return [dict(lr=LOG(0.005, 0.8, rng.random()), mu=LOG(1e-3, 1.0, rng.random()), batch_size=int(rng.choice([16, 32, 64])), local_epochs=int(rng.choice([1, 2, 3, 5])),
                 hidden=int(rng.choice([16, 32, 64])), server_lr=float(rng.choice([0.5, 1.0, 1.5]))) for _ in range(n)]

def _algo_cfg(c: dict, algo: str) -> dict:
    c = dict(c)
    if algo != "scaffold":
        c["server_lr"] = 1.0
    if algo != "fedprox":
        c["mu"] = 0.0
    return c

def _syn_runner(K, kind):
    def run(data, algo, seed, rounds, **c):
        parts = E_._parts(data.y_train, K, kind, seed)
        return run_fl(FLConfig(algo=algo, rounds=rounds, seed=seed, **_algo_cfg(c, algo)), data, parts)
    return run

def task_4_11_random_search() -> dict:
    STORY = "fl_bench_004"
    n_cfg = scale(N_RANDOM, 3); rounds = scale(30, 4)
    cfgs = sample_configs(n_cfg, 7)
    tune_seeds = [100, 101][: scale(2, 1)]; eval_seeds = list(range(scale(10, 3))); splits = list(range(scale(5, 2)))
    prov = provenance("FL-BENCH-004.11", dict(search_space=dict(lr="log-uniform [0.005, 0.8]", mu="log-uniform [1e-3, 1] (FedProx)", batch_size=[16, 32, 64], local_epochs=[1, 2, 3, 5], hidden=[16, 32, 64], server_lr="[0.5, 1, 1.5] (SCAFFOLD)"),
                                               n_random_configs=n_cfg, grid_configs=dict(fedavg_like=len(E_.LR_GRID), fedprox=len(E_.LR_GRID) * len(E_.MU_GRID)), rounds=rounds, tune_seeds=tune_seeds, eval_seeds=eval_seeds, natural_site_splits=splits,
                                               selection="mean validation F1 on 25% of TRAIN; test untouched", compute_note="local_epochs and hidden width change per-round compute, so budgets are matched in number of configurations, NOT in FLOPs"))
    rows, chosen = [], []
    # ---- synthetic, Dirichlet 0.3, K=10 (same setting as 4.8)
    K, kind = 10, "0.3"
    data = make_dataset(seed=0, n_samples=scale(12000, 2500)); val = split_validation(data, 0.25, 0)
    run = _syn_runner(K, kind)
    for a in ALGOS:
        grid_best, _ = _tune(val, a, run, [dict(c, local_epochs=2) for c in E_._configs(a)], tune_seeds, rounds)
        rnd_best, tab = _tune(val, a, run, [_algo_cfg(c, a) for c in cfgs], tune_seeds, rounds)
        rnd_best = _algo_cfg(rnd_best, a)
        chosen.append(dict(dataset="synthetic", algo=a, method="grid(lr,mu)", **grid_best)); chosen.append(dict(dataset="synthetic", algo=a, method="random(6 params)", **rnd_best))
        for tag, c in (("default", dict(lr=DEFAULT["lr"], mu=DEFAULT["mu"], local_epochs=2)), ("grid", dict(grid_best, local_epochs=2)), ("random", rnd_best)):
            for s in eval_seeds:
                rows.append(dict(dataset="synthetic", algo=a, setting=tag, rep=s, **_fin(run(data, a, s, rounds, **c))))
        print(f"  synthetic {a} done", flush=True)
    # ---- natural 4-site heart, protocol A, nested per split
    for s in splits:
        sd = load_sites(s); vs = site_validation(sd, 0.25, s)
        for a in ALGOS:
            grid_best, _ = _tune(vs, a, _site_run, E_._configs(a), [i + s for i in tune_seeds], rounds)
            rnd_best, _ = _tune(vs, a, _site_run, [_algo_cfg(c, a) for c in cfgs], [i + s for i in tune_seeds], rounds); rnd_best = _algo_cfg(rnd_best, a)
            chosen.append(dict(dataset="heart-4sites", split=s, algo=a, method="grid(lr,mu)", **grid_best)); chosen.append(dict(dataset="heart-4sites", split=s, algo=a, method="random(6 params)", **rnd_best))
            for tag, c in (("default", dict(lr=DEFAULT["lr"], mu=DEFAULT["mu"])), ("grid", grid_best), ("random", rnd_best)):
                rows.append(dict(dataset="heart-4sites", algo=a, setting=tag, rep=s, **_fin(_site_run(sd, a, s, rounds, **c))))
        print(f"  heart split {s} done", flush=True)
    save_csv(STORY, "task_4_11_runs", rows); save_csv(STORY, "task_4_11_selected_configs", chosen)
    summ = _summ(rows, ("dataset", "algo", "setting"), ("acc", "f1", "auc"))
    tests = []
    for ds in ("synthetic", "heart-4sites"):
        for a in ALGOS:
            g = lambda t: np.array([r["f1"] for r in rows if r["dataset"] == ds and r["algo"] == a and r["setting"] == t])
            for x, y in (("random", "grid"), ("random", "default"), ("grid", "default")):
                if len(g(x)) < 3: continue
                pt = S.paired_tests(g(x), g(y)); pt.update(dataset=ds, algo=a, contrast=f"{x} - {y}", metric="f1"); tests.append(pt)
    save_csv(STORY, "task_4_11_summary", summ); save_csv(STORY, "task_4_11_tests", tests)
    edge = [dict(dataset=c["dataset"], algo=c["algo"], lr=c["lr"], at_lr_edge=bool(c["lr"] >= 0.7 or c["lr"] <= 0.006)) for c in chosen if c["method"].startswith("random")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, ds in zip(axs, ("synthetic", "heart-4sites")):
        for i, t in enumerate(("default", "grid", "random")):
            ax.bar(np.arange(4) + (i - 1) * 0.27, [next(s["f1"] for s in summ if s["dataset"] == ds and s["algo"] == a and s["setting"] == t) for a in ALGOS], 0.27, label=t)
        ax.set_xticks(range(4)); ax.set_xticklabels(ALGOS); ax.set_ylim(0.5, 1); ax.set_ylabel("test F1"); ax.set_title(ds); ax.legend(fontsize=7)
    savefig(fig, STORY, "task_4_11_random_search", prov, f"{n_cfg} random configs vs lr/mu grid vs default")
    out = dict(provenance=prov, summary=summ, tests=tests, lr_edge=edge,
               caveat=f"{n_cfg} random configurations per algorithm is a small budget; validation-set selection may overfit on small data (heart: 25% of ~640 rows); budgets matched in #configs, not compute")
    save_json(STORY, "task_4_11_random_search", out); return out

ATK_UNT = ("none", "alie", "minmax", "minsum", "fang", "agr_tailored")

ATK_BD = ("none_clean", "static_boost", "minmax_backdoor", "agr_backdoor")

def task_3_8_stronger_adaptive() -> dict:
    STORY = "fl_attack_003"
    import pandas as pd
    from federated_learning.pipelines.experiments.attack003 import TRIGGER_CANDIDATES
    seeds = list(range(scale(5, 1)))
    sel = pd.read_csv(f"{OUT}/{STORY}/task_3_1_trigger_selection.csv")
    trig = TRIGGER_CANDIDATES[sel.sort_values("clean_model_asr").iloc[0]["trigger"]]
    syn = make_dataset(seed=0, n_samples=scale(12000, 2500))
    settings = [dict(name="syn", data=syn, K=20, f=4, rounds=scale(20, 3), E=1, lr=0.05),
                dict(name="syn_f2", data=syn, K=20, f=2, rounds=scale(20, 3), E=1, lr=0.05, only=("none", "minmax", "agr_tailored")),
                dict(name="syn_f8", data=syn, K=20, f=8, rounds=scale(20, 3), E=1, lr=0.05, only=("none", "minmax", "agr_tailored"))]
    if not scale(0, 1):
        settings.append(dict(name="wdbc", data=load_real("wdbc", 0), K=10, f=2, rounds=scale(30, 3), E=2, lr=0.05))
    prov = provenance("FL-ATTACK-003.8", dict(threat_model="omniscient colluding attackers that also KNOW the server aggregation rule and simulate it (upper bound; not a realistic attacker)",
                                               settings=[{k: v for k, v in s.items() if k != "data"} for s in settings], seeds=seeds, defenses=E_.DEFENSES, untargeted=ATK_UNT, backdoor=ATK_BD, trigger=str(trig),
                                               agr_tailored="candidates: 3 directions x 10 gammas (x ||mean||) + ALIE + MIN-MAX + MIN-SUM + FANG; score = <simulated aggregate, benign mean direction>"))
    unt, bd = [], []
    for st in settings:
        atks = st.get("only", ATK_UNT)
        for sd in seeds:
            for dfn in E_.DEFENSES:
                for atk in atks:
                    fin = E_._adaptive_run(st["data"], st["K"], st["f"], sd, st["rounds"], st["E"], st["lr"], atk, dfn)
                    unt.append(dict(dataset=st["name"], seed=sd, defense=dfn, attack=atk, acc=fin["acc"], f1=fin["f1"], auc=fin["auc"], diverged=fin["diverged"]))
        print(f"  stronger attacks untargeted {st['name']} done", flush=True)
    BD = dict(trig=trig, poison_frac=0.5, boost=3.0); s0 = settings[0]
    for sd in seeds:
        for dfn in E_.DEFENSES:
            for atk in ATK_BD:
                fin = E_._adaptive_run(s0["data"], s0["K"], s0["f"], sd, s0["rounds"], s0["E"], s0["lr"], atk, dfn, backdoor=BD)
                bd.append(dict(dataset="syn", seed=sd, defense=dfn, attack=atk, asr=fin["asr"], clean_acc=fin["clean_acc"], diverged=fin["diverged"]))
    print("  stronger attacks backdoor done", flush=True)
    save_csv(STORY, "task_3_8_untargeted_runs", unt); save_csv(STORY, "task_3_8_backdoor_runs", bd)
    su = _summ(unt, ("dataset", "defense", "attack"), ("acc", "f1", "auc")); sb = _summ(bd, ("defense", "attack"), ("asr", "clean_acc"))
    worst = []
    for ds in sorted({r["dataset"] for r in unt}):
        for dfn in E_.DEFENSES:
            rows_ = [s for s in su if s["dataset"] == ds and s["defense"] == dfn and s["attack"] != "none"]
            w = min(rows_, key=lambda s: s["acc"]); clean = next(s["acc"] for s in su if s["dataset"] == ds and s["defense"] == dfn and s["attack"] == "none")
            worst.append(dict(dataset=ds, defense=dfn, clean_acc=clean, worst_attack=w["attack"], worst_acc=w["acc"], drop=clean - w["acc"],
                              agr_tailored_acc=next((s["acc"] for s in rows_ if s["attack"] == "agr_tailored"), float("nan"))))
    save_csv(STORY, "task_3_8_untargeted_summary", su); save_csv(STORY, "task_3_8_backdoor_summary", sb); save_csv(STORY, "task_3_8_worst_case", worst)
    # does AGR-tailored beat the best fixed attack (paired by seed, accuracy)?
    cmp_ = []
    for ds in sorted({r["dataset"] for r in unt}):
        for dfn in E_.DEFENSES:
            def acc(atk): return np.array([r["acc"] for r in sorted(unt, key=lambda r: r["seed"]) if r["dataset"] == ds and r["defense"] == dfn and r["attack"] == atk])
            fixed = [a for a in ("alie", "minmax", "minsum", "fang") if len(acc(a))]
            best_fixed = min(fixed, key=lambda a: acc(a).mean()) if fixed else None
            if best_fixed and len(acc("agr_tailored")) >= 3:
                pt = S.paired_tests(acc("agr_tailored"), acc(best_fixed)); pt.update(dataset=ds, defense=dfn, best_fixed_attack=best_fixed, note="agr_tailored - best fixed attack (accuracy; negative = stronger)")
                cmp_.append(pt)
    save_csv(STORY, "task_3_8_agr_vs_best_fixed", cmp_)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.7))
    for i, atk in enumerate(ATK_UNT):
        axs[0].bar(np.arange(len(E_.DEFENSES)) + (i - 2.5) * 0.14, [next(s["acc"] for s in su if s["dataset"] == "syn" and s["defense"] == d and s["attack"] == atk) for d in E_.DEFENSES], 0.14, label=atk)
    axs[0].set_xticks(range(len(E_.DEFENSES))); axs[0].set_xticklabels(E_.DEFENSES, rotation=40, fontsize=7); axs[0].set_ylabel("test accuracy"); axs[0].set_title("Stronger untargeted attacks (synthetic, 20% attackers)"); axs[0].legend(fontsize=6, ncol=2)
    for i, atk in enumerate(ATK_BD):
        axs[1].bar(np.arange(len(E_.DEFENSES)) + (i - 1.5) * 0.2, [next(s["asr"] for s in sb if s["defense"] == d and s["attack"] == atk) for d in E_.DEFENSES], 0.2, label=atk)
    axs[1].set_xticks(range(len(E_.DEFENSES))); axs[1].set_xticklabels(E_.DEFENSES, rotation=40, fontsize=7); axs[1].set_ylabel("backdoor ASR"); axs[1].set_title("AGR-aware backdoor"); axs[1].legend(fontsize=7)
    savefig(fig, STORY, "task_3_8_stronger_adaptive", prov, f"{len(seeds)} seeds; omniscient + rule-aware attacker (upper bound)")
    out = dict(provenance=prov, untargeted=su, backdoor=sb, worst_case=worst, agr_vs_best_fixed=cmp_,
               caveat="rule-aware omniscient attacker = upper bound; AGR-tailored searches a fixed candidate set, not a global optimum, so it can be beaten by a better attacker; no loss-aware/learning-based attacker was implemented; synthetic/small data")
    save_json(STORY, "task_3_8_stronger_adaptive", out); return out
