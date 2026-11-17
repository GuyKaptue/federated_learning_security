"""Stage-2 experiments: four literature defences (FLAME, RLR, FoolsGold, SignGuard) against the existing attack suite, plus model baselines.

Everything is MEASURED on SYN-IDS-v1 (completely synthetic) or on the public four-site UCI heart-disease cohorts with virtual clients.
The attackers are the omniscient/colluding simulation attackers of ``attacks/adaptive.py`` (an upper bound, not a realistic adversary);
no defence is tuned per attack; parameters of the new defences are the paper's where the paper gives one and otherwise ours (stated in the class docstrings).
Detection metrics (TPR/FPR of rejected rows) exist only for rules that reject updates; they are averaged over rounds.
"""
from __future__ import annotations

import numpy as np

from federated_learning import attacks as A
from federated_learning.aggregators.base import Aggregator
from federated_learning.aggregators.defenses import AnomalyFiltered
from federated_learning.aggregators.foolsgold_indicative import make_foolsgold
from federated_learning.aggregators.literature_defenses import FLAME, FoolsGold, RobustLearningRate, SignGuard
from federated_learning.aggregators.mean import WeightedMean
from federated_learning.aggregators.robust import CoordinateMedian, Krum, TrimmedMean
from federated_learning.attacks.adaptive import AdaptiveAttackAggregator, MedianNormClip
from federated_learning.pipelines.experiments import ext as E_
from federated_learning.pipelines.experiments.common import OUT, mean_ci, plt, provenance, save_csv, save_json, savefig, scale
from federated_learning.pipelines.experiments.stage1 import _contrasts
from federated_learning.preprocessing.multisite_heart import load_sites
from federated_learning.servers.orchestrator import run_fl
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset
from federated_learning.training.centralized import centralized_baseline
from federated_learning.training.config import FLConfig

STORY = "fl_stage2"
K, F = 20, 4                                   # 20 clients, 4 colluding attackers (20%); rows 0..3 are the attackers
OLD = ["mean", "median", "trimmed_mean", "krum", "multi_krum", "clip_median_norm", "anomaly_filter"]
NEW = ["flame", "flame_noise", "rlr", "foolsgold", "signguard"]
DEFS = OLD + NEW
METRICS = ("acc", "f1", "auc")


def make_defence(name: str, n: int = K, f: int = F, seed: int = 0) -> Aggregator:
    return {"mean": WeightedMean, "median": CoordinateMedian, "trimmed_mean": lambda: TrimmedMean(min(0.45, f / n + 0.05)),
            "krum": lambda: Krum(f, 1), "multi_krum": lambda: Krum(f, n - f), "clip_median_norm": MedianNormClip, "anomaly_filter": AnomalyFiltered,
            "flame": lambda: FLAME(noise_lambda=0.0, seed=seed), "flame_noise": lambda: FLAME(noise_lambda=0.001, seed=seed),
            "rlr": lambda: RobustLearningRate(theta_frac=0.4), "foolsgold": FoolsGold, "foolsgold_corrected": lambda: make_foolsgold("allfeat_corrected"), "signguard": lambda: SignGuard(seed=seed)}[name]()


class DetectionRecorder(Aggregator):
    """Outermost wrapper: delegates, and records which rows the inner rule rejected (needs full participation: row == client id)."""

    def __init__(self, inner: Aggregator, mal_ids: set):
        super().__init__()
        self.inner, self.mal, self.name = inner, set(mal_ids), inner.name
        self.tp = self.fn = self.fp = self.tn = 0
        self.detecting = False

    def aggregate(self, deltas, weights, rnd=0):
        out = self.inner.aggregate(deltas, weights, rnd)
        st = self.inner.last_stats or {}
        n = len(deltas)
        rej = None
        if "dropped_idx" in st:
            rej = set(st["dropped_idx"])
        elif "selected" in st and len(st["selected"]) > 1:
            rej = set(range(n)) - set(st["selected"])
        self.last_stats = dict(st)
        if rej is not None:
            self.detecting = True
            ids = self.ctx_ids or list(range(n))
            for row in range(n):
                bad = ids[row] in self.mal
                if row in rej:
                    self.tp += bad; self.fp += (not bad)
                else:
                    self.fn += bad; self.tn += (not bad)
        return out

    def rates(self) -> tuple[float, float]:
        if not self.detecting:
            return float("nan"), float("nan")
        return self.tp / max(1, self.tp + self.fn), self.fp / max(1, self.fp + self.tn)


def _run(data, parts, seed, defence, attack, backdoor, rounds, E, lr, arch="mlp", knowledge="draw", jitter=0.0, make=None, n_mal=F):
    """attack: none | sign_flip | alie | minmax | agr_tailored | (backdoor) none_clean | static_boost | norm_matched | agr_backdoor"""
    mal = set(range(n_mal))
    base = (make or make_defence)(defence, len(parts), F, seed)
    kw: dict = {}
    hook = None
    if backdoor is not None:
        rng = np.random.default_rng(seed + 99); trig = backdoor["trig"]
        if attack != "none_clean":
            kw["client_data"] = {k: A.poison_client_data(data.X_train[parts[k]], data.y_train[parts[k]], backdoor["poison_frac"], 0, rng, *trig)[:2] for k in mal}
        kw["eval_fn"] = lambda m, w: A.backdoor_metrics(m, w, data.X_test, data.y_test, 0, *trig)
        if attack == "static_boost":
            hook = A.boost_hook(mal, backdoor["boost"]); inner = base
        elif attack == "none_clean":
            inner = base
        else:
            inner = AdaptiveAttackAggregator(base, mal, attack, knowledge=knowledge, jitter=jitter, seed=seed)
    elif attack == "sign_flip":
        hook = A.byzantine_hook("sign_flip", mal, 5.0, seed=seed); inner = base
    elif attack == "none":
        inner = base
    else:
        inner = AdaptiveAttackAggregator(base, mal, attack, knowledge=knowledge, jitter=jitter, seed=seed)
    rec = DetectionRecorder(inner, mal)   # for 'none' the designated rows behave honestly, so TPR is not meaningful there (reported for attack scenarios only)
    try:
        r = run_fl(FLConfig(rounds=rounds, local_epochs=E, lr=lr, seed=seed, arch=arch), data, parts, aggregator=rec, client_hook=hook, **kw)
        out = dict(r.final, diverged=0.0)
    except ValueError as e:
        if "non-finite" not in str(e):
            raise
        acc = float((data.y_test == 0).mean())
        out = dict(acc=acc, f1=0.0, auc=0.5, asr=1.0, clean_acc=acc, diverged=1.0)
    out["tpr"], out["fpr"] = rec.rates()
    return out


def _trigger():
    import pandas as pd
    from federated_learning.pipelines.experiments.attack003 import TRIGGER_CANDIDATES
    f = f"{OUT}/fl_attack_003/task_3_1_trigger_selection.csv"
    try:
        sel = pd.read_csv(f)
    except FileNotFoundError:           # e.g. a scratch results root in tests: fall back to the first candidate (documented in the cfg)
        return TRIGGER_CANDIDATES["f0-3=+4"]
    return TRIGGER_CANDIDATES[sel.sort_values("clean_model_asr").iloc[0]["trigger"]]


def _summ(rows, keys, ms):
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rr in groups.items():
        row = dict(zip(keys, g), n=len(rr), diverged=float(np.mean([r["diverged"] for r in rr])))
        for m in ms:
            v = [r[m] for r in rr if not (isinstance(r[m], float) and np.isnan(r[m]))]
            if v:
                a, lo, hi = mean_ci(v); row[m] = a; row[m + "_lo"] = lo; row[m + "_hi"] = hi
            else:
                row[m] = row[m + "_lo"] = row[m + "_hi"] = float("nan")
        out.append(row)
    return out


def _tests(rows, metric, refs=("mean", "median")):
    t = []
    for ref in refs:
        for x in _contrasts(rows, ("scenario", "partition"), "defence", ref, metric):
            t.append(dict(x, family=f"vs {ref}"))
    return t


def _plot(summ, scen_list, metric, title, fname, prov, cap):
    fig, axs = plt.subplots(1, len(scen_list), figsize=(3.6 * len(scen_list), 3.4), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, sc in zip(axs, scen_list):
        v = [next((x for x in summ if x["scenario"] == sc and x["partition"] == "iid" and x["defence"] == d), None) for d in DEFS]
        ys = [x[metric] if x else np.nan for x in v]
        err = [[(x[metric] - x[metric + "_lo"]) if x else 0 for x in v], [(x[metric + "_hi"] - x[metric]) if x else 0 for x in v]]
        ax.bar(range(len(DEFS)), ys, yerr=err, capsize=1.5, color=["#888"] * len(OLD) + ["#c44"] * len(NEW))
        ax.set_xticks(range(len(DEFS))); ax.set_xticklabels(DEFS, rotation=60, ha="right", fontsize=6); ax.set_title(sc, fontsize=8)
    axs[0].set_ylabel(title)
    savefig(fig, STORY, fname, prov, cap)


def task_s2_1_untargeted_defences() -> dict:
    seeds = list(range(scale(10, 2))); rounds = scale(20, 3)
    data = make_dataset(seed=0, n_samples=scale(12000, 2500))
    scen = [("none", "iid"), ("sign_flip", "iid"), ("alie", "iid"), ("minmax", "iid"), ("agr_tailored", "iid"), ("none", "dirichlet0.3"), ("sign_flip", "dirichlet0.3")]
    cfg = dict(K=K, f=F, rounds=rounds, seeds=len(seeds), defences=DEFS, scenarios=scen, E=1, lr=0.05, attacker="omniscient colluding (upper bound)", sign_flip_scale=5.0,
               new_defence_params=dict(flame_lambda=[0.0, 0.001], rlr_theta_frac=0.4, foolsgold_kappa=1.0, signguard=dict(L=0.1, R=3.0, coord_frac=0.1)))
    prov = provenance("FL-S2-001", cfg)
    rows = []
    for sd in seeds:
        for atk, part in scen:
            parts = E_._parts(data.y_train, K, "iid" if part == "iid" else "0.3", sd)
            for d in DEFS:
                fin = _run(data, parts, sd, d, atk, None, rounds, 1, 0.05)
                rows.append(dict(scenario=atk, partition=part, defence=d, seed=sd, **{m: fin[m] for m in METRICS}, tpr=fin["tpr"], fpr=fin["fpr"], diverged=fin["diverged"]))
        print(f"  s2.1 seed {sd} done", flush=True)
    summ = _summ(rows, ("scenario", "partition", "defence"), METRICS + ("tpr", "fpr"))
    tests = _tests(rows, "acc")
    save_csv(STORY, "s2_1_runs", rows); save_csv(STORY, "s2_1_summary", summ); save_csv(STORY, "s2_1_tests", tests)
    _plot(summ, ["none", "sign_flip", "alie", "minmax", "agr_tailored"], "acc", "test accuracy (95% CI)", "s2_1_untargeted", prov, f"{len(seeds)} seeds, 20% omniscient attackers; red = new")
    out = dict(provenance=prov, summary=summ, tests=tests,
               caveat="omniscient attacker is an upper bound; new defences use the paper's parameters where stated, otherwise ours; detection rates exist only for rejecting rules")
    save_json(STORY, "s2_1_untargeted_defences", out)
    return out


def task_s2_2_backdoor_defences() -> dict:
    seeds = list(range(scale(10, 2))); rounds = scale(20, 3)
    data = make_dataset(seed=0, n_samples=scale(12000, 2500))
    trig = _trigger(); BD = dict(trig=trig, poison_frac=0.5, boost=3.0)
    scen = [("none_clean", "iid"), ("static_boost", "iid"), ("norm_matched", "iid"), ("agr_backdoor", "iid"), ("none_clean", "dirichlet0.3"), ("static_boost", "dirichlet0.3")]
    cfg = dict(K=K, f=F, rounds=rounds, seeds=len(seeds), defences=DEFS, scenarios=scen, E=1, lr=0.05, trigger=str(trig), poison_frac=0.5, boost=3.0,
               attacker="data-poisoning clients (+ omniscient update replacement for adaptive variants; upper bound)")
    prov = provenance("FL-S2-002", cfg)
    rows = []
    for sd in seeds:
        for atk, part in scen:
            parts = E_._parts(data.y_train, K, "iid" if part == "iid" else "0.3", sd)
            for d in DEFS:
                fin = _run(data, parts, sd, d, atk, BD, rounds, 1, 0.05)
                rows.append(dict(scenario=atk, partition=part, defence=d, seed=sd, asr=fin["asr"], clean_acc=fin["clean_acc"], acc=fin["clean_acc"],
                                 tpr=fin["tpr"], fpr=fin["fpr"], diverged=fin["diverged"]))
        print(f"  s2.2 seed {sd} done", flush=True)
    summ = _summ(rows, ("scenario", "partition", "defence"), ("asr", "clean_acc", "tpr", "fpr"))
    tests = []
    for ref in ("mean", "median"):
        for x in _contrasts(rows, ("scenario", "partition"), "defence", ref, "asr"):
            tests.append(dict(x, family=f"vs {ref}"))
    save_csv(STORY, "s2_2_runs", rows); save_csv(STORY, "s2_2_summary", summ); save_csv(STORY, "s2_2_tests", tests)
    _plot(summ, ["none_clean", "static_boost", "norm_matched", "agr_backdoor"], "asr", "backdoor ASR (95% CI)", "s2_2_backdoor", prov, f"{len(seeds)} seeds; red = new")
    out = dict(provenance=prov, summary=summ, tests=tests, caveat="ASR measured with the stored trigger of task 3.1; omniscient adaptive variants are upper bounds; synthetic data only")
    save_json(STORY, "s2_2_backdoor_defences", out)
    return out


BD_DEFAULT = dict(poison_frac=0.5, boost=3.0)
ARCH_LIST = ("logreg", "mlp", "deep_mlp")


def task_s2_3_baselines() -> dict:
    seeds = list(range(scale(10, 2))); rounds = scale(40, 3); K_ = 10
    data = make_dataset(seed=0, n_samples=scale(12000, 2500))
    cfg = dict(archs=ARCH_LIST, K_synthetic=K_, rounds=rounds, lr=0.1, local_epochs=2, hidden=32, seeds=len(seeds), same_budget_for_all_archs=True,
               datasets=["synthetic dirichlet(0.3)", "heart-4sites"], robust_probe=dict(K=K, f=F, attack="sign_flip x5", aggregators=["mean", "median"], rounds=scale(20, 3), lr=0.05, E=1))
    prov = provenance("FL-S2-003", cfg)
    rows = []
    for sd in seeds:
        parts = E_._parts(data.y_train, K_, "0.3", sd)
        sites = load_sites(sd)
        iid = E_._parts(data.y_train, K, "iid", sd)
        for a in ARCH_LIST:
            for ds, d, pt in (("synthetic", data, parts), ("heart-4sites", sites, sites.parts)):
                r = run_fl(FLConfig(rounds=rounds, local_epochs=2, lr=0.1, seed=sd, arch=a), d, pt)
                rows.append(dict(dataset=ds, setting="fedavg", arch=a, seed=sd, **{m: r.final[m] for m in METRICS}, n_params=r.totals["n_params"], comm_mb=r.totals["comm_mb"]))
                c = centralized_baseline(d, epochs=2 * rounds, lr=0.1, seed=sd, arch=a)
                rows.append(dict(dataset=ds, setting="centralised", arch=a, seed=sd, **{m: c[m] for m in METRICS}, n_params=r.totals["n_params"], comm_mb=0.0))
            for dn in ("mean", "median"):
                fin = _run(data, iid, sd, dn, "sign_flip", None, scale(20, 3), 1, 0.05, arch=a)
                rows.append(dict(dataset="synthetic", setting=f"{dn}+signflip20%", arch=a, seed=sd, **{m: fin[m] for m in METRICS}, n_params=0, comm_mb=0.0))
        print(f"  s2.3 seed {sd} done", flush=True)
    summ = _summ([dict(r, diverged=0.0) for r in rows], ("dataset", "setting", "arch"), METRICS)
    tests = []
    for x in _contrasts(rows, ("dataset", "setting"), "arch", "mlp", "f1"):
        tests.append(x)
    save_csv(STORY, "s2_3_runs", rows); save_csv(STORY, "s2_3_summary", summ); save_csv(STORY, "s2_3_tests", tests)
    out = dict(provenance=prov, summary=summ, tests=tests,
               caveat="same lr/rounds/epochs for every architecture (no per-architecture tuning); centralised = pooled-data SGD with 2 x rounds epochs (the same number of local passes per sample as FedAvg)")
    save_json(STORY, "s2_3_baselines", out)
    return out
