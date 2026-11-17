"""E3 Phase 4A: legacy vs paper-faithful (indicative-feature) FoolsGold under no attack, identical colluders and the *existing* jitter-1.0 colluders.

No new attacker: the AGR-backdoor attacker (12 candidates), its `jitter` option and all settings are the ones already used by FL-S3-002 / FL-E3-BASE.
Arms: legacy | paper_soft (primary) | paper_hard (p=0.1, sensitivity) | allfeat_corrected (diagnostic: legacy similarity, corrected alpha=0 handling).
Seeds 0..9, 15 rounds, K=20, F=4, same data, partitions iid (primary) and Dirichlet(0.3) (secondary). Outputs: outputs/research/adaptive_attacks/e3/fg_validation/.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy import stats

from federated_learning.aggregators.base import Aggregator
from federated_learning.aggregators.foolsgold_indicative import make_foolsgold, output_layer_from_model
from federated_learning.evaluation.statistics import cliffs_delta, cohens_dz, holm, t_ci
from federated_learning.pipelines.experiments import ext as E_
from federated_learning.pipelines.experiments.common import ROOT, _json_default, provenance, scale
from federated_learning.pipelines.experiments.stage2 import BD_DEFAULT, F, K, _run, _trigger
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset
from federated_learning.training.models import build_model

ARMS = ["legacy", "paper_soft", "paper_hard", "allfeat_corrected"]
SCEN = [("none_clean", "none_clean", 0.0), ("agr_identical", "agr_backdoor", 0.0), ("agr_jitter1.0", "agr_backdoor", 1.0)]
PARTS = [("iid", "iid"), ("dirichlet0.3", "0.3")]
OUTDIR = ROOT / "outputs" / "research" / "adaptive_attacks" / "e3" / "fg_validation"


def _cos_mean(R: np.ndarray) -> float:
    if len(R) < 2:
        return float("nan")
    U = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    C = U @ U.T
    return float((C.sum() - len(R)) / (len(R) * (len(R) - 1)))


class StatsTap(Aggregator):
    """Transparent wrapper around the defence under test: records, per real aggregation, each row's alpha and the round-update similarity."""

    def __init__(self, inner: Aggregator, mal_ids):
        super().__init__()
        self.inner, self.mal, self.name = inner, set(mal_ids), inner.name
        self.rows: list[dict] = []

    def aggregate(self, deltas, weights, rnd=0):
        out = self.inner.aggregate(deltas, weights, rnd)
        st = self.inner.last_stats or {}
        self.last_stats = dict(st)
        ids = self.ctx_ids or list(range(len(deltas)))
        m = np.array([i in self.mal for i in ids])
        D = np.asarray(deltas, dtype=np.float64)
        a = np.asarray(st.get("alpha", np.full(len(ids), np.nan)), dtype=float)
        self.rows.append(dict(rnd=rnd, a_mal=float(a[m].mean()) if m.any() else np.nan, a_ben=float(a[~m].mean()),
                              mal_lt_half=float((a[m] < 0.5).mean()) if m.any() else np.nan, ben_lt_half=float((a[~m] < 0.5).mean()),
                              cos_mal=_cos_mean(D[m]) if m.sum() > 1 else np.nan, cos_ben=_cos_mean(D[~m])))
        return out

    def summary(self) -> dict:
        d = pd.DataFrame(self.rows)
        return {c: float(d[c].mean()) for c in ("a_mal", "a_ben", "mal_lt_half", "ben_lt_half", "cos_mal", "cos_ben")}


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    m, lo, hi = t_ci(d) if len(d) > 1 and d.std() > 0 else (float(d.mean()), float(d.mean()), float(d.mean()))
    res = dict(n=len(d), mean_diff=m, sd_diff=float(d.std(ddof=1)), ci_lo=lo, ci_hi=hi, dz=cohens_dz(a, b),
               cliff_paired=float(((d > 0).sum() - (d < 0).sum()) / len(d)), n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()))
    res["ttest_p"] = float(stats.ttest_rel(a, b).pvalue) if np.any(d != 0) else 1.0
    try:
        res["wilcoxon_p"] = float(stats.wilcoxon(a, b, method="exact").pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        res["wilcoxon_p"] = 1.0
    return res


def analyse(runs: pd.DataFrame) -> pd.DataFrame:
    """Primary family per partition (Holm): jitter effect within each arm (4) and its difference against legacy for the three other arms (3)."""
    out = []
    for part in runs.partition.unique():
        r = runs[runs.partition == part]
        wide = lambda arm, sc, col="asr": r[(r.arm == arm) & (r.scenario == sc)].set_index("seed")[col].sort_index()   # noqa: E731
        fam = []
        for arm in ARMS:
            fam.append(dict(partition=part, family="primary", comparison=f"jitter - identical | {arm}", **_paired(wide(arm, "agr_jitter1.0").values, wide(arm, "agr_identical").values)))
        leg = wide("legacy", "agr_jitter1.0") - wide("legacy", "agr_identical")
        for arm in ARMS[1:]:
            x = wide(arm, "agr_jitter1.0") - wide(arm, "agr_identical")
            fam.append(dict(partition=part, family="primary", comparison=f"(jitter - identical): {arm} - legacy", **_paired(x.values, leg.values)))
        for rec, p in zip(fam, holm([f["wilcoxon_p"] for f in fam]), strict=True):
            rec["wilcoxon_p_holm"] = p
        for rec, p in zip(fam, holm([f["ttest_p"] for f in fam]), strict=True):
            rec["ttest_p_holm"] = p
        out += fam
        for arm in ARMS:                                        # descriptive, not in the Holm family
            for sc in ("agr_identical", "agr_jitter1.0"):
                out.append(dict(partition=part, family="descriptive", comparison=f"{sc} - none_clean | {arm}", **_paired(wide(arm, sc).values, wide(arm, "none_clean").values)))
        for sc in ("none_clean", "agr_identical", "agr_jitter1.0"):
            for col in ("clean_acc",):
                for arm in ARMS[1:]:
                    out.append(dict(partition=part, family="descriptive", comparison=f"{col}: {arm} - legacy | {sc}", **_paired(wide(arm, sc, col).values, wide("legacy", sc, col).values)))
    return pd.DataFrame(out)


def task_e3_fg_validation() -> dict:
    seeds = list(range(scale(10, 2))); rounds = scale(15, 3)
    data = make_dataset(seed=0, n_samples=scale(8000, 2500))
    BD = dict(BD_DEFAULT, trig=_trigger())
    model = build_model(data.X_train.shape[1], arch="mlp")
    out_idx, n_out = output_layer_from_model(model)
    cfg = dict(K=K, f=F, rounds=rounds, seeds=seeds, arms=ARMS, scenarios=[s[0] for s in SCEN], partitions=[p[0] for p in PARTS], arch="mlp", n_params=int(sum(p.numel() for p in model.parameters())),
               output_layer_weight_index=[int(out_idx[0]), int(out_idx[-1])], n_out=int(n_out), hard_proportion=0.1, kappa=1.0, importance="|W_out| per-class max-normalised (adaptation; the paper gives no formula)",
               knowledge="draw", poison_frac=BD["poison_frac"], boost=BD["boost"], trigger=str(BD["trig"]), local_epochs=1, lr=0.05, new_attacker=False)
    prov = provenance("FL-E3-FGVAL", cfg)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "e3_fg_config.json").write_text(json.dumps(cfg, indent=1, default=_json_default))
    rows, t_all = [], time.time()
    for sd in seeds:
        for pname, pkind in PARTS:
            parts = E_._parts(data.y_train, K, pkind, sd)
            for arm in ARMS:
                for lab, atk, jit in SCEN:
                    holder: dict = {}

                    def make(name, n, f, seed, arm=arm, holder=holder):
                        tap = StatsTap(make_foolsgold(arm, out_idx, n_out), set(range(F)))
                        holder["tap"] = tap
                        return tap
                    t0 = time.time()
                    fin = _run(data, parts, sd, "foolsgold", atk, BD, rounds, 1, 0.05, knowledge="draw", jitter=jit, make=make)
                    rows.append(dict(partition=pname, arm=arm, scenario=lab, seed=sd, runtime_s=round(time.time() - t0, 2),
                                     **{m: fin[m] for m in ("acc", "f1", "auc", "asr", "clean_acc", "tpr", "fpr", "diverged") if m in fin}, **holder["tap"].summary()))
        pd.DataFrame(rows).to_csv(OUTDIR / "e3_fg_runs.csv", index=False)
        print(f"  e3 fg-validation seed {sd} done ({time.time() - t_all:.0f}s)", flush=True)
    runs = pd.DataFrame(rows)
    cols = ["asr", "clean_acc", "f1", "tpr", "fpr", "a_mal", "a_ben", "mal_lt_half", "cos_mal", "cos_ben", "runtime_s"]
    g = runs.groupby(["partition", "arm", "scenario"])[cols]
    summ = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()
    summ.to_csv(OUTDIR / "e3_fg_summary.csv", index=False)
    stats_df = analyse(runs)
    stats_df.to_csv(OUTDIR / "e3_fg_paired_stats.csv", index=False)
    # reproduction checks of the legacy arm against stored results (the tap must not change numerics)
    chk = {}
    base = ROOT / "outputs" / "research" / "adaptive_attacks" / "e3" / "baseline" / "e3_baseline_runs.csv"
    if base.exists():
        b = pd.read_csv(base); b = b[b.defence == "foolsgold"].assign(scenario=lambda x: x.scenario.replace({"agr_backdoor": "agr_identical"}))
        m = runs[(runs.arm == "legacy") & (runs.partition == "iid")].merge(b, on=["scenario", "seed"], suffixes=("_new", "_base"))
        dv = (m.asr_new - m.asr_base).abs()
        chk["vs_e3_baseline_iid"] = dict(n_pairs=int(len(m)), max_abs_diff=float(dv.max()) if len(m) else None, n_exact=int((dv < 1e-12).sum()))
    old = ROOT / "experiments" / "results" / "fl_stage3" / "s3_2_runs.csv"
    if old.exists():
        o = pd.read_csv(old); o = o[(o.partition == "iid") & (o.defence == "foolsgold") & (o.scenario == "agr_bd+jit")].assign(scenario="agr_jitter1.0")
        m = runs[(runs.arm == "legacy") & (runs.partition == "iid")].merge(o, on=["scenario", "seed"], suffixes=("_new", "_old"))
        dv = (m.asr_new - m.asr_old).abs()
        chk["vs_s3_002_jitter_iid"] = dict(n_pairs=int(len(m)), max_abs_diff=float(dv.max()) if len(m) else None, n_exact=int((dv < 1e-12).sum()))
    res = dict(provenance=prov, reproduction_checks=chk, total_runtime_s=round(time.time() - t_all, 1))
    (OUTDIR / "e3_fg_validation.json").write_text(json.dumps(res, indent=1, default=_json_default))
    return res


__all__ = ["task_e3_fg_validation"]
