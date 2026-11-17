"""E3 baseline reproduction gate: ordinary collusion (AGR-backdoor, unchanged attacker) with 10 seeds. NO diversified attacker is run here.

Same configuration as FL-S3-002 (K=20, F=4, 15 rounds, 8000 samples, i.i.d. clients, stored trigger, poison_frac 0.5, boost 3.0, knowledge 'draw',
jitter 0) restricted to the seven non-committee rules; seeds 0..9. Seeds 0..4 are compared value by value with the stored FL-S3-002 runs.
Outputs go to outputs/research/adaptive_attacks/e3/baseline/ and never touch experiments/results/.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from federated_learning.pipelines.experiments import ext as E_
from federated_learning.pipelines.experiments.common import ROOT, _json_default, provenance, scale
from federated_learning.pipelines.experiments.stage2 import BD_DEFAULT, F, K, _run, _summ, _trigger, make_defence
from federated_learning.pipelines.experiments.stage3 import BASE
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset

SCEN = ["none_clean", "static_boost", "norm_matched", "agr_backdoor"]
OUTDIR = ROOT / "outputs" / "research" / "adaptive_attacks" / "e3" / "baseline"


def task_e3_baseline() -> dict:
    seeds = list(range(scale(10, 2))); rounds = scale(15, 3)
    data = make_dataset(seed=0, n_samples=scale(8000, 2500))
    BD = dict(BD_DEFAULT, trig=_trigger())
    cfg = dict(K=K, f=F, rounds=rounds, seeds=seeds, defences=BASE, scenarios=SCEN, partition="iid", knowledge="draw", jitter=0.0,
               poison_frac=BD["poison_frac"], boost=BD["boost"], trigger=str(BD["trig"]), local_epochs=1, lr=0.05, n_samples=int(len(data.y_train) + len(data.y_test)) if hasattr(data, "y_test") else None)
    prov = provenance("FL-E3-BASE", cfg)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    t_all = time.time()
    for sd in seeds:
        parts = E_._parts(data.y_train, K, "iid", sd)
        for atk in SCEN:
            for d in BASE:
                t0 = time.time()
                fin = _run(data, parts, sd, d, atk, BD, rounds, 1, 0.05, knowledge="draw", jitter=0.0, make=make_defence)
                rows.append(dict(scenario=atk, defence=d, seed=sd, runtime_s=round(time.time() - t0, 2),
                                 **{m: fin[m] for m in ("acc", "f1", "auc", "asr", "clean_acc", "tpr", "fpr", "diverged") if m in fin}))
        pd.DataFrame(rows).to_csv(OUTDIR / "e3_baseline_runs.csv", index=False)       # checkpoint after every seed
        print(f"  e3 baseline seed {sd} done ({time.time() - t_all:.0f}s)", flush=True)
    for r in rows:
        r.setdefault("clean_acc", r.get("acc"))
    summ = _summ(rows, ("scenario", "defence"), ("asr", "clean_acc", "tpr", "fpr"))
    pd.DataFrame(summ).to_csv(OUTDIR / "e3_baseline_summary.csv", index=False)
    # reproduction check against the stored FL-S3-002 seeds 0..4 (same names: agr_bd == agr_backdoor, partition iid, draw knowledge)
    chk = {}
    old = ROOT / "experiments" / "results" / "fl_stage3" / "s3_2_runs.csv"
    if old.exists():
        o = pd.read_csv(old); o = o[(o.partition == "iid") & o.scenario.isin(["none_clean", "static_boost", "norm_matched", "agr_bd"]) & o.defence.isin(BASE)]
        o = o.assign(scenario=o.scenario.replace({"agr_bd": "agr_backdoor"}))
        n = pd.DataFrame(rows)
        m = o.merge(n, on=["scenario", "defence", "seed"], suffixes=("_old", "_new"))
        for col in ("asr", "clean_acc"):
            dv = (m[col + "_old"] - m[col + "_new"]).abs()
            chk[col] = dict(n_pairs=int(len(m)), max_abs_diff=float(dv.max()), n_exact=int((dv < 1e-12).sum()))
    out = dict(provenance=prov, summary=summ, reproduction_check_vs_FL_S3_002=chk, total_runtime_s=round(time.time() - t_all, 1))
    (OUTDIR / "e3_baseline.json").write_text(json.dumps(out, indent=1, default=_json_default))
    (OUTDIR / "e3_config.json").write_text(json.dumps(cfg, indent=1, default=_json_default))
    return out


__all__ = ["task_e3_baseline", "np"]
