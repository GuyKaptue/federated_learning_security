"""Stage-3 experiments: randomised rule committee (RRC) under an attacker-knowledge ladder, detection metrics and ablations.

Setting (all MEASURED on SYN-IDS-v1, completely synthetic; virtual clients): K=20, f=4 colluding omniscient attackers (rows 0..3), 20 rounds.
Knowledge ladder for rule-simulating attackers:
  draw  = attacker simulates the exact committee that will run (shares the private random draw)  -> worst case, == a deterministic committee;
  pool  = attacker knows pool, k and vote rule but each of its simulations sees an independent random draw.
Jitter = the colluders add independent noise of relative size 1.0 to their rows (pairwise cosine ~0.5), which removes the
trivial 'all malicious rows are identical' signature of the idealised AGR attackers.
Not new in concept: randomised/hybrid aggregation exists (Nabavirazavi et al. 2023; Yue et al. 2024). Only the measurements here are ours.
"""
from __future__ import annotations

import numpy as np

from federated_learning.aggregators.base import Aggregator
from federated_learning.aggregators.committee import RuleCommittee
from federated_learning.pipelines.experiments import ext as E_
from federated_learning.pipelines.experiments.common import plt, provenance, save_csv, save_json, savefig, scale
from federated_learning.pipelines.experiments.stage1 import _contrasts
from federated_learning.pipelines.experiments.stage2 import (
    BD_DEFAULT,
    F,
    K,
    METRICS,
    _plot,
    _run,
    _summ,
    _trigger,
    make_defence,
)
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset

STORY = "fl_stage3"
POOL = ("anomaly_filter", "multi_krum", "flame", "signguard")
BASE = ["mean", "median", "multi_krum", "flame", "signguard", "foolsgold", "anomaly_filter"]
COMMITTEES = {"rc_all_v1": dict(k=4, vote=1, randomise=False), "rc_all_v2": dict(k=4, vote=2, randomise=False),
              "rrc_k2_v1": dict(k=2, vote=1, randomise=True), "rrc_k3_v2": dict(k=3, vote=2, randomise=True)}
DEFS = BASE + list(COMMITTEES)
RANDOMISED = [d for d, c in COMMITTEES.items() if c["randomise"]]


def make3(name: str, n: int = K, f: int = F, seed: int = 0, **override) -> Aggregator:
    if name in COMMITTEES:
        c = dict(COMMITTEES[name], **override)
        return RuleCommittee([make_defence(m, n, F, seed) for m in POOL], seed=seed * 31 + 7, **c)
    return make_defence(name, n, f, seed)


def _scenarios_u():
    s = [("none", "none", "draw", 0.0), ("sign_flip", "sign_flip", "draw", 0.0), ("alie", "alie", "draw", 0.0),
         ("agr", "agr_tailored", "draw", 0.0), ("agr+jit", "agr_tailored", "draw", 1.0)]
    return s


def _scenarios_b():
    return [("none_clean", "none_clean", "draw", 0.0), ("static_boost", "static_boost", "draw", 0.0), ("norm_matched", "norm_matched", "draw", 0.0),
            ("agr_bd", "agr_backdoor", "draw", 0.0), ("agr_bd+jit", "agr_backdoor", "draw", 1.0)]


def _pool_scen(base: list) -> list:
    return [(f"{n}[pool]", a, "pool", j) for n, a, _k, j in base if n.startswith("agr")]


def _loop(task_id, tag, scen, extra_scen, noniid, data, bd, seeds, rounds):
    rows = []
    for sd in seeds:
        for part_name, sc_list in (("iid", scen), ("dirichlet0.3", noniid)):
            parts = E_._parts(data.y_train, K, "iid" if part_name == "iid" else "0.3", sd)
            for lab, atk, kn, jit in sc_list:
                for d in DEFS:
                    if "[pool]" in lab and d not in RANDOMISED:
                        continue                                    # for deterministic rules draw-aware == pool-aware
                    fin = _run(data, parts, sd, d, atk, bd, rounds, 1, 0.05, knowledge=kn, jitter=jit, make=make3)
                    rows.append(dict(scenario=lab, partition=part_name, defence=d, seed=sd, **{m: fin[m] for m in fin if m in ("acc", "f1", "auc", "asr", "clean_acc", "tpr", "fpr", "diverged")}))
        print(f"  {tag} seed {sd} done", flush=True)
    return rows


def task_s3_1_untargeted() -> dict:
    seeds = list(range(scale(5, 2))); rounds = scale(15, 3)
    data = make_dataset(seed=0, n_samples=scale(8000, 2500))
    sc = _scenarios_u(); sc = sc + _pool_scen(sc)
    cfg = dict(K=K, f=F, rounds=rounds, seeds=len(seeds), defences=DEFS, pool=POOL, committees=COMMITTEES, scenarios=[s[0] for s in sc], jitter=1.0,
               attacker="omniscient colluding (upper bound); knowledge ladder draw|pool")
    prov = provenance("FL-S3-001", cfg)
    rows = _loop("FL-S3-001", "s3.1", sc, None, [("none", "none", "draw", 0.0), ("agr+jit", "agr_tailored", "draw", 1.0)], data, None, seeds, rounds)
    summ = _summ(rows, ("scenario", "partition", "defence"), METRICS + ("tpr", "fpr"))
    tests = []
    for ref in ("median", "foolsgold"):
        tests += [dict(x, family=f"vs {ref}") for x in _contrasts(rows, ("scenario", "partition"), "defence", ref, "acc")]
    save_csv(STORY, "s3_1_runs", rows); save_csv(STORY, "s3_1_summary", summ); save_csv(STORY, "s3_1_tests", tests)
    out = dict(provenance=prov, summary=summ, tests=tests, caveat="omniscient upper-bound attackers; synthetic data; committee pool and thresholds fixed a priori, not tuned per attack")
    save_json(STORY, "s3_1_untargeted", out)
    return out


def task_s3_2_backdoor() -> dict:
    seeds = list(range(scale(5, 2))); rounds = scale(15, 3)
    data = make_dataset(seed=0, n_samples=scale(8000, 2500))
    BD = dict(BD_DEFAULT, trig=_trigger())
    sc = _scenarios_b(); sc = sc + _pool_scen(sc)
    cfg = dict(K=K, f=F, rounds=rounds, seeds=len(seeds), defences=DEFS, pool=POOL, committees=COMMITTEES, scenarios=[s[0] for s in sc], jitter=1.0, trigger=str(BD["trig"]),
               poison_frac=BD["poison_frac"], boost=BD["boost"])
    prov = provenance("FL-S3-002", cfg)
    rows = _loop("FL-S3-002", "s3.2", sc, None, [("none_clean", "none_clean", "draw", 0.0), ("static_boost", "static_boost", "draw", 0.0)], data, BD, seeds, rounds)
    for r in rows:
        r.setdefault("acc", r.get("clean_acc"))
    summ = _summ(rows, ("scenario", "partition", "defence"), ("asr", "clean_acc", "tpr", "fpr"))
    tests = []
    for ref in ("median", "foolsgold"):
        tests += [dict(x, family=f"vs {ref}") for x in _contrasts(rows, ("scenario", "partition"), "defence", ref, "asr")]
    save_csv(STORY, "s3_2_runs", rows); save_csv(STORY, "s3_2_summary", summ); save_csv(STORY, "s3_2_tests", tests)
    out = dict(provenance=prov, summary=summ, tests=tests, caveat="omniscient upper-bound attackers; stored task-3.1 trigger; synthetic data")
    save_json(STORY, "s3_2_backdoor", out)
    return out


def task_s3_3_ablations() -> dict:
    seeds = list(range(scale(4, 2))); rounds = scale(15, 3)
    data = make_dataset(seed=0, n_samples=scale(8000, 2500))
    BD = dict(BD_DEFAULT, trig=_trigger())
    ks = scale([1, 2, 3, 4], [1, 4]); votes = scale([1, 2, 3], [1, 2]); fs = scale([2, 4, 6, 8], [2, 4])
    cfg = dict(K=K, rounds=rounds, seeds=len(seeds), attacks=["agr_tailored+jit", "agr_backdoor+jit"], k_values=ks, vote_values=votes, f_values=fs,
               defender_krum_f=F, knowledge=["draw", "pool"], jitter=1.0)
    prov = provenance("FL-S3-003", cfg)
    rows = []
    for sd in seeds:
        parts = E_._parts(data.y_train, K, "iid", sd)
        for atk, bd, lab in (("agr_tailored", None, "untargeted"), ("agr_backdoor", BD, "backdoor")):
            for kn in ("draw", "pool"):
                for k in ks:                                                  # (a) committee size x knowledge (random draw, vote 1)
                    fin = _run(data, parts, sd, "rrc", atk, bd, rounds, 1, 0.05, knowledge=kn, jitter=1.0,
                               make=lambda n, nn, ff, ss, k=k: make3("rrc_k2_v1", seed=ss, k=k, vote=1, randomise=True))
                    rows.append(dict(axis="k", level=k, attack=lab, knowledge=kn, seed=sd, **{m: fin[m] for m in ("acc", "asr", "clean_acc", "tpr", "fpr", "diverged") if m in fin}))
            for v in votes:                                                   # (b) vote threshold, all four members, deterministic
                fin = _run(data, parts, sd, "rc", atk, bd, rounds, 1, 0.05, knowledge="draw", jitter=1.0,
                           make=lambda n, nn, ff, ss, v=v: make3("rc_all_v1", seed=ss, k=4, vote=v, randomise=False))
                rows.append(dict(axis="vote", level=v, attack=lab, knowledge="draw", seed=sd, **{m: fin[m] for m in ("acc", "asr", "clean_acc", "tpr", "fpr", "diverged") if m in fin}))
            for f_ in fs:                                                     # (c) attacker fraction (defender assumes f=4)
                for nm, kn in (("rc_all_v1", "draw"), ("rrc_k2_v1", "pool")):
                    fin = _run(data, parts, sd, nm, atk, bd, rounds, 1, 0.05, knowledge=kn, jitter=1.0, make=make3, n_mal=f_)
                    rows.append(dict(axis="f", level=f_, attack=lab, knowledge=kn, arm=nm, seed=sd, **{m: fin[m] for m in ("acc", "asr", "clean_acc", "tpr", "fpr", "diverged") if m in fin}))
        print(f"  s3.3 seed {sd} done", flush=True)
    for r in rows:
        r.setdefault("arm", "rrc" if r["axis"] == "k" else "rc_all")
        for m in ("acc", "asr", "clean_acc", "tpr", "fpr"):
            r.setdefault(m, float("nan"))
    summ = _summ(rows, ("axis", "level", "attack", "knowledge", "arm"), ("acc", "asr", "clean_acc", "tpr", "fpr"))
    save_csv(STORY, "s3_3_runs", rows); save_csv(STORY, "s3_3_summary", summ)
    out = dict(provenance=prov, summary=summ, caveat="one attack family at a time; defender's Krum assumption fixed at f=4 in the attacker-fraction sweep (misspecified for f=6, 8)")
    save_json(STORY, "s3_3_ablations", out)
    return out


__all__ = ["task_s3_1_untargeted", "task_s3_2_backdoor", "task_s3_3_ablations", "plt", "savefig", "_plot", "np"]
