# 00 — Research Implementation Plan

Scope: EPIC-001 only. Traceability chain per task: research question → reference → math → module → notebook → result → conclusion.

## Architecture (new modules; nothing pre-existing to extend)

```
src/federated_learning/ (formerly src/research_sprint/)
  data.py            synthetic IDS-style dataset, IID / Dirichlet partitioning
  models.py          small MLP + flat-parameter helpers
  fl.py              FL simulator: FedAvg, FedProx, SCAFFOLD, FedNova, pluggable aggregation + client hooks
  monitoring.py      resource / communication monitor, MLflow logging helper
  defenses.py        median, trimmed mean, Krum/Multi-Krum, clipping, DP noise, anomaly filter
  attacks.py         backdoor, Byzantine, membership inference, DLG/iDLG + analytic inversion
  stats.py           bootstrap CI, Wilcoxon, paired t, Holm, effect sizes, Friedman
  crypto/he.py       CKKS + BFV (TenSEAL), Paillier (phe)
  crypto/secagg.py   Bonawitz-style pairwise masking with Shamir dropout recovery
  crypto/smpc.py     additive sharing + SPDZ-style MAC check + Beaver multiplication
  experiments/       one function per notebook; each returns & saves metrics/figures
```

## Execution order and exit criteria

| Phase | Story | Exit criteria (all must hold before "COMPLETE") |
|---|---|---|
| 0 | Assessment | this document + assessment written |
| 1 | FL-SEC-001 | HE/SecAgg/SMPC unit + protocol tests pass; 5 notebooks executed; comparison table from measured data |
| 2 | FL-PERF-002 | monitor tests pass; 10/50/100/500-client runs measured (virtual clients); MLflow runs logged |
| 3 | FL-ATTACK-003 | 4 attacks × 3 intensities; defenses before/after; KPI table with honest pass/fail |
| 4 | FL-BENCH-004 | 4 algorithms, IID + non-IID, ≥5 seeds, Wilcoxon + Holm + effect sizes |
| 5 | FL-PUB-005 | reproducibility audit, ablation, threat model, report, manuscript draft, ZIP verified |

## Honesty rules applied to every artefact

* Every number in docs is read from a generated JSON/CSV in `experiments/results/`.
* Labels used: **MEASURED** (timed/counted in this run), **SIMULATED** (virtual clients), **ESTIMATED** (formula on measured inputs), **THEORETICAL**.
* Anything not run is written as `NOT_RUN` with the reason.
* No claim of regulatory compliance; no claim of cryptographic security from "decrypts correctly".

## Known constraints

2 CPU cores · 7 GB RAM · no GPU · no real datasets · no MP-SPDZ binaries · no direct access to arXiv/Crossref from the shell. Experiment sizes are chosen so the whole sprint is reproducible on this machine in a bounded time; larger scales are reported as limitations rather than extrapolated.
