# Implementation inventory

Status keys: IMPLEMENTED (exists, runs), TESTED (independent test), PARTIAL, PLANNED, MISSING, UNVERIFIED.

| Component | Location | Status | Note |
|---|---|---|---|
| FedAvg/FedProx/SCAFFOLD/FedNova | `training/` | IMPLEMENTED, TESTED | |
| Mean, median, trimmed, Krum, Multi-Krum, clipped, anomaly filter | `aggregators/robust.py`, `defenses.py` | IMPLEMENTED, TESTED | |
| FLAME, FoolsGold, SignGuard, RLR | `aggregators/literature_defenses.py` | IMPLEMENTED, TESTED | Re-implemented from the papers; not checked against the authors' code. FoolsGold history is stateful. |
| RuleCommittee | `aggregators/committee.py` | IMPLEMENTED, TESTED (7 tests) | Exploratory; negative result |
| ALIE, MIN-MAX, MIN-SUM, FANG | `attacks/adaptive.py` | IMPLEMENTED, TESTED | Re-implemented; not validated against reference code |
| AGR-tailored, AGR-backdoor | `attacks/adaptive.py` | IMPLEMENTED, TESTED | Fixed candidate search; omniscient |
| Knowledge ladder (draw/pool), jitter | `attacks/adaptive.py` | IMPLEMENTED, TESTED | Jitter added after the choice; not optimised |
| Multi-round attacker (cross-round consistency) | - | MISSING | PoisonedFL is the published reference |
| Learned (RL/optimisation) attacker | - | MISSING | NeurIPS 2022 and AAAI 2025 contain RL attackers |
| Diversification-aware Sybil attacker | - | MISSING | jitter is a weak substitute |
| Detection metrics TPR/FPR | `pipelines/experiments/stage2.py` | IMPLEMENTED | Only for rejecting rules |
| Worst-client performance, PR-AUC under attack | - | MISSING | Brier/ECE exist (`evaluation/calibration.py`) |
| Non-IID partitions (Dirichlet) and natural sites | `experiments/ext.py`, heart loader | IMPLEMENTED | Quantity skew, participation variability: PARTIAL |
| Secure aggregation, HE, SMPC, DP | `privacy/` | IMPLEMENTED, TESTED | Not combined with robust rules |
| Secure-and-robust aggregation | - | MISSING | Postponed by the user since Stage 3 |
| SecAgg malicious-server / consistency round | - | MISSING | |
| Notebooks for this work package | `notebooks/research/adaptive_attacks/` | MISSING | Ten generic notebooks exist for the earlier work |
| Real multi-site clinical data | - | MISSING | No credentialed clinical repository available |
