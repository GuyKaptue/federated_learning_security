# Project audit (Gate 1) - adaptive-attacks work package

Date: 2026-11-13. Scope: the repository in `fl_sprint/` and the manuscript `manuscript_Kaptue.pdf` (14 pages, revised 2026-11-13). Evidence: files and tests in the repository, stored result CSV/JSON files, and this conversation's logs. Nothing below is taken from the attached prompt unless it was checked against the code.

## What the manuscript and repository establish
* In-process simulator `run_fl` plus a networked server/client (mTLS, validated on localhost only); strategies FedAvg, FedProx, SCAFFOLD, FedNova.
* Aggregators: mean, median, trimmed mean, Krum, Multi-Krum, clipped, anomaly filter, FLAME, FoolsGold, SignGuard, robust learning rate (RLR), and `RuleCommittee` (exploratory, 5 seeds).
* Attackers: sign flip, noise, ALIE, MIN-MAX, MIN-SUM, FANG, static/boosted/norm-matched backdoor, and two rule-aware attackers (`AGR-tailored`, `AGR-backdoor`) in `attacks/adaptive.py`, with `knowledge` (draw/pool) and `jitter` options added in Stage 3.
* Data: synthetic SYN-TAB-40 (code name SYN-IDS-v1), three small public medical tables, UCI heart disease with four natural sites; Dirichlet label skew.
* Statistics: paired Student-t intervals, Cohen's d_z, Cliff's delta, exact Wilcoxon with Holm correction; provenance JSON; 25 tables traced; 211 tests passing.

## Correcting statements in the work-package prompt
| Prompt statement | Checked against | Finding |
|---|---|---|
| "fixed 30-candidate strategy" | `GAMMA_GRID` (10 values) x 3 directions + ALIE, MIN-MAX, MIN-SUM, FANG | 34 candidates for untargeted, 12 for the backdoor attacker. Fixed set confirmed; the number is not exactly 30. |
| "Attackers' knowledge: benign updates and aggregation-rule knowledge" | `AdaptiveAttackAggregator` | Correct: omniscient, sees all benign updates of the round, simulates the rule. Stage 3 added a knowledge ladder for randomised rules only. |
| "diversified Sybil attacks were not tested" | Stage 3 jitter option | **Partly tested**: independent noise of norm equal to the update norm was added to the colluders' rows (not optimised against any defence). With FoolsGold, backdoor ASR rose from 0.512 [0.159, 0.864] to 0.804 [0.400, 1.208] (5 seeds, intervals very wide, upper bound above 1 is a t-interval artefact). Suggestive only. |
| "Four malicious out of 20" | `K=20, F=4` constants | Correct for Stages 2 and 3; sensitivity grids use other fractions. |
| Defenses "RLR" | `RobustLearningRate` | Present (re-implemented from the paper in Stage 2). |

## Reusable, to improve, to build
* Reusable: harness (`run_all`, provenance, CSV/JSON, scratch `RS_ROOT`), `_run` experiment driver, `DetectionRecorder` (TPR/FPR), Stage-1 statistics helpers, defences, jitter hook.
* To improve: attacker candidate set is fixed and per-round; attackers see benign updates (omniscient); no attack state across rounds; no cross-round defence state except FoolsGold; no per-client (worst-client) metrics in the adaptive experiments.
* Missing: multi-round-consistency attacker, learned (RL/optimisation) attacker, diversification-aware Sybil attacker, privacy-compatible robust aggregation, worst-client and PR-AUC metrics under attack, notebooks for this work package.
* Must be repeated before comparing: AGR-tailored baselines under the new metrics, with 10 seeds where earlier work used 3-5.
* Scientifically insufficient so far: backdoor ASR comparisons at 5 seeds (intervals overlap), single synthetic benchmark for all attack experiments, committee evaluation.

## Quality-gate status
Gate 1 (understanding): complete for the repository and manuscript. See `implementation_inventory.md`.
