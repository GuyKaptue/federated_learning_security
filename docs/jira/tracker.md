# RESEARCH SPRINT TRACKER — EPIC-001
Generated 2026-11-07T09:25:43Z by package_research_sprint.py.

| Story | Status | Notebooks | Docs | Results |
|---|---|---|---|---|
| FL-SEC-001 | COMPLETE_WITH_LIMITATIONS | 5/5 notebooks executed OK | docs/jira/fl_sec_001/ | experiments/results/fl_sec_001/ |
| FL-PERF-002 | COMPLETE_WITH_LIMITATIONS (KPI overhead targets NOT MET; SecAgg complete graph @500 NOT_RUN) | 5/5 notebooks executed OK | docs/jira/fl_perf_002/ | experiments/results/fl_perf_002/ |
| FL-ATTACK-003 | COMPLETE_WITH_LIMITATIONS (backdoor & gradient-leakage KPIs NOT MET undefended; rule-aware attackers defeat the undefended mean and degrade robust rules at 40% attackers) | 8/8 notebooks executed OK | docs/jira/fl_attack_003/ | experiments/results/fl_attack_003/ |
| FL-BENCH-004 | COMPLETE_WITH_LIMITATIONS (lr/mu grid + 6-parameter random search; 3 small tables + 4 natural-site heart cohorts, n=920; no live federation) | 11/11 notebooks executed OK | docs/jira/fl_bench_004/ | experiments/results/fl_bench_004/ |
| FL-PUB-005 | COMPLETE_WITH_LIMITATIONS (full manuscript draft compiled with IEEEtran.cls; not peer reviewed; 2 references unverified) | 5/5 notebooks executed OK | docs/jira/fl_pub_005/ | experiments/results/fl_pub_005/ |

* Automated tests: **110 passed** (0 skipped; see reports/pytest_junit.xml).
* Notebooks: 34/34 executed without error; all have the 18 mandatory sections (34/34 verified programmatically).
* Phases: 0 assessment ✔ · 1 FL-SEC-001 ✔ · 2 FL-PERF-002 ✔ · 3 FL-ATTACK-003 ✔ · 4 FL-BENCH-004 ✔ · 5 FL-PUB-005 ✔ (drafts) · packaging ✔ (see ZIP_DELIVERY_VALIDATION.md).
* Status vocabulary: COMPLETE_WITH_LIMITATIONS means implemented, tested, executed and documented, but with listed unmet KPIs / blockers. Nothing is labelled "publication-ready".

## Blockers and honest gaps
| Item | Status | Reason |
|---|---|---|
| Real datasets | PARTIAL | 3 small public tables with synthetic partitions (tasks 3.7/4.9) and the 4 natural UCI heart-disease cohorts as clients (task 4.10, n=920, 1980s research data, virtual clients); larger datasets, CIC-IDS-2017 and any live multi-site clinical data NOT available |
| Real hospitals / network | NOT_APPLICABLE | virtual clients in one process |
| Baseline tuning | DONE (grid lr/mu in 4.8-4.10; 24-config random search over 6 parameters in 4.11) | FedAvg lr optimum at grid edge on synthetic data; nested tuning did not help on real tables or natural sites; random-search budget small, compute not matched |
| Adaptive attacks | DONE (3.7: ALIE, MIN-MAX, norm/MIN-MAX backdoor; 3.8: AGR-tailored, MIN-SUM, Fang, AGR-backdoor; omniscient, rule-aware) | upper-bound attackers with fixed candidate sets; no learning-based attacker |
| GPU metrics | NOT_MEASURED | no GPU |
| MP-SPDZ | BLOCKED | binaries unavailable; own SPDZ-style implementation (trusted dealer) |
| Flower secaggplus end-to-end | NOT_RUN | primitives inspected only |
| SecAgg complete graph @500 clients | NOT_RUN | cost; sparse k=14 measured |
| SecAgg malicious-server consistency round | NOT_IMPLEMENTED | out of scope; documented |
| Reference verification | 41 of 48 METADATA_VERIFIED_ON_SOURCE_PAGE (title/authors/dates; bodies not re-read; venue_verified per entry); 4 legal acts TEXT_PARTIALLY_READ; Wilcoxon 1945 existence only; 2 REFERENCE_NOT_VERIFIED (Pima source page, Smith 1988: PMC page behind captcha) | classic papers verified via bibliographic records, not publisher pages |
| German / EU legal analysis from official sources | PARTIAL | read: BDSG s.22(1); GDPR Art. 9 and 32(1) (legislation.gov.uk copy of adopted text, not EUR-Lex OJ); NIS2 Art. 21 and Annex I health (EUR-Lex consolidated); AI Act Art. 15 (unofficial rendering). NOT read: AI Act Art. 6/Annexes, remaining articles, national transposition; no legal analysis or compliance claim |
| Git commit id | NOT_AVAILABLE | workspace is not a git repository |
| IEEEtran | DONE (class) / SUBSTITUTE (bst) | compiled with genuine IEEEtran.cls V1.8b (kept in latex/vendor/, not shipped); IEEEtran.bst unavailable so ieeetr.bst used; PDF in LATEX/ (author Guy M. Kaptue T.) |
| KPI overhead <30% / <40% | NOT MET | measured (see fl_perf_002 results) |

## Stage-0 audit (2026-11-11)
* Table II/V captions claimed 10 and 5 seeds; the stored runs had 3. Re-run with 10 seeds each; captions are now computed from the runs files (`scripts/generate_traceability.py` fails if they disagree). The scalability table lists its seeds per row (10/5/5/3 for 10/50/100/500 clients).
* Table II vs III communication differed exactly 2x: Table II is uplink-only, Table III is uplink+downlink. Captions now say so.
* Negative training overheads (median -5.7%, DP -2.5%) were a run-order artefact (configurations ran in blocks). Timing now uses a seed-outer, randomised order with a warm-up run; the median shows +10.9% [-1.2, 23.1] and an independent interleaved check gave +8.6% [3.4, 14.0].
* `tenseal`/`phe` imports are now lazy; the package and the networked server start without them (regression test added).

## Stage 1 (2026-11-12): federated preprocessing, calibration, sensitivity grids
| Item | Status | Evidence |
|---|---|---|
| Federated mean/std via secure aggregation vs pooled | DONE | `preprocessing/federated_scaler.py`, `FL-S1-001`, `tests/unit/preprocessing/test_federated_scaler.py` |
| CIs + paired effect sizes (d_z, Cliff's delta) + Holm | DONE for the new experiments; older tables keep their original intervals/tests | `stage1._contrasts`, `fl_stage1/*_tests.csv` |
| Calibration (Brier, ECE, reliability) | DONE | `evaluation/calibration.py`, `FL-S1-002` |
| Extra ablations (malicious fraction 0-40%, alpha 0.1-1, participation 10-100%, hidden width) | DONE | `FL-S1-003` (10 seeds) |
| Manuscript: explicit contributions, no-credentialed-repository statement | DONE | `scripts/generate_manuscript.py`, `scripts/manuscript_stage1.py` |
| Heart headline results still use pooled-median preprocessing | OPEN (stated in paper) | federated variant evaluated separately |
| Main-text tables 1-5 CI/effect-size retrofit | OPEN | not all older tables report effect sizes |
Not done / not claimed: a defence contribution (Stage 3), FLAME/RLR/FoolsGold/SignGuard (Stage 2), real clinical data (none available).

## Stage 2 (2026-11-12): literature defences and baselines
| Item | Status | Evidence |
|---|---|---|
| FLAME, RLR, FoolsGold, SignGuard from the papers, references verified | DONE | `aggregators/literature_defenses.py`, `docs/defenses.md`, 12 unit tests |
| Logistic regression + deeper MLP baselines | DONE | `training/models.py`, `FL-S2-003` |
| 12-rule evaluation, detection rates, label-skew cost | DONE (10 seeds) | `FL-S2-001/002`, manuscript Tables s2u/s2b/s2d/s2n/s2a |
| Attacker that decorrelates updates (vs FoolsGold) | OPEN (Stage 3) | not implemented |
| Novel defence contribution | OPEN (Stage 3) | not started |

## Stage 3 (2026-11-13): rule committee and attacker-knowledge ladder
| Item | Status | Evidence |
|---|---|---|
| RuleCommittee + knowledge ladder + jitter attacker | DONE | `aggregators/committee.py`, `attacks/adaptive.py`, 7 unit tests |
| FL-S3-001/002/003 experiments | DONE (reduced scale: 5 seeds, 15 rounds) | `experiments/results/fl_stage3/` |
| Threat analysis (2 elementary propositions) | DONE | `docs/threat_analysis.md` |
| Novelty claim | LIMITED | randomisation/hybrid rules already published; claim = measurements only |
| Secure-and-robust aggregation | POSTPONED | not started |

## Stage 4 (2026-11-13): journal targeting
| Item | Status | Evidence |
|---|---|---|
| Venue shortlist with verified facts and source tags | DONE | `docs/journal_targeting.md` |
| Primary target | Journal of Cybersecurity / IEEE Access (named author required; TMLR is double-blind) | author decision 2026-11-13 |
| Submission | NOT DONE | awaiting author decision |

## Adaptive-attacks work package (2026-11-13)
| Item | Status | Evidence |
|---|---|---|
| Gate 1 audit | DONE | `docs/research/adaptive_attacks/project_audit.md` |
| Gate 2 literature | PARTIAL (8 sources verified, not systematic) | `literature_review.md` |
| Gate 3 formalisation | DRAFT | `threat_model.md` |
| Experiments E1-E5 | NOT STARTED | awaiting user confirmation of priorities |

## E3 work package (2026-11-13 to 2026-11-14), same repository
| Item | Status | Evidence |
|---|---|---|
| Phase 0 audits (repository, inventory, preliminary result) | DONE | `docs/research/adaptive_attacks/e3_repository_audit.md`, `e3_implementation_inventory.md`, `e3_preliminary_results_audit.md` |
| Phase 1 literature and gap | DONE (limited; several sources FULL TEXT NOT VERIFIED) | `e3_literature_review.md`, `e3_literature_matrix.csv`, `e3_research_gap.md` |
| Phase 2 threat model, protocol (H1-H5) | DONE | `e3_threat_model.md`, `e3_experimental_protocol.md` |
| Phase 3 baseline, 10 seeds, 140/140 exact reproduction | DONE | `e3_baseline_validation.md`, `outputs/research/adaptive_attacks/e3/baseline/` |
| Phase 4 go/no-go | DONE (superseded by 4A) | `e3_go_no_go_report.md` |
| Phase 4A FoolsGold verification, audit, faithful variant, 10-seed validation | DONE | `foolsgold_*.md`, `e3_gap_reassessment.md`, `e3_phase4a_completion_report.md`, `outputs/research/adaptive_attacks/e3/fg_validation/` |
| Authors' FoolsGold code / full PDF to settle S_t | NOT DONE | no network route from the workspace |
| Utility-cost ablation, more heterogeneity levels and seeds | NOT DONE | awaiting approval |
| Diversified attacker, E1, E2 | NOT STARTED | not approved |

## FoolsGold defect audit (2026-11-14)
| Item | Status | Evidence |
|---|---|---|
| ln(0) defect characterised (reachable under untargeted identical-row attackers) | DONE | `docs/research/adaptive_attacks/foolsgold_defect_and_manuscript_audit.md` |
| Read-only manuscript cross-check | DONE (14 locations; manuscript not edited) | same file, section 4 |
| Fix of legacy FoolsGold / re-run of affected S2-001, S3-001 cells | NOT DONE | blocked until explicit approval |
| `.PHONY` fix for `make experiments` | DONE | Makefile line 1 |
