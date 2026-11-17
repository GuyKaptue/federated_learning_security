# E3 Phase 4 – Go / no-go report (STOP: awaiting explicit approval)

Note: my notes after the context reset did not keep the prompt's 14 numbered items verbatim; the sections below cover the content they asked for. Please tell me if an item is missing.

## 1. Recommendation
**Conditional GO, narrowed scope.** Novelty is not claimed, only a controlled measurement. Go ahead only with the conditions in section 9.

## 2. What is established
Harness reproduces stored results exactly (140/140 pairs). FoolsGold neutralises identical colluders (TPR 1.00, FPR 0.003, ASR 0.428 vs no-attack 0.383, 10 seeds). Jitter 1.0 raised ASR against FoolsGold by +0.29 in 5 seeds (exploratory, unadjusted p = 0.039, does not survive Holm over seven defences).

## 3. What is not established
Any novelty; the full FoolsGold paper's adaptive-attacker results (checked only via fetch extraction); non-IID behaviour; effect with indicative-feature weighting; sources L2, L5–L9 unread.

## 4. Main risk: confound
Our FoolsGold lacks indicative-feature weighting. The effect could be an artefact of our variant. Prior literature (extraction only) suggests noise attacks were already studied against FoolsGold.

## 5. Proposed implementation (needs approval)
(a) indicative-feature FoolsGold as an ablation variant with unit tests; (b) logging of the four diversification measures; (c) a jitter-level attacker, conditions C then D, via the existing `jitter` parameter first; an optimised diversifying attacker only if C shows a signal.

## 6. Files to be touched
`aggregators/literature_defenses.py` (new variant class), `attacks/adaptive.py` (measure logging only), `pipelines/experiments/e3.py` (new tasks), `run_all.py` (registry), `tests/unit/aggregators/test_literature_defenses.py`, `tests/integration/reproducibility/test_e3_*.py`, new `outputs/research/adaptive_attacks/e3/<task>/`, docs.

## 7. Cost
Baseline took 321 s for 280 runs; conditions A–C × 2 partitions × 7 defences × 10 seeds ≈ 20–30 minutes. No heavy compute needed.

## 8. Not done and not planned
No E1, no E2, no insertion into the manuscript, no real clinical data (no PhysioNet credentials), no new attacker run so far.

## 9. Conditions for GO
1. Fix or ablate the FoolsGold variant first and re-run the baseline for it.
2. Author reads FoolsGold (arXiv 1808.04866 / RAID 2020) fully and confirms its adaptive-attacker section.
3. Hypotheses H1–H5 stay as pre-registered; 10+ seeds; Holm over the whole family.
4. Report null or negative outcomes.

## 10. NO-GO triggers
If the indicative-feature variant removes the jitter effect (H4 true), E3 becomes a negative or replication result and should be reported as such, not as an attack contribution.

Status: **stopped. No diversified attacker, E1 or E2 was run.**

## Update 2026-11-14 (Phase 4A) – the recommendation above is superseded by `e3_phase4a_completion_report.md`
Condition 1 (fix or ablate the FoolsGold variant) is met. NO-GO trigger 10 (the indicative-feature variant removes the jitter effect) fired for iid data. See the completion report.
