# E3 Phase 3 – Baseline validation (existing attackers only, 10 seeds)

Task `FL-E3-BASE` (`pipelines/experiments/e3.py`). Outputs: `outputs/research/adaptive_attacks/e3/baseline/` (`e3_baseline_runs.csv`, `e3_baseline_summary.csv`, `e3_baseline.json`, `e3_config.json`). No diversified attacker was run. Config: iid partition, K=20, f=4, 15 rounds, 8000 samples, seeds 0–9, 7 defences, 4 scenarios (none_clean, static_boost, norm_matched, agr_backdoor), draw-aware, jitter 0. Runtime 321 s.

## Checks
| Check | Result |
|---|---|
| All runs present | 280 = 10 seeds × 4 scenarios × 7 defences, no missing rows |
| Divergence | 0 diverged runs |
| NaN in outcome columns | none in acc, f1, auc, asr, clean_acc (tpr/fpr are NaN only for defences without a detection output: mean, median) |
| Reproduction of stored S3-002 (seeds 0–4) | **140/140 pairs exact** for ASR (max |diff| 1.1e-16) and for clean accuracy (0.0) |
| Seeds 5–9 | new, not previously computed |
| Quick-mode regression test | `tests/integration/reproducibility/test_e3_baseline.py` passes |

The reproduction gate is passed. (The prompt's own acceptance list was not re-quoted verbatim in my notes after the context reset; the checks above are the ones I could verify against it, and the criteria should be compared in the go/no-go review.)

## Results that matter for E3 (10 seeds, 95% t-intervals in the summary CSV)
- **FoolsGold vs identical colluders is effectively neutralised.** TPR = 1.00 in all 10 seeds, FPR 0.003. AGR-backdoor ASR 0.428 [0.211, 0.645] versus 0.383 [0.183, 0.583] with no attack. Paired difference +0.045 (t p = 0.018, Wilcoxon p = 0.0137, unadjusted); in 8 of 10 seeds it is below 0.13.
- The earlier 5-seed value 0.512 becomes 0.428 with 10 seeds (seeds 5–9 are lower: 0.452, 0.169, 0.112, 0.879, 0.111). Seed-to-seed spread of ASR is very large (0.07–0.88), so intervals are wide.
- Clean ASR (no attack) is about 0.31–0.38 for every defence, so ASR must be read against this floor.
- FoolsGold costs benign accuracy even without attack: clean_acc 0.725 vs 0.743 for mean (about −1.8 points).
- Distance/outlier defences (Multi-Krum, FLAME, SignGuard, anomaly filter) are beaten by AGR-backdoor (ASR 0.76–0.89) but FoolsGold is not, which is the contrast E3 relies on.
- FLAME has high FPR on clean data (0.44) and under AGR-backdoor (0.53).

## Limits
iid only (Dirichlet not run here), synthetic data, omniscient attacker, all-feature FoolsGold variant (no indicative-feature weighting), 15 rounds. The baseline validates the harness, not any claim about diversified attackers.
