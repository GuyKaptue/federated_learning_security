# E3 Phase 0 – Repository audit

Scope: only what bears on E3 (diversified colluding clients vs similarity-based Sybil defences). No new attack was run for this document.

## Inspected
- `src/federated_learning/aggregators/literature_defenses.py` (FoolsGold, FLAME, SignGuard, RLR)
- `src/federated_learning/attacks/adaptive.py` (AdaptiveAttackAggregator, AGR-tailored/backdoor, jitter)
- `src/federated_learning/pipelines/experiments/stage3.py`, `stage2.py` (`_run`), `run_all.py`
- stored results `experiments/results/fl_stage3/s3_2_runs.csv`
- `docs/threat_analysis.md`

## Verified facts
1. FoolsGold follows Algorithm 1 of Fung et al. (history sums, cosine, pardoning, logit, kappa=1). It keeps state per client id.
2. **Indicative-feature (output-layer) weighting is NOT implemented.** Our FoolsGold is the all-feature variant. The original paper applies this weighting and discusses noise-based adaptive Sybils against it. Any statement about "FoolsGold" in E3 must say "all-feature FoolsGold variant".
3. Malicious rows are identical by construction (`np.tile`), so ordinary collusion is the maximally detectable case for a similarity defence.
4. Jitter adds independent noise of norm `jitter * ||D[i]||` after the attack vector is chosen. It is not optimised against any defence. Expected pairwise cosine is about 1/(1+jitter^2).
5. Attacker is omniscient (sees honest updates, simulates the server). Results are upper bounds, not deployed-attacker estimates.
6. No aggregator here is compatible with secure aggregation (all see plaintext updates).
7. No dedicated diversified-Sybil attacker exists; jitter is the only decorrelation mechanism.

## Gaps relevant to E3
- No measurement of malicious pairwise cosine, dispersion or temporal consistency is logged (needed for H-measures, Phase 2).
- Defence-side logging for FoolsGold records alpha and dropped indices (usable for TPR/FPR).
- No non-IID-aware FoolsGold evidence beyond Stage 3.

## Not claimed
No first diversified Sybil attack, no first collusion attack, no first FoolsGold bypass.
