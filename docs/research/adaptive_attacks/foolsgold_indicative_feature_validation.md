# Controlled validation: legacy vs paper-faithful FoolsGold (Phase 4A, Phase E)

Task `FL-E3-FGVAL` (`pipelines/experiments/e3_fg.py`); outputs in `outputs/research/adaptive_attacks/e3/fg_validation/` (runs, summary, paired stats, config). The existing baseline outputs were not touched. Runtime 239 s, 240 runs (10 seeds × 2 partitions × 4 arms × 3 scenarios), no divergence, no missing values.

**Design.** Same data, splits, K=20, f=4, 15 rounds, backdoor task, local training and the unchanged AGR-backdoor attacker (draw-aware); no new attacker. Scenarios: no attack; identical colluders; the existing jitter 1.0 colluders. Arms: `legacy`; `paper_soft` (primary); `paper_hard` (top 10%, sensitivity); `allfeat_corrected` (diagnostic: legacy similarity with corrected α=0 clipping). Partitions: iid (primary, as in S3-002) and Dirichlet(0.3) (secondary). Primary family per partition, Holm-corrected: the jitter effect within each of 4 arms, and the difference of that effect against legacy for the 3 other arms (7 tests).

**Reproduction checks.** Legacy arm matches the stored FL-E3-BASE iid values in 20/20 pairs and S3-002 jitter seeds 0–4 in 5/5 pairs (max difference 1e-16), so the added logging does not change numerics. The diagnostic arm equals legacy in all 60 paired runs.

## iid results (ASR, mean over 10 seeds, 95% t-interval of paired differences)
| Arm | none | identical | jitter 1.0 | jitter − identical | Holm p (Wilcoxon, exact) | seeds with +/− |
|---|---|---|---|---|---|---|
| legacy | 0.383 | 0.428 | 0.721 | **+0.293 [0.141, 0.445]**, d_z 1.38 | 0.014 | 10/0 |
| paper_soft | 0.577 | 0.594 | 0.603 | +0.009 [−0.007, 0.024], d_z 0.40 | 0.375 | 4/0 (6 ties-to-zero) |
| paper_hard | 0.627 | 0.632 | 0.662 | +0.030 [−0.026, 0.085], d_z 0.38 | 0.438 | 5/2 |
| allfeat_corrected | 0.383 | 0.428 | 0.721 | +0.293 (identical to legacy) | 0.014 | 10/0 |

Contrast against legacy: (jitter − identical) is **−0.284 [−0.430, −0.139]** for paper_soft (10/10 seeds, Holm p = 0.014) and −0.263 [−0.409, −0.118] for paper_hard (Holm p = 0.014).

Reading ASR against each arm's own no-attack floor: legacy attack − none is +0.045 (identical) and +0.338 [0.183, 0.493] (jitter); paper_soft is +0.018 [−0.066, 0.101] and +0.026 [−0.058, 0.110]; paper_hard +0.005 and +0.035.

## The cost: benign utility (iid)
| | clean acc (none) | mean benign weight | none-attack: rows 0–3 weight 0 in rounds |
|---|---|---|---|
| legacy | 0.725 | 0.79 | 1% |
| paper_soft | 0.648 (−0.077 [−0.097, −0.058], 10/10 seeds) | 0.34 | 47% |
| paper_hard | 0.627 (−0.098 [−0.120, −0.076], 10/10) | 0.24 | 80% |
Even without any attack, the faithful variant down-weights most clients on the iid split and the model is worse and its no-attack ASR is higher (0.58–0.63 vs 0.38). Clean accuracy stays 4.6–9.4 points below legacy under attack as well. Note that "weight 0" is a weight outcome, not proof of malicious detection: in the no-attack runs those rows are benign.

Detection versus weight (identical colluders, iid): TPR (weight exactly 0) is 1.00 for legacy, 0.998 for soft, 0.99 for hard; with jitter 1.0 it falls to 0.49 (legacy), 0.88 (soft), 0.90 (hard); malicious mean weight 0.37 (legacy), 0.09 (soft), 0.07 (hard). Mean pairwise cosine of the malicious round updates: 0.995 (identical) and 0.50 (jitter 1.0) for all arms.

## Dirichlet(0.3) results (secondary; inconclusive)
All arms are far weaker here: ASR with identical colluders is 0.90 (legacy), 0.90 (soft), 0.89 (hard), clean accuracy about 0.47–0.53. The jitter effect is +0.096 [−0.118, 0.309] for legacy and soft, +0.015 for hard, with 2 positive and 2 negative non-tied seeds; none is significant (Holm p = 1.0) and the intervals span ±0.3. No difference between arms in the jitter effect can be claimed (contrasts 0.000 [−0.32, 0.32] and −0.080 [−0.48, 0.32]). With n = 10 and per-seed SD about 0.3, this is **inconclusive**, and the stronger finding is descriptive: FoolsGold, in any variant, did not stop the backdoor on this heterogeneous split (attack − none ≈ +0.4 to +0.56, p ≈ 0.004–0.03 unadjusted).

## Limits
Synthetic SYN-IDS-v1, one architecture, 15 rounds, one trigger, omniscient AGR attacker. S_t is our adaptation because the paper gives no formula; the cost seen here may differ from the authors' implementation, which I could not inspect. The iid split violates the paper's non-IID assumption, which likely explains the benign down-weighting. Seed 0–9 ASR spread is large (SD about 0.3–0.4).
