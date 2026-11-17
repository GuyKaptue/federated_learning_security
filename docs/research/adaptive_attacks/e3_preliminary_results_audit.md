# E3 Phase 0 – Audit of the preliminary 0.512 vs 0.804 result

Source: `experiments/results/fl_stage3/s3_2_runs.csv` (S3-002), iid partition, FoolsGold, backdoor ASR, 5 seeds (0–4). Recomputed here from the stored runs; nothing was re-run.

## Comparability
Both numbers come from the same task, same seeds, same data, same 15 rounds, same defence and the same attacker (`agr_backdoor`, draw-aware). The only change is jitter 1.0 (independent noise of norm equal to the row norm, added after the attack choice). So the pair is comparable and paired by seed. Caveats: (a) the jitter row is not a diversification-optimised attacker; (b) the setup uses the all-feature FoolsGold variant; (c) n = 5.

## Paired per-seed values (ASR)
| seed | agr_bd | agr_bd+jit | diff |
|---|---|---|---|
| 0 | 0.418 | 1.000 | +0.582 |
| 1 | 0.072 | 0.237 | +0.165 |
| 2 | 0.566 | 1.000 | +0.434 |
| 3 | 0.716 | 0.959 | +0.243 |
| 4 | 0.786 | 0.824 | +0.037 |

Mean paired difference +0.292 (SD 0.217), paired t p = 0.039, d_z = 1.35; exact Wilcoxon p = 0.0625 (the minimum possible with n = 5). Direction is the same in 5/5 seeds. The earlier 0.804 interval upper bound above 1 is a t-interval artefact on a bounded quantity.

## Cross-defence context (same seeds, jitter minus no jitter, unadjusted)
| defence | diff in ASR | paired t p |
|---|---|---|
| foolsgold | +0.292 | 0.039 |
| median | -0.003 | 0.57 |
| multi_krum | -0.472 | 0.001 |
| flame | -0.462 | 0.001 |
| signguard | -0.220 | 0.042 |
| anomaly_filter | -0.263 | 0.055 |
| mean | 0.000 | – |

Jitter helps the attacker only against the similarity-based defence and hurts it against distance/outlier defences, which is the pattern an E3 hypothesis would predict, but this is a post-hoc reading.

## Interpretation limits
- Seven defences were compared; with Holm correction over them the FoolsGold p of 0.039 does not survive. The result is a **signal, not a confirmed effect**.
- Seed-to-seed spread is large (clean-run ASR itself ranges 0.07–0.66), so 5 seeds are underpowered.
- The effect may partly come from our FoolsGold lacking indicative-feature weighting; the original paper reports noise attacks against the full version. This is an unresolved confound.
- Clean-model ASR without attack is non-trivial (about 0.3–0.4 on average), so ASR must always be read against the no-attack baseline.

Conclusion: the preliminary result justifies a pre-registered, adequately powered study (10+ seeds) and does not support any novelty or bypass claim.
