# E3 Phase 0 – Implementation inventory

| Component | Status | Location | Note |
|---|---|---|---|
| FoolsGold, all-feature (Alg. 1) | implemented, unit-tested | aggregators/literature_defenses.py | no indicative-feature weighting |
| FoolsGold, indicative-feature variant | **not implemented** | – | candidate confound fix / ablation |
| Mean, median, Multi-Krum, FLAME, SignGuard, anomaly filter | implemented | aggregators/ | |
| Omniscient colluding attackers (ALIE, MIN-MAX, MIN-SUM, FANG, AGR-tailored, AGR-backdoor, norm-matched) | implemented | attacks/adaptive.py | identical malicious rows |
| Independent jitter noise | implemented | attacks/adaptive.py | post-hoc, not optimised |
| Diversification-aware attacker (optimises against similarity) | **not implemented** | – | would be E3 implementation, awaits approval |
| Detection recorder (TPR/FPR) | implemented | stage2/3 | |
| Pairwise-cosine / dispersion logging | **not implemented** | – | needed for E3 |
| E3 baseline reproduction task `FL-E3-BASE` | implemented (this work package) | pipelines/experiments/e3.py | existing attackers only |
| Stats (paired t, d_z, Cliff delta, exact Wilcoxon, Holm) | implemented | evaluation/statistics.py | Wilcoxon needs n>=6 |

Tests: unit tests exist for aggregators and adaptive attacks; an integration test for e3.py is still to be added.

## Update 2026-11-14 (Phase 4A) – history above is unchanged
| Component | Status | Location |
|---|---|---|
| FoolsGold, indicative-feature variant (soft/hard) | **implemented** as a documented adaptation (S_t is not given by the paper) | aggregators/foolsgold_indicative.py |
| FoolsGold all-feature with corrected α=0 clipping (diagnostic) | implemented | same, `mode="all"` |
| Output-layer index helper (mlp, logreg, deep_mlp) | implemented, tested | same |
| Legacy FoolsGold | unchanged; latent −inf → 1.0 defect documented | literature_defenses.py |
| Task `FL-E3-FGVAL` (legacy vs variants, 10 seeds) | implemented, run | pipelines/experiments/e3_fg.py |
| Diversification-aware attacker, pairwise-cosine logging of colluders | round-update cosine now logged (StatsTap); attacker still not implemented | e3_fg.py |
