# E3 Phase 4A completion report – STOP: awaiting explicit approval

## Recommendation: **NO-GO for E3 as originally framed; CONDITIONAL GO only for a reframed E3** (see 10)

## 1. Paper verification
Fung, Yoon, Beschastnikh, arXiv:1808.04866 v5 / RAID 2020. Read through three text extractions (arXiv PDF, ar5iv, RAID PDF), **not** a page-by-page read of the typeset paper; authors' code not reachable. Details and the table in `foolsgold_paper_method_verification.md`.

## 2. Indicative-feature formulation
Explicit: output-layer parameter magnitudes of the global model, hard filter or soft weight, normalised across classes, similarity weighted by S_t. **Not given: any formula for S_t**, the soft weights, or a default threshold. Our choice (adaptation): |W_out| per-class max-normalised; soft = weights; hard = top 10%. Not an exact reproduction.

## 3. Legacy audit
Missing the indicative-feature mechanism; one latent defect (ln 0 = −inf set to 1.0, so exactly identical sybils could receive full weight; a toy case shows [1,1,1] vs [0,0,1]); everything else matches Algorithm 1. The defect did not alter any result (diagnostic arm identical in 60/60 runs). `foolsgold_implementation_audit.md`.

> **Correction 2026-11-14:** item 3 above ("did not alter any result") holds for the E3 runs only; the defect is reachable under untargeted identical-row attackers (S2-001, S3-001). See `foolsgold_defect_and_manuscript_audit.md`.

## 4. New architecture
`aggregators/foolsgold_indicative.py`: `FoolsGoldIndicative(FoolsGold)` (modes soft/hard/all), `make_foolsgold(variant)`, `output_layer_from_model`. Reuses the legacy `_cos_matrix`, the `Aggregator` context mechanism and the existing `_run` harness; legacy untouched. Single-head models only (multi-head raises).

## 5. Tests
Unit: `tests/unit/aggregators/test_foolsgold_indicative.py` (9 tests, hand-computed values, legacy regression pinned). Integration: `test_e3_fg_validation.py`, `test_e3_baseline.py`. Ruff clean. Full suite: 222 passed, 0 failed (run with `PYTHONPATH=src`; the cryptography subprocess test needs it because the package is not installed in this environment).

## 6–7. Ten-seed results and statistics
`foolsgold_indicative_feature_validation.md`. Headline (iid): legacy jitter effect +0.293 [0.141, 0.445], 10/10 seeds; paper_soft +0.009 [−0.007, 0.024]; difference −0.284 [−0.430, −0.139], Holm p = 0.014. Cost: clean accuracy −0.077 [−0.097, −0.058], benign weight 0.34 vs 0.79. Dirichlet(0.3): inconclusive, no variant stops the backdoor.

## 8. The original jitter observation
Real for the legacy all-feature FoolsGold (now with 10 seeds), explained by the missing indicative-feature weighting for the iid case. It is not evidence against the paper's defence.

## 9. Revised gap
`e3_gap_reassessment.md`: iid Outcome A, Dirichlet Outcome C. A FoolsGold-bypass contribution is not supported.

## 10. Recommended direction
Reframe E3 as a measurement of the cost and failure region of similarity-based defences under heterogeneity (utility cost on iid, failure under Dirichlet, sensitivity to S_t choice). Before any new attacker: (a) obtain and read the authors' code or full PDF to settle S_t, (b) more heterogeneity levels and seeds (Dirichlet CI ±0.3 is too wide), (c) check whether the iid utility cost is an artefact of our S_t. A diversified attacker should be approved only if it targets the output-layer features, which needs a separate decision.

## 11. Files
Created: `src/federated_learning/aggregators/foolsgold_indicative.py`, `src/federated_learning/pipelines/experiments/e3_fg.py`, `tests/unit/aggregators/test_foolsgold_indicative.py`, `tests/integration/reproducibility/test_e3_fg_validation.py`, `outputs/research/adaptive_attacks/e3/fg_validation/*`, and in `docs/research/adaptive_attacks/`: `foolsgold_paper_method_verification.md`, `foolsgold_implementation_audit.md`, `foolsgold_indicative_feature_validation.md`, `e3_gap_reassessment.md`, `e3_phase4a_completion_report.md`.
Modified (appended, history kept): `run_all.py` (registry entry `FL-E3-FGVAL`), `e3_implementation_inventory.md`, `e3_research_gap.md`, `e3_experimental_protocol.md`, `e3_go_no_go_report.md`. Not modified: manuscript, baseline outputs, legacy FoolsGold.

## 12. Limitations
Single synthetic dataset and architecture; 15 rounds; S_t is our adaptation; paper read via extraction; indicative features were designed for non-IID data and tested here on an iid split; Dirichlet arm underpowered; sources L2 and L5–L9 still unread; no E1, E2, RL, secure aggregation or new attacker was run.
