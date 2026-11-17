# Legacy FoolsGold implementation audit (Phase 4A, Phase B)

Code: `aggregators/literature_defenses.py::FoolsGold` (unchanged by this work package); new: `aggregators/foolsgold_indicative.py`.

| Component | Paper requirement | Legacy behaviour | Difference | Consequence | Correction | Tests |
|---|---|---|---|---|---|---|
| Update representation | per-client update/gradient history | per-client sum of flat updates, all parameters | none | none | – | existing |
| History | H_i = Σ updates | kept per client id across rounds | none | none | – | `test_history_accumulates_across_rounds` |
| Indicative features | similarity on output-layer features weighted by S_t | **absent**: cosine over all 1378 parameters | missing mechanism | hidden-layer noise dilutes the similarity; diversification evades | new class | importance, noise-diversification toy |
| Output layer mapping | output layer = last layer | n/a | – | – | index helper | `test_output_layer_mapping_*` |
| Client mapping | per client | `ctx_ids`, requires full context | none | none | – | edge tests |
| Multiple heads | not discussed | n/a | – | unsupported | new class raises | `FoolsGoldIndicative(np.arange(5), 2)` |
| Normalise / clip | α/max α, logit, clip to [0,1] | same, **except** non-finite → 1.0 | ln(0) = −inf becomes 1.0, not 0 | **latent defect**: exactly identical sybils (cosine exactly 1.0) would get full weight | clip to 0 in the new class; legacy left unchanged | `test_alpha_exact_identical_sybils_clipped...` (pins both) |
| Edge cases | not specified | n<2 → ones; all α=0 → ones; zero-norm rows → cosine 0 | not in paper | none observed | same conventions in new class | `test_alpha_bounds_and_edge_cases` |
| Update rule | w + Σ α_i Δ_i | Σ α_i p_i Δ_i | FedAvg weights (documented) | none | – | – |

## Answers
1. Same update representation? Yes (sums of per-client updates), but the whole vector rather than indicative features.
2. History preserved correctly? Yes.
3. Indicative-feature mechanism? **No.**
4. Output-layer parameters handled? No (not distinguished).
5. Parameters mapped to clients? Yes, by client id.
6. Multiple heads? Not applicable; neither version supports them.
7. Normalise and clip? Yes, except that −inf becomes 1.0 in legacy.
8. Edge cases consistent? Mostly; the −inf case is the exception.
9. Other differences: FedAvg weights in the update; α=1 → 0.99 trick (harmless).

**Correction 2026-11-14:** the statements below are true only for the E3 runs. The defect is not latent: it triggers under untargeted identical-row attackers (see `foolsgold_defect_and_manuscript_audit.md`).

**Did the −inf defect affect results?** The `allfeat_corrected` arm (legacy similarity, corrected clipping) matched legacy exactly in all 60 paired runs (30 per partition; ASR and clean accuracy, maximum difference 0.0), so it did not change any result reported here. It remains a latent defect for exactly identical colluders.
