# FoolsGold primary-source verification (Phase 4A, Phase A)

## Source and access
- Title: *Mitigating Sybils in Federated Learning Poisoning*. Authors: Clement Fung, Chris J. M. Yoon, Ivan Beschastnikh (University of British Columbia). Used version: arXiv:1808.04866 **v5** (stamp 15 Jul 2020). Venue version: *The Limitations of Federated Learning in Sybil Settings*, RAID 2020 (USENIX listing). Stable URLs: arxiv.org/abs/1808.04866; cs.ubc.ca/~bestchai/papers/foolsgold-raid2020.pdf.
- **Access limitation (stated plainly):** the shell has no route to arXiv, so I could not open a PDF myself. I read the paper through the fetch tool, which returns a text extraction of the arXiv PDF, the ar5iv HTML (first 100,000 of about 157,500 characters, then the rest) and the RAID PDF. The three extractions agree on the points below, but this is **not** a page-by-page reading of the typeset paper; equations and figures can be garbled. Page numbers below come from the arXiv extraction and are approximate. The authors' code repository (github.com/DistributedML/FoolsGold, URL given in the RAID text) could not be fetched, so the reference implementation was not inspected.

## What the paper states explicitly
| Original paper component | Mathematical description | Existing (legacy) implementation | Required correction | Evidence |
|---|---|---|---|---|
| Update history | H_i = Σ_t Δ_{i,t} | Same, per client id, all parameters | none | Alg. 1, p. 5 |
| Similarity | cs_ij = cosine(H_i, H_j) **weighted by S_t**, "the weight of indicative features at iteration t" | Plain cosine over the **whole** parameter vector | restrict and weight by S_t | Alg. 1; Sec. 5 |
| Indicative features | "found by measuring the magnitude of model parameters in the output layer of the global model"; can be filtered (hard) or re-weighed (soft); "normalized across all classes" | not implemented | implement | Sec. 5 (RAID Sec. 6.2); Sec. 6.4 |
| Max similarity | v_i = max_j cs_ij | same | none | Alg. 1 |
| Pardoning | if v_j > v_i: cs_ij ← cs_ij · v_i / v_j | same | none | Alg. 1; Sec. 5 |
| Learning rate | α_i = 1 − max_j cs_ij; α ← α / max α ("the maximum adaption ... is 1") | same | none | Alg. 1; p. 6–7 |
| Logit | α ← κ (ln[α/(1−α)] + 0.5); κ = 1 in all experiments | same, but any non-finite value, including ln(0) = −inf, is set to **1.0** | clip instead: out-of-range values are "clipped and rounded to the boundary" | Alg. 1; p. 7 |
| Update rule | w_{t+1} = w_t + Σ_i α_i Δ_{i,t} | sum_i α_i p_i Δ_i with FedAvg weights p_i (adaptation noted in the legacy docstring) | none (documented adaptation) | Alg. 1 (garbled in the RAID extraction) |
| α = 1 in the logit | undefined in the text | set to 0.99 first (result clipped to 1) | none | not in paper |

## Ambiguities in the paper (our implementation choices, not the authors' method)
1. **No equation for S_t** in any of the three extractions; it is given in words only ("weight of important features at iteration t").
2. Soft weighting is described only as "proportional" to the output-layer parameter magnitude; no formula.
3. Hard filtering: the paper varies the kept proportion from 0.001 to 1 (MNIST, Fig. 10) and reports that the intelligent-noise attack fails below 10%; it gives no default.
4. "Normalized across all classes" has no stated procedure.
5. Which output-layer tensors (weights only or biases too) is not stated.
6. Algorithm 1 text contains typos in the extraction (cs_i vs cs_ij).

## Our choices for the faithful variant (documented adaptations)
S_t = |W_out| of the global model **before** the round, weights only, each class row divided by its row maximum; soft: use as feature weights on the histories before the cosine; hard: keep the top 10% (ties kept) with weight 1 and set the rest to 0; histories stored on the output-layer weights only (S_t is 0 elsewhere).

## Paper evaluation relevant to E3 (from the extractions)
- Intelligent noise (Sec. 6.4, Fig. 8): sybil pairs send a1+ζ and a2−ζ so the sum is unchanged. With all features the attack succeeds; with the indicative-feature proportion below 0.1 the attack rate is near 0; at 0.01 honest clients start to be flagged. Soft weighting is reported as comparable to the best hard threshold.
- Adaptive threshold sybils (Sec. 6.4, Alg. 2, Fig. 9): attackers that send poison only when their inter-sybil similarity is below a threshold M can succeed with enough sybils; "a sufficiently large number of sybils and a low threshold can subvert the defense".
- Limitations (Sec. 7): single attacker not handled; informed attackers with many sybils can succeed.
- Threat-model assumptions: at least one honest client per class; non-IID honest data; no secure aggregation; attackers cannot see individual honest gradients.

## Compatibility with our model (Phase A 3.3)
Model: `MLP(d_in=40, hidden=32, n_out=2)`, single head, binary task, flat `parameters_to_vector` update. Output layer weights are indices 1312–1375 of 1378 parameters (tested). The mapping is direct for this model. **Not supported and raises:** layouts whose output weights are not divisible by `n_out` (multi-head or multi-task). With two classes the per-class normalisation is over only two rows; the paper's setting (10-class MNIST, deep nets) differs, so this is an adaptation. Our server also cannot be used with secure aggregation, as in the paper.
