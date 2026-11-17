# Scientific gap analysis (Gate 2)

Classification keys: Established / Incremental / Potentially novel / Novelty not established / Not scientifically justified. These are judgements from the eight sources in `literature_review.md`, not from a systematic search; any "novelty" label is provisional.

| # | Candidate contribution | Existing literature | Remaining gap | Proposed investigation | Novelty status |
|---|---|---|---|---|---|
| G1 | Learned or optimisation-based attacker stronger than a fixed-candidate AGR attacker | RL attackers (Li et al. 2022; Wang et al. 2025 RL-attack) | Does learning add anything over our 34-candidate search when both are omniscient, at K=20 cross-silo? Headroom of the fixed set is unmeasured | **E1**: add a local-search/gradient-free attacker over a continuous direction-and-scale space in the same simulated-server loop; compare worst-case accuracy drop and ASR to the fixed set across 12 rules, 10 seeds | Incremental (new setting and headroom measurement, not a new attacker class) |
| G2 | Multi-round consistent attacker vs per-round AGR attacker | PoisonedFL (CVPR 2025): no benign/rule knowledge, cross-device; authors say cross-silo is harder | Does multi-round consistency help or hurt against rules with small K and when an omniscient per-round attacker is available? | **E2**: implement PoisonedFL-style sign-consistent attacker (our own re-implementation, validated on a trend/shape basis only), compare to AGR-tailored at f in {2,4,6,8} | Incremental |
| G3 | Diversified Sybil collusion vs similarity defences | FoolsGold paper (Sybil taxonomy; adaptive evasion not verified because only the abstract was read) | Our weak jitter already moved FoolsGold ASR from 0.512 to 0.804 (5 seeds, wide intervals). A diversification-aware attacker that optimises the diversity needed to pass the similarity test is untested | **E3**: sweep the jitter strength; add an attacker that adds the minimum noise needed to bring pairwise cosine below FoolsGold's threshold; measure ASR vs honest-client cost | Novelty not established until the FoolsGold paper body and later follow-ups are read |
| G4 | Non-IID heterogeneity and detection reliability in clinical-site data | FLPoison (Dirichlet 0.5), Li et al. 2023 (Dirichlet 0.1), Djemaa et al. 2026 survey | Quantitative effect on detection TPR/FPR with natural UCI sites, attacker present, small data | **E4**: run rules under natural-site clients (K=4) plus Dirichlet grid with attackers; report TPR/FPR and per-site accuracy. K=4 limits f to 1 | Incremental, limited by data (303-920 rows) |
| G5 | Defence generalisation to unseen attacks | FLPoison compares many pairs; AdaAggRL generalisation unclear | A transfer matrix (attack optimised against rule A, evaluated on rule B) at small K | **E5**: re-use the candidate-choice machinery with the simulated rule different from the deployed rule | Incremental |
| G6 | Robust detection under secure aggregation | no verified source addresses it | Needs a protocol design and security analysis; was postponed by the user in Stage 3 | not planned now | Potentially novel, high cost, postponed; no formal security claim possible without proof |
| G7 | New defence (e.g. learned aggregation) | AdaAggRL (AAAI 2025) | The committee experiment was a negative result | not justified now | Not scientifically justified |

## Falsifiable hypotheses (not confirmed)
* H1 (adaptive evasion): defences evaluated only against static attacks overestimate robustness; already supported in the manuscript against the omniscient AGR attacker under one benchmark; to be re-tested with 10 seeds.
* H2 (multi-round): a sign-consistent attacker reaches a different (not necessarily higher) damage than the per-round attacker at f/K = 0.2.
* H3 (non-IID camouflage): detection TPR drops and FPR rises with decreasing Dirichlet alpha.
* H4 (Sybil): FoolsGold ASR increases with the diversification strength and returns to the Multi-Krum level.
* H5 (generalisation): transfer-matrix off-diagonal entries are smaller than diagonal.
* H6 (utility): robust rules lose clean accuracy and worsen calibration/worst-client accuracy relative to mean.
Each will be tested with paired contrasts and Holm correction; exploratory versus confirmatory runs will be labelled.
