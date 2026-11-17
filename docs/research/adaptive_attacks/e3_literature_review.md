# E3 Phase 1 – Literature review (limited, source-honest)

Rule applied: nothing is cited as evidence unless its identifier was seen this session; every source not read in full carries FULL TEXT NOT VERIFIED. Per-source status is in `e3_literature_matrix.csv`. No claim of a first diversified Sybil attack, first collusion attack or first FoolsGold bypass is made or supported.

## What the verified sources establish
- **FoolsGold (Fung, Yoon, Beschastnikh; arXiv:1808.04866).** Abstract (read): flags poisoning sybils by the diversity of client updates; no bound on the number of attackers; evaluated against sybil label-flipping and backdoors across data distributions and sybil strategies. Via the fetch-tool extraction (not manually checked), the paper itself evaluates adaptive sybils that add noise or adapt thresholds. If this holds, noise-based diversification against FoolsGold is **not new**, and the paper's full defence uses output-layer feature selection, which our implementation lacks.
- **FLPoison benchmark (Zhang et al.; arXiv:2502.03801).** Evaluates 15 attacks and 17 defences across FL algorithms and heterogeneity levels; includes FoolsGold; reports that FoolsGold's non-adaptive hyperparameters limit its generalisability. No dedicated diversified-Sybil attack was confirmed (the page was read up to 100,000 of 107,188 characters).
- **PoisonedFL (Xie, Fang, Gong; arXiv:2404.15611, CVPR 2025).** Attacker injects fake clients; enforces consistency across rounds without genuine-client knowledge; breaks eight defences (Multi-Krum, Median, Trimmed Mean, Norm Bound, FLTrust, FLAME, FLCert, FLDetector). FoolsGold not seen in the visible text. It is about multi-round consistency, not diversification.

## What the review cannot say
Sources L2, L5–L9 were not read. A search for "FoolsGold evasion" returned titles only. The absence of a diversified-colluder study in healthcare cross-silo FL is therefore **not established**; it is only "not found in the sources checked".

## Consequences for E3
1. Novelty must be framed narrowly: a controlled, pre-registered measurement of the evasion/utility trade-off under healthcare-motivated heterogeneity, not a new attack.
2. The FoolsGold variant must be named (all-feature) and an indicative-feature variant added as an ablation or fix before E3 conclusions.
3. Before any manuscript use, L1 and L2 must be read in full by the author.
