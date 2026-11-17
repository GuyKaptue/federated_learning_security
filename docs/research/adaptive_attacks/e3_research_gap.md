# E3 Phase 1 – Research gap statement

Classification (evidence: verified sources only; our repository for "implemented/tested").

| Item | Class |
|---|---|
| Sybil clients can be detected by update similarity (FoolsGold) | established (L1) |
| Adaptive sybils with noise/thresholds exist in FoolsGold's own evaluation | established **if** extraction holds; manual check pending |
| FoolsGold is included in a modern benchmark | established (L3) |
| Fake-client attacks exploiting multi-round consistency | established (L4), different mechanism |
| Our identical-row colluders vs FoolsGold | implemented, tested (S3-002) |
| Our independent-jitter colluders vs FoolsGold (5 seeds, iid) | tested, partial, underpowered |
| Indicative-feature FoolsGold | not implemented |
| Pairwise-cosine / dispersion logging of colluders | not implemented |
| Non-IID (Dirichlet) jitter vs FoolsGold | not tested |
| Evasion vs benign-utility trade-off curve over a diversification budget | not tested |
| Healthcare-motivated cross-silo setting | not tested for this question (synthetic data only) |
| Any novelty of "diversified colluders" as a concept | **not claimed** |

## Residual gap (potentially novel, not guaranteed)
A paired, pre-registered, adequately powered measurement of how much diversification a colluding group needs to evade similarity-based defences in heterogeneous cross-silo settings and what it costs the attacker's backdoor effect and benign utility. Whether this is unpublished is unverified.

## Central question
Under what conditions can diversified colluding clients evade similarity-based Sybil defences in heterogeneous, cross-silo healthcare federated learning, and what is the resulting trade-off between backdoor suppression and benign-client utility? (Restricted to the synthetic SYN-IDS-v1 benchmark with virtual clients; no PhysioNet or other credentialed clinical data were available.)

## Update 2026-11-14 (Phase 4A)
The row "Our independent-jitter colluders vs FoolsGold (5 seeds, iid) – partial" is now: legacy all-feature FoolsGold, 10 seeds, iid: **tested, +0.293 [0.141, 0.445], 10/10 seeds**. Indicative-feature FoolsGold (our adaptation): **tested, jitter effect +0.009 [−0.007, 0.024] iid; inconclusive under Dirichlet(0.3)**. See `e3_gap_reassessment.md`. The residual gap is narrower than stated above and is no longer a FoolsGold-bypass gap.
