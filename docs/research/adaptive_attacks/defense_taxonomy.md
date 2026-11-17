# Defence taxonomy (as used in this project)
| Family | Rules in repo | Sees individual updates | Stateful | Notes |
|---|---|---|---|---|
| Distance/rank statistics | median, trimmed mean, Krum, Multi-Krum | yes | no | |
| Clipping/norm | clipped mean | yes | no | |
| Clustering + noise | FLAME | yes | no | re-implemented |
| Sign/statistics clustering | SignGuard | yes | no | re-implemented |
| Similarity / Sybil | FoolsGold | yes | yes (history) | excluded from committee pool |
| Sign voting | RLR | yes | no | |
| Anomaly filter | anomaly_filter | yes | no | |
| Committee | RuleCommittee | yes | partly | exploratory, no benefit |
| Learned aggregation | - | yes | yes | AdaAggRL (AAAI 2025), not implemented |
All of these need plaintext access to individual updates; none is compatible with secure aggregation as implemented.
