# Attack taxonomy (as used in this project)
| Family | Variants in repo | Needs benign updates | Needs rule | Multi-round state | Reference status |
|---|---|---|---|---|---|
| Byzantine static | sign flip, noise | no | no | no | classic |
| Statistical adaptive | ALIE, MIN-MAX, MIN-SUM, FANG | yes (omniscient version) | no / partly | no | re-implemented |
| Rule-aware | AGR-tailored | yes | yes (simulated) | no | ours, fixed 34-candidate search |
| Backdoor | static, boosted, norm-matched, AGR-backdoor | partly | AGR-backdoor yes | no | re-implemented / ours |
| Diversified collusion | jitter option | yes | no | no | weak; Sybil literature (FoolsGold paper) |
| Multi-round consistent | - | no | no | yes | PoisonedFL (not implemented) |
| Learned (RL) | - | varies | varies | yes | Li et al. 2022; Wang et al. 2025 (not implemented) |
