| Dataset | Samples | Features | Positive rate | Source | License | Federated setup | Purpose in experiments |
|---|---|---|---|---|---|---|---|
| SYN-IDS-v1 (synthetic) | 12000 | 40 | 0.402 | generated (scikit-learn) | n/a (generated, no restrictions) | simulated IID / Dirichlet / shard clients | FL benchmarks, crypto overhead, secure aggregation, attacks, defences, scalability, HPO |
| Breast Cancer Wisconsin (Diagnostic) | 569 | 30 | 0.627 | UCI, via scikit-learn (bundled) | CC BY 4.0 | simulated IID / Dirichlet clients | validation on real, de-identified tables |
| Pima Indians Diabetes | 768 | 8 | 0.349 | public GitHub mirror (original: NIDDK) | not verified (no licence statement found; mirror unknown) | simulated IID / Dirichlet clients | validation on real, de-identified tables |
| Heart Disease, Cleveland (processed) | 302 (303 raw, 1 duplicate removed) | 13 | 0.543 | UCI, via public GitHub mirror | CC BY 4.0 (original UCI dataset); mirror unverified | simulated IID / Dirichlet clients | validation on real, de-identified tables |
| UCI Heart Disease, four sites | 920 | 10 | 0.553 | UCI / public GitHub mirror (pinned commit) | CC BY 4.0 (original); mirror unverified | 4 natural site clients (Cleveland, Hungary, Switzerland, VA Long Beach) | natural heterogeneity; within-site and leave-one-site-out validation |
