# data/

| Folder | Content | In git? |
|---|---|---|
| `raw/<dataset>/` | files downloaded from public sources, SHA-256 verified | no (README only) |
| `processed/` | derived arrays (none are required; loaders preprocess on the fly) | no |
| `synthetic/syn_ids_v1/` | SYN-IDS-v1 arrays (`.npz`) for seeds 0 and 1 | yes (fully synthetic) |
| `synthetic/validation/`, `synthetic/metadata/` | validation/quality reports and dataset cards | yes |
| `synthetic/generated/` | scratch space for ad-hoc generation | no |
| `external/` | data supplied by the user | no |

**Statements that apply to everything here.** SYN-IDS-v1 is completely synthetic despite its name (not real intrusion data, not patient data). No real hospital data is used and no patient-identifiable data exists in the project. All federated clients are simulated locally. Raw source datasets are not redistributed; some come from public third-party mirrors whose licences were not verified. CIC-IDS-2017 was not available and is not used.

Dataset catalogue, checksums, licences and preprocessing: [`docs/datasets/dataset_catalog.md`](../docs/datasets/dataset_catalog.md).

## Summary table
<!-- dataset-table:start -->
| Dataset | Samples | Features | Positive rate | Source | License | Federated setup | Purpose in experiments |
|---|---|---|---|---|---|---|---|
| SYN-IDS-v1 (synthetic) | 12000 | 40 | 0.402 | generated (scikit-learn) | n/a (generated, no restrictions) | simulated IID / Dirichlet / shard clients | FL benchmarks, crypto overhead, secure aggregation, attacks, defences, scalability, HPO |
| Breast Cancer Wisconsin (Diagnostic) | 569 | 30 | 0.627 | UCI, via scikit-learn (bundled) | CC BY 4.0 | simulated IID / Dirichlet clients | validation on real, de-identified tables |
| Pima Indians Diabetes | 768 | 8 | 0.349 | public GitHub mirror (original: NIDDK) | not verified (no licence statement found; mirror unknown) | simulated IID / Dirichlet clients | validation on real, de-identified tables |
| Heart Disease, Cleveland (processed) | 302 (303 raw, 1 duplicate removed) | 13 | 0.543 | UCI, via public GitHub mirror | CC BY 4.0 (original UCI dataset); mirror unverified | simulated IID / Dirichlet clients | validation on real, de-identified tables |
| UCI Heart Disease, four sites | 920 | 10 | 0.553 | UCI / public GitHub mirror (pinned commit) | CC BY 4.0 (original); mirror unverified | 4 natural site clients (Cleveland, Hungary, Switzerland, VA Long Beach) | natural heterogeneity; within-site and leave-one-site-out validation |
<!-- dataset-table:end -->
