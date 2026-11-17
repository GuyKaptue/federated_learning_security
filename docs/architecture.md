# Architecture and design decisions

## Package layout (`src/federated_learning`)
| Package | Responsibility | Why separate |
|---|---|---|
| `clients/` | `FederatedClient` (private data, local SGD) | clients never see other clients' data; seeding per (seed, round, client) gives reproducibility |
| `strategies/` | `Strategy` + FedAvg/FedProx/SCAFFOLD/FedNova | algorithm logic (client proximal term, control variates, server step) is independent of how updates are combined or hidden |
| `servers/` | `run_fl` simulator + `FederatedServer` (real TCP/mTLS server) | same round logic in two transports; the networked server adds join/authentication, timeouts and dropout handling |
| `aggregators/` | mean, coordinate median, trimmed mean, Krum, clipping, anomaly filter, audit | aggregation rules / defences share one interface (`Aggregator.aggregate`) |
| `privacy/` | HE (CKKS/BFV/Paillier), SecAgg, SMPC, client-side DP | confidentiality mechanisms are drop-in aggregators |
| `communication/` | wire protocol, mutual-TLS helpers, payload accounting, framework adapters | the only place that touches sockets/certificates |
| `attacks/` | backdoor, Byzantine, MIA, gradient inversion, adaptive | adversaries plug in as client hooks or data poisoners |
| `synthetic_data/` | generators, validators, distributions, quality metrics, drift analysis, metadata, pipelines | one reproducible, documented, validated benchmark |
| `preprocessing/`, `feature_engineering/` | registry (source + SHA-256 + licence), download, loaders; imputation/scaling transformers | integrity and no train/test leakage |
| `data_partitioning/` | IID, Dirichlet, label shards, stratified, quantity skew, site-based, missing-data simulation | federation simulation independent of the dataset |
| `training/` | `FLConfig`, MLP, centralised baseline, model registry | configuration is a validated dataclass fed from `configs/*.yaml` |
| `evaluation/`, `explainability/` | metrics, paired statistics, permutation importance | evaluation never touches training state |
| `monitoring/` | resource/communication meters, MLflow tracking | measurement is separate from the algorithms |
| `pipelines/` | experiment scripts that regenerate the stored results | reproduction entry points |
| `utils/` | paths, hashing, config, logging | no domain logic |
| `causal_inference/` | **scaffold only** | present in the target architecture; nothing implemented |

## Decisions
* **src layout + installable package**: no path hacks in code; tests import the installed (or `PYTHONPATH=src`) package.
* **Strategy / Aggregator / Client separation** mirrors the structure of Flower and FedML so the pieces can be mapped onto them. Compatibility is by interface shape only (`NumPyClientAdapter`); it was not run against those frameworks.
* **Lower-case, snake_case names everywhere**, including story folders (`fl_sec_001`); hyphenated task identifiers (`FL-SEC-001.1`) are kept inside documents because they are Jira keys.
* **Data are never committed unless fully synthetic.** Raw files are downloaded and verified by hash; mirrors may carry unknown licences.
* **Behaviour-preserving refactor**: modules were split mechanically and verified with golden hashes (dataset arrays, partitions, FedAvg/FedProx/SCAFFOLD/FedNova runs, aggregators, loaders) that are identical before and after.
* **Configs are code-checked**: tests assert that every YAML key maps onto a dataclass field, so configs cannot silently drift.

## Missing components (honest list)
* Networked mode is tested on localhost only (separate processes, mTLS): no WAN testing, certificate revocation/rotation, rate limiting, Sybil-resistant enrollment, or distributed secure-aggregation message flow; no gRPC; no `hetero_epochs` over the network.
* No Flower / TensorFlow Federated / FedML integration run; only a duck-typed adapter.
* `causal_inference/` is empty; `explainability/` has only permutation importance.
* No GPU measurements; HE/SecAgg overhead measured on a 2-core CPU.
* SecAgg malicious-server consistency round not implemented; MP-SPDZ unavailable (own SPDZ-style implementation with a trusted dealer).
* No DP accounting library integration beyond clipping + Gaussian noise (Opacus is listed as an optional extra).
* No pre-commit hooks or type-checking (mypy) configuration yet.
* Data versioning (DVC) is not set up; integrity relies on SHA-256 pins.
