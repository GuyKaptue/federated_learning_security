# Project report — Federated Learning Security research platform

**Author:** Guy M. Kaptue T. · guykaptue24@gmail.com · https://github.com/GuyKaptue/
**Status:** research prototype, simulated federation. Not peer reviewed. No compliance claim of any kind.

## 1. What this is
A simulated federated-learning platform (virtual clients in one process, PyTorch MLP) used to study the cost of secure aggregation / homomorphic encryption / SMPC / differential privacy, and the effect of attacks and defences. It originated as an EPIC-001 research sprint (five stories: confidentiality mechanisms, performance, attacks, benchmarking, publication) and was then refactored into a production-style Python package with reproducible data pipelines.

## 2. Data used
<!-- dataset-table:start -->
| Dataset | Samples | Features | Positive rate | Source | License | Federated setup | Purpose in experiments |
|---|---|---|---|---|---|---|---|
| SYN-IDS-v1 (synthetic) | 12000 | 40 | 0.402 | generated (scikit-learn) | n/a (generated, no restrictions) | simulated IID / Dirichlet / shard clients | FL benchmarks, crypto overhead, secure aggregation, attacks, defences, scalability, HPO |
| Breast Cancer Wisconsin (Diagnostic) | 569 | 30 | 0.627 | UCI, via scikit-learn (bundled) | CC BY 4.0 | simulated IID / Dirichlet clients | validation on real, de-identified tables |
| Pima Indians Diabetes | 768 | 8 | 0.349 | public GitHub mirror (original: NIDDK) | not verified (no licence statement found; mirror unknown) | simulated IID / Dirichlet clients | validation on real, de-identified tables |
| Heart Disease, Cleveland (processed) | 302 (303 raw, 1 duplicate removed) | 13 | 0.543 | UCI, via public GitHub mirror | CC BY 4.0 (original UCI dataset); mirror unverified | simulated IID / Dirichlet clients | validation on real, de-identified tables |
| UCI Heart Disease, four sites | 920 | 10 | 0.553 | UCI / public GitHub mirror (pinned commit) | CC BY 4.0 (original); mirror unverified | 4 natural site clients (Cleveland, Hungary, Switzerland, VA Long Beach) | natural heterogeneity; within-site and leave-one-site-out validation |
<!-- dataset-table:end -->

* SYN-IDS-v1 is completely synthetic despite its name; it is not real intrusion data and not patient data.
* No real hospital data and no patient-identifiable data are used. All federated clients are simulated locally.
* Raw datasets are not redistributed; mirror licences were not verified (Pima has no verified licence).
* CIC-IDS-2017 was not available and is not used.
* No credentialed clinical repository (e.g. PhysioNet MIMIC-IV, eICU) was available; none is used in the paper or the code.
* The Cleveland mirror's label 1 appears to mean *absence* of disease (164 of 302 rows equals the UCI count of patients without disease); this was inferred from counts and does not affect binary-classification metrics.

## 3. Where the results are
Stored full-scale results with provenance: `experiments/results/<story>/` (CSV/JSON/figures). Per-task documentation: `docs/jira/<story>/` (literature, mathematics, methodology, results and limitations). Manuscript: `latex/manuscript_ieee.pdf`. Numbers in this report are not repeated here on purpose; quote them from the stored files.

Headline limitations from the study (see `docs/jira/tracker.md`):
* The secure-aggregation overhead KPIs (< 30 % training time, < 40 % communication) were **not met** in the measured configurations.
* Undefended FedAvg is vulnerable to backdoor and gradient-leakage attacks; rule-aware adaptive attackers defeat the undefended mean and degrade the tested robust rules at 40 % attackers.
* Nested hyper-parameter tuning did not help on the real tables or natural sites; the random-search budget was small.
* 41 of 48 references were checked against source pages (title/authors/dates); legal texts were partially read; two references remain `REFERENCE_NOT_VERIFIED`.
* No GPU measurements; no live Flower/TFF/FedML run. All stored experiment results come from the in-process simulator; the networked mode (real server and client processes, mutual TLS) was added afterwards and is validated on localhost for equivalence with the simulator, not re-run for the experiments.

## 4. The refactor (this change)
* Package `federated_learning` under `src/` with client / server / strategy / aggregator / privacy / communication separation, snake_case and lower-case names everywhere.
* Behaviour preserved: golden hashes of datasets, partitions, all four algorithms, aggregators and real-data loaders are identical before and after (`tests/integration/reproducibility/test_golden_reference.py` keeps the recorded values).
* New: synthetic-data pipeline (generate → validate → quality → drift → dataset card), partitioning module (IID, Dirichlet, shards, stratified, quantity skew, site-based, MCAR/MAR/MNAR), dataset registry with SHA-256-verified downloads, YAML configs wired to code, model registry, permutation importance, Flower-style adapter (duck-typed), 14 numbered notebooks, dataset documentation, CI workflows.
* Networked federation: `FederatedServer` / `NetworkClient` over TCP with mutual TLS, bit-identical to the simulator in tests.
* Test-suite: see `reports/pytest_junit.xml`.

## 5. Open items
See "Missing components" in `docs/architecture.md`.
