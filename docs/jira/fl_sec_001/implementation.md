# Implementation — fl_sec_001

## FL-SEC-001.1 Cryptographic approach selection (HE / SecAgg / SMPC)
**Objective.** Select, with explicit criteria and evidence, the cryptographic aggregation mechanisms to implement for healthcare FL, and document assumptions. Output: a decision matrix with theoretical properties separated from measured costs.

**Modules.** `crypto/he.py`

**Notebook.** `notebooks/sprint_tasks/fl_sec_001/FL_SEC_001_1.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-SEC-001.1`

## FL-SEC-001.2 Homomorphic-encryption aggregation (CKKS, BFV, Paillier)
**Objective.** Implement and validate encrypted weighted aggregation in a key-holder model, measuring precision, cost and FL utility.

**Modules.** `crypto/he.py`, `defenses.Audited`

**Notebook.** `notebooks/sprint_tasks/fl_sec_001/FL_SEC_001_2.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-SEC-001.2`

## FL-SEC-001.3 Secure aggregation with dropout recovery
**Objective.** Implement Bonawitz-style secure aggregation (X25519, HKDF, ChaCha20-Poly1305, Shamir, double masking) with dropout recovery and security checks.

**Modules.** `crypto/secagg.py`

**Notebook.** `notebooks/sprint_tasks/fl_sec_001/FL_SEC_001_3.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-SEC-001.3`

## FL-SEC-001.4 SMPC aggregation with authenticated shares (SPDZ-style)
**Objective.** Implement additive secret sharing with information-theoretic MACs, Beaver triples and MAC-checked opening; detect share tampering.

**Modules.** `crypto/smpc.py`

**Notebook.** `notebooks/sprint_tasks/fl_sec_001/FL_SEC_001_4.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-SEC-001.4`

## FL-SEC-001.5 Comparative analysis of HE vs SecAgg vs SMPC
**Objective.** Compare plaintext, CKKS, BFV, SecAgg (with/without dropout) and SMPC on accuracy, time, communication and aggregation error across seeds.

**Modules.** `fl.py`, `crypto/*`, `monitoring.py`

**Notebook.** `notebooks/sprint_tasks/fl_sec_001/FL_SEC_001_5.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-SEC-001.5`
