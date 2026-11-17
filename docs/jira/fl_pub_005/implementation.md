# Implementation — fl_pub_005

## FL-PUB-005.1 Reproducibility audit
**Objective.** Verify determinism, re-execution, provenance consistency and artifact integrity of all stored results.

**Modules.** `experiments/common.py`, `fl.py`

**Notebook.** `notebooks/sprint_tasks/fl_pub_005/FL_PUB_005_1.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-PUB-005.1`

## FL-PUB-005.2 Ablation study
**Objective.** Ablate confidentiality (HE/SecAgg/SMPC), DP and robust aggregation and their feasible combinations; document incompatible combinations.

**Modules.** `fl.py`, `defenses.py`, `crypto/*`

**Notebook.** `notebooks/sprint_tasks/fl_pub_005/FL_PUB_005_2.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-PUB-005.2`

## FL-PUB-005.3 Healthcare threat model and security architecture
**Objective.** Document adversaries, assets, STRIDE mapping and security architecture for healthcare FL.

**Modules.** `scripts/generate_publication.py`

**Notebook.** `notebooks/sprint_tasks/fl_pub_005/FL_PUB_005_3.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-PUB-005.3`

## FL-PUB-005.4 Scientific report
**Objective.** Produce a scientific report from stored results.

**Modules.** `scripts/generate_publication.py`

**Notebook.** `notebooks/sprint_tasks/fl_pub_005/FL_PUB_005_4.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-PUB-005.4`

## FL-PUB-005.5 IEEE/ACM manuscript draft
**Objective.** Produce an IEEE-style manuscript draft and verified-status bibliography.

**Modules.** `scripts/generate_publication.py`

**Notebook.** `notebooks/sprint_tasks/fl_pub_005/FL_PUB_005_5.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-PUB-005.5`
