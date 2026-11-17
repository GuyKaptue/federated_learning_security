# Implementation — fl_bench_004

## FL-BENCH-004.1 FedAvg baseline
**Objective.** Implement and validate FedAvg; sweep rounds/epochs/participation.

**Modules.** `fl.py`, `stats.py`

**Notebook.** `notebooks/sprint_tasks/fl_bench_004/FL_BENCH_004_1.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-BENCH-004.1`

## FL-BENCH-004.2 FedProx
**Objective.** Implement FedProx and measure drift reduction vs μ.

**Modules.** `fl.py`, `stats.py`

**Notebook.** `notebooks/sprint_tasks/fl_bench_004/FL_BENCH_004_2.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-BENCH-004.2`

## FL-BENCH-004.3 SCAFFOLD
**Objective.** Implement SCAFFOLD (option II control variates) and evaluate under partial participation.

**Modules.** `fl.py`, `stats.py`

**Notebook.** `notebooks/sprint_tasks/fl_bench_004/FL_BENCH_004_3.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-BENCH-004.3`

## FL-BENCH-004.4 FedNova
**Objective.** Implement FedNova and verify objective-inconsistency correction on a closed-form quadratic toy.

**Modules.** `fl.py`, `stats.py`

**Notebook.** `notebooks/sprint_tasks/fl_bench_004/FL_BENCH_004_4.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-BENCH-004.4`

## FL-BENCH-004.5 Four-algorithm comparison grid
**Objective.** Compare FedAvg/FedProx/SCAFFOLD/FedNova (+ proposed FedAvg+CKKS) across IID, Dirichlet α∈{0.3,0.1}, 50 clients, E=5, partial participation, and sign-flip attackers.

**Modules.** `fl.py`, `stats.py`

**Notebook.** `notebooks/sprint_tasks/fl_bench_004/FL_BENCH_004_5.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-BENCH-004.5`

## FL-BENCH-004.6 Statistical analysis
**Objective.** Friedman + pairwise Wilcoxon/paired-t with Holm correction and effect sizes; bootstrap CIs.

**Modules.** `fl.py`, `stats.py`

**Notebook.** `notebooks/sprint_tasks/fl_bench_004/FL_BENCH_004_6.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-BENCH-004.6`

## FL-BENCH-004.7 Leaderboard
**Objective.** Multi-criteria leaderboard (utility, convergence, communication, robustness, cost).

**Modules.** `fl.py`, `stats.py`

**Notebook.** `notebooks/sprint_tasks/fl_bench_004/FL_BENCH_004_7.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-BENCH-004.7`

## FL-BENCH-004.10 Natural multi-site benchmark: UCI heart disease (4 collection sites as clients)
**Objective.** Replace synthetic client partitions by NATURAL site heterogeneity (Cleveland, Hungary, Switzerland, VA Long Beach), with within-site and leave-one-site-out protocols, nested tuning, pooled/local baselines and a secure-aggregation equivalence check.

**Modules.** `real_sites.py`, `experiments/ext2.py`

**Notebook.** `notebooks/sprint_tasks/fl_bench_004/FL_BENCH_004_10.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-BENCH-004.10`

## FL-BENCH-004.11 Broader tuning: six-parameter random search vs lr/mu grid
**Objective.** Test whether tuning more than learning rate and FedProx mu (batch size, local epochs, hidden width, SCAFFOLD server step) changes conclusions, on synthetic data and on the four natural sites.

**Modules.** `experiments/ext2.py`

**Notebook.** `notebooks/sprint_tasks/fl_bench_004/FL_BENCH_004_11.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-BENCH-004.11`

## FL-BENCH-004.8 Validation-based tuning of the four FL baselines
**Objective.** Remove the 'untuned baseline' objection: select lr (all) and mu (FedProx) by validation F1 on a split of the TRAIN set, then compare tuned vs default on the untouched test set.

**Modules.** `experiments/ext.py`, `real_data.split_validation`, `stats.py`

**Notebook.** `notebooks/sprint_tasks/fl_bench_004/FL_BENCH_004_8.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-BENCH-004.8`

## FL-BENCH-004.9 Benchmark on real public de-identified medical tables
**Objective.** Rerun the four-algorithm benchmark, the secure-aggregation equivalence check and a Byzantine check on real public tabular medical data (WDBC, Pima diabetes, Cleveland heart disease) with nested tuning.

**Modules.** `real_data.py`, `experiments/ext.py`

**Notebook.** `notebooks/sprint_tasks/fl_bench_004/FL_BENCH_004_9.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-BENCH-004.9`
