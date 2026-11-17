# Implementation — fl_perf_002

## FL-PERF-002.1 Resource and communication monitoring framework
**Objective.** Build and validate the monitor (CPU, memory, comm, MLflow) used by all benchmarks.

**Modules.** `monitoring.py`, `fl.py`, `crypto/*`

**Notebook.** `notebooks/sprint_tasks/fl_perf_002/FL_PERF_002_1.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-PERF-002.1`

## FL-PERF-002.2 Benchmark with 10 virtual clients
**Objective.** Measure secure-aggregation overhead at 10 clients.

**Modules.** `monitoring.py`, `fl.py`, `crypto/*`

**Notebook.** `notebooks/sprint_tasks/fl_perf_002/FL_PERF_002_2.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-PERF-002.2`

## FL-PERF-002.3 Benchmark with 50 virtual clients
**Objective.** Measure overhead at 50 clients.

**Modules.** `monitoring.py`, `fl.py`, `crypto/*`

**Notebook.** `notebooks/sprint_tasks/fl_perf_002/FL_PERF_002_3.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-PERF-002.3`

## FL-PERF-002.4 Benchmark with 100 and 500 virtual clients
**Objective.** Measure overhead at 100 and 500 virtual clients.

**Modules.** `monitoring.py`, `fl.py`, `crypto/*`

**Notebook.** `notebooks/sprint_tasks/fl_perf_002/FL_PERF_002_4.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-PERF-002.4`

## FL-PERF-002.5 Scalability analysis and KPI check
**Objective.** Fit scaling exponents, check story KPIs, and discuss limits; cost figures use labelled illustrative price assumptions.

**Modules.** `monitoring.py`, `fl.py`, `crypto/*`

**Notebook.** `notebooks/sprint_tasks/fl_perf_002/FL_PERF_002_5.ipynb`

**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task FL-PERF-002.5`
