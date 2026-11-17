# Experimental methodology — fl_perf_002

* Data: synthetic SYN-IDS-v1 (40 features, binary, 70:30); IID and Dirichlet partitions; **virtual clients**.
* Seeds fixed; unit of replication = seed; 95% CIs (t/bootstrap); non-parametric tests where applicable (Wilcoxon needs ≥6 seeds for p<0.05).
* Provenance in every result JSON (experiment id, config SHA-256, environment, no git commit).
* Labels: MEASURED / SIMULATED / ESTIMATED / THEORETICAL / NOT_RUN.
* Hardware: 2 CPU cores, no GPU. Timing experiments were run alone on the machine.

## Research questions and hypotheses
### FL-PERF-002.1
- RQ: What are the training-time, communication, memory and CPU overheads of each secure aggregation mechanism?
- RQ: Are the story KPIs (<30% training overhead, <40% communication overhead, <1 pp accuracy loss) met?
- H: Secure aggregation overhead exceeds the 30%/40% KPIs for at least some mechanisms in simulation.
### FL-PERF-002.2
- RQ: What are the training-time, communication, memory and CPU overheads of each secure aggregation mechanism?
- RQ: Are the story KPIs (<30% training overhead, <40% communication overhead, <1 pp accuracy loss) met?
- H: Secure aggregation overhead exceeds the 30%/40% KPIs for at least some mechanisms in simulation.
### FL-PERF-002.3
- RQ: What are the training-time, communication, memory and CPU overheads of each secure aggregation mechanism?
- RQ: Are the story KPIs (<30% training overhead, <40% communication overhead, <1 pp accuracy loss) met?
- H: Secure aggregation overhead exceeds the 30%/40% KPIs for at least some mechanisms in simulation.
### FL-PERF-002.4
- RQ: What are the training-time, communication, memory and CPU overheads of each secure aggregation mechanism?
- RQ: Are the story KPIs (<30% training overhead, <40% communication overhead, <1 pp accuracy loss) met?
- H: Secure aggregation overhead exceeds the 30%/40% KPIs for at least some mechanisms in simulation.
### FL-PERF-002.5
- RQ: What are the training-time, communication, memory and CPU overheads of each secure aggregation mechanism?
- RQ: Are the story KPIs (<30% training overhead, <40% communication overhead, <1 pp accuracy loss) met?
- H: Secure aggregation overhead exceeds the 30%/40% KPIs for at least some mechanisms in simulation.
