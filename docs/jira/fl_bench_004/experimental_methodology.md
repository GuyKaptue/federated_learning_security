# Experimental methodology — fl_bench_004

* Data: synthetic SYN-IDS-v1 (40 features, binary, 70:30); IID and Dirichlet partitions; **virtual clients**.
* Seeds fixed; unit of replication = seed; 95% CIs (t/bootstrap); non-parametric tests where applicable (Wilcoxon needs ≥6 seeds for p<0.05).
* Provenance in every result JSON (experiment id, config SHA-256, environment, no git commit).
* Labels: MEASURED / SIMULATED / ESTIMATED / THEORETICAL / NOT_RUN.
* Hardware: 2 CPU cores, no GPU. Timing experiments were run alone on the machine.

## Research questions and hypotheses
### FL-BENCH-004.1
- RQ: Does the implementation satisfy its validation identities?
- RQ: How does it behave under IID vs non-IID data?
- H: Validation identities hold to numerical precision.
### FL-BENCH-004.2
- RQ: Does the implementation satisfy its validation identities?
- RQ: How does it behave under IID vs non-IID data?
- H: Validation identities hold to numerical precision.
### FL-BENCH-004.3
- RQ: Does the implementation satisfy its validation identities?
- RQ: How does it behave under IID vs non-IID data?
- H: Validation identities hold to numerical precision.
### FL-BENCH-004.4
- RQ: Does the implementation satisfy its validation identities?
- RQ: How does it behave under IID vs non-IID data?
- H: Validation identities hold to numerical precision.
### FL-BENCH-004.5
- RQ: Does the implementation satisfy its validation identities?
- RQ: How does it behave under IID vs non-IID data?
- H: Validation identities hold to numerical precision.
### FL-BENCH-004.6
- RQ: Does the implementation satisfy its validation identities?
- RQ: How does it behave under IID vs non-IID data?
- H: Validation identities hold to numerical precision.
### FL-BENCH-004.7
- RQ: Does the implementation satisfy its validation identities?
- RQ: How does it behave under IID vs non-IID data?
- H: Validation identities hold to numerical precision.
### FL-BENCH-004.10
- RQ: Do the algorithm rankings from synthetic partitions hold when the clients are natural cohorts?
- RQ: How well does a model trained on three sites transfer to the fourth (external validation)?
- RQ: Does federated training beat local-only training and approach pooled training?
- H: Differences between FL algorithms are within noise at n=920 with four clients.
- H: Transfer to Switzerland (positive rate 0.93, cholesterol missing) is clearly worse than to Cleveland in balanced accuracy.
### FL-BENCH-004.11
- RQ: Does a 24-configuration random search over six parameters beat the lr/mu grid on test F1?
- RQ: Does it change which algorithm is best?
- H: Gains over the grid are small relative to seed variability.
### FL-BENCH-004.8
- RQ: Does tuning change the ranking of FedAvg/FedProx/SCAFFOLD/FedNova?
- RQ: Is the selected hyper-parameter interior to the grid or at its boundary?
- H: Tuned learning rates exceed the default 0.05 because the round budget is fixed.
- H: Rankings shrink toward ties after tuning.
### FL-BENCH-004.9
- RQ: Do the algorithm rankings from synthetic data hold on real tables?
- RQ: Do CKKS and secure aggregation reproduce plaintext accuracy on real data?
- RQ: Does coordinate median still protect against a sign-flip client?
- H: Differences between algorithms are within noise on these tiny datasets.
- H: Secure aggregation is exact up to quantisation on real data as on synthetic data.
