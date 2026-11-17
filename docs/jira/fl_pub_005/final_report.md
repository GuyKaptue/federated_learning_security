# Scientific report (DRAFT) — Secure Federated Learning Framework: cybersecurity validation in a simulated healthcare-motivated setting

**Status:** draft generated from stored results. Not peer reviewed; not claimed publication-ready. Reference status is per entry in REFERENCES_MASTER (41 of 48 metadata-verified on source pages; bodies not re-read; legal acts partially read). No patient data and no hospitals were involved: **virtual clients** on one machine, **synthetic** tabular data (SYN-IDS-v1). No regulatory compliance claim is made.

## 1. Summary
We implemented and measured three confidentiality mechanisms for federated aggregation (HE, secure aggregation, SMPC), four FL algorithms (FedAvg, FedProx, SCAFFOLD, FedNova), four attack families (backdoor, Byzantine, membership inference, gradient leakage) with defences, a scalability study at 10–500 virtual clients, a reproducibility audit and an ablation.

## 2. Key measured findings
1. **Accuracy is unaffected by confidentiality mechanisms; cost is not** (task 1.5):

| config | acc | train_overhead_pct | comm_overhead_pct | acc_loss_pp |
|---|---|---|---|---|
| plaintext | 0.8 | 0.0 | 0.0 | 0.0 |
| he_ckks | 0.8 | 119.4 | 5912.55 | 0.0 |
| he_bfv | 0.8 | 102.79 | 7745.51 | 0.01 |
| secagg | 0.8 | 34.09 | 141.07 | 0.0 |
| secagg_10pct_dropout | 0.81 | 32.18 | 119.83 | -0.05 |
| smpc_3party | 0.8 | 144.5 | 500.0 | 0.0 |
2. **Overhead KPIs (<30% training, <40% communication) are mostly not met** in simulation (task 2.5):

| config | n_clients | training_overhead_pct | target_lt_30 | comm_overhead_pct | target_lt_40 | accuracy_loss_pp |
|---|---|---|---|---|---|---|
| he_ckks | 10 | 134.7 | False | 2956.2 | False | 0.0 |
| he_ckks | 50 | 415.7 | False | 2955.8 | False | 0.0 |
| he_ckks | 100 | 751.3 | False | 2956.5 | False | 0.0 |
| he_ckks | 500 | 1747.7 | False | 2955.4 | False | 0.0 |
| secagg_complete | 10 | 30.0 | True | 89.5 | False | 0.0 |
| secagg_complete | 50 | 608.8 | False | 259.3 | False | 0.0 |
| secagg_complete | 100 | 2353.6 | False | 471.6 | False | 0.0 |
| smpc_3party | 10 | 157.8 | False | 250.0 | False | 0.0 |
| smpc_3party | 50 | 531.8 | False | 250.0 | False | 0.0 |
| smpc_3party | 100 | 938.5 | False | 250.0 | False | 0.0 |
| smpc_3party | 500 | 2296.0 | False | 250.0 | False | 0.0 |
| secagg_sparse14 | 100 | 287.1 | False | 110.7 | False | 0.0 |
| secagg_sparse14 | 500 | 711.2 | False | 110.7 | False | 0.0 |
3. **Backdoors succeed without defence; confidentiality tools do not stop them** (task 3.1 / 3.5):

| level | asr | clean_acc |
|---|---|---|
| none(clean FL) | 0.046 | 0.78 |
| low | 0.741 | 0.772 |
| medium | 0.997 | 0.741 |
| high | 0.999 | 0.69 |
4. **Membership inference** is weak in the evaluated regime (task 3.3), reducing further with DP noise at utility cost:

| config | auc | utility_acc |
|---|---|---|
| none | 0.577 | 0.841 |
| clip_only(C=0.5) | 0.578 | 0.841 |
| dp_sigma=0.1 | 0.581 | 0.839 |
| dp_sigma=0.3 | 0.564 | 0.827 |
| dp_sigma=1.0 | 0.521 | 0.754 |
5. **KPI verification (attacks):**

| kpi | measured | met |
|---|---|---|
| Backdoor success rate < 5% | 0.999 (high, undefended); best defended 0.054 | False |
| Gradient leakage success < 10% | B=1 undefended 1.00; secagg K=10 (B_eff=20) 0.00 | False |
| Membership inference AUC < 0.60 | undefended 0.577; lowest 0.521 | True |
6. **Algorithm benchmark under Dirichlet α=0.3** (task 4.7): SCAFFOLD and FedNova lead on F1 and rounds-to-target; FedProx (untuned μ) does not beat FedAvg. 13 pairwise comparisons survive Holm correction (task 4.6).

| algorithm | utility_f1 | convergence_rounds | communication_mb_to_target | mean_rank |
|---|---|---|---|---|
| FedAvg | 0.843 | 16.8 | 1.766 | 2.2 |
| SCAFFOLD | 0.873 | 11.4 | 2.397 | 2.2 |
| FedNova | 0.864 | 12.8 | 1.346 | 2.2 |
| FedProx | 0.839 | 18.0 | 1.892 | 3.4 |
| Proposed framework (FedAvg + CKKS HE aggregation) | 0.843 | 16.8 | 53.981 | nan |
7. **Ablation** (task 5.2):

| config | acc | train_overhead_pct | acc_under_signflip | backdoor_asr |
|---|---|---|---|---|
| plain FL | 0.84 | 0.0 | 0.406 | 0.998 |
| +HE (CKKS) | 0.84 | 116.032 | 0.406 | 0.998 |
| +SecAgg | 0.84 | 59.605 | 0.406 | 0.998 |
| +SMPC (3-party) | 0.84 | 149.076 | 0.406 | 0.998 |
| +DP (central, sigma=0.5) | 0.814 | 9.208 | 0.772 | 0.888 |
| +robust (median) | 0.838 | 10.938 | 0.81 | 0.245 |
| SecAgg + client-DP | 0.815 | 64.21 | 0.772 | 0.889 |
| HE + client-DP | 0.815 | 123.216 | 0.772 | 0.889 |
| SMPC + norm-bound | 0.84 | 568.273 | 0.838 | 0.112 |
8. **Reproducibility:** 10 of 11 reproducibility checks passed; failed: ['git commit identifier available'].
9. **Baseline tuning (task 4.8)** — validation-selected lr/mu, synthetic Dirichlet 0.3, 10 seeds. Tuning narrows the gaps; no pairwise difference between tuned algorithms survives Holm correction, so the untuned ranking in item 6 is weaker than it looked (tuned FedProx gains about 1.8 pp F1; FedAvg's selected lr 0.8 is at the grid edge):

| algo | setting | lr | mu | f1 | f1_ci |
|---|---|---|---|---|---|
| fedavg | default | 0.05 | 0.05 | 0.8431 | [0.8124, 0.8737] |
| fedavg | tuned | 0.8 | 0.0 | 0.8481 | [0.8128, 0.8834] |
| fedprox | default | 0.05 | 0.05 | 0.8391 | [0.8088, 0.8693] |
| fedprox | tuned | 0.2 | 0.01 | 0.8574 | [0.8379, 0.8769] |
| scaffold | default | 0.05 | 0.05 | 0.8729 | [0.8606, 0.8852] |
| scaffold | tuned | 0.05 | 0.0 | 0.8729 | [0.8606, 0.8852] |
| fednova | default | 0.05 | 0.05 | 0.8637 | [0.8471, 0.8803] |
| fednova | tuned | 0.1 | 0.0 | 0.8633 | [0.8478, 0.8788] |
10. **Real public tables (task 4.9)** — WDBC, Pima diabetes, Cleveland heart; 5 virtual clients (synthetic partitions); nested tuning over 10 splits. Algorithms are statistically indistinguishable (Friedman p below), tuning did not help on average, CKKS/SecAgg reproduce plaintext parameters exactly:

| dataset | algo | setting | acc | f1 | auc |
|---|---|---|---|---|---|
| wdbc | centralized(pooled) | reference | 0.97 | 0.976 | 0.995 |
| wdbc | fedavg | tuned | 0.968 | 0.974 | 0.995 |
| wdbc | fedprox | tuned | 0.97 | 0.976 | 0.995 |
| wdbc | scaffold | tuned | 0.97 | 0.976 | 0.995 |
| wdbc | fednova | tuned | 0.969 | 0.975 | 0.995 |
| pima | centralized(pooled) | reference | 0.76 | 0.621 | 0.833 |
| pima | fedavg | tuned | 0.744 | 0.577 | 0.827 |
| pima | fedprox | tuned | 0.748 | 0.598 | 0.826 |
| pima | scaffold | tuned | 0.741 | 0.613 | 0.807 |
| pima | fednova | tuned | 0.748 | 0.593 | 0.822 |
| heart | centralized(pooled) | reference | 0.837 | 0.85 | 0.912 |
| heart | fedavg | tuned | 0.822 | 0.838 | 0.907 |
| heart | fedprox | tuned | 0.824 | 0.84 | 0.907 |
| heart | scaffold | tuned | 0.821 | 0.831 | 0.901 |
| heart | fednova | tuned | 0.834 | 0.848 | 0.905 |

| dataset | friedman_p |
|---|---|
| wdbc | 0.326 |
| pima | 0.836 |
| heart | 0.234 |
11. **Adaptive attacks (task 3.7; omniscient attacker)** — adaptive backdoors defeat defences that stopped the static one (ASR on synthetic data); untargeted ALIE/MIN-MAX did little damage in this setting:

| defense | none_clean | static_boost | norm_matched | minmax_backdoor |
|---|---|---|---|---|
| anomaly_filter | 0.107 | 0.106 | 0.498 | 0.649 |
| clip_median_norm | 0.111 | 0.788 | 0.775 | 0.639 |
| krum | 0.097 | 0.12 | 0.184 | 0.996 |
| mean | 0.106 | 0.998 | 0.753 | 0.636 |
| median | 0.113 | 0.261 | 0.257 | 0.266 |
| multi_krum | 0.104 | 0.106 | 0.557 | 0.763 |
| trimmed_mean | 0.114 | 0.288 | 0.29 | 0.311 |

| defense | none | alie | minmax |
|---|---|---|---|
| anomaly_filter | 0.777 | 0.777 | 0.777 |
| clip_median_norm | 0.774 | 0.775 | 0.753 |
| krum | 0.768 | 0.774 | 0.749 |
| mean | 0.777 | 0.777 | 0.756 |
| median | 0.777 | 0.775 | 0.758 |
| multi_krum | 0.775 | 0.776 | 0.757 |
| trimmed_mean | 0.776 | 0.776 | 0.758 |

## 3. Threats to validity
Mostly synthetic data (plus three small public tables with synthetic client partitions); virtual clients; two CPU cores; no network; adaptive attackers are omniscient upper bounds; a single trigger family; trusted-dealer SMPC; honest key-holder HE; semi-honest SecAgg without consistency round; only lr/mu tuned (boundary optimum for FedAvg); real-data test sets of 91-231 samples; percentage overheads are relative to a very cheap in-process baseline.

## 4. What is NOT shown
Behaviour on large real clinical data, real hospital heterogeneity or federations (only three small public tables with synthetic partitions were used), real networks, malicious-server SecAgg security, regulatory compliance, MP-SPDZ based MPC, GPU performance, complete-graph SecAgg at 500 clients.

## 5. Next steps
Larger real datasets with natural site splits (with approval); tuning beyond lr/mu; attacker adaptivity beyond ALIE/MIN-MAX; real transport and key management; formal security analysis.
