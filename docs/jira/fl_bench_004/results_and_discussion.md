# Results and discussion — fl_bench_004: FedAvg / FedProx / SCAFFOLD / FedNova benchmark

All numbers below are read from the stored result files (MEASURED unless stated; virtual clients; synthetic data).

## Leaderboard IID (task 4.7)

| algorithm | utility_f1 | convergence_rounds | communication_mb_to_target | robustness_acc_under_signflip | training_cost_s | mean_rank |
|---|---|---|---|---|---|---|
| FedAvg | 0.884 | 11.8 | 1.2406 | 0.6609 | 8.2729 | 1.4 |
| SCAFFOLD | 0.8849 | 11.8 | 2.4811 | 0.8147 | 8.3072 | 2.0 |
| FedNova | 0.884 | 11.8 | 1.2406 | 0.6229 | 8.3269 | 3.0 |
| FedProx | 0.8824 | 12.6 | 1.3247 | 0.6585 | 8.5312 | 3.6 |
| Proposed framework (FedAvg + CKKS HE aggregation) | 0.8841 | 11.8 | 37.9104 | nan | 11.2263 | nan |

## Leaderboard Dirichlet α=0.3 (task 4.7)

| algorithm | utility_f1 | convergence_rounds | communication_mb_to_target | robustness_acc_under_signflip | training_cost_s | mean_rank |
|---|---|---|---|---|---|---|
| FedAvg | 0.8431 | 16.8 | 1.7662 | 0.6609 | 7.9543 | 2.2 |
| SCAFFOLD | 0.8729 | 11.4 | 2.397 | 0.8147 | 8.1876 | 2.2 |
| FedNova | 0.8637 | 12.8 | 1.3457 | 0.6229 | 8.0003 | 2.2 |
| FedProx | 0.8391 | 18.0 | 1.8924 | 0.6585 | 8.059 | 3.4 |
| Proposed framework (FedAvg + CKKS HE aggregation) | 0.8431 | 16.8 | 53.9809 | nan | 11.2549 | nan |

## Selected hyper-parameters (task 4.8; validation F1)

| algo | lr | mu | val_f1 | at_lr_grid_edge |
|---|---|---|---|---|
| fedavg | 0.8 | 0.0 | 0.8675 | True |
| fednova | 0.1 | 0.0 | 0.8312 | False |
| fedprox | 0.2 | 0.01 | 0.864 | False |
| scaffold | 0.05 | 0.0 | 0.8719 | False |

## Tuned vs untuned, synthetic Dirichlet 0.3, 10 seeds (task 4.8)

| algo | setting | lr | mu | acc | f1 | f1_ci | auc |
|---|---|---|---|---|---|---|---|
| fedavg | default | 0.05 | 0.05 | 0.8839 | 0.8431 | [0.8124, 0.8737] | 0.955 |
| fedavg | tuned | 0.8 | 0.0 | 0.8831 | 0.8481 | [0.8128, 0.8834] | 0.9472 |
| fedprox | default | 0.05 | 0.05 | 0.8808 | 0.8391 | [0.8088, 0.8693] | 0.9526 |
| fedprox | tuned | 0.2 | 0.01 | 0.8917 | 0.8574 | [0.8379, 0.8769] | 0.9573 |
| scaffold | default | 0.05 | 0.05 | 0.8986 | 0.8729 | [0.8606, 0.8852] | 0.9593 |
| scaffold | tuned | 0.05 | 0.0 | 0.8986 | 0.8729 | [0.8606, 0.8852] | 0.9593 |
| fednova | default | 0.05 | 0.05 | 0.8948 | 0.8637 | [0.8471, 0.8803] | 0.9592 |
| fednova | tuned | 0.1 | 0.0 | 0.8948 | 0.8633 | [0.8478, 0.8788] | 0.9608 |

## Paired test tuned - default (task 4.8; uncorrected p)

| algo | mean_diff | dz | wilcoxon_p | ttest_p |
|---|---|---|---|---|
| fedavg | 0.005 | 0.2196 | 0.1934 | 0.5049 |
| fedprox | 0.0183 | 0.9901 | 0.0059 | 0.0121 |
| scaffold | 0.0 | 0.0 | 1.0 | 1.0 |
| fednova | -0.0004 | -0.0256 | 0.9219 | 0.9373 |

## Pairwise tuned algorithms (task 4.8, Holm)

| a | b | mean_diff | wilcoxon_p | holm_p |
|---|---|---|---|---|
| fedavg | fedprox | -0.0093 | 0.9219 | 1.0 |
| fedavg | scaffold | -0.0248 | 0.1309 | 0.7852 |
| fedavg | fednova | -0.0152 | 0.2754 | 1.0 |
| fedprox | scaffold | -0.0155 | 0.375 | 1.0 |
| fedprox | fednova | -0.0059 | 0.6953 | 1.0 |
| scaffold | fednova | 0.0096 | 0.1934 | 0.9668 |

## REAL public tables: test performance, 10 outer splits, 5 virtual clients (task 4.9)

| dataset | algo | setting | acc | f1 | f1_ci | auc |
|---|---|---|---|---|---|---|
| wdbc | centralized(pooled) | reference | 0.9696 | 0.9758 | nan | 0.9952 |
| wdbc | fedavg | default | 0.9678 | 0.9745 | [0.9685, 0.9805] | 0.9945 |
| wdbc | fedavg | tuned | 0.9678 | 0.9744 | [0.9671, 0.9817] | 0.9949 |
| wdbc | fedprox | default | 0.9678 | 0.9745 | [0.9685, 0.9805] | 0.9945 |
| wdbc | fedprox | tuned | 0.9702 | 0.9763 | [0.9692, 0.9834] | 0.995 |
| wdbc | scaffold | default | 0.9673 | 0.9738 | [0.9666, 0.9811] | 0.9949 |
| wdbc | scaffold | tuned | 0.9702 | 0.9762 | [0.9700, 0.9825] | 0.9954 |
| wdbc | fednova | default | 0.969 | 0.9753 | [0.9705, 0.9802] | 0.9948 |
| wdbc | fednova | tuned | 0.969 | 0.9753 | [0.9692, 0.9815] | 0.9954 |
| pima | centralized(pooled) | reference | 0.7602 | 0.6207 | nan | 0.8333 |
| pima | fedavg | default | 0.7489 | 0.5866 | [0.5453, 0.6279] | 0.8319 |
| pima | fedavg | tuned | 0.7442 | 0.5771 | [0.5225, 0.6317] | 0.8273 |
| pima | fedprox | default | 0.7485 | 0.5862 | [0.5451, 0.6272] | 0.832 |
| pima | fedprox | tuned | 0.7476 | 0.5979 | [0.5518, 0.6441] | 0.8256 |
| pima | scaffold | default | 0.7593 | 0.6377 | [0.6047, 0.6707] | 0.8247 |
| pima | scaffold | tuned | 0.7411 | 0.6131 | [0.5688, 0.6573] | 0.8069 |
| pima | fednova | default | 0.7606 | 0.6225 | [0.6015, 0.6435] | 0.8326 |
| pima | fednova | tuned | 0.7485 | 0.5932 | [0.5516, 0.6349] | 0.8218 |
| heart | centralized(pooled) | reference | 0.8374 | 0.8501 | nan | 0.9125 |
| heart | fedavg | default | 0.8363 | 0.8532 | [0.8341, 0.8722] | 0.9102 |
| heart | fedavg | tuned | 0.822 | 0.8385 | [0.8207, 0.8562] | 0.9074 |
| heart | fedprox | default | 0.8363 | 0.8532 | [0.8341, 0.8722] | 0.91 |
| heart | fedprox | tuned | 0.8242 | 0.8401 | [0.8216, 0.8586] | 0.9071 |
| heart | scaffold | default | 0.8209 | 0.8317 | [0.8029, 0.8605] | 0.9092 |
| heart | scaffold | tuned | 0.8209 | 0.831 | [0.8028, 0.8592] | 0.9008 |
| heart | fednova | default | 0.8341 | 0.8494 | [0.8277, 0.8711] | 0.9112 |
| heart | fednova | tuned | 0.8341 | 0.8476 | [0.8244, 0.8707] | 0.905 |

## REAL data: tuned - default paired by split (task 4.9; uncorrected p)

| dataset | algo | mean_diff | wilcoxon_p |
|---|---|---|---|
| wdbc | fedavg | -0.0001 | 0.625 |
| wdbc | fedprox | 0.0018 | 0.875 |
| wdbc | scaffold | 0.0024 | 0.1641 |
| wdbc | fednova | -0.0 | 1.0 |
| pima | fedavg | -0.0095 | 0.4688 |
| pima | fedprox | 0.0118 | 0.3125 |
| pima | scaffold | -0.0246 | 0.0781 |
| pima | fednova | -0.0293 | 0.0391 |
| heart | fedavg | -0.0147 | 0.0938 |
| heart | fedprox | -0.013 | 0.0625 |
| heart | scaffold | -0.0007 | 1.0 |
| heart | fednova | -0.0018 | 1.0 |

## REAL data: Friedman across tuned algorithms (task 4.9)

| dataset | friedman_stat | friedman_p | mean_f1_fedavg | mean_f1_fedprox | mean_f1_scaffold | mean_f1_fednova |
|---|---|---|---|---|---|---|
| wdbc | 3.4615 | 0.3258 | 0.9744 | 0.9763 | 0.9762 | 0.9753 |
| pima | 0.8571 | 0.8358 | 0.5771 | 0.5979 | 0.6131 | 0.5932 |
| heart | 4.2632 | 0.2344 | 0.8385 | 0.8401 | 0.831 | 0.8476 |

## REAL data: secure-aggregation equivalence (task 4.9)

| dataset | mechanism | f1 | acc | max_param_diff_vs_plain | n |
|---|---|---|---|---|---|
| heart | ckks | 0.8436 | 0.8264 | 0.0 | 5 |
| heart | plain | 0.8436 | 0.8264 | nan | 5 |
| heart | secagg | 0.8436 | 0.8264 | 0.0 | 5 |
| pima | ckks | 0.569 | 0.7446 | 0.0 | 5 |
| pima | plain | 0.569 | 0.7446 | nan | 5 |
| pima | secagg | 0.569 | 0.7446 | 0.0 | 5 |
| wdbc | ckks | 0.9705 | 0.9626 | 0.0 | 5 |
| wdbc | plain | 0.9705 | 0.9626 | nan | 5 |
| wdbc | secagg | 0.9705 | 0.9626 | 0.0 | 5 |

## REAL data: one sign-flip client of five (task 4.9)

| dataset | aggregator | acc | f1 |
|---|---|---|---|
| heart | krum(f=1) | 0.7626 | 0.7926 |
| heart | mean | 0.6484 | 0.6515 |
| heart | median | 0.8154 | 0.8386 |
| heart | trimmed(0.2) | 0.8088 | 0.8302 |
| pima | krum(f=1) | 0.6364 | 0.3149 |
| pima | mean | 0.6442 | 0.5584 |
| pima | median | 0.6996 | 0.3914 |
| pima | trimmed(0.2) | 0.7013 | 0.467 |
| wdbc | krum(f=1) | 0.9123 | 0.9313 |
| wdbc | mean | 0.4292 | 0.3782 |
| wdbc | median | 0.9497 | 0.9612 |
| wdbc | trimmed(0.2) | 0.9509 | 0.9619 |

## Natural sites, protocol A: pooled test, 10 splits (task 4.10)

| algo | setting | acc | f1 | f1_ci | auc | bacc |
|---|---|---|---|---|---|---|
| fedavg | default | 0.7852 | 0.806 | [0.7950, 0.8170] | 0.8592 | 0.783 |
| fedavg | tuned | 0.7773 | 0.7993 | [0.7877, 0.8108] | 0.8531 | 0.7746 |
| fedprox | default | 0.7852 | 0.806 | [0.7950, 0.8170] | 0.8591 | 0.783 |
| fedprox | tuned | 0.7776 | 0.8008 | [0.7907, 0.8109] | 0.8537 | 0.7743 |
| scaffold | default | 0.7968 | 0.8242 | [0.8163, 0.8322] | 0.8576 | 0.7891 |
| scaffold | tuned | 0.7892 | 0.8147 | [0.8023, 0.8271] | 0.8471 | 0.7834 |
| fednova | default | 0.7917 | 0.8167 | [0.8061, 0.8273] | 0.8598 | 0.7862 |
| fednova | tuned | 0.7877 | 0.8141 | [0.7987, 0.8294] | 0.8545 | 0.7814 |

## Natural sites: baselines (task 4.10)

| method | acc | f1 | auc | bacc |
|---|---|---|---|---|
| centralized(pooled) | 0.7888 | 0.8135 | 0.8596 | 0.7837 |
| local-only(mean of 4 site models, pooled test) | 0.7044 | 0.7594 | 0.8173 | 0.6893 |

## Natural sites, protocol A: per-site test metrics, tuned (task 4.10)

| algo | site | acc | f1 | auc |
|---|---|---|---|---|
| fedavg | cleveland | 0.7725 | 0.7583 | 0.86 |
| fedavg | hungary | 0.8079 | 0.7393 | 0.8728 |
| fedavg | switzerland | 0.7649 | 0.8588 | 0.7229 |
| fedavg | va_long_beach | 0.7467 | 0.8381 | 0.6964 |
| fedprox | cleveland | 0.7714 | 0.7577 | 0.8629 |
| fedprox | hungary | 0.8034 | 0.7361 | 0.8718 |
| fedprox | switzerland | 0.7622 | 0.8587 | 0.6943 |
| fedprox | va_long_beach | 0.7583 | 0.846 | 0.7007 |
| scaffold | cleveland | 0.7868 | 0.7737 | 0.8554 |
| scaffold | hungary | 0.8034 | 0.7498 | 0.865 |
| scaffold | switzerland | 0.8054 | 0.8859 | 0.69 |
| scaffold | va_long_beach | 0.7617 | 0.85 | 0.6793 |
| fednova | cleveland | 0.7758 | 0.7678 | 0.8604 |
| fednova | hungary | 0.809 | 0.7505 | 0.8723 |
| fednova | switzerland | 0.8054 | 0.886 | 0.72 |
| fednova | va_long_beach | 0.7633 | 0.8519 | 0.6953 |

## Natural sites: tuned - default paired by split (task 4.10; uncorrected p)

| algo | mean_diff | dz | wilcoxon_p | ttest_p |
|---|---|---|---|---|
| fedavg | -0.0067 | -0.4976 | 0.1484 | 0.1501 |
| fedprox | -0.0052 | -0.5347 | 0.2754 | 0.1251 |
| scaffold | -0.0096 | -0.5957 | 0.0977 | 0.0922 |
| fednova | -0.0026 | -0.233 | 0.6406 | 0.4799 |

## Natural sites: pairwise tuned algorithms (task 4.10, Holm)

| a | b | mean_diff | wilcoxon_p | holm_p |
|---|---|---|---|---|
| fedavg | fedprox | -0.0015 | 0.8438 | 1.0 |
| fedavg | scaffold | -0.0154 | 0.0645 | 0.3223 |
| fedavg | fednova | -0.0148 | 0.0645 | 0.3223 |
| fedprox | scaffold | -0.0139 | 0.1309 | 0.3926 |
| fedprox | fednova | -0.0133 | 0.0273 | 0.1641 |
| scaffold | fednova | 0.0006 | 0.9219 | 1.0 |

## Natural sites, protocol B (leave-one-site-out): macro over held-out sites (task 4.10)

| algo | setting | acc | f1 | auc | bacc |
|---|---|---|---|---|---|
| fedavg | default | 0.7537 | 0.7882 | 0.7943 | 0.7202 |
| fedavg | tuned | 0.7546 | 0.7879 | 0.8003 | 0.7237 |
| fedprox | default | 0.7539 | 0.7885 | 0.7944 | 0.7207 |
| fedprox | tuned | 0.7515 | 0.788 | 0.7948 | 0.7191 |
| scaffold | default | 0.75 | 0.7924 | 0.7897 | 0.7145 |
| scaffold | tuned | 0.749 | 0.7923 | 0.7914 | 0.7145 |
| fednova | default | 0.7516 | 0.7918 | 0.7927 | 0.7162 |
| fednova | tuned | 0.754 | 0.7933 | 0.796 | 0.7189 |

## Natural sites, protocol B: per held-out site (task 4.10)

| held_out | algo | setting | acc | f1 | auc | bacc |
|---|---|---|---|---|---|---|
| cleveland | fedavg | default | 0.7482 | 0.7461 | 0.8491 | 0.7526 |
| cleveland | fedavg | tuned | 0.7663 | 0.7568 | 0.8654 | 0.7684 |
| cleveland | fedprox | default | 0.7488 | 0.7467 | 0.8494 | 0.7533 |
| cleveland | fedprox | tuned | 0.7422 | 0.7472 | 0.8441 | 0.7489 |
| cleveland | scaffold | default | 0.7449 | 0.7572 | 0.8548 | 0.7542 |
| cleveland | scaffold | tuned | 0.7449 | 0.7577 | 0.8609 | 0.7543 |
| cleveland | fednova | default | 0.7446 | 0.7526 | 0.8518 | 0.7523 |
| cleveland | fednova | tuned | 0.7594 | 0.7622 | 0.8658 | 0.7656 |
| hungary | fedavg | default | 0.7912 | 0.7391 | 0.8835 | 0.7974 |
| hungary | fedavg | tuned | 0.7912 | 0.7391 | 0.8835 | 0.7974 |
| hungary | fedprox | default | 0.7925 | 0.7406 | 0.8837 | 0.7987 |
| hungary | fedprox | tuned | 0.7922 | 0.74 | 0.8843 | 0.7982 |
| hungary | scaffold | default | 0.768 | 0.7321 | 0.8792 | 0.7921 |
| hungary | scaffold | tuned | 0.7667 | 0.733 | 0.881 | 0.7931 |
| hungary | fednova | default | 0.784 | 0.7419 | 0.8848 | 0.8007 |
| hungary | fednova | tuned | 0.784 | 0.7419 | 0.8848 | 0.8007 |
| switzerland | fedavg | default | 0.7309 | 0.8367 | 0.7108 | 0.6816 |
| switzerland | fedavg | tuned | 0.722 | 0.8304 | 0.71 | 0.6768 |
| switzerland | fedprox | default | 0.7293 | 0.8355 | 0.7102 | 0.6808 |
| switzerland | fedprox | tuned | 0.7285 | 0.835 | 0.7099 | 0.6803 |
| switzerland | scaffold | default | 0.7374 | 0.8412 | 0.7034 | 0.6851 |
| switzerland | scaffold | tuned | 0.735 | 0.8396 | 0.7025 | 0.6838 |
| switzerland | fednova | default | 0.7341 | 0.839 | 0.7072 | 0.6834 |
| switzerland | fednova | tuned | 0.7293 | 0.8356 | 0.7064 | 0.6808 |
| va_long_beach | fedavg | default | 0.7445 | 0.831 | 0.7339 | 0.6493 |
| va_long_beach | fedavg | tuned | 0.739 | 0.8255 | 0.7422 | 0.652 |
| va_long_beach | fedprox | default | 0.745 | 0.8313 | 0.7341 | 0.6502 |
| va_long_beach | fedprox | tuned | 0.743 | 0.8297 | 0.7408 | 0.6489 |
| va_long_beach | scaffold | default | 0.7495 | 0.8391 | 0.7213 | 0.6268 |
| va_long_beach | scaffold | tuned | 0.7495 | 0.8391 | 0.7213 | 0.6268 |
| va_long_beach | fednova | default | 0.7435 | 0.8337 | 0.7271 | 0.6286 |
| va_long_beach | fednova | tuned | 0.7435 | 0.8337 | 0.7271 | 0.6286 |

## Natural sites: secure-aggregation equivalence (task 4.10)

| split | max_abs_param_diff_secagg_vs_plain | f1_plain | f1_secagg |
|---|---|---|---|
| 0.0 | 0.0 | 0.8232 | 0.8232 |
| 1.0 | 0.0 | 0.7855 | 0.7855 |
| 2.0 | 0.0 | 0.7907 | 0.7907 |

## Broader tuning: test metrics for default / grid / random search (task 4.11)

| dataset | algo | setting | acc | f1 | f1_ci | auc |
|---|---|---|---|---|---|---|
| synthetic | fedavg | default | 0.8839 | 0.8431 | [0.8124, 0.8737] | 0.955 |
| synthetic | fedavg | grid | 0.8831 | 0.8481 | [0.8128, 0.8834] | 0.9472 |
| synthetic | fedavg | random | 0.891 | 0.8589 | [0.8449, 0.8730] | 0.9558 |
| synthetic | fedprox | default | 0.8808 | 0.8391 | [0.8088, 0.8693] | 0.9526 |
| synthetic | fedprox | grid | 0.8914 | 0.8581 | [0.8425, 0.8736] | 0.9567 |
| synthetic | fedprox | random | 0.8892 | 0.8554 | [0.8461, 0.8648] | 0.9552 |
| synthetic | scaffold | default | 0.8986 | 0.8729 | [0.8606, 0.8852] | 0.9593 |
| synthetic | scaffold | grid | 0.8225 | 0.8012 | [0.7093, 0.8931] | 0.8684 |
| synthetic | scaffold | random | 0.8613 | 0.8143 | [0.6994, 0.9293] | 0.9006 |
| synthetic | fednova | default | 0.8948 | 0.8637 | [0.8471, 0.8803] | 0.9592 |
| synthetic | fednova | grid | 0.885 | 0.8549 | [0.8342, 0.8756] | 0.9587 |
| synthetic | fednova | random | 0.9018 | 0.8734 | [0.8642, 0.8826] | 0.9623 |
| heart-4sites | fedavg | default | 0.7863 | 0.8078 | [0.7871, 0.8286] | 0.8603 |
| heart-4sites | fedavg | grid | 0.7841 | 0.8075 | [0.7823, 0.8326] | 0.8491 |
| heart-4sites | fedavg | random | 0.7733 | 0.7965 | [0.7581, 0.8349] | 0.8516 |
| heart-4sites | fedprox | default | 0.7856 | 0.8073 | [0.7867, 0.8279] | 0.8602 |
| heart-4sites | fedprox | grid | 0.7812 | 0.8046 | [0.7954, 0.8137] | 0.857 |
| heart-4sites | fedprox | random | 0.787 | 0.8141 | [0.7887, 0.8395] | 0.8563 |
| heart-4sites | scaffold | default | 0.7978 | 0.8263 | [0.8144, 0.8382] | 0.8601 |
| heart-4sites | scaffold | grid | 0.7913 | 0.819 | [0.8083, 0.8297] | 0.8536 |
| heart-4sites | scaffold | random | 0.7913 | 0.8196 | [0.8059, 0.8333] | 0.8585 |
| heart-4sites | fednova | default | 0.7899 | 0.816 | [0.7995, 0.8324] | 0.8608 |
| heart-4sites | fednova | grid | 0.7812 | 0.8124 | [0.7963, 0.8284] | 0.8516 |
| heart-4sites | fednova | random | 0.7834 | 0.8149 | [0.7967, 0.8332] | 0.8554 |

## Broader tuning: paired tests (task 4.11; uncorrected p)

| dataset | algo | contrast | mean_diff | wilcoxon_p | ttest_p |
|---|---|---|---|---|---|
| synthetic | fedavg | random - grid | 0.0108 | 0.8457 | 0.4612 |
| synthetic | fedavg | random - default | 0.0159 | 0.1602 | 0.1447 |
| synthetic | fedavg | grid - default | 0.005 | 0.1934 | 0.5049 |
| synthetic | fedprox | random - grid | -0.0026 | 0.6953 | 0.625 |
| synthetic | fedprox | random - default | 0.0163 | 0.1602 | 0.1988 |
| synthetic | fedprox | grid - default | 0.019 | 0.0273 | 0.0387 |
| synthetic | scaffold | random - grid | 0.0131 | 0.1934 | 0.7036 |
| synthetic | scaffold | random - default | -0.0585 | 0.4316 | 0.2664 |
| synthetic | scaffold | grid - default | -0.0717 | 0.1934 | 0.0803 |
| synthetic | fednova | random - grid | 0.0185 | 0.0488 | 0.0491 |
| synthetic | fednova | random - default | 0.0097 | 0.0273 | 0.0352 |
| synthetic | fednova | grid - default | -0.0088 | 0.4316 | 0.3654 |
| heart-4sites | fedavg | random - grid | -0.011 | 0.1875 | 0.2538 |
| heart-4sites | fedavg | random - default | -0.0114 | 0.4375 | 0.3785 |
| heart-4sites | fedavg | grid - default | -0.0004 | 0.875 | 0.9239 |
| heart-4sites | fedprox | random - grid | 0.0095 | 0.375 | 0.3056 |
| heart-4sites | fedprox | random - default | 0.0068 | 0.8125 | 0.4588 |
| heart-4sites | fedprox | grid - default | -0.0027 | 0.625 | 0.7057 |
| heart-4sites | scaffold | random - grid | 0.0006 | 0.625 | 0.7593 |
| heart-4sites | scaffold | random - default | -0.0067 | 0.3125 | 0.1445 |
| heart-4sites | scaffold | grid - default | -0.0073 | 0.25 | 0.0875 |
| heart-4sites | fednova | random - grid | 0.0025 | 0.8125 | 0.6742 |
| heart-4sites | fednova | random - default | -0.001 | 1.0 | 0.5574 |
| heart-4sites | fednova | grid - default | -0.0036 | 0.625 | 0.5156 |


### Acceptance verdicts
* Four algorithms implemented from the original papers and **validated** by identity tests (weighted aggregation, 1-client = local SGD, μ=0 ⇒ FedAvg, FD gradient check of the proximal term, SCAFFOLD invariant, FedNova equal-τ ⇒ FedAvg, closed-form quadratic toy): MET.
* Benchmark across IID/non-IID, scale, local epochs, partial participation, sign-flip attackers with 10 seeds (main conditions) or 5 seeds (others), Friedman + Wilcoxon + Holm + effect sizes: MET (task 4.6 table lists which differences survive Holm correction).
* Under IID data differences are tiny; under Dirichlet α=0.3 SCAFFOLD/FedNova lead FedAvg/FedProx on F1 and rounds-to-target in the **untuned** leaderboard (tasks 4.5-4.7).
* **Tuning (task 4.8)** shrinks that gap: tuned FedProx gains ≈ 1.8 pp F1 (paired Wilcoxon p = 0.006 uncorrected, d_z ≈ 1.0), FedAvg's selected lr is at the grid edge (0.8; unreliable), SCAFFOLD's default was already best; after Holm correction **no pairwise difference between tuned algorithms is significant** (10 seeds). The earlier "SCAFFOLD/FedNova beat FedAvg/FedProx" statement is therefore an untuned-baseline result and is **weaker than it looked**.
* **Real public tables (task 4.9; WDBC, Pima diabetes, Cleveland heart, nested tuning, 10 splits):** the four algorithms are statistically indistinguishable (Friedman p = 0.33 / 0.84 / 0.23); validation-based tuning did **not** improve test F1 on average (validation sets of only ~50-135 samples are too small to select reliably; mean tuned-minus-default F1 differences range from -0.029 to +0.012, and the single nominally significant one, FedNova on Pima, p = 0.039 uncorrected, is in the *wrong* direction). CKKS and SecAgg reproduce plaintext parameters exactly (max difference 0.0). One sign-flip client of five collapses plain averaging on WDBC (F1 0.38 vs 0.96 for median); Pima results under the attack are not informative (minority-class F1 unstable; see accuracy in the CSV).
* The "proposed framework" row is FedAvg + CKKS aggregation (not a novel algorithm): robustness **NOT_MEASURED**, hence unranked.
* Real-data caveat: clients are synthetic partitions of one small public table; Pima/Heart files come from hash-pinned third-party GitHub mirrors (licence of the mirrors not verified; UCI pages state CC BY 4.0 for WDBC and Heart). The heart table was de-duplicated (1 exact duplicate removed); Pima zeros in glucose/BP/skin/insulin/BMI were treated as missing and imputed with train medians.

### Task 4.10 verdicts (natural sites: Cleveland / Hungary / Switzerland / VA, n=920; computed from the tables above)
* With the four natural cohorts as clients, federated training matches pooled centralised training (pooled-test F1 about 0.80-0.82 vs 0.814) and clearly beats local-only models (0.759).
* SCAFFOLD/FedNova are slightly ahead of FedAvg/FedProx (Friedman p = 0.014 over tuned algorithms) but **no pairwise difference survives Holm correction** (smallest adjusted p 0.16).
* Validation-based tuning did **not** improve test F1 for any algorithm (all mean differences -0.003 to -0.010, uncorrected p >= 0.09).
* Leave-one-site-out: macro AUC about 0.79-0.80 for all algorithms; transfer to Switzerland and VA (AUC about 0.70-0.72) is much worse than to Cleveland/Hungary (0.85-0.88). Seeds replicate only initialisation and only four sites exist: descriptive only.
* Secure aggregation reproduces plaintext parameters to float rounding (max abs difference about 1.5e-7, identical F1). Switzerland has 8 negatives (positive rate 0.93): F1/accuracy there are inflated; use AUC/balanced accuracy.
* Scope: four 1980s public research cohorts, virtual clients in one process, pooled preprocessing statistics; not a live federation; no compliance claim.

### Task 4.11 verdicts (24-configuration random search over lr, mu, batch size, local epochs, hidden width, SCAFFOLD server step; computed from the tables above)
* Random search vs the lr/mu grid: mean F1 differences are between -0.011 and +0.018; only FedNova on synthetic data reaches uncorrected p = 0.049 (n = 10 seeds); with 24 comparisons about one such result is expected by chance, so **no reliable advantage of broader tuning is shown**.
* Tuning can hurt: SCAFFOLD's default (F1 0.873) beat both grid (0.801) and random (0.814) selections on synthetic data (differences not significant, n = 10); selection on a 25% validation split does not reliably transfer to test.
* Natural sites (heart, 5 splits): every default/grid/random difference is <= 0.011 F1 and none is significant (Wilcoxon cannot reach p < 0.05 with 5 pairs anyway).
* FedAvg's random-search learning rate (0.78) is again near the top of the search range: its optimum is not established.
* Budgets are matched in number of configurations, not compute; 24 configurations is a small search; no best-algorithm claim is made from this experiment.

## Limitations
- Only lr and μ tuned; boundary optimum for FedAvg lr; real datasets tiny (test sets 91-231 samples); sign-flip robustness for SCAFFOLD/FedNova uses plain averaging of corrected updates and is not a defence claim.
- Synthetic data, virtual clients, 2 CPU cores, no GPU, no real network.
