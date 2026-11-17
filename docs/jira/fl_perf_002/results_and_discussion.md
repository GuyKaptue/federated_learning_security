# Results and discussion — fl_perf_002: Performance and scalability at 10/50/100/500 virtual clients

All numbers below are read from the stored result files (MEASURED unless stated; virtual clients; synthetic data).

## KPI check (task 2.5)

| config | n_clients | training_overhead_pct | target_lt_30 | comm_overhead_pct | target_lt_40 | accuracy_loss_pp | target_lt_1 |
|---|---|---|---|---|---|---|---|
| secagg_complete | 10 | 15.6761 | True | 89.4775 | False | 0.0 | True |
| secagg_complete | 50 | 507.0469 | False | 259.2888 | False | 0.0 | True |
| secagg_complete | 100 | 2050.0639 | False | 471.553 | False | 0.0 | True |
| he_ckks | 10 | 77.1199 | False | 2955.3429 | False | 0.0 | True |
| he_ckks | 50 | 361.4532 | False | 2956.9802 | False | 0.0 | True |
| he_ckks | 100 | 570.2023 | False | 2955.7148 | False | 0.0 | True |
| he_ckks | 500 | 1325.5695 | False | 2956.4369 | False | 0.0 | True |
| smpc_3party | 10 | 104.5352 | False | 250.0 | False | 0.0 | True |
| smpc_3party | 50 | 490.7526 | False | 250.0 | False | 0.0 | True |
| smpc_3party | 100 | 962.8699 | False | 250.0 | False | 0.0 | True |
| smpc_3party | 500 | 2074.5006 | False | 250.0 | False | 0.0 | True |
| secagg_sparse14 | 100 | 242.9421 | False | 110.7039 | False | 0.0 | True |
| secagg_sparse14 | 500 | 585.0599 | False | 110.7039 | False | 0.0 | True |

## Scaling exponents (task 2.5)

| config | loglog_slope | points | note |
|---|---|---|---|
| plain | 0.2557 | 4 | slope~1 linear in n; slope~2 quadratic (empirical, 3-4 points) |
| secagg_complete | 1.3309 | 3 | slope~1 linear in n; slope~2 quadratic (empirical, 3-4 points) |
| he_ckks | 0.7905 | 4 | slope~1 linear in n; slope~2 quadratic (empirical, 3-4 points) |
| smpc_3party | 0.8699 | 4 | slope~1 linear in n; slope~2 quadratic (empirical, 3-4 points) |

## Scalability table (task 2.5)

| config | n_clients | runs | train_wall_s | mean_round_s | client_time_s | agg_time_s | up_mb | down_mb | comm_mb_per_round | cpu_time_s | cpu_util_mean_pct | rss_peak_mb | throughput_samples_per_s | acc | train_overhead_pct | comm_overhead_pct | acc_loss_pp | agg_share_of_round | est_cost_usd_per_run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain | 10 | 3 | 1.4471 | 0.1447 | 1.432 | 0.002 | 0.5257 | 0.5257 | 0.1051 | 1.44 | 99.5664 | 899.1198 | 59383.6345 | 0.7774 | nan | nan | nan | 0.0014 | 0.0001 |
| plain | 50 | 3 | 1.6474 | 0.1647 | 1.6302 | 0.0041 | 2.6283 | 2.6283 | 0.5257 | 1.64 | 99.147 | 998.3255 | 51993.2012 | 0.6805 | nan | nan | nan | 0.0025 | 0.0005 |
| plain | 100 | 2 | 1.8427 | 0.1843 | 1.8176 | 0.0084 | 5.2567 | 5.2567 | 1.0513 | 1.815 | 98.0632 | 1006.8945 | 46241.4013 | 0.6214 | nan | nan | nan | 0.0046 | 0.0009 |
| plain | 500 | 1 | 3.98 | 0.398 | 3.9082 | 0.0387 | 26.2833 | 26.2833 | 5.2567 | 3.91 | 97.9923 | 1027.6484 | 21493.3536 | 0.535 | nan | nan | nan | 0.0097 | 0.0047 |
| secagg_complete | 10 | 3 | 1.6555 | 0.1655 | 1.3812 | 0.2632 | 1.2672 | 0.7248 | 0.1992 | 1.65 | 99.5432 | 934.7109 | 60831.4277 | 0.7774 | 15.6761 | 89.4775 | 0.0 | 0.159 | 0.0002 |
| secagg_complete | 50 | 3 | 9.8501 | 0.985 | 1.8267 | 8.0071 | 10.8376 | 8.049 | 1.8887 | 9.82 | 100.0655 | 999.207 | 46402.1618 | 0.6805 | 507.0469 | 259.2888 | 0.0 | 0.8129 | 0.0018 |
| secagg_complete | 100 | 2 | 39.5823 | 3.9582 | 1.9105 | 37.6548 | 32.9285 | 27.1606 | 6.0089 | 39.41 | 100.2844 | 1010.2344 | 43996.1418 | 0.6214 | 2050.0639 | 471.553 | 0.0 | 0.9513 | 0.0057 |
| he_ckks | 10 | 3 | 2.5495 | 0.255 | 1.4499 | 1.0845 | 31.5961 | 0.5257 | 3.2122 | 2.5333 | 101.7924 | 978.474 | 58151.6139 | 0.7774 | 77.1199 | 2955.3429 | 0.0 | 0.4254 | 0.0029 |
| he_ckks | 50 | 3 | 7.5447 | 0.7545 | 1.8077 | 5.7209 | 158.0665 | 2.6283 | 16.0695 | 7.48 | 100.2505 | 1004.9661 | 46474.421 | 0.6805 | 361.4532 | 2956.9802 | 0.0 | 0.7583 | 0.0142 |
| he_ckks | 100 | 2 | 12.3395 | 1.2339 | 1.8834 | 10.4357 | 316.0 | 5.2567 | 32.1257 | 12.235 | 99.6433 | 1026.4102 | 44603.7472 | 0.6214 | 570.2023 | 2955.7148 | 0.0 | 0.8457 | 0.0284 |
| he_ckks | 500 | 1 | 56.7382 | 5.6738 | 3.8267 | 52.8817 | 1580.3795 | 26.2833 | 160.6663 | 56.13 | 99.7023 | 1157.2891 | 21951.119 | 0.535 | 1325.5695 | 2956.4369 | 0.0 | 0.932 | 0.1418 |
| smpc_3party | 10 | 3 | 2.9337 | 0.2934 | 1.3808 | 1.5397 | 3.154 | 0.5257 | 0.368 | 2.9133 | 99.1944 | 982.1784 | 60983.5135 | 0.7774 | 104.5352 | 250.0 | 0.0 | 0.5248 | 0.0004 |
| smpc_3party | 50 | 3 | 9.6503 | 0.965 | 1.7208 | 7.9165 | 15.77 | 2.6283 | 1.8398 | 9.5733 | 99.2838 | 1006.9036 | 49078.7683 | 0.6805 | 490.7526 | 250.0 | 0.0 | 0.8203 | 0.0017 |
| smpc_3party | 100 | 2 | 19.575 | 1.9575 | 1.9721 | 17.5861 | 31.5399 | 5.2567 | 3.6797 | 19.44 | 99.3603 | 1027.6484 | 42594.7078 | 0.6214 | 962.8699 | 250.0 | 0.0 | 0.8984 | 0.0035 |
| smpc_3party | 500 | 1 | 86.546 | 8.6546 | 3.7235 | 82.7902 | 157.6996 | 26.2833 | 18.3983 | 85.68 | 99.1302 | 1157.2773 | 22559.6199 | 0.535 | 2074.5006 | 250.0 | 0.0 | 0.9566 | 0.0171 |
| secagg_sparse14 | 100 | 2 | 6.3343 | 0.6334 | 1.813 | 4.5044 | 13.7978 | 8.3542 | 2.2152 | 6.315 | 100.0381 | 1010.2344 | 46989.5437 | 0.6214 | 242.9421 | 110.7039 | 0.0 | 0.7111 | 0.002 |
| secagg_sparse14 | 500 | 1 | 27.2657 | 2.7266 | 4.212 | 23.0236 | 68.9888 | 41.7709 | 11.076 | 27.11 | 100.4143 | 1027.6484 | 19943.0743 | 0.535 | 585.0599 | 110.7039 | 0.0 | 0.8444 | 0.01 |


### Acceptance verdicts
* Monitoring of time, CPU, memory, communication with MLflow logging (SQLite backend): MET (GPU: not available, recorded as such).
* Benchmarks at 10, 50, 100, 500 **virtual** clients: MET, with the exception **complete-graph SecAgg at 500 clients = NOT_RUN** (a sparse k=14 graph was measured instead and is labelled as such).
* KPIs training overhead < 30%, communication overhead < 40%, accuracy loss < 1 pp: accuracy KPI MET everywhere; the other two are **NOT MET** in most rows of the KPI table above (the table is the verdict; rows with `True` are the exceptions).
* Cost estimates use labelled *illustrative* price assumptions (ESTIMATED), not quotes.

## Limitations
- Virtual clients share one 2-core machine: wall time is serial; no network. Overheads are relative to a plain in-process baseline that is very cheap, which inflates percentage overheads.
- Synthetic data, virtual clients, 2 CPU cores, no GPU, no real network.
