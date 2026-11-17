# Results and discussion — fl_sec_001: Cryptographic validation: HE, Secure Aggregation, SMPC

All numbers below are read from the stored result files (MEASURED unless stated; virtual clients; synthetic data).

## Plaintext vs HE vs SecAgg vs SMPC (task 1.5, MEASURED, mean over seeds)

| config | acc | train_overhead_pct | comm_overhead_pct | acc_loss_pp | max_agg_err |
|---|---|---|---|---|---|
| plaintext | 0.8104 | 0.0 | 0.0 | 0.0 | 0.0 |
| he_ckks | 0.8104 | 108.1138 | 5912.4371 | 0.0 | 0.0 |
| he_bfv | 0.8104 | 84.3907 | 7745.5632 | 0.0 | 0.0 |
| secagg | 0.8104 | 33.8723 | 141.074 | 0.0 | 0.0 |
| secagg_10pct_dropout | 0.811 | 28.0219 | 119.8258 | -0.0648 | 0.0 |
| smpc_3party | 0.8104 | 121.9874 | 500.0 | 0.0 | 0.0 |

## SecAgg dropout (task 1.3)

| stage | drop_fraction | trials | completed | aborted | max_abs_err | threshold_frac | n |
|---|---|---|---|---|---|---|---|
| before_masking | 0.0 | 5 | 5 | 0 | 0.0 | 0.67 | 20 |
| before_masking | 0.1 | 5 | 5 | 0 | 0.0 | 0.67 | 20 |
| before_masking | 0.2 | 5 | 5 | 0 | 0.0 | 0.67 | 20 |
| before_masking | 0.3 | 5 | 5 | 0 | 0.0 | 0.67 | 20 |
| before_masking | 0.4 | 5 | 0 | 5 | nan | 0.67 | 20 |
| before_unmask | 0.0 | 5 | 5 | 0 | 0.0 | 0.67 | 20 |
| before_unmask | 0.1 | 5 | 5 | 0 | 0.0 | 0.67 | 20 |
| before_unmask | 0.2 | 5 | 5 | 0 | 0.0 | 0.67 | 20 |
| before_unmask | 0.3 | 5 | 5 | 0 | 0.0 | 0.67 | 20 |
| before_unmask | 0.4 | 5 | 0 | 5 | nan | 0.67 | 20 |

## SMPC by number of parties (task 1.4)

| parties_m | clients | collusion_threshold_parties | time_s | max_abs_err | online_up_kb_per_client | offline_dealer_kb_total | party_to_party_kb | plaintext_kb_per_client | online_overhead_x |
|---|---|---|---|---|---|---|---|---|---|
| 2.0 | 10.0 | 1.0 | 0.0992 | 0.0 | 22.5312 | 788.5938 | 22.5625 | 5.6328 | 4.0 |
| 3.0 | 10.0 | 2.0 | 0.1594 | 0.0 | 33.7969 | 1126.5625 | 67.6406 | 5.6328 | 6.0 |
| 5.0 | 10.0 | 4.0 | 0.2597 | 0.0 | 56.3281 | 1802.5 | 225.3906 | 5.6328 | 10.0 |


### Acceptance verdicts (computed from the tables above)
* HE (CKKS/BFV/Paillier), SecAgg with dropout recovery and SMPC with MAC checks are **implemented and tested** (see `TESTS/`): MET.
* Accuracy loss < 1 pp: MET for every mechanism in the 1.5 table (`acc_loss_pp` column; the 95% CIs include 0).
* Training/communication overhead KPIs (< 30% / < 40%, from the Jira epic): **NOT MET** for HE and SMPC, and for SecAgg at the configuration measured (see `comm_overhead_pct`, `train_overhead_pct`). These are *measured simulation* costs with virtual clients.
* Security: correctness-to-plaintext is **not** a security proof. Server-view statistics (correlation, high-bit fraction) and abort/tamper tests are sanity checks only.

## Limitations
- SecAgg: semi-honest server, no consistency round; SMPC: trusted-dealer offline phase, own implementation (MP-SPDZ unavailable); HE: honest key-holder; Flower secaggplus inspected not run end-to-end (NOT_RUN).
- Synthetic data, virtual clients, 2 CPU cores, no GPU, no real network.
