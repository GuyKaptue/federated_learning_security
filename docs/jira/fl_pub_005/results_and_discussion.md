# Results and discussion — fl_pub_005: Reproducibility, ablation, threat model and publication draft

All numbers below are read from the stored result files (MEASURED unless stated; virtual clients; synthetic data).

## Ablation (task 5.2)

| config | acc | train_overhead_pct | acc_under_signflip | backdoor_asr | server_sees_individual_updates |
|---|---|---|---|---|---|
| plain FL | 0.8477 | 0.0 | 0.4156 | 0.9972 | True |
| +HE (CKKS) | 0.8477 | 60.2495 | 0.4156 | 0.9972 | False |
| +SecAgg | 0.8477 | 29.8722 | 0.4156 | 0.9972 | False |
| +SMPC (3-party) | 0.8477 | 92.749 | 0.4156 | 0.9972 | False |
| +DP (central, sigma=0.5) | 0.8211 | -2.5362 | 0.7762 | 0.855 | True |
| +robust (median) | 0.846 | -5.6885 | 0.8189 | 0.1335 | True |
| SecAgg + client-DP | 0.8225 | 38.6802 | 0.7806 | 0.857 | False |
| HE + client-DP | 0.8225 | 70.98 | 0.7806 | 0.857 | False |
| SMPC + norm-bound | 0.8478 | 394.763 | 0.8442 | 0.0458 | False |

## Incompatible combinations (task 5.2)

| combination | reason | status |
|---|---|---|
| SecAgg + median/Krum/trimmed-mean | robust rules need per-client updates; masked updates are uniformly random individually | NOT_RUN (incompatible by construction) |
| HE (CKKS) + Krum/median | needs encrypted comparisons/sorting (deep circuits; not provided by additive CKKS aggregation used here) | NOT_RUN |
| plain + DP + robust (median) | composition would need a separate privacy analysis for median sensitivity | NOT_RUN |


### Acceptance verdicts
* Reproducibility audit executed (task 5.1): determinism, re-execution, provenance-hash and consistency checks pass; **git commit identifier unavailable** (check fails by design) → SHA-256 manifest used instead.
* Ablation with 9 feasible configurations plus documented incompatible combinations (NOT_RUN with reason): MET.
* Threat model and healthcare context documents written; **no compliance claim**; legal sources are `REFERENCE_NOT_VERIFIED`.
* Journal-style report and IEEE draft: **drafts**, not submission-ready (see `PUBLICATION/`).

## Limitations
- Gradient-leakage and MIA columns in the ablation are reused from tasks 3.3/3.4.
- Synthetic data, virtual clients, 2 CPU cores, no GPU, no real network.
