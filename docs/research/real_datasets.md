# Real public datasets used (tasks 3.7, 4.9)

| Dataset | Rows x features | Obtained from | Integrity | Terms | De-identification note |
|---|---|---|---|---|---|
| WDBC (Breast Cancer Wisconsin Diagnostic) | 569 x 30 | bundled with scikit-learn (no download) | n/a | UCI page: CC BY 4.0 (page read 2026-11-06) | UCI page does not discuss de-identification; sklearn copy has no patient IDs |
| Pima Indians Diabetes | 768 x 8 | public GitHub mirror (jbrownlee/Datasets), URL in `src/federated_learning/preprocessing/registry.py` | SHA-256 pinned in code | licence of the mirror NOT verified; original: NIDDK, donated by V. Sigillito (mirror page of `pima.names` read) | not documented on the pages read |
| Cleveland heart disease | 302 x 13 (+target) after removing 1 exact duplicate of 303 | public GitHub mirror (kb22/Heart-Disease-Prediction), URL in `src/federated_learning/preprocessing/registry.py` | SHA-256 pinned in code | UCI page: CC BY 4.0 for the original; mirror licence NOT verified | UCI page states names and SSNs were removed and replaced by dummy values |

* The downloaded Pima/Heart files are **not** included in the ZIP (licence of the mirrors unverified); `load_real()` downloads them to `data/raw/<dataset>/` and refuses files whose SHA-256 differs from the pinned value.
* Preprocessing: stratified 70:30 split per seed; Pima zeros in glucose, blood pressure, skin-fold, insulin, BMI treated as missing; imputation (train median) and standardisation use TRAIN statistics only; validation for tuning is carved from TRAIN.
* These are small research tables, not hospital federations. 'Clients' are synthetic Dirichlet partitions of one table; the datasets have 91-231 test samples, so intervals are wide. Nothing here is evidence about performance on a real multi-site deployment.
* Label semantics are used as provided (binary); the Heart mirror's target encoding was not re-derived from the UCI 'num' column.
* Dataset sources (UCI WDBC, UCI Heart) are listed in REFERENCES_MASTER (`uci_wdbc`, `uci_heart`, `pima_nidd`, `smith1988`).
