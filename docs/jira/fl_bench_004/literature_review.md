# Literature review — fl_bench_004

Source status is per entry in REFERENCES_MASTER.md (most are `REFERENCE_NOT_VERIFIED`; no access to publisher hosts); statements summarise well-known results in the author's words and must be checked against the papers.

## FL-BENCH-004.1 FedAvg baseline
FedAvg averages local models weighted by data size [mcmahan2017].

## FL-BENCH-004.2 FedProx
A proximal term limits client drift under heterogeneity [li2020fedprox].

## FL-BENCH-004.3 SCAFFOLD
Control variates correct client drift [karimireddy2020]; communication doubles.

## FL-BENCH-004.4 FedNova
Heterogeneous local steps bias FedAvg's fixed point [wang2020fednova].

## FL-BENCH-004.5 Four-algorithm comparison grid
See tasks 4.1–4.4.

## FL-BENCH-004.6 Statistical analysis
Non-parametric multi-algorithm comparison [demsar2006, holm1979, wilcoxon1945].

## FL-BENCH-004.7 Leaderboard
Mean-rank aggregation of per-criterion ranks [demsar2006].

## FL-BENCH-004.10 Natural multi-site benchmark: UCI heart disease (4 collection sites as clients)
Real federations differ by site in prevalence, measurement practice and missingness [kairouz2021, rieke2020]; benchmark rankings depend on how heterogeneity is created [hsu2019, li2021niid]. Data: UCI Heart Disease [uci_heart], four site files fetched from a commit-pinned mirror and hash-verified.

## FL-BENCH-004.11 Broader tuning: six-parameter random search vs lr/mu grid
Hyper-parameter sensitivity can reverse algorithm rankings in FL benchmarks [li2021niid, hsu2019]; random search is a standard budget-matched alternative to grids.

## FL-BENCH-004.8 Validation-based tuning of the four FL baselines
Baseline comparisons are only meaningful if every method gets comparable tuning effort [li2020fedprox, karimireddy2020, wang2020fednova, li2021niid]. Validation data is carved from train; test data is never used for selection.

## FL-BENCH-004.9 Benchmark on real public de-identified medical tables
Small public clinical tables allow a first check that conclusions from synthetic data are not artefacts of the generator. They are NOT hospital federations: clients are synthetic partitions of a single table. Sources: UCI repository pages [uci_wdbc, uci_heart, pima_nidd, smith1988].

Keys used: `demsar2006`, `hsu2019`, `kairouz2021`, `karimireddy2020`, `li2020fedprox`, `li2021niid`, `mcmahan2017`, `pima_nidd`, `smith1988`, `uci_heart`, `uci_wdbc`, `wang2020fednova`
