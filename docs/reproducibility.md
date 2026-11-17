# Reproducibility guide
1. `pip install -r requirements.txt && pip install -e . --no-deps`
2. Data: `make data` (downloads and verifies raw files) and `make synthetic` (regenerates SYN-IDS-v1; compare the printed SHA-256 with `data/synthetic/metadata/`).
3. Tests: `make test` (211 tests, about 140 s).
4. Notebooks: `make notebooks` (14 numbered notebooks at demonstration scale; figures to `reports/figures/`, summaries to `reports/notebooks/`).
5. Full-scale experiments: `make experiments` (about 2.5 h on 2 CPU cores, run alone for valid timings; results to `experiments/results/`).
6. Manuscript: `cd latex && ./build.sh` (needs `pdflatex`, `bibtex` and the IEEEtran class, see `latex/README.md`).

Determinism holds on one software stack: other BLAS/torch builds can change low-order bits (the regression test then compares accuracy with a tolerance).
