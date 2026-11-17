# Adaptive attacks work package – index

Everything here is part of the same repository as the code under `src/` and results under `experiments/results/` and `outputs/research/`.

| Stage | Documents |
|---|---|
| Original work package (Gates 1–3) | `project_audit.md`, `literature_review.md`, `threat_model.md`, `research_gap.md`, and the other files without the `e3_` prefix |
| E3 Phases 0–4 | `e3_repository_audit.md`, `e3_implementation_inventory.md`, `e3_preliminary_results_audit.md`, `e3_literature_review.md`, `e3_literature_matrix.csv`, `e3_research_gap.md`, `e3_threat_model.md`, `e3_experimental_protocol.md`, `e3_baseline_validation.md`, `e3_go_no_go_report.md` |
| E3 Phase 4A | `foolsgold_paper_method_verification.md`, `foolsgold_implementation_audit.md`, `foolsgold_indicative_feature_validation.md`, `e3_gap_reassessment.md`, `e3_phase4a_completion_report.md` (current status and recommendation) |

Code: `src/federated_learning/aggregators/foolsgold_indicative.py`, `pipelines/experiments/e3.py`, `e3_fg.py`. Tests: `tests/unit/aggregators/test_foolsgold_indicative.py`, `tests/integration/reproducibility/test_e3_*.py`. Results: `outputs/research/adaptive_attacks/e3/{baseline,fg_validation}/`. Tracker: `docs/jira/tracker.md`.
Run: `PYTHONPATH=src python -m federated_learning.pipelines.experiments.run_all --task FL-E3-BASE` or `FL-E3-FGVAL`.
