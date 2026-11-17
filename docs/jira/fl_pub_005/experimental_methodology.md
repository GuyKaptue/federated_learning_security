# Experimental methodology — fl_pub_005

* Data: synthetic SYN-IDS-v1 (40 features, binary, 70:30); IID and Dirichlet partitions; **virtual clients**.
* Seeds fixed; unit of replication = seed; 95% CIs (t/bootstrap); non-parametric tests where applicable (Wilcoxon needs ≥6 seeds for p<0.05).
* Provenance in every result JSON (experiment id, config SHA-256, environment, no git commit).
* Labels: MEASURED / SIMULATED / ESTIMATED / THEORETICAL / NOT_RUN.
* Hardware: 2 CPU cores, no GPU. Timing experiments were run alone on the machine.

## Research questions and hypotheses
### FL-PUB-005.1
- RQ: Are results bit-reproducible with a fixed seed?
- RQ: Do stored summaries equal recomputation from raw runs?
- H: All determinism and consistency checks pass; the git-commit check fails (no repository).
### FL-PUB-005.2
- RQ: What does each component cost and protect against?
- RQ: Which combinations are incompatible?
- H: Confidentiality components do not change robustness; robust/clipping components do.
### FL-PUB-005.3
- RQ: Does the document state its assumptions, limits and sources honestly and consistently with the stored results?
- H: Every quantitative statement is traceable to a stored result file; no compliance claim appears.
### FL-PUB-005.4
- RQ: Does the document state its assumptions, limits and sources honestly and consistently with the stored results?
- H: Every quantitative statement is traceable to a stored result file; no compliance claim appears.
### FL-PUB-005.5
- RQ: Does the document state its assumptions, limits and sources honestly and consistently with the stored results?
- H: Every quantitative statement is traceable to a stored result file; no compliance claim appears.
