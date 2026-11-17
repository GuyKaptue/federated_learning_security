# Experimental methodology — fl_sec_001

* Data: synthetic SYN-IDS-v1 (40 features, binary, 70:30); IID and Dirichlet partitions; **virtual clients**.
* Seeds fixed; unit of replication = seed; 95% CIs (t/bootstrap); non-parametric tests where applicable (Wilcoxon needs ≥6 seeds for p<0.05).
* Provenance in every result JSON (experiment id, config SHA-256, environment, no git commit).
* Labels: MEASURED / SIMULATED / ESTIMATED / THEORETICAL / NOT_RUN.
* Hardware: 2 CPU cores, no GPU. Timing experiments were run alone on the machine.

## Research questions and hypotheses
### FL-SEC-001.1
- RQ: Which mechanisms satisfy the story's requirements (server learns only the aggregate; correctness; dropout tolerance)?
- RQ: What are the measured per-update costs of the HE schemes at the model size used?
- H: CKKS gives the lowest ciphertext expansion among HE schemes for float vectors, BFV is exact up to quantisation, Paillier is the most expensive.
- H: SecAgg is cheapest in bandwidth but requires dropout machinery.
### FL-SEC-001.2
- RQ: Does encrypted aggregation reproduce plaintext FedAvg accuracy?
- RQ: How do error and cost scale with vector length and magnitude?
- H: Max aggregation error of CKKS < 1e-5 for update magnitudes used.
- H: Final accuracy equals plaintext FedAvg (loss ≈ 0 pp).
### FL-SEC-001.3
- RQ: Is the aggregate exact under dropout?
- RQ: What does the server view look like, and when does the protocol abort?
- H: Aggregate error < 1e-6 with ≤ (1−threshold) dropout.
- H: Server view is statistically unrelated to plaintext.
### FL-SEC-001.4
- RQ: Is the aggregate exact?
- RQ: Is share tampering detected?
- RQ: How do costs scale with parties and clients?
- H: Aggregate error < 1e-5.
- H: Any single-share modification triggers MAC failure.
### FL-SEC-001.5
- RQ: Does any mechanism change accuracy?
- RQ: What are the training-time and communication overheads?
- H: Accuracy loss < 1 pp for all mechanisms.
- H: Overheads differ by orders of magnitude between mechanisms.
