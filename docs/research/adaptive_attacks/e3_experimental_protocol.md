# E3 Phase 2 – Experimental protocol (pre-registration draft)

Registered before any diversified attacker is run. Seeds 0–9 minimum; dataset seed 0; 15 rounds; iid and Dirichlet(0.3) partitions; f = 4 of K = 20. Statistics: paired differences by seed, Cohen's d_z, Cliff's delta, paired t-interval and exact Wilcoxon, Holm correction over the pre-registered family (all five hypotheses and defence comparisons counted), significance 0.05. No seed, scenario or defence is dropped after seeing results; all runs are reported.

## Hypotheses (null in parentheses)
- **H1.** Against the all-feature FoolsGold, jitter raises backdoor ASR relative to ordinary collusion (B). Outcome: ASR(C) − ASR(B), paired. (H0: no difference.) Prior evidence: +0.292 at 5 seeds, unadjusted p = 0.039, exploratory only.
- **H2.** Evasion is paid for by the attacker: higher diversification lowers FoolsGold's malicious detection (TPR) but also lowers backdoor strength when measured without a defence (mean aggregation) and against distance defences. Outcome: TPR and ASR as functions of d. (H0: monotone-free, no trade-off.)
- **H3.** Under Dirichlet(0.3) heterogeneity FoolsGold has higher FPR and lower benign utility than under iid, and diversified colluders hide better. Outcome: FPR, clean accuracy, ASR by partition. (H0: no partition effect.)
- **H4.** With indicative-feature weighting, FoolsGold resists jitter at least as well as the all-feature variant. Outcome: ASR(indicative) − ASR(all-feature) under C. (H0: no difference.) Requires the variant to be implemented first.
- **H5.** Successful evasion costs the defender benign utility: where FoolsGold suppresses less backdoor, clean accuracy and benign-client weight change; report the suppression versus utility frontier. Outcome: clean accuracy, per-client weight alpha for benign clients. (H0: no relation.)

## Primary and secondary outcomes
Primary: backdoor ASR, with no-attack ASR floor (about 0.38) reported alongside. Secondary: clean accuracy, TPR/FPR of the detector, the four diversification measures, mean alpha for benign versus malicious clients.

## Control conditions
Every scenario is run for A, B, C (D conditional) under each defence; ASR is always compared to A on the same seed.

## Gates
Baseline gate passed (see `e3_baseline_validation.md`). Nothing beyond the baseline is run until explicit approval.

## Update 2026-11-14 (Phase 4A)
H4 (indicative-feature FoolsGold resists jitter at least as well as the all-feature variant) was evaluated on iid and Dirichlet(0.3) with the existing jitter-1.0 colluders, 10 seeds: supported on iid (difference −0.284 [−0.430, −0.139], Holm p = 0.014), inconclusive on Dirichlet. H1 for the legacy variant is supported with 10 seeds (+0.293, Holm p = 0.014 within the Phase 4A family). Hypotheses H2, H3, H5 were not tested. The Phase 4A family (7 tests per partition) is separate from the original pre-registered family; this is stated, not hidden.
