# Experimental methodology — fl_attack_003

* Data: synthetic SYN-IDS-v1 (40 features, binary, 70:30); IID and Dirichlet partitions; **virtual clients**.
* Seeds fixed; unit of replication = seed; 95% CIs (t/bootstrap); non-parametric tests where applicable (Wilcoxon needs ≥6 seeds for p<0.05).
* Provenance in every result JSON (experiment id, config SHA-256, environment, no git commit).
* Labels: MEASURED / SIMULATED / ESTIMATED / THEORETICAL / NOT_RUN.
* Hardware: 2 CPU cores, no GPU. Timing experiments were run alone on the machine.

## Research questions and hypotheses
### FL-ATTACK-003.1
- RQ: How does ASR grow with attacker fraction/boost?
- RQ: What is the ASR of the trigger on a clean model?
- H: ASR rises with intensity; secure aggregation does not reduce it.
### FL-ATTACK-003.2
- RQ: Which aggregator preserves accuracy under which behaviour and fraction?
- RQ: What is the clean-run cost of robustness?
- H: Mean aggregation degrades under sign-flip; robust rules do not.
### FL-ATTACK-003.3
- RQ: Does the attack exceed AUC 0.6 on an overfit target?
- RQ: What do clipping and noise change (AUC, TPR@1%FPR, utility, ε)?
- H: DP noise lowers AUC and utility monotonically.
### FL-ATTACK-003.4
- RQ: When does reconstruction succeed (batch size, hidden width, noise)?
- RQ: Does aggregation over clients protect?
- H: Success is high for B=1,2 plaintext and falls with batch size/aggregation/noise.
### FL-ATTACK-003.5
- RQ: Which defences reduce ASR/accuracy damage and at what clean-accuracy cost?
- H: Secure aggregation and HE leave backdoor ASR unchanged; robust rules/norm bounds reduce it.
### FL-ATTACK-003.6
- RQ: Which KPIs are met?
- H: Not all KPIs are met.
### FL-ATTACK-003.7
- RQ: Do median, trimmed mean, Krum/Multi-Krum, median-norm clipping and the anomaly filter keep accuracy under ALIE/MIN-MAX?
- RQ: Do norm- or distance-constrained backdoors defeat the defences that stopped the static backdoor?
- H: At least one defence that performed well against static attacks degrades under an adaptive attack.
- H: A norm-matched backdoor passes norm-based defences.
### FL-ATTACK-003.8
- RQ: Is there any tested rule whose accuracy stays near the no-attack level under the strongest attack?
- RQ: Does AGR-tailored beat the best fixed attack (paired by seed)?
- RQ: Do AGR-aware backdoors raise ASR above the distance-constrained backdoor of task 3.7?
- H: Every rule has at least one attack that lowers its accuracy by more than 1 pp.
- H: AGR-tailored is at least as damaging as the best fixed attack on most rules.
