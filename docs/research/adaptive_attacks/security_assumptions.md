# Security assumptions
1. The server is honest-but-fallible and runs the stated rule; it is not an attacker.
2. Attackers are simulated omniscient upper bounds (C, E) or knowledge-limited (D, F); results are not estimates of real attack prevalence.
3. Client data are synthetic or small public tables; no patient data and no credentialed clinical repository were used.
4. Plaintext updates are visible to the defence; secure aggregation is not combined with robust rules in this work package (postponed).
5. No differential-privacy or cryptographic security claim is made.
6. Statistical claims use paired tests over seeds; with fewer than 6 seeds exact Wilcoxon cannot reach p < 0.05.
