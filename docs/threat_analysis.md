# Lightweight threat analysis for the rule committee (Stage 3)

Scope: this is an elementary, assumption-explicit analysis that explains *what the experiments measure*. It is not a convergence proof, not a certified-robustness result and not a security proof. All empirical statements refer to the synthetic SYN-IDS-v1 benchmark with simulated (virtual) clients; no hospital or patient data were used and no credentialed clinical repository (e.g. PhysioNet) was available.

## 1. Threat model

* K = 20 clients, f = 4 of them (rows 0..3) are malicious and **collude**; the server is honest-but-fallible (it runs the aggregation rule faithfully).
* Attacker capability ("omniscient upper bound"): sees all honest updates of the round, knows the pool of rules and may simulate candidate attack vectors against a copy of the server. This is stronger than any deployed attacker, so results are upper bounds on attack success.
* Goals: (i) untargeted accuracy degradation, (ii) a backdoor with a fixed trigger (attack success rate, ASR).
* Out of scope: malicious server, Sybil creation beyond f, secure-aggregation interaction (secure-and-robust aggregation is postponed), adaptive defender tuning per attack.

## 2. Defence under analysis

`RuleCommittee(members, k, vote, randomise)`: each round, draw k rules from a pool (anomaly filter, multi-Krum, FLAME, SignGuard). A row is rejected if at least `min(vote, #voting members)` members flag it. Survivors are combined by a clipped weighted mean. If no row survives, the coordinate-wise median is used (fail-open).
Pool and thresholds were fixed a priori and not tuned per attack.

## 3. Attacker-knowledge ladder

| Level | What the attacker knows | Implementation |
|---|---|---|
| rule-blind | nothing about the server rule | baseline attacks (sign flip, ALIE) |
| pool-aware (`pool`) | pool, k, vote rule; **not** this round's draw | each candidate is simulated under an independent random draw |
| draw-aware (`draw`) | the exact draw that will run | the simulated server copy has the same RNG state (equivalent to a deterministic committee) |

A deterministic committee (`randomise=False`) makes draw-aware and pool-aware coincide.

## 4. Two elementary statements

**Proposition 1 (union filter shrinks the feasible set).** Fix an honest-update set H and a candidate malicious vector u. Let A_m(H,u) be the set of rows member m accepts. With `vote = 1` the committee accepts exactly the intersection of the A_m over drawn members. Hence for committees M ⊆ M', the accepted set of M' is a subset of the accepted set of M, and the set of vectors u that pass unnoticed (the attacker's feasible set) can only shrink as members are added.
*Limits:* the statement concerns acceptance of the malicious rows only. It says nothing about damage of the survivors, and a larger union also rejects more honest rows (measured false-positive rate), which costs accuracy and, under non-IID data, fairness to minority clients. The fail-open fallback to the median is outside the proposition.

**Proposition 2 (randomisation cannot help the attacker).** Let D(u, r) be the damage of attack vector u when the realised draw is r, and let r be random. Then
max_u E_r[D(u, r)] ≤ E_r[max_u D(u, r)].
The left side is a pool-aware attacker who must commit before the draw, the right side a draw-aware attacker. So the pool-aware attacker is never better off in expectation than the draw-aware one, and the experiment reports both; the gap between them is what randomisation buys *in these settings*.
*Limits:* this is only an ordering, not a lower bound on the gap (the gap can be zero), it holds for expected damage and not for a single round, and it assumes the draw is private and fresh each round. If the draw is predictable (reused or leaked seed) the committee degenerates to the deterministic one.

## 5. What is and is not claimed

* The concepts "randomised aggregation" and "hybrid/multi-rule aggregation" are not new (Nabavirazavi et al. 2023; Yue et al. 2024). The Stage-3 contribution is limited to the measured study: committee variants under the knowledge ladder with detection metrics (TPR/FPR), a jitter attacker that removes the identical-rows signature, and defence ablations (k, vote rule, attacker fraction).
* The jitter noise is added *after* the attack vector is chosen (the attacker does not optimise against the jitter), which is a weakness of the attacker, not of the defence.
* FoolsGold keeps state across rounds and is therefore excluded from the committee pool.
* With f = 6 and 8 the defender's Krum assumption stays at f = 4 (misspecified on purpose); results there show degradation under misspecification, not a tuned defence.
* Statistical statements use paired tests over seeds with Holm correction; with few seeds Wilcoxon cannot reach p < 0.05.
