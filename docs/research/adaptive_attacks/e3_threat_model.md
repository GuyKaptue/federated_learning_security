# E3 Phase 2 – Threat model

Scope: simulation only; virtual clients on synthetic SYN-IDS-v1 (reported as SYN-TAB-40). No hospital or patient data; no PhysioNet or other credentialed clinical repository was available.

- **System:** cross-silo horizontal FL, K = 20 virtual clients, honest-but-fallible server running a similarity-based Sybil defence (FoolsGold, all-feature variant; indicative-feature variant planned as an ablation) with distance-based defences as comparators. Plaintext updates (not secure-aggregation compatible; out of scope).
- **Attacker:** f = 4 colluding clients (20%), sees all honest updates and simulates the server (omniscient upper bound), knows the defence class. Goal: backdoor with a fixed trigger (ASR). Controls only its own updates; no server or data access beyond that.
- **Diversification:** attackers may vary their updates to lower pairwise similarity. The attacker's budget is a diversification level d (initially independent noise of relative norm d), constrained by a backdoor-utility floor.
- **Defender knowledge:** fixed hyperparameters, not tuned per attack.
- **Out of scope:** malicious server, extra Sybil identities beyond f, adaptive defender retuning, real-world deployment claims.

## Diversification measures (to be logged)
1. mean pairwise cosine among malicious updates (per round and on accumulated history, the quantity FoolsGold uses);
2. mean pairwise Euclidean distance relative to benign dispersion;
3. dispersion of malicious updates versus benign dispersion (ratio);
4. temporal consistency: cosine between a client's consecutive updates and between its history and the global update.

## Conditions
A benign (no attack); B ordinary collusion (identical rows, existing baseline); C weak jitter (e.g. 0.25, 0.5); D stronger jitter (1.0, 2.0) run **only if** C shows a signal and the extra run is justified in the go/no-go.
