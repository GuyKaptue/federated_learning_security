# Healthcare FL threat model (STRIDE-oriented, simulation scope)

## Assets
Local patient records (never leave a site in FL); model updates (derived from records); global model (integrity, IP); keys; audit logs.

## Adversaries
| ID | Adversary | Capability | Evaluated in sprint? |
|---|---|---|---|
| A1 | Honest-but-curious aggregation server | sees all uploads, follows protocol | Yes: HE / SecAgg / SMPC (SEC-001) |
| A2 | Malicious server | deviates from protocol, forges/omits messages | **Partially**: SecAgg consistency round NOT implemented; SMPC MAC check detects share tampering; no malicious-server proof for SecAgg |
| A3 | Malicious client(s) | submit poisoned/arbitrary updates | Yes: backdoor, Byzantine (ATTACK-003) |
| A4 | Colluding clients (+ server) | pool views | Only via SecAgg threshold reasoning / SMPC non-collusion assumption (documented, not attacked empirically) |
| A5 | External network attacker | eavesdrop/tamper in transit | **No** (no network layer simulated; TLS assumed out of scope) |
| A6 | Insider with model access | black-box/white-box queries | Membership inference (shadow-model, loss baseline) |
| A7 | Gradient-observing attacker | sees one client's plaintext update | Gradient leakage DLG/iDLG + analytic inversion |

## STRIDE summary
Spoofing: client identity/attestation not simulated (open). Tampering: Byzantine/backdoor updates (tested), share tampering (SMPC MAC, SecAgg AEAD tag). Repudiation: audit logging only as provenance hashes. Information disclosure: A1/A6/A7 (tested). Denial of service: dropout/straggler cost (tested), malicious abort (SecAgg/SMPC abort on failed checks – a DoS vector by design). Elevation: not applicable in simulation.

## Key assumptions and limits
Trusted-dealer offline phase in SMPC (not a real preprocessing protocol); key-holder model for HE (server holds public context only; decryption party assumed honest and non-colluding with server); SecAgg semi-honest server, no consistency round; no differential-privacy guarantee for SecAgg/HE alone; SecAgg/HE/SMPC do **not** defend against poisoning (verified empirically where measured).
