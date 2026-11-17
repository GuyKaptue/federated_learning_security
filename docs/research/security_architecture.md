# Reference security architecture (design discussion, not an implemented/certified system)

```
 Site (hospital) boundary                     Aggregation zone                     Key/decryption zone
 +-----------------------------+          +---------------------------+          +--------------------+
 | local data (never exported) |          | server: public ctx only   |          | key holder / threshold |
 | local training              |  masked/ |  - verify update bounds   |  ct sum  | decrypts AGGREGATE only|
 | clip + (opt.) DP noise      | encrypted|  - robust/secure aggregate| -------> |                        |
 | encrypt / mask update       | -------> |  - audit log (hashes)     |          +--------------------+
 +-----------------------------+          +---------------------------+
          mTLS + site attestation (out of simulation scope)
```
Layers: (1) transport security and site authentication (assumed, not simulated); (2) update confidentiality (HE/SecAgg/SMPC – evaluated); (3) update integrity (clipping, robust aggregation, norm bounds – evaluated; note incompatibility: SecAgg/HE hide individual updates, so server-side robust statistics need SMPC-style bound checks or client-side clipping – see ablation in FL-PUB-005); (4) output privacy (client-side DP – evaluated with RDP accountant); (5) governance (DPIA, legal basis, key management, logging, incident response – out of scope; required for any real deployment).
Residual risks are listed in fl_pub_005/threat_model.md and the final report.
