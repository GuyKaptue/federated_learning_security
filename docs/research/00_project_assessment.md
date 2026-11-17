# 00 — Project Assessment (Phase 0 baseline)

Date: 2026-11-05 · Epic: EPIC-001 · Written **before** any implementation.

## 1. What was actually found

| Item | Finding |
|---|---|
| Workspace contents at start | One file only: `PBSE027E_ch12.pdf` (the source chapter). No repository, no source, no tests, no notebooks, no configs, no logs, no CI, no git history. |
| Git | Not a git repository → no commit identifiers available; experiments record `git_commit: null` and a SHA-256 of the configuration instead. |
| Hardware | 2 CPU cores, 7 GB RAM, **no GPU**, Python 3.13. |
| Network from the shell | Restricted. Package registries (PyPI) work; arXiv, Crossref, Flower docs and GitHub releases are **not reachable** from the shell. |
| Dataset access | CIC-IDS-2017 (~50 GB PCAP / CSV) and the "IoT Healthcare Security" dataset are **not available** in the workspace and cannot be downloaded. |
| Installed for the sprint | `flwr 1.43.0`, `torch 2.15.0` (CPU path), `tenseal 0.3.18`, `phe 1.5.0`, `mlflow 3.18.0`, `opacus 1.6.0`, `nbconvert`, `pytest`. |

## 2. What the source chapter actually contains

Read from the supplied PDF text (Chapter 12, *Cybersecurity-enabled federated learning approach for digital healthcare*):

* **No code, no repository link, no hyperparameters for federation** (number of clients, rounds, local epochs, aggregation rule, participation rate are all unspecified).
* The "algorithm" in §12.7 is **pseudocode** (`HomomorphicEncryption(localModels)` or `SecureMultiPartyComputation(localModels)` are single placeholder calls). No scheme, parameters, key management or protocol is specified.
* The experimental model is a small 1-D CNN (TensorFlow, Adam, binary cross-entropy, 30 epochs, 70:30 split). The text also mentions "recurrent neural network" but never describes it.
* Reported results (Table 12.3): CIC-IDS-2017 accuracy 97.5 → 99.5 (centralized → "proposed"); IoT dataset 98.2 → 99.2.
* The task is **network intrusion detection on IoT/network-flow data**, not clinical prediction on patient records. "Healthcare" enters only through the IoT-healthcare framing. Conclusions about clinical models cannot be inferred.

### Observations that limit generalisation (to be reported honestly, not "fixed")

1. A federated model beating its centralized counterpart by ~2 points is unexplained; no federated setup, client split, or seed variance is given, and no repeated runs or significance test are reported.
2. Datasets are described with 15 classes (1 benign + 14 attacks), yet the loss named is *binary* cross-entropy; the label handling is unspecified.
3. A citation `[33]` is used for CIC-IDS-2017 in §12.8.1.1 but the reference list ends at `[32]`; `[31]` (Tang et al.) is attached to the dataset in §12.8.2 instead. The dataset provenance is therefore unclear.
4. §12.7 contains text about "emotional nuances" and "sentiments" that is unrelated to the topic (a drafting artefact).
5. No cryptographic mechanism is implemented or measured; no attack is evaluated; no overhead is reported (this matches the Jira "Business Problem").

## 3. Research gap analysis (statuses per the master prompt)

| Component | Existing implementation (verified) | Required implementation | Status |
|---|---|---|---|
| Federated Learning | Chapter pseudocode only; no code | Validate FL pipeline | **NOT_IMPLEMENTED** (DOCUMENTED_ONLY in chapter) |
| Flower / TFF in use | Neither; no project exists | Flower-compatible pipeline | **NOT_IMPLEMENTED** |
| Simulation backend (Ray etc.) | None | Controlled simulation | **NOT_IMPLEMENTED** |
| Homomorphic Encryption | Mentioned as one pseudocode call | Implement + benchmark | **DOCUMENTED_ONLY** |
| Secure Aggregation | Mentioned, no protocol | Implement + benchmark | **DOCUMENTED_ONLY** |
| SMPC | Mentioned, no protocol | Implement + benchmark | **DOCUMENTED_ONLY** |
| Differential Privacy | Named once as option | Implement if feasible | **DOCUMENTED_ONLY** |
| Authentication / TLS / IDPS modules | Pseudocode (§12.7) | Document as architecture; implement only what is in sprint scope | **DOCUMENTED_ONLY** |
| Performance monitoring | None | Implement | **NOT_IMPLEMENTED** |
| Security attacks (4 families) | None | Implement | **NOT_IMPLEMENTED** |
| FL algorithms (FedAvg/FedProx/SCAFFOLD/FedNova) | None | Implement + benchmark | **NOT_IMPLEMENTED** |
| Statistical analysis | None (single point estimates) | Implement | **NOT_IMPLEMENTED** |
| Reproducibility (seeds, configs, env) | None | Implement | **NOT_IMPLEMENTED** |
| Scientific documentation | Chapter only | Complete | **DOCUMENTED_ONLY** |

Because nothing exists, this sprint **builds** the research prototype rather than extending it. Statuses in the tracker will only move beyond these after code, tests and executed notebooks exist.

## 4. Consequences for the experimental design (decisions taken now)

| Decision | Reason |
|---|---|
| Use a **synthetic intrusion-detection-style tabular dataset** (`SYN-IDS-v1`, generated deterministically) | Real datasets unavailable; also keeps offensive experiments free of real patient data as required. Results are **not** comparable with the chapter's CIC-IDS-2017 numbers and are never presented as such. |
| Build an **in-process FL simulator** on PyTorch for FedAvg/FedProx/SCAFFOLD/FedNova | Flower's simulation needs Ray + a long runtime; its own aggregation strategies for SCAFFOLD/FedNova are not in the core package. Flower's SecAgg+ primitives are inspected, but a full Flower deployment is not claimed. |
| Implement Secure Aggregation (Bonawitz-style) and an SMPC protocol **from first principles** with audited primitives (`cryptography`: X25519, HKDF, ChaCha20-Poly1305, SHAKE-256) | No suitable ready end-to-end implementation is reachable offline. Independence from simulation shortcuts: real key agreement, real secret sharing, real authenticated encryption. |
| **MP-SPDZ is blocked** (GitHub release download returns 403; building needs a toolchain + source download) | Documented as a blocker; an SPDZ-style MAC-authenticated additive-sharing protocol with Beaver multiplication is implemented instead and labelled as such. |
| "Clients" are **virtual clients in one process** | Never presented as independent hospitals. 500 clients is simulated on 2 CPUs. |
| No GPU | GPU metrics recorded as `unavailable`, not estimated. |

## 5. Reference-verification limitation

Direct fetches of arXiv/Crossref/DOI pages were refused by the session allow-list and shell. Existence of each listed paper was confirmed through search-result titles and publisher hosts only. Per-reference status (`EXISTENCE_VERIFIED_VIA_SEARCH` / `REFERENCE_NOT_VERIFIED`) is recorded in `references/references.json`. DOIs not seen in a retrieved page are left blank rather than guessed.
