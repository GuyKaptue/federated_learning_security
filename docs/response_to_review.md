# Response to the peer review (2026-11-13)

Verdict received: major revision. Status of each point, honestly stated.

## Done in the manuscript
| Review point | What changed |
|---|---|
| Overhead claim inconsistent (abstract vs VI-A) | Reconciled using stored times. Communication overhead exceeds the 40% budget for every mechanism (CKKS 5913%, secure aggregation 141% uplink); for CKKS/BFV/SPDZ this is expansion that grows with the model, for secure aggregation part is a fixed key-exchange cost that was not measured at larger models. Training-time overhead (32%-145%) is measured against a very cheap in-process baseline (about 1 s per run); if the added time stayed fixed the 30% budget would be met once training cost 1.1x (secure aggregation) to 4.8x (SPDZ) more. The text now calls training time a lower bound and communication the firmer result. |
| "do nothing against a backdoor" reads as a failed mechanism | Reworded to "no integrity protection"; threat model now states that confidentiality and integrity are measured separately by design. |
| Abstract too long and a table of contents | Rewritten around the findings and shortened (still about 200 words, not half). |
| Omniscient qualifier missing | Added in abstract, discussion and conclusion. |
| Committee: 5 seeds, point estimates before caveats | Section titled exploratory; opens with the null result and the seed counts; pool-vs-draw sentence now carries the overlap caveat. Seeds were not raised (see below). |
| Federated preprocessing caveat | Abstract says "with the same imputation" and that headline results still use pooled preprocessing; results text states imputation is matched across compared arms (federated equals pooled-mean exactly; the median-to-mean change alone moves F1 by -0.006). |
| Question 3 (matched imputation) | Already in Table `t:fp`: the +0.114 / -0.076 F1 contrasts are local vs pooled-mean, both matched. Clarified in the text. |
| Switzerland AUC (8 negative cases) | Flagged where reported in Results. |
| LOSO inference | Stated in Results (cannot support inference about new sites), plus protocol-B qualification of "hard to tell apart". |
| "8 of 35" misread risk | Rephrased: Holm is applied within scenario, so it is a count of tests, not a family-wise result. |
| Random search single draw | Limitations states the variance of the draw was not estimated, so grid-vs-random swings (e.g. SCAFFOLD) are not attributable to search strategy. |
| "stated convention" vague | Restated: non-finite weights score as constant class-0 prediction, AUC 0.5. |
| SYN-IDS-v1 name | Paper now calls it SYN-TAB-40 and says the code identifier is SYN-IDS-v1. The code and data folders were NOT renamed (golden hashes and docs depend on it). |
| Contributions order; secure-aggregation float sentence; contributions item 1 | Reordered (rule-aware attacker, preprocessing first); sentence moved to Q1; item 1 rewritten. |

## Not done (and why)
* **10 seeds for the committee comparison.** Not rerun; labelled exploratory instead (the reviewer's alternative). The cost is hours of compute.
* **Move overlapping tables to an appendix; restructure sections** so the attacker and preprocessing are the centrepieces. Only the contribution order and abstract were changed.
* **Candidate-set sensitivity of the AGR attacker (Q6)**, including a loss-aware candidate. This is the first experiment of the new adaptive-attack work package.
* **Variance of the random-search draw (Q4).** Stated as a limitation, not estimated.
* **Broader non-IID benchmarking citations.** No references were added because none were verified.
* **Abstract at half length.**
