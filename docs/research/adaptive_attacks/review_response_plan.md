# Review response plan (two reviewer reports, both "Major Revision")

Status: plan and proposed wording only. No manuscript edit has been made for this document. Every claim-bearing change below waits for the author's approval. Written 2026-11-17 against manuscript sha256 9854b286f8f5565fc7bd4712fd5503179c39f470ed37f82ef82104a214d7f556.

## 1. A point the reviewers did not raise, and it is stronger than their novelty comment

"AGR-tailored" is not our term. Shejwalkar and Houmansadr (NDSS 2021, already cited as `shejwalkar2021`) define AGR-tailored adversaries as those who know the server's aggregation rule and tailor the attack objective to it, and AGR-agnostic ones as those who do not; their Min-Max and Min-Sum are the agnostic attacks (source: the paper's Section IV, read at https://par.nsf.gov/servlets/purl/10286354). Our manuscript says "*AGR-tailored* (new here)" in Section IV-D and lists "a rule-aware attacker" as contribution 1. The idea of a rule-aware attacker is therefore not new, and the name is borrowed. What is ours is narrower: a gradient-free, candidate-search version (simulate the server rule on 30 scaled-direction candidates plus ALIE, MIN-MAX, MIN-SUM and FANG, submit the one that does most damage), applied to twelve rules including four re-implemented defences and a committee, with a draw-aware versus pool-aware split. **Proposal P1 (recommended, needed regardless of the reviewers):** drop "new here", say the term and the threat model come from Shejwalkar and Houmansadr, and describe ours as a discrete candidate-search instance of it. This also answers reviewer Major Comment 2 without any new experiment.

## 2. Comment-by-comment triage

Legend: V = valid, P = partly valid, A = already in the paper. Effort: T = text only, D = documentation, X = new experiment (none is mandatory).

| Reviewer item | Verdict | What the manuscript has now | Proposed action | Effort | Touches a claim |
|---|---|---|---|---|---|
| Major 1: healthcare framing stronger than evidence (both reports) | V | Intro, Limitations and Ethics already say no EHR data, no hospital; title and keywords still say healthcare | P2: one sentence in the abstract and the Discussion that conclusions apply to simulation-based FL benchmarking; keep the title (see minor 1) | T | yes, narrows |
| Major 2: AGR-tailored novelty | V, understated | "new here" | P1 above, plus a short paragraph comparing with ALIE, FANG and Shejwalkar-Houmansadr (what each knows, what each optimises, what ours searches) | T | yes, narrows |
| Major 3: systems results from one process, two cores | P | Limitations and Section VI-A already caveat; the cost section already gives the "budget met once training costs 1.1x to 4.8x" argument | P3: one paragraph relating the numbers to a deployment (what transfers: ciphertext and share expansion, quadratic masking; what does not: absolute times, network). The TCP/mTLS federation exists in the code but was only validated on one machine; a measured run with real sockets is an optional X, not run | T (X optional) | no |
| Major 4: partial re-implementations | V | prose in Section IV-H | P4: a small table, one row per defence: published component, our version, expected consequence. Consequence column must say "untested" except where measured (FoolsGold feature weighting: iid ASR jitter effect +0.293 all-feature versus +0.009 with weighting, from the E3 validation, which the manuscript does not report yet; the defect and its marked cells) | D | yes, adds scope |
| Major 5: uneven statistical evidence | P | Section IV-K (Statistics) and the committee section already call themselves exploratory; 3-seed and 4-site results are labelled descriptive in places | P5: a single table or paragraph classifying every headline result as inferential (10 seeds, paired tests) or descriptive (3 seeds, 4 sites, 5-seed committee) | T | no |
| Major 6: federated preprocessing, scalability, malicious statistics, related work | P | misreporting by a client is mentioned in one sentence in Section IV-F; no scalability or literature positioning | P6: a paragraph on cost (two rounds, per-feature payload) and on the misreporting risk; related work positioning needs real references, which I will not invent: the author must supply or approve them | T plus references | no |
| Minor 1: title journalistic | P | "What Secure Federated Learning Costs and What It Misses: ..." | optional neutral alternative: "Costs and Gaps of Secure Federated Learning: A Simulation Study on Healthcare-Motivated Tabular Data" (author's choice) | T | no |
| Minor 2: abstract separates primary and exploratory | V | the abstract already marks the committee as exploratory | P7: one clause marking the 3-seed and committee results as descriptive | T | no |
| Minor 3: hypotheses | P | four research questions | P8: state them as expectations; do not call them pre-registered, because they were not | T | no |
| Minor 4, 9: too many experiments, move to appendix | V | single body | optional: move the sensitivity grids and Tables for model baselines to a supplement; layout change, needs the author | D | no |
| Minor 5: effect sizes next to tables | P | d_z and Cliff's delta are in the stored test files and in text where tests are discussed | add one sentence per table family pointing to the effect-size file, or a column where room allows | T | no |
| Minor 6: regulatory text too long | P | one paragraph in Background plus the Ethics section | optional shortening; keep the "pointers only, no compliance claim" sentence | T | no |
| Minor 7: crowded figures | V | 6 figures | author review of the figures; I have not re-drawn any | D | no |
| Minor 8: Q1-Q4 summary table | V | none | P9: a four-row table, question, answer, evidence strength | T | no |
| Minor 10: references formatted and verified | A/P | 47 of 54 entries checked, 2 flagged as unverifiable in the archive | list the flagged entries and resolve or drop them before submission | D | no |

## 3. Notes on the reports themselves
- Both reports are generic in places. Their section references (for example IV-D, IV-H, VI-A, VIII) do exist in the manuscript, but they were not checked one by one against the content cited; check each against the current PDF before replying to a real editor.
- The first report has a blank item ("Relevant visual examples include:") and the second gives acceptance probabilities (40-55 %, 70-85 %, 85-90 %) with no basis; do not quote them.
- Both reports credit the FoolsGold defect disclosure as honest reporting; that is accurate as of the current manuscript.
- Neither report mentions the FoolsGold-specific limits that remain: the AGR-tailored FoolsGold cell is withdrawn, and Table XXI's FoolsGold row was judged unaffected, not measured.

## 4. Draft wording (proposals, not applied)

**P1, Section IV-D replacement for "(new here)"** (keep the rest of the paragraph): "*AGR-tailored* adversaries, in the terminology of Shejwalkar and Houmansadr~\cite{shejwalkar2021}, know the server's aggregation rule and tailor the attack to it. Our instance is a discrete search: ..."

**P1, contribution 1, first sentence:** "A rule-aware attacker. Following the AGR-tailored threat model of Shejwalkar and Houmansadr~\cite{shejwalkar2021}, our attacker simulates the server's aggregation rule before it submits; ours is a gradient-free search over a fixed candidate set."

**P2, abstract (added to the last sentence):** "...we make no claim of regulatory compliance, and the conclusions concern simulation-based FL benchmarking, not clinical deployment."

**P5, caption or paragraph:** "Inferential results (10 seeds, paired tests with Holm correction): the cost of confidentiality, the twelve-rule comparison. Descriptive results: backdoor, Byzantine and membership-inference experiments (3 seeds), the four natural sites, and the rule committee (5 seeds)."

## 5. Suggested order
1. Approve P1 and P2 (narrowing, no new evidence needed).
2. P5, P7, P8, P9 (text only).
3. P4 and P3 (a table and a paragraph).
4. P6 (needs references from the author).
5. Optional: title, appendix move, figure review, the real-socket cost run.

## 6. Constraints that still hold
No run, no code change, no numbers changed. Nothing here changes the correction track.
