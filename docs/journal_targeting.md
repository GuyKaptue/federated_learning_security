# Stage 4 - Journal targeting (checked 2026-11-13)

Purpose: choose a venue for the manuscript in `latex/manuscript_ieee.pdf` given what it actually is: a reproducible **measurement study** of federated-learning security (confidentiality mechanisms, adaptive attackers, twelve defences, preprocessing, a benchmark protocol) on synthetic and small public data, with a **negative** Stage-3 result (a rule committee that did not beat its best member). No credentialed clinical data were used.

## How facts were verified
Each fact below is tagged **[official]** (read on the publisher's or journal's own page today), **[secondary]** (read on a third-party copy or aggregator; treat as provisional) or **[not found]**. Nothing tagged otherwise is asserted. Fees change; re-check on the submission page before submitting. Rankings and "fit" statements are my judgement, not facts.

## Verified facts per candidate
| Venue | Scope / policy facts | Fees | Review facts | Source tags |
|---|---|---|---|---|
| **TMLR** (Transactions on Machine Learning Research) | First criterion: claims supported by accurate and convincing evidence; novelty/"significance" explicitly not a criterion; work is not rejected for lacking state-of-the-art results; reproducibility-style studies that systematically examine a published method can qualify. | **No fees to authors** [official] | Double-blind; action editor within a week; at least three reviewers; no decision deadline stated; no page limit or template stated; code optional; arXiv preprints allowed; **no expanded conference papers, no parallel submission** | [official] jmlr.org/tmlr acceptance-criteria and editorial-policies |
| **IEEE Access** | Multidisciplinary, open access; applications-oriented work; "negative results" within scope for applied engineering | Page names no amount; $2,160 reported by third-party sites **[secondary]** | Binary accept/reject; "rapid peer review"; submission-to-publication 4-6 weeks (journal's claim); no first-decision time stated | [official] ieeeaccess.ieee.org/about; fee [secondary] |
| **Journal of Cybersecurity** (OUP) | Full research manuscripts up to 15,000 words; abstract max 350 words; fully open access | A charge applies; **amount not found** on the pages read (the "3625 USD" on the instructions page sits next to "Book review charge", so I do not use it); LMIC waiver/discount exists | Single-anonymised; ~2 reviewers; editors may desk-reject; "fast first decisions", no time stated; code/data strongly encouraged, not required | [official] academic.oup.com/cybersecurity |
| **IEEE TDSC** | Page limit 12 pages | OA fee $2,800 (a $2,240 consortium rate also shown) and overlength $220/page, both from a **library-hosted copy of the IEEE price list dated 12/30/2025 [secondary]**; IEEE's own page confirms overlength charges are mandatory, 5%/20% member discounts apply to OA fees only, and a low-income-country program exists for fully OA journals only | not found | mixed |
| **IEEE TIFS** | Scope covers information forensics, security, biometrics, surveillance; encourages publishing code/data; ScholarOne submission | Same list: OA $2,800, overlength $220/page, page limit shown ambiguously as "10 pages and 2 pages" **[secondary]** | not found | [official] scope; fees [secondary] |
| **Journal of Information Security and Applications** (Elsevier) | Practice-driven security research; **excludes routine applications of off-the-shelf AI/ML models**, purely descriptive reviews and purely theoretical work; impact factor 4.4 as shown | not found | not found | [official] shop.elsevier.com |
| **Computers & Security** (Elsevier) | **Page states a moratorium on submissions with AI/ML as a significant component since early 2024 and that security of AI/ML systems is out of scope**; cryptology excluded | not found | not found | [official] shop.elsevier.com (could not open the ScienceDirect scope page to confirm wording) |
| **ACM DTRAP** | The only page I could read is a 2022 special-issue call, which welcomes measurement studies and asks reviewers to rate reproducibility. **I could not verify the current general scope or fees.** | not found | not found | [official but dated] |

## Update 2026-11-13: author name must appear
The author requires the name "Guy M. Kaptue T." on the manuscript. TMLR is **double-blind** [official], so a named manuscript would not be accepted there. TMLR is therefore dropped as the primary target unless the author later agrees to submit anonymously. No anonymous version is part of the deliverables.

## Recommendation (judgement)
**With the author name on the manuscript, the order is: (1) Journal of Cybersecurity (single-anonymised review, i.e. authors are visible to reviewers [official]; open-access fee to be looked up), (2) IEEE Access (review model for author visibility not checked on its page; confirm in the submission instructions), (3) TDSC/TIFS after more work.** The TMLR reasoning below is kept for reference only.
1. **Primary: TMLR.** The paper's strengths are exactly what TMLR says it judges: evidence for every claim, paired statistics, honest negative results and full traceability. Its weaknesses (no new method, synthetic data) are not rejection grounds there by its own criteria. No fee. Risks: reviewers may regard network-security measurement as marginal to ML; double-blind rules mean the manuscript must be anonymised; nothing may be submitted in parallel elsewhere.
2. **Fallback: IEEE Access** if TMLR rejects or a faster outcome matters. Accepts applied work including negative results; costs about $2,160 if the secondary figure is right (confirm before paying; waivers or member discounts may apply). Lower selectivity, so lower signalling value (judgement).
3. **Alternative: Journal of Cybersecurity** if a security-community, open-access venue is preferred and the author can pay or obtain a waiver; fee must be looked up first.
4. **Stretch only after more work: IEEE TDSC / TIFS.** These venues typically expect a technical contribution. Stage 3 produced a negative result, so acceptance odds are low (judgement); I would only go there with a defence that demonstrably works against the omniscient attackers, or real multi-site data. Budget about $2,800 for OA plus overlength charges if over 12 pages (secondary figures).
5. **Do not target: Computers & Security** (stated AI/ML moratorium); **JISA** only if an editor confirms the topic is not treated as "routine application of off-the-shelf ML".

## Submission checklist (generic; the TMLR-specific anonymisation items no longer apply)
- [ ] Keep the author name "Guy M. Kaptue T." on the manuscript; confirm the chosen venue's review model allows named authors.
- [ ] Confirm no part has been published or is under review elsewhere (preprint on arXiv is allowed).
- [ ] State novelty honestly in the abstract and cover letter: measurement study, not a new principle (randomised/hybrid aggregation exists: Nabavirazavi et al. 2023; Yue et al. 2024).
- [ ] Keep the statements that no credentialed clinical repository was available and that all clients are simulated.
- [ ] Re-run the traceability check and attach the artefact (ZIP, SHA-256); consider a DOI-bearing archive after anonymous review.
- [ ] Re-check stored ASR intervals: they are wide; do not describe ordering claims as established.
- [ ] Decide the format (TMLR names no page limit or template on the page read; its LaTeX style is on the site, not checked here).
- [ ] Note the manuscript is IEEEtran two-column, 14 pages: fine for TMLR, over the TDSC limit if sent there.
- [ ] Check each venue's current policy on disclosure of AI-writing assistance (not verified here).

## Known weaknesses a reviewer will raise (unchanged by Stage 4)
Synthetic main benchmark and small public tables; omniscient upper-bound attackers; reduced Stage-3 scale (5 seeds); no WAN/real-framework run; secure-and-robust aggregation postponed; headline heart results still use pooled-median preprocessing; some older tables lack paired effect sizes; empty `causal_inference` scaffold.
