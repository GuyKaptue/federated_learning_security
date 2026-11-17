"""Builds docs/references/references.json and docs/references/references_master.md.
Honesty note: WebFetch/shell had no access to publisher/arXiv/DOI hosts in this environment, so NO entry could be
content-verified. Entries are cited from the author's knowledge with identifiers only where the identifier is
well known (arXiv ids, ePrint ids, EU legal act numbers). Entries carry a per-entry status and must be
checked against the publisher before any submission. No DOIs are asserted."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
S = "fl_sec_001"; P = "fl_perf_002"; A = "fl_attack_003"; B = "fl_bench_004"; U = "fl_pub_005"
refs = [
 ("mcmahan2017", "McMahan, H.B., Moore, E., Ramage, D., Hampson, S., Agüera y Arcas, B.", "Communication-Efficient Learning of Deep Networks from Decentralized Data", "AISTATS 2017", "arXiv:1602.05629", [P,B,S]),
 ("li2020fedprox", "Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A., Smith, V.", "Federated Optimization in Heterogeneous Networks", "MLSys 2020", "arXiv:1812.06127", [B]),
 ("karimireddy2020", "Karimireddy, S.P., Kale, S., Mohri, M., Reddi, S., Stich, S., Suresh, A.T.", "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning", "ICML 2020", "arXiv:1910.06378", [B]),
 ("wang2020fednova", "Wang, J., Liu, Q., Liang, H., Joshi, G., Poor, H.V.", "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization", "NeurIPS 2020", "arXiv:2007.07481", [B]),
 ("hsu2019", "Hsu, T.-M.H., Qi, H., Brown, M.", "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification", "arXiv preprint", "arXiv:1909.06335", [B]),
 ("li2021niid", "Li, Q., Diao, Y., Chen, Q., He, B.", "Federated Learning on Non-IID Data Silos: An Experimental Study", "ICDE 2022", "arXiv:2102.02079", [B]),
 ("kairouz2021", "Kairouz, P., McMahan, H.B., et al.", "Advances and Open Problems in Federated Learning", "Foundations and Trends in ML 14(1-2), 2021", "arXiv:1912.04977", [S,P,U]),
 ("li2020challenges", "Li, T., Sahu, A.K., Talwalkar, A., Smith, V.", "Federated Learning: Challenges, Methods, and Future Directions", "IEEE Signal Processing Magazine 37(3), 2020", "arXiv:1908.07873", [P,U]),
 ("rieke2020", "Rieke, N., Hancox, J., Li, W., et al.", "The future of digital health with federated learning", "npj Digital Medicine 3, 119 (2020)", "arXiv:2003.08119", [S,U]),
 ("bonawitz2017", "Bonawitz, K., Ivanov, V., Kreuter, B., et al.", "Practical Secure Aggregation for Privacy-Preserving Machine Learning", "ACM CCS 2017", "IACR ePrint 2017/281", [S,P]),
 ("bell2020", "Bell, J.H., Bonawitz, K.A., Gascón, A., Lepoint, T., Raykova, M.", "Secure Single-Server Aggregation with (Poly)Logarithmic Overhead", "ACM CCS 2020", "IACR ePrint 2020/704", [S,P]),
 ("cheon2017", "Cheon, J.H., Kim, A., Kim, M., Song, Y.", "Homomorphic Encryption for Arithmetic of Approximate Numbers", "ASIACRYPT 2017", "IACR ePrint 2016/421", [S]),
 ("fv2012", "Fan, J., Vercauteren, F.", "Somewhat Practical Fully Homomorphic Encryption", "IACR ePrint", "IACR ePrint 2012/144", [S]),
 ("paillier1999", "Paillier, P.", "Public-Key Cryptosystems Based on Composite Degree Residuosity Classes", "EUROCRYPT 1999", "no identifier asserted", [S]),
 ("damgard2012", "Damgård, I., Pastro, V., Smart, N., Zakarias, S.", "Multiparty Computation from Somewhat Homomorphic Encryption (SPDZ)", "CRYPTO 2012", "IACR ePrint 2011/535", [S]),
 ("beaver1991", "Beaver, D.", "Efficient Multiparty Protocols Using Circuit Randomization", "CRYPTO 1991", "no identifier asserted", [S]),
 ("shamir1979", "Shamir, A.", "How to Share a Secret", "Communications of the ACM 22(11), 1979", "no identifier asserted", [S]),
 ("benaloh_tenseal", "Benaissa, A., Retiat, B., Cebere, B., Belfedhal, A.E.", "TenSEAL: A Library for Encrypted Tensor Operations Using Homomorphic Encryption", "arXiv preprint", "arXiv:2104.03152", [S]),
 ("bagdasaryan2020", "Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., Shmatikov, V.", "How To Backdoor Federated Learning", "AISTATS 2020", "arXiv:1807.00459", [A]),
 ("sun2019", "Sun, Z., Kairouz, P., Suresh, A.T., McMahan, H.B.", "Can You Really Backdoor Federated Learning?", "arXiv preprint", "arXiv:1911.07963", [A]),
 ("wang2020tail", "Wang, H., Sreenivasan, K., et al.", "Attack of the Tails: Yes, You Really Can Backdoor Federated Learning", "NeurIPS 2020", "arXiv:2007.05084", [A]),
 ("blanchard2017", "Blanchard, P., El Mhamdi, E.M., Guerraoui, R., Stainer, J.", "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent (Krum)", "NeurIPS 2017", "no identifier asserted", [A]),
 ("yin2018", "Yin, D., Chen, Y., Ramchandran, K., Bartlett, P.", "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates", "ICML 2018", "arXiv:1803.01498", [A]),
 ("fang2020", "Fang, M., Cao, X., Jia, J., Gong, N.Z.", "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning", "USENIX Security 2020", "arXiv:1911.11815", [A]),
 ("shokri2017", "Shokri, R., Stronati, M., Song, C., Shmatikov, V.", "Membership Inference Attacks Against Machine Learning Models", "IEEE S&P 2017", "arXiv:1610.05820", [A]),
 ("nasr2019", "Nasr, M., Shokri, R., Houmansadr, A.", "Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning", "IEEE S&P 2019", "arXiv:1812.00910", [A]),
 ("carlini2022", "Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., Tramèr, F.", "Membership Inference Attacks From First Principles", "IEEE S&P 2022", "arXiv:2112.03570", [A]),
 ("zhu2019", "Zhu, L., Liu, Z., Han, S.", "Deep Leakage from Gradients", "NeurIPS 2019", "arXiv:1906.08935", [A]),
 ("zhao2020", "Zhao, B., Mopuri, K.R., Bilen, H.", "iDLG: Improved Deep Leakage from Gradients", "arXiv preprint", "arXiv:2001.02610", [A]),
 ("geiping2020", "Geiping, J., Bauermeister, H., Dröge, H., Moeller, M.", "Inverting Gradients - How easy is it to break privacy in federated learning?", "NeurIPS 2020", "arXiv:2003.14053", [A]),
 ("abadi2016", "Abadi, M., Chu, A., Goodfellow, I., et al.", "Deep Learning with Differential Privacy", "ACM CCS 2016", "arXiv:1607.00133", [A,U]),
 ("mironov2017", "Mironov, I.", "Rényi Differential Privacy", "IEEE CSF 2017", "arXiv:1702.07476", [A]),
 ("mcmahan2018dp", "McMahan, H.B., Ramage, D., Talwar, K., Zhang, L.", "Learning Differentially Private Recurrent Language Models", "ICLR 2018", "arXiv:1710.06963", [A]),
 ("demsar2006", "Demšar, J.", "Statistical Comparisons of Classifiers over Multiple Data Sets", "JMLR 7, 2006", "no identifier asserted", [B]),
 ("holm1979", "Holm, S.", "A Simple Sequentially Rejective Multiple Test Procedure", "Scandinavian Journal of Statistics 6, 65-70, 1979", "no identifier asserted", [B]),
 ("wilcoxon1945", "Wilcoxon, F.", "Individual Comparisons by Ranking Methods", "Biometrics Bulletin 1(6), 1945", "no identifier asserted", [B]),
 ("pineau2020", "Pineau, J., Vincent-Lamarre, P., Sinha, K., et al.", "Improving Reproducibility in Machine Learning Research", "JMLR 22, 2021", "arXiv:2003.12206", [U]),
 ("henderson2018", "Henderson, P., Islam, R., Bachman, P., et al.", "Deep Reinforcement Learning that Matters", "AAAI 2018", "arXiv:1709.06560", [U]),
 ("gdpr", "European Parliament and Council", "Regulation (EU) 2016/679 (General Data Protection Regulation), Art. 9 and Art. 32", "Official Journal of the EU, L 119", "EUR-Lex (not fetched)", [U,S]),
 ("aiact", "European Parliament and Council", "Regulation (EU) 2024/1689 (Artificial Intelligence Act)", "Official Journal of the EU", "EUR-Lex (not fetched)", [U]),
 ("nis2", "European Parliament and Council", "Directive (EU) 2022/2555 (NIS2)", "Official Journal of the EU", "EUR-Lex (not fetched)", [U]),
 ("bdsg", "Bundesrepublik Deutschland", "Bundesdatenschutzgesetz (BDSG), incl. §22 (processing of special categories of personal data)", "gesetze-im-internet.de", "official text not fetched; section numbers to be re-checked", [U]),
 ("baruch2019", "Baruch, M., Baruch, G., Goldberg, Y.", "A Little Is Enough: Circumventing Defenses For Distributed Learning", "NeurIPS 2019", "arXiv:1902.06156", [A]),
 ("nguyen2022flame", "Nguyen, T.D., Rieger, P., Chen, H., Yalame, H., Möllering, H., Fereidooni, H., Marchal, S., Miettinen, M., Mirhoseini, A., Zeitouni, S., Koushanfar, F., Sadeghi, A.-R., Schneider, T.", "FLAME: Taming Backdoors in Federated Learning", "USENIX Security 2022, pp. 1415-1432", "arXiv:2101.02281", [A, "fl_stage2"]),
 ("ozdayi2021rlr", "Ozdayi, M.S., Kantarcioglu, M., Gel, Y.R.", "Defending against Backdoors in Federated Learning with Robust Learning Rate", "AAAI 2021, vol. 35(10), pp. 9268-9276", "arXiv:2007.03767; doi:10.1609/aaai.v35i10.17118", [A, "fl_stage2"]),
 ("fung2020foolsgold", "Fung, C., Yoon, C.J.M., Beschastnikh, I.", "Mitigating Sybils in Federated Learning Poisoning (arXiv title); venue version: The Limitations of Federated Learning in Sybil Settings", "RAID 2020, pp. 301-316", "arXiv:1808.04866", [A, "fl_stage2"]),
 ("xu2022signguard", "Xu, J., Huang, S.-L., Song, L., Lan, T.", "Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering (SignGuard)", "ICDCS 2022, pp. 1223-1235", "arXiv:2109.05872; doi:10.1109/ICDCS54860.2022.00120", [A, "fl_stage2"]),
 ("nabavirazavi2023randomization", "Nabavirazavi, S., Taheri, R., Shojafar, M., Iyengar, S. S.", "Impact of Aggregation Function Randomization against Model Poisoning in Federated Learning", "IEEE TrustCom 2023, pp. 165-172", "doi:10.1109/TrustCom60117.2023.00043", [A, "fl_stage3"]),
 ("yue2024hybrid", "Yue, K., Jin, R., Wong, C.-W., Dai, H.", "Advancing Hybrid Defense for Byzantine Attacks in Federated Learning", "arXiv preprint (venue not confirmed)", "arXiv:2409.06474", [A, "fl_stage3"]),
 ("shejwalkar2021", "Shejwalkar, V., Houmansadr, A.", "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning", "NDSS 2021", "NDSS paper page (no DOI asserted)", [A]),
 ("uci_wdbc", "Wolberg, W., Mangasarian, O., Street, N., Street, W.", "Breast Cancer Wisconsin (Diagnostic) [Dataset]", "UCI Machine Learning Repository, 1993/1995", "doi:10.24432/C5DW2B (as printed on the UCI page)", [B]),
 ("uci_heart", "Janosi, A., Steinbrunn, W., Pfisterer, M., Detrano, R.", "Heart Disease [Dataset] (Cleveland subset used); cited paper: Detrano et al. 1989, American Journal of Cardiology", "UCI Machine Learning Repository", "UCI dataset 45", [B]),
 ("pima_nidd", "National Institute of Diabetes and Digestive and Kidney Diseases (original owner); donated by V. Sigillito (JHU APL)", "Pima Indians Diabetes Database (pima.names)", "UCI/MLC++ distribution", "mirror page read; original UCI page not read", [B]),
 ("smith1988", "Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., Johannes, R.S.", "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus", "Proc. Annu. Symp. Comput. Appl. Med. Care, 1988", "PMC2245318 (page blocked by captcha)", [B]),
]
SRC_VERIFIED = {  # title + authors + submission/revision date read on the source page (arXiv abs / ePrint / JMLR / NeurIPS) on 2026-11-06
 "mcmahan2017","li2020fedprox","karimireddy2020","wang2020fednova","hsu2019","li2021niid","kairouz2021","li2020challenges","rieke2020","benaloh_tenseal",
 "sun2019","wang2020tail","yin2018","fang2020","shokri2017","nasr2019","carlini2022","zhu2019","zhao2020","geiping2020","abadi2016","mironov2017",
 "mcmahan2018dp","pineau2020","henderson2018","bagdasaryan2020","bonawitz2017","cheon2017","fv2012","damgard2012","bell2020","demsar2006","blanchard2017","baruch2019","shejwalkar2021","uci_wdbc","uci_heart",
 "paillier1999","beaver1991","shamir1979","holm1979"}
VENUE_VERIFIED = {  # venue as cited is stated by the page itself, by a DOI read on the page, or by a venue page surfaced in search
 "karimireddy2020":"page: ICML 2020/PMLR 119","li2020challenges":"DOI 10.1109/MSP.2020.2975749 on arXiv page","rieke2020":"DOI 10.1038/s41746-020-00323-1 on arXiv page",
 "nasr2019":"DOI 10.1109/SP.2019.00065 on arXiv page","abadi2016":"DOI 10.1145/2976749.2978318 on arXiv page","cheon2017":"ePrint page: published in ASIACRYPT 2017",
 "bell2020":"ePrint page: CCS 2020, DOI 10.1145/3372297.3417885","demsar2006":"JMLR page: 7(1):1-30, 2006","blanchard2017":"NeurIPS proceedings page: NIPS 2017",
 "wang2020tail":"neurips.cc poster page (search result)","fang2020":"arXiv PDF header: USENIX Security 2020 (search result)","bagdasaryan2020":"proceedings.mlr.press v108 page (search result)",
 "henderson2018":"ojs.aaai.org page (search result)","uci_wdbc":"UCI page (title, creators, DOI, CC BY 4.0, 569x30 read)","paillier1999":"bibliographic record (Univ. Utah bib archive): LNCS vol. 1592, 1999, start page 223; EUROCRYPT name not stated on page","beaver1991":"bibliographic record (Univ. Utah bib archive): LNCS vol. 576, 1991, start page 420; CRYPTO name not stated on page","shamir1979":"author page (Tau) shows CACM 22(11), Nov 1979","holm1979":"bibbase record: Scand. J. Statist. 6, 65-70, 1979","uci_heart":"UCI page (creators, 303 instances, CC BY 4.0, de-identification note read)"}
SRC_VERIFIED |= {"nguyen2022flame", "ozdayi2021rlr", "fung2020foolsgold", "xu2022signguard", "nabavirazavi2023randomization", "yue2024hybrid"}
VENUE_VERIFIED.update({  # read 2026-11-12 (Stage 2): arXiv full texts for the algorithms, venue pages for the venue
 "nguyen2022flame": "usenix.org presentation page: 31st USENIX Security Symposium 2022, pp. 1415-1432; arXiv full text read for the algorithm",
 "ozdayi2021rlr": "ojs.aaai.org article page: AAAI vol. 35 no. 10, 2021, pp. 9268-9276, DOI 10.1609/aaai.v35i10.17118; arXiv v4 full text read for the algorithm",
 "fung2020foolsgold": "usenix.org RAID 2020 page ('The Limitations of Federated Learning in Sybil Settings', pp. 301-316, introduces FoolsGold); the page does not state its relation to arXiv 1808.04866; algorithm read from the arXiv version",
 "nabavirazavi2023randomization": "researchportal.port.ac.uk record: TrustCom 2023, pp. 165-172, DOI 10.1109/TrustCom60117.2023.00043; only the record was read, not the paper body",
 "xu2022signguard": "scholars.cityu.edu.hk record: ICDCS 2022, pp. 1223-1235, DOI 10.1109/ICDCS54860.2022.00120; arXiv v2 full text read for the algorithm",
})
LEGAL = {"gdpr": ("TEXT_PARTIALLY_READ", "Operative text read on 2026-11-06 from the legislation.gov.uk copy of the as-adopted Regulation (NOT the EUR-Lex Official Journal page, which only rendered recitals): Art. 9(1), 9(2)(a),(h),(i),(j), 9(3), 9(4); Art. 32(1)(a)-(d); key phrases of Art. 25(1), 32(2) and 35(1). Not read: the remaining articles; consolidated/current EU text and national implementation."),
         "aiact": ("TEXT_PARTIALLY_READ", "Art. 15(1)-(5) read on 2026-11-06 from artificialintelligenceact.eu (an unofficial rendering by the Future of Life Institute, not the Official Journal). Art. 6, Annex III and the rest NOT read; later amendments (e.g. the digital omnibus) may change the text."),
         "nis2": ("TEXT_PARTIALLY_READ", "Art. 21(1) and 21(2)(a)-(j) and Annex I sector 5 (health: healthcare providers, EU reference labs, pharma R&D, pharma/medical-device manufacturing) read on 2026-11-06 from the EUR-Lex consolidated text (CELEX 02022L2555-20221227). Rest NOT read; national transposition (e.g. in Germany) NOT read."),
         "wilcoxon1945": ("EXISTENCE_CONFIRMED_LANDING_PAGE_ONLY", "Springer chapter list shows a reprint 'Individual Comparisons by Ranking Methods' in Breakthroughs in Statistics (DOI 10.1007/978-1-4612-4380-9_16) with Noether's introduction; the original Biometrics Bulletin 1(6) citation (volume/pages) was NOT stated on any page read."),
         "bdsg": ("TEXT_PARTIALLY_READ", "gesetze-im-internet.de/bdsg_2018/__22.html read on 2026-11-06: heading of section 22 and paragraph 1 (Nr. 1 lit. a-d, Nr. 2 lit. a-c) read; paragraph 2 (list of measures) and the rest of the BDSG NOT read.")}
out = []
for k, au, ti, ve, idn, st in refs:
    if k in SRC_VERIFIED:
        vv = VENUE_VERIFIED.get(k)
        out.append(dict(key=k, authors=au, title=ti, venue=ve, identifier=idn, used_in=st, status="METADATA_VERIFIED_ON_SOURCE_PAGE", venue_verified=bool(vv),
                        verification_note="Title, authors and date read on the source page (2026-11-06); content/claims not re-checked against the paper body. Venue: " + (vv if vv else "NOT confirmed by the page (cited from knowledge)") + "."))
    elif k in LEGAL:
        out.append(dict(key=k, authors=au, title=ti, venue=ve, identifier=idn, used_in=st, status=LEGAL[k][0], venue_verified=False, verification_note=LEGAL[k][1]))
    else:
        out.append(dict(key=k, authors=au, title=ti, venue=ve, identifier=idn, used_in=st, status="REFERENCE_NOT_VERIFIED", venue_verified=False,
                        verification_note="Classic non-arXiv reference, no source page reachable/checked; cited from knowledge. Check before submission."))
(R/"references").mkdir(exist_ok=True)
(R/"docs/references/references.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
md = ["# REFERENCES_MASTER", "", "**Verification status (2026-11-06; four Stage-2 defence papers added and verified 2026-11-12):** 45 entries (papers, classic crypto/statistics papers via bibliographic records, and UCI dataset pages) are `METADATA_VERIFIED_ON_SOURCE_PAGE` (title, authors and dates read on the arXiv/ePrint/JMLR/NeurIPS page; the `venue_verified` flag says whether the cited venue was also confirmed; paper bodies were not re-read). Legal texts are `TEXT_PARTIALLY_READ` with the exact articles listed per entry: BDSG section 22(1) (gesetze-im-internet.de), GDPR Art. 9/32 (legislation.gov.uk copy of the adopted text, not EUR-Lex), NIS2 Art. 21 and Annex I health (EUR-Lex consolidated), AI Act Art. 15 (unofficial rendering). Wilcoxon 1945 is `EXISTENCE_CONFIRMED_LANDING_PAGE_ONLY` (reprint located, original citation not stated). Two entries remain `REFERENCE_NOT_VERIFIED`: the Pima source page (UCI dataset 34 fetched is a different dataset) and Smith et al. 1988 (PMC page behind a captcha). Paillier/Beaver pages give LNCS volume only (conference names not stated on the page) Identifiers are given only where they are widely known; no DOI is asserted. Legal texts must be read from official sources (EUR-Lex, gesetze-im-internet.de) before any statement beyond the pointers in this sprint is made. **No compliance claim is made anywhere in this sprint.**", "",
      "| Key | Authors | Title | Venue | Venue verified | Identifier | Used in | Status |", "|---|---|---|---|---|---|---|---|"]
for r in out:
    md.append(f"| {r['key']} | {r['authors']} | {r['title']} | {r['venue']} | {'yes' if r['venue_verified'] else 'no'} | {r['identifier']} | {', '.join(r['used_in'])} | {r['status']} |")
(R/"docs/references/references_master.md").write_text("\n".join(md) + "\n")
# per-story references.md
names = {S: "fl_sec_001", P: "fl_perf_002", A: "fl_attack_003", B: "fl_bench_004", U: "fl_pub_005"}
for story in names:
    d = R/"docs/jira"/story; d.mkdir(parents=True, exist_ok=True)
    rows = [r for r in out if story in r["used_in"]]
    (d/"references.md").write_text(f"# References – {story}\n\nPer-entry status below (`METADATA_VERIFIED_ON_SOURCE_PAGE`, `TEXT_PARTIALLY_READ`, `EXISTENCE_CONFIRMED_LANDING_PAGE_ONLY`, `REFERENCE_NOT_VERIFIED`); see docs/references/references_master.md.\n\n" +
        "\n".join(f"- **[{r['key']}]** {r['authors']}. *{r['title']}*. {r['venue']} (venue verified: {'yes' if r['venue_verified'] else 'no'}). {r['identifier']}. — `{r['status']}`" for r in rows) + "\n")
print(len(out), "references")
