"""Builds latex/manuscript_ieee.tex (full IEEEtran conference-style paper) from stored results. Every number in tables comes from result files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manuscript_stage1 as S1
import manuscript_stage2 as S2
import manuscript_stage3 as S3
from manuscript_lib import *

cite = lambda *k: "\\cite{" + ",".join(k) + "}"
refs = json.load(open(R / "docs/references/references.json")); nref = len(refs)
nver = sum(r["status"] == "METADATA_VERIFIED_ON_SOURCE_PAGE" for r in refs); nunv = sum(r["status"] == "REFERENCE_NOT_VERIFIED" for r in refs)

# ---------------------------------------------------------------- measured inputs
s15 = rd("fl_sec_001", "task_1_5_comparison_summary.csv")
kpi = rd("fl_perf_002", "task_2_5_kpi_check.csv")
bd = rd("fl_attack_003", "task_3_1_summary.csv")
mia = rd("fl_attack_003", "task_3_3_summary.csv")
lb = rd("fl_bench_004", "task_4_7_leaderboard_dir0.3.csv")
tun = rd("fl_bench_004", "task_4_8_summary.csv")
real = rd("fl_bench_004", "task_4_9_real_summary.csv")
rank9 = rd("fl_bench_004", "task_4_9_real_ranking.csv"); rank9 = rank9[rank9.friedman_p.notna()]
ab = rd("fl_pub_005", "task_5_2_summary.csv")
repro = js("fl_pub_005", "task_5_1_reproducibility.json")["checks"]; nrep = sum(c["passed"] for c in repro)
sig = rd("fl_bench_004", "task_4_6_pairwise_tests.csv"); nsig = int(sig["significant_holm_0.05"].sum())
adb = rd("fl_attack_003", "task_3_7_backdoor_summary.csv"); adu = rd("fl_attack_003", "task_3_7_untargeted_summary.csv")
A10 = rd("fl_bench_004", "task_4_10_A_summary.csv"); A10b = rd("fl_bench_004", "task_4_10_A_baselines_summary.csv"); A10s = rd("fl_bench_004", "task_4_10_A_site_summary.csv")
B10 = rd("fl_bench_004", "task_4_10_B_summary.csv"); B10m = rd("fl_bench_004", "task_4_10_B_macro.csv"); B10b = rd("fl_bench_004", "task_4_10_B_baselines_summary.csv")
sec10 = rd("fl_bench_004", "task_4_10_secagg_equivalence.csv"); n10 = js("fl_bench_004", "task_4_10_natural_sites.json")
rs = rd("fl_bench_004", "task_4_11_summary.csv"); rst = rd("fl_bench_004", "task_4_11_tests.csv"); rsj = js("fl_bench_004", "task_4_11_random_search.json")
wc = rd("fl_attack_003", "task_3_8_worst_case.csv"); su8 = rd("fl_attack_003", "task_3_8_untargeted_summary.csv"); sb8 = rd("fl_attack_003", "task_3_8_backdoor_summary.csv")
run8 = rd("fl_attack_003", "task_3_8_untargeted_runs.csv"); ndiv = int(run8.diverged.sum())
cmp8 = rd("fl_attack_003", "task_3_8_agr_vs_best_fixed.csv")
import glob as _glob


def nseed(story, fname, col="seed"):
    """Number of distinct seeds actually present in a stored runs file (captions are derived from this, never typed by hand)."""
    return int(pd.read_csv(O / story / fname)[col].nunique())
N_SEC = nseed("fl_sec_001", "task_1_5_comparison_runs.csv"); N_AB = nseed("fl_pub_005", "task_5_2_runs.csv")
_kr = pd.concat([pd.read_csv(f) for f in sorted(_glob.glob(str(O / "fl_perf_002" / "fl_perf_002.*_runs.csv")))])
_kr = _kr[_kr.get("status", "ok") == "ok"] if "status" in _kr else _kr
_ks = _kr.groupby(["config", "n_clients"]).seed.nunique().rename("seeds").reset_index()
def ci_cell(df, col, nd=0):
    return [f"{float(m):.{nd}f} {c}" for m, c in zip(df[col], df[col + "_ci"])]
AL = ("fedavg", "fedprox", "scaffold", "fednova"); AN = dict(fedavg="FedAvg", fedprox="FedProx", scaffold="SCAFFOLD", fednova="FedNova")
get = lambda df, **kw: df.query(" and ".join(f"{k}=={v!r}" for k, v in kw.items()))
pp = lambda x: f"{100 * x:.1f}"

# ---------------------------------------------------------------- derived narrative facts
best_ovh = s15[s15.config != "plaintext"].sort_values("train_overhead_pct").iloc[0]
ckks = s15[s15.config == "he_ckks"].iloc[0]; sa = s15[s15.config == "secagg"].iloc[0]; sm = s15[s15.config == "smpc_3party"].iloc[0]
bdh = bd[bd.level == "high"].iloc[0]; bdc = bd[bd.level.str.startswith("none")].iloc[0]
mean_bd = get(adb, defense="mean", attack="static_boost").iloc[0]
# adaptive backdoor: defences that stopped the static attack (<=0.30) but fail (>=0.50) against some adaptive variant
stopped = []
for d in sorted(adb.defense.unique()):
    st = get(adb, defense=d, attack="static_boost").iloc[0]["asr"]
    ad = adb[(adb.defense == d) & adb.attack.isin(["norm_matched", "minmax_backdoor"])]["asr"].max()
    if st <= 0.30 and ad >= 0.50: stopped.append((d, st, ad))
unt_min = adu[adu.dataset == "syn"].groupby("attack")["acc"].min()
# 4.10 facts
Af1 = {a: get(A10, algo=a, setting="tuned").iloc[0] for a in AL}; Ad = {a: get(A10, algo=a, setting="default").iloc[0] for a in AL}
cen = A10b[A10b.method.str.startswith("central")].iloc[0]; loc = A10b[A10b.method.str.startswith("local")].iloc[0]
Bm = {(r.algo, r.setting): r for r in B10m.itertuples()}
secmax = float(sec10["max_abs_param_diff_secagg_vs_plain"].max())
fr_A = n10["A"]["friedman_tuned"]; fr_B = n10["B"]["friedman_site_means_auc"]
# 4.11 facts
def rsv(ds, a, t, m="f1"): return float(get(rs, dataset=ds, algo=a, setting=t).iloc[0][m])
gain_rg = {(ds, a): rsv(ds, a, "random") - rsv(ds, a, "grid") for ds in ("synthetic", "heart-4sites") for a in AL}
sigt = rst[(rst.contrast == "random - grid") & (rst.wilcoxon_p < 0.05)] if "wilcoxon_p" in rst else rst.iloc[0:0]
# 3.8 facts
agr_rows = wc.copy(); agr_stronger = cmp8[(cmp8.get("mean_diff", 0) < 0)] if len(cmp8) else cmp8
def accv(ds, d, a): return float(get(su8, dataset=ds, defense=d, attack=a).iloc[0]["acc"])
def asrv(d, a): return float(get(sb8, defense=d, attack=a).iloc[0]["asr"])
DEF = ["mean", "median", "trimmed_mean", "krum", "multi_krum", "clip_median_norm", "anomaly_filter"]
worst_syn = get(wc, dataset="syn").sort_values("drop", ascending=False)
bd_agr = {d: asrv(d, "agr_backdoor") for d in DEF}; bd_mm = {d: asrv(d, "minmax_backdoor") for d in DEF}


DATATABLE = table(pd.DataFrame([
    ["Synthetic (SYN-TAB-40)", "generated", 12000, 40, "0.40", "scikit-learn make_classification", "main experiments"],
    ["Breast cancer (WDBC)", "public, UCI", 569, 30, "0.63", "bundled with scikit-learn", "synthetic splits"],
    ["Pima diabetes", "public", 768, 8, "0.35", "GitHub mirror, SHA-256 pinned", "synthetic splits"],
    ["Heart disease (Cleveland)", "public, UCI", 302, 13, "0.54", "GitHub mirror, SHA-256 pinned", "synthetic splits"],
    ["Heart disease, 4 sites", "public, UCI", 920, 10, "0.36--0.93", "GitHub mirror, SHA-256 pinned", "natural site = client"]],
    columns=["Data set", "Type", "Rows", "Features", "Positive rate", "Source", "Used as"]),
    "Data sources. All clients are virtual. The positive rate for the four-site data is the range over sites.", "t:data", 0, None, None, True, "lllrrll")

# ---------------------------------------------------------------- text
T = []
T.append(r"""\documentclass[conference]{IEEEtran}
\usepackage{cite,graphicx,booktabs,hyperref,amsmath,amssymb}
\hypersetup{hidelinks}
\begin{document}
\title{What Secure Federated Learning Costs and What It Misses: A Simulation Study on Healthcare-Motivated Tabular Data}
\author{\IEEEauthorblockN{Guy M. Kaptue T.}}
\maketitle""")
T.append(r"""\begin{abstract}
Hiding model updates from a federated-learning (FL) server is not the same as checking them. We measured what that gap costs in simulation, using virtual clients in one process, one synthetic tabular benchmark and four small public medical tables. Pairwise-mask secure aggregation, CKKS/BFV homomorphic encryption and an SPDZ-style three-party sum all left accuracy unchanged and gave no integrity protection against poisoning. Their communication overhead exceeded our 40\% budget in every configuration, while their training-time overhead exceeded the 30\% budget only against a very cheap in-process baseline (if the added time stayed fixed, it would be met once local training cost about 1 to 5 times more).
Against omniscient attackers, robust aggregators that hold up against a static backdoor were often beaten by attackers that simulate the server's rule, which we call AGR-tailored. We also show that global feature means and variances can be computed through the same secure aggregation without pooling rows; on the four natural heart-disease sites this matched pooled standardisation with the same imputation, although the headline site results still use pooled preprocessing. On small real data the four FL algorithms were not reliably separable after tuning. Four re-implemented published defences did not close that gap, and an exploratory rule committee did not beat its best member. Everything here comes from simulation; no patient data, hospital or real network took part, and we make no claim of regulatory compliance.
\end{abstract}
\begin{IEEEkeywords}
federated learning, secure aggregation, homomorphic encryption, secret sharing, poisoning attacks, adaptive attacks, hyper-parameter tuning, calibration, federated preprocessing, healthcare data, reproducibility
\end{IEEEkeywords}""")
T.append(r"""\section{Introduction}
Healthcare data are sensitive and fragmented across institutions, which motivates federated learning~""" + cite("rieke2020", "kairouz2021", "mcmahan2017") + r""". FL is, however, not a security mechanism by itself: model updates can reveal training examples~""" + cite("zhu2019", "geiping2020", "shokri2017") + r""", and a few malicious participants can implant backdoors or degrade the model~""" + cite("bagdasaryan2020", "blanchard2017", "fang2020") + r""". Cryptographic aggregation protects the \emph{confidentiality} of individual updates but not their \emph{integrity}, and robust aggregation protects integrity only inside a threat model that adaptive attackers can leave~""" + cite("baruch2019", "shejwalkar2021") + r""".

This paper is a measurement study, and we built it so that it can be run again. It starts from four questions. What do confidentiality mechanisms cost compared with plain aggregation, and does that cost grow with the number of clients? Which attacks do they stop, together with a few common robust aggregators, and which do they leave open? How much do conclusions about FL algorithms depend on tuning, and on whether the heterogeneity between clients is synthetic or natural? And how much stronger does an attacker get once it knows the defence?

Our contributions, in brief:
\begin{enumerate}
\item \textbf{A rule-aware attacker.} The AGR-tailored attacker simulates the server's aggregation rule before it submits. We compare it with ALIE, MIN-MAX, MIN-SUM and FANG on backdoor and untargeted goals, and use it to put twelve aggregation rules side by side, including four published defences (FLAME, robust learning rate, FoolsGold, SignGuard) that we re-implemented from their papers. For each rule we report rejection rates and the accuracy lost under label skew. The result is a map of where fixed defences fail, not a defence that closes the gap.
\item \textbf{Preprocessing without a trusted centre.} Global feature means and variances, with missing values, from two secure-aggregation rounds. They match the pooled statistics up to fixed-point precision, and we report the communication cost and the effect on accuracy.
\item \textbf{The cost of confidentiality, measured.} CKKS/BFV, Paillier, pairwise-mask secure aggregation and an SPDZ-style three-party sum, all implemented and tested, set against plain aggregation. To remove run-order artefacts we put seeds in the outer loop, randomise the run order, include a warm-up run, and report 10-seed confidence intervals.
\item \textbf{A benchmark protocol that discourages optimistic conclusions.} Nested tuning, grid against random search, natural site splits, calibration, paired effect sizes with Holm-corrected tests, and one-factor grids for malicious fraction, heterogeneity, participation and model size.
\item \textbf{An artefact you can trace.} Every table is generated from stored result files and listed in a table-to-source sheet. There are over 200 automated tests and a real TCP/mutual-TLS federation, which we validated on one machine only.
\end{enumerate}
We also report an exploratory rule committee that draws a few of these rules at random each round, tested against attackers who know the draw or only the pool. Randomised and hybrid aggregation are not new~\cite{nabavirazavi2023randomization,yue2024hybrid}, and here the committee did not beat its best member or the median; we report it as a negative result with five seeds only.
The scope is narrow. All results come from simulations with virtual clients on synthetic or small public data. We had no access to a credentialed clinical repository such as PhysioNet's MIMIC-IV or eICU, so no electronic-health-record data appear anywhere in this paper or its code, and nothing here is evidence about real hospital deployments or legal compliance.
The scope is limited: all results are simulations with virtual clients on synthetic or small public data; we had no access to a credentialed clinical repository (for example PhysioNet's MIMIC-IV or eICU), so no electronic-health-record data are used and none of this is evidence about real hospital deployments or legal compliance.""")
T.append(r"""\section{Background and Related Work}
\textbf{Algorithms.} FedAvg~""" + cite("mcmahan2017") + r""" averages client updates; FedProx~""" + cite("li2020fedprox") + r""" adds a proximal term; SCAFFOLD~""" + cite("karimireddy2020") + r""" uses control variates against client drift; FedNova~""" + cite("wang2020fednova") + r""" normalises by the number of local steps. Benchmarks of non-IID FL~""" + cite("hsu2019", "li2021niid") + r""" show that rankings depend on partitioning and tuning; we therefore tune every algorithm with the same protocol.

\textbf{Confidentiality.} Secure aggregation~""" + cite("bonawitz2017", "bell2020") + r""" hides individual updates behind pairwise masks and secret-shared seeds. Homomorphic encryption~""" + cite("cheon2017", "fv2012", "paillier1999") + r""" lets the server add ciphertexts. SPDZ-style MPC~""" + cite("damgard2012", "beaver1991", "shamir1979") + r""" computes the sum on secret shares with authenticated tags. Differential privacy~""" + cite("abadi2016", "mironov2017", "mcmahan2018dp") + r""" bounds membership inference at a utility cost.

\textbf{Attacks and defences.} Backdoors~""" + cite("bagdasaryan2020", "sun2019", "wang2020tail") + r""", Byzantine updates and defences (median, trimmed mean, Krum)~""" + cite("blanchard2017", "yin2018") + r""", membership inference~""" + cite("shokri2017", "nasr2019", "carlini2022") + r""", and gradient inversion~""" + cite("zhu2019", "zhao2020", "geiping2020") + r""". ALIE~""" + cite("baruch2019") + r""", the optimisation-based attacks of Shejwalkar and Houmansadr~""" + cite("shejwalkar2021") + r""" and the directed-deviation attack of Fang et al.~""" + cite("fang2020") + r""" show that statistically hidden updates defeat distance- and rank-based rules.

\textbf{Reproducibility and statistics.} We follow reporting advice for ML reproducibility~""" + cite("pineau2020", "henderson2018") + r""" and use paired non-parametric tests across seeds with Holm correction~""" + cite("demsar2006", "holm1979", "wilcoxon1945") + r""".

\textbf{Regulatory context (pointers only).} GDPR Art.~32(1)(a) lists pseudonymisation and encryption among possible security measures, Art.~9 restricts processing of health data, NIS2 Art.~21 requires risk-management measures including cryptography policies and lists healthcare providers in Annex~I, and AI Act Art.~15(5) refers to data poisoning and adversarial inputs for high-risk systems~""" + cite("gdpr", "nis2", "aiact") + r""". We read these provisions (see Section~\ref{sec:ethics} for exactly what was and was not read) but make \emph{no} claim that any mechanism here satisfies any legal requirement.""")
T.append(r"""\section{Threat Model}
We want to protect the clients' training data, their model updates, the global model, and the integrity of that model. We consider three adversaries. The first is a semi-honest aggregation server (or an eavesdropper) that sees whatever it is sent. The second is a group of $f$ colluding malicious clients that poison their data or their updates. The third is a stronger version of the second, used for the adaptive attacks: the malicious clients also observe every benign update of the round (they are omniscient) and, for the rule-aware variants, know the server's aggregation rule and may query it offline. This is deliberately an upper bound on a realistic attacker, so that the value of a defence is bounded from below.

Confidentiality and integrity are measured separately on purpose. The confidentiality mechanisms do not try to detect malicious updates, and the robust aggregators do not try to hide updates, so ``secure aggregation does nothing against a backdoor'' is a statement about scope and not about a failed mechanism; combining the two is an open design problem that we do not attempt. We assume an honest key holder for homomorphic encryption and a trusted dealer for the multiplication triples of the SPDZ-style protocol; secure aggregation is semi-honest, without a consistency round, and we do not simulate network links. Malicious servers against secure aggregation, side channels, model extraction, fairness and legal compliance are outside our scope.""")
T.append(r"""\section{Methods}
\subsection{Simulation platform}
All clients are \emph{virtual}: Python/PyTorch processes inside one interpreter on two CPU cores with a small MLP (one hidden layer, ReLU) on flat parameter vectors. Rounds, local epochs, batch size, learning rate and seeds are explicit configuration; every result file carries a provenance record (configuration hash, package versions, seed list). Communication is accounted as logical payload bytes, not network traffic.
\subsection{Confidentiality mechanisms}
CKKS and BFV via TenSEAL, Paillier via \texttt{phe}, pairwise-mask secure aggregation with X25519, HKDF, ChaCha20-Poly1305, Shamir sharing and double masking, and a SPDZ-style three-party sum over $\mathbb{F}_{2^{61}-1}$ with MAC checks and Beaver triples from a trusted dealer. Each is validated by equivalence tests against plaintext sums and by negative tests (tampered shares, below-threshold dropout).
\subsection{Attacks and defences}
Static backdoor (pixel-pattern analogue trigger on tabular features, three strengths), sign-flip and noise Byzantine updates, a loss-threshold membership-inference attack, and gradient inversion (DLG/iDLG). Defences: coordinate median, trimmed mean, Krum / Multi-Krum, norm clipping, an anomaly filter, and client-side DP.
\subsection{Adaptive attacks}
All adaptive attackers replace the malicious rows of the update matrix after seeing the benign rows. \emph{ALIE}: $\mu-z\sigma$ with $z=\Phi^{-1}((n-s)/n)$, $s=\lfloor n/2+1\rfloor-f$~""" + cite("baruch2019") + r""". \emph{MIN-MAX} and \emph{MIN-SUM}: $\mu+\gamma p$ with the largest $\gamma$ such that the maximum (resp. summed squared) distance to benign updates does not exceed the benign spread~""" + cite("shejwalkar2021") + r""". \emph{FANG}: per-coordinate push beyond the benign extreme, opposite to the benign direction~""" + cite("fang2020") + r""". \emph{Norm-matched} and \emph{MIN-MAX backdoor}: genuine backdoor updates rescaled to the benign median norm, or pushed to the distance bound. \emph{AGR-tailored} (new here): the attacker simulates the server rule on 30 scaled-direction candidates plus ALIE, MIN-MAX, MIN-SUM and FANG, and submits the candidate minimising $\langle \mathrm{agg},\hat\mu\rangle$; \emph{AGR-backdoor} does the same along the true backdoor direction. The $\gamma$ grid is unconstrained for the scaled candidates, so this is an upper bound; it searches a fixed candidate set and is not a global optimum.
\subsection{Data}
We use five data sources (Table~\ref{t:data}). The synthetic set (called SYN-TAB-40 here; its identifier in the code and data folders is SYN-IDS-v1, a historical name, and it contains no real intrusion data or any other real data) comes from scikit-learn's \texttt{make\_classification}: 40 features, two classes in a 60/40 ratio and three clusters per class. It carries the cost, scaling, attack and benchmark experiments, because there we control everything. To check that conclusions are not an artefact of synthetic data, we also use three small public medical tables, namely Wisconsin diagnostic breast cancer (bundled with scikit-learn), Pima diabetes (zeros in physiologically impossible columns are treated as missing) and Cleveland heart disease (one duplicate row removed)~""" + cite("uci_wdbc", "uci_heart", "pima_nidd", "smith1988") + r""". On these, virtual clients are created by IID or Dirichlet label-skew partitions, so the heterogeneity is synthetic.

The fifth source gives heterogeneity that was not made by us: the four UCI heart-disease cohorts (Cleveland, Hungary, Switzerland, VA Long Beach; 920 patients), with each site acting as one client. We keep ten features and take ``any angiographic disease'' as the target. The sites differ strongly in prevalence (positive rates 0.46, 0.36, 0.93 and 0.74), and cholesterol is recorded as 0 for every Swiss patient, which we treat as missing. Imputation and standardisation of the headline site results are fitted on the pooled training rows; this is a centralised step that a real federation would have to replace with secure statistics, and Section~\ref{sec:fp} measures that replacement. The Pima, Cleveland and site files were fetched from public GitHub mirrors pinned to a commit and checked against SHA-256 hashes. We could not verify the mirrors' licences, so the raw files are not redistributed with our code. None of the data is patient-level clinical data from a hospital we worked with, and all clients run inside one process. We had no access to a credentialed clinical repository such as PhysioNet's MIMIC-IV or eICU, so no electronic-health-record data are used anywhere in this paper or in the accompanying code.
""" + DATATABLE + r"""
""" + S1.methods() + "\n" + S2.methods() + "\n" + S3.methods() + r"""
\subsection{Tuning and evaluation protocols}
For every dataset split, hyper-parameters are chosen on a validation set carved from the training data (25\%, stratified), by mean validation F1 with ties broken by validation loss; test data are never used for selection (nested protocol). \emph{Grid}: learning rate $\in\{0.01,\dots,0.8\}$ (7 values), plus $\mu\in\{0.001,\dots,1\}$ for FedProx (42 configurations). \emph{Random search}: 24 configurations drawn once and shared by all algorithms over learning rate (log-uniform), $\mu$, batch size, local epochs, hidden width and SCAFFOLD server step. Natural-site \emph{Protocol A}: each site split 70/30, test on the union and on each site's test part. \emph{Protocol B}: leave one site out; train on the other three, test on all rows of the held-out site. Baselines: pooled centralised training and local-only training.
\subsection{Statistics}
Seeds (or splits) are the unit of replication; we report means with bootstrap/\,$t$ intervals, paired Wilcoxon and $t$ tests with Cohen's $d_z$, Friedman tests across algorithms, and Holm correction within families. A Wilcoxon test cannot reach $p<0.05$ with fewer than six pairs, so we use at least ten seeds or splits for headline tests. Where only descriptive evidence is possible (four held-out sites) we say so.""")

# ---------------------------------------------------------------- results
Rr = []
Rr.append(r"\section{Experimental Setup}" + "\n" + r"""Unless stated otherwise we use 10 seeds (or data splits) for headline comparisons and 5 for sensitivity analyses; the exceptions are the backdoor, Byzantine and membership-inference experiments (3 seeds), and the scalability benchmark, which uses 1--3 seeds per row because rows with up to 500 clients are expensive (the number is listed in Table~\\ref{t:kpi}), 20 to 40 rounds, one or two local epochs and full participation. The synthetic benchmark has 12\,000 samples and $K=10$ clients with Dirichlet label skew $\alpha=0.3$; the adaptive-attack benchmark uses $K=20$ clients of which $f\in\{2,4,8\}$ are malicious. Every table in this paper was generated by a script from the stored result files. The accompanying archive contains the code, the test suite, executed notebooks and a SHA-256 manifest.""")
Rr.append(r"\section{Results}")
Rr.append(r"\subsection{Cost of confidentiality (Q1)}")
_be = (s15[s15.config != "plaintext"].set_index("config").train_overhead_pct / 30.0)
Rr.append(f"Accuracy is identical across plaintext, CKKS, BFV, secure aggregation and 3-party SPDZ-style aggregation to within numerical error (largest accuracy change {s15.acc_loss_pp.abs().max():.2f} pp, confidence intervals include zero), and secure aggregation reproduces plaintext parameters to float rounding (max abs difference {secmax:.1e}). The cost is elsewhere (Table~\\ref{{t:sec}}). Communication first: CKKS adds {ckks.comm_overhead_pct:.0f}\\% and secure aggregation {sa.comm_overhead_pct:.0f}\\% uplink, and the 40\\% budget is missed by every mechanism. For CKKS, BFV and the SPDZ-style sum this comes from ciphertext or share expansion, which grows with the model; for secure aggregation part of it is a fixed key-exchange cost that would shrink relative to a larger model, which we did not measure. Training time second: the smallest overhead is {best_ovh.train_overhead_pct:.0f}\\% ({esc(best_ovh.config)}) and the largest {s15.train_overhead_pct.max():.0f}\\%, so the 30\\% budget is missed too. This comparison is against a very cheap in-process baseline (about {float(s15[s15.config == 'plaintext'].train_wall_s.iloc[0]):.1f}\\,s for the whole run), so it overstates the overhead of real training. If the extra time of each mechanism stayed the same while local training became more expensive, the 30\\% budget would be met once training cost grew by a factor of {_be.min():.1f} (secure aggregation) to {_be.max():.1f} (SPDZ-style); the factors for the other mechanisms lie in between. We read the training-time result as a lower bound on how far from budget the mechanisms are, and the communication result as the firmer one.")
s15 = s15.copy(); s15["train_ovh_ci"] = ci_cell(s15, "train_overhead_pct", 0)
Rr.append(table(s15, f"Confidentiality mechanisms vs plaintext (measured, virtual clients, {N_SEC} seeds, randomised run order; overheads are relative to plaintext, with 95\\% bootstrap intervals; communication counts the uplink only).", "t:sec", 2, ["config", "acc", "train_ovh_ci", "comm_overhead_pct", "acc_loss_pp"], {"train_ovh_ci": "train ovh % [95% CI]", "comm_overhead_pct": "uplink ovh %", "acc_loss_pp": "acc loss pp"}, fit=True))
k = kpi.copy(); Rr.append(f"Scaling (Table~\\ref{{t:kpi}}) is unfavourable: complete-graph secure aggregation grows from {k[(k.config=='secagg_complete')&(k.n_clients==10)].training_overhead_pct.iloc[0]:.0f}\\% to {k[(k.config=='secagg_complete')&(k.n_clients==100)].training_overhead_pct.iloc[0]:.0f}\\% training overhead between 10 and 100 virtual clients (pairwise masking is quadratic), sparse neighbour graphs reduce but do not remove this, and CKKS and SPDZ stay far above budget up to 500 clients. The same caveat applies: percentage overheads are relative to the cheap in-process baseline, and absolute times are reported in the artifact.")
kpi = kpi.merge(_ks, on=["config", "n_clients"], how="left")
Rr.append(table(kpi, "Overhead KPIs versus number of virtual clients, relative to plain FedAvg with the same number of clients. Communication counts uplink and downlink (the downlink is unchanged by the mechanisms, so these values are lower than the uplink-only values of Table~\\ref{t:sec}); \\emph{seeds} is the number of repetitions behind each row.", "t:kpi", 1, ["config", "n_clients", "seeds", "training_overhead_pct", "comm_overhead_pct", "accuracy_loss_pp"], {"training_overhead_pct": "train %", "comm_overhead_pct": "comm (up+down) %", "accuracy_loss_pp": "acc loss pp"}, fit=True))
Rr.append(r"\subsection{Attacks and defences (Q2)}")
Rr.append(f"Without defence, the backdoor succeeds (attack success rate {bdh.asr:.3f} at high intensity versus {bdc.asr:.3f} for the clean model, Table~\\ref{{t:bd}}); none of the confidentiality mechanisms changes this because they leave the aggregate unchanged (ablation, Table~\\ref{{t:ab}}). The loss-threshold membership-inference attack is weak in our regime (AUC {mia[mia.config=='none'].auc.iloc[0]:.3f} undefended, {mia.auc.min():.3f} at the strongest DP noise), which is a statement about this overfitting level on synthetic data only. Gradient inversion succeeds on small-batch plaintext updates and not on aggregated ones (task~3.4 in the artifact).")
Rr.append(table(bd, "Backdoor without defence (3 intensities).", "t:bd", 3, ["level", "asr", "clean_acc"]))
ab = ab.copy(); ab["train_ovh_ci"] = ci_cell(ab, "train_overhead_pct", 1)
Rr.append(table(ab, f"Ablation of security layers (measured; {N_AB} seeds, randomised run order; training overhead with 95\\% bootstrap interval, an interval that contains 0 means no measurable overhead).", "t:ab", 3, ["config", "acc", "train_ovh_ci", "acc_under_signflip", "backdoor_asr"], {"train_ovh_ci": "train ovh % [95% CI]", "acc_under_signflip": "acc, sign-flip"}, fit=True))
Rr.append(r"\subsection{Algorithms, tuning and heterogeneity (Q3)}")
Rr.append(f"With default hyper-parameters and Dirichlet $\\alpha=0.3$ SCAFFOLD and FedNova lead on F1 (Table~\\ref{{t:lb}}), but {nsig} pairwise differences survive Holm correction in the untuned benchmark. Selecting the learning rate (and $\\mu$) on a validation split narrows the gaps (Table~\\ref{{t:tun}}); FedAvg's selected rate sits at the edge of the grid, so its optimum is not established.")
Rr.append(table(lb[lb.key != "proposed"], "Untuned leaderboard, Dirichlet 0.3.", "t:lb", 3, ["algorithm", "utility_f1", "convergence_rounds", "communication_mb_to_target", "mean_rank"], {"utility_f1": "F1", "convergence_rounds": "rounds", "communication_mb_to_target": "MB to target", "mean_rank": "mean rank"}))
Rr.append(table(tun, "Default vs validation-tuned (synthetic, 10 seeds).", "t:tun", 3, ["algo", "setting", "lr", "mu", "acc", "f1", "auc"]))
r9 = real[real.setting != "default"]
Rr.append(f"On the three small public tables (clients are synthetic partitions) the four algorithms are statistically indistinguishable (Friedman $p$ = {', '.join(f'{p:.2f}' for p in rank9.friedman_p)} for WDBC, Pima, heart), validation tuning did not help on average, and CKKS and secure aggregation reproduce plaintext training exactly (Table~\\ref{{t:real}}).")
Rr.append(table(r9, "Public tables, nested tuning, 10 splits (tuned setting; pooled-data reference).", "t:real", 3, ["dataset", "algo", "acc", "f1", "auc"]))
Rr.append(r"\subsubsection{Natural sites}")
sp = " / ".join(f"{x:.2f}" for x in n10["meta"]["site_positive_rate"])
Rr.append(f"Using the four heart-disease cohorts as clients (training rows: {n10['meta']['n_train']}, positive rates {sp} for Cleveland / Hungary / Switzerland / VA), the pooled-test F1 of the best untuned algorithm is {max(Ad[a].f1 for a in AL):.3f} and of pooled centralised training {cen.f1:.3f}, against {loc.f1:.3f} for local-only models (Table~\\ref{{t:A10}}): federation recovers the pooled result and clearly beats isolated sites. SCAFFOLD and FedNova are slightly ahead of FedAvg and FedProx (Friedman across tuned algorithms $p$ = {fr_A['p']:.3f}) but no pairwise difference survives Holm correction; tuning did not improve test F1 for any algorithm (mean change between {min(float(x) for x in rd('fl_bench_004','task_4_10_A_tuned_vs_default_tests.csv').mean_diff):.3f} and {max(float(x) for x in rd('fl_bench_004','task_4_10_A_tuned_vs_default_tests.csv').mean_diff):.3f}). Secure aggregation reproduces plaintext parameters to float rounding (max abs difference {secmax:.1e}). Per-site performance differs strongly: AUC is {get(A10s, algo='fedavg', site='cleveland').auc.iloc[0]:.2f}--{get(A10s, algo='fedavg', site='hungary').auc.iloc[0]:.2f} at Cleveland and Hungary but {get(A10s, algo='fedavg', site='va_long_beach').auc.iloc[0]:.2f}--{get(A10s, algo='fedavg', site='switzerland').auc.iloc[0]:.2f} at VA and Switzerland, where F1 is inflated by the high prevalence. Switzerland contributes only 8 negative test cases, so its AUC is too uncertain to compare with Cleveland's; we report it for completeness.")
a10 = pd.concat([A10[A10.setting == "tuned"].assign(method=lambda d: d.algo + " (tuned)"), A10[A10.setting == "default"].assign(method=lambda d: d.algo + " (default)"), A10b.assign(method=A10b.method)])[["method", "acc", "f1", "auc", "bacc"]]
Rr.append(table(a10, "Natural sites, protocol A (within-site holdout), pooled test, 10 splits.", "t:A10", 3, None, {"bacc": "bal. acc"}, fit=True))
Rr.append(f"Under leave-one-site-out (protocol B) the macro-averaged AUC over the four held-out sites is {B10m[B10m.setting=='tuned'].auc.min():.3f}--{B10m[B10m.setting=='tuned'].auc.max():.3f} for all algorithms; transfer to Switzerland and VA is markedly worse than to Cleveland and Hungary (Table~\\ref{{t:B10}}), and the algorithm ranking is not stable (Friedman over four site means $p$ = {fr_B['p']:.2f}; this is descriptive because the seeds only replicate initialisation and only four sites exist, so these numbers cannot support inference about new sites). Under protocol B the differences between algorithms on single sites (for example FedProx and FedAvg at Cleveland) are not established as meaningful or as negligible.")
bt = B10[B10.setting == "tuned"].pivot_table(index="held_out", columns="algo", values="auc").reset_index()
Rr.append(table(bt, "Protocol B: AUC on the held-out site (tuned, 10 init seeds).", "t:B10", 3))
Rr.append(figure("fl_bench_004/figures/task_4_10_natural_sites.png", "Natural-site benchmark. Left: pooled F1 (protocol A). Right: AUC on the held-out site (protocol B).", "f:sites", star=True))
Rr.append(r"\subsubsection{Tuning more than the learning rate and $\mu$}")
mx = max(gain_rg.items(), key=lambda kv: kv[1]); mn = min(gain_rg.items(), key=lambda kv: kv[1])
Rr.append(f"A 24-configuration random search over learning rate, $\\mu$, batch size, local epochs, hidden width and SCAFFOLD server step changes test F1 relative to the lr/$\\mu$ grid by between {mn[1]:+.3f} ({mn[0][0]}, {AN[mn[0][1]]}) and {mx[1]:+.3f} ({mx[0][0]}, {AN[mx[0][1]]}) (Table~\\ref{{t:rs}}); {len(sigt)} of {len(rst[rst.contrast=='random - grid'])} paired random-vs-grid comparisons have uncorrected $p<0.05$. Budgets are matched in the number of configurations, not in compute, and selection on small validation sets can overfit, so we draw no conclusion about the best FL algorithm from this experiment.")
Rr.append(table(rs.pivot_table(index=["dataset", "algo"], columns="setting", values="f1").reset_index()[["dataset", "algo", "default", "grid", "random"]], "Test F1: default, lr/$\\mu$ grid, six-parameter random search.", "t:rs", 3))
Rr.append(r"\subsection{Adaptive attackers (Q4)}\label{sec:adapt}")
st = "; ".join(f"{esc(d)} ({a:.2f}$\\to${b:.2f})" for d, a, b in stopped) or "none"
Rr.append(f"The tested defences are not robust to attackers that adapt (synthetic data, $K=20$, $f=4$). Against a static boosted backdoor the mean has ASR {mean_bd.asr:.2f}, while median, trimmed mean, Krum, Multi-Krum and the anomaly filter keep it low; adaptive variants raise it again for: {st} (static$\\to$adaptive ASR, Table~\\ref{{t:adb}}). Untargeted ALIE and MIN-MAX alone barely moved accuracy in this setting (minimum accuracy {unt_min.min():.3f} versus about 0.777 without attack).")
Rr.append(table(adb.pivot_table(index="defense", columns="attack", values="asr").reset_index()[["defense", "none_clean", "static_boost", "norm_matched", "minmax_backdoor"]], "Backdoor ASR, static vs adaptive attacks (synthetic).", "t:adb", 3, fit=True))
w8 = get(wc, dataset="syn")
Rr.append(f"Rule-aware attackers that simulate the server rule are stronger (Table~\\ref{{t:wc}}): the worst-case accuracy drop per rule ranges from {w8['drop'].min():.3f} to {w8['drop'].max():.3f}, the largest for {esc(w8.sort_values('drop').iloc[-1].defense)} under {esc(w8.sort_values('drop').iloc[-1].worst_attack)}; the AGR-aware backdoor reaches ASR above 0.5 on {sum(v > 0.5 for v in bd_agr.values())} of 7 rules (versus {sum(v > 0.5 for v in bd_mm.values())} for the distance-constrained backdoor). AGR-tailored was the strongest attack for {int((w8.worst_attack == 'agr_tailored').sum())} of 7 rules in the untargeted setting. Models destroyed outright (non-finite weights) are scored as constant predictors and counted; {ndiv} of {len(run8)} untargeted runs ended with non-finite weights (all under AGR-tailored against the undefended mean).")
Rr.append(table(w8[["defense", "clean_acc", "worst_attack", "worst_acc", "drop"]], "Worst-case untargeted attack per aggregation rule (synthetic, $f=4$ of 20).", "t:wc", 3))
Rr.append(figure("fl_attack_003/figures/task_3_8_stronger_adaptive.png", "Rule-aware adaptive attacks. Left: accuracy under untargeted attacks. Right: backdoor ASR.", "f:adapt", star=True))
Rr += S1.results()
Rr += S2.results()
Rr += S3.results()
Rr.append(r"\subsection{Reproducibility}" + (f"\n{nrep} of {len(repro)} automated reproducibility checks pass; " + ("the one that fails is the git-commit check because the bundle is not a repository (a SHA-256 manifest is used instead)." if [c["check"] for c in repro if not c["passed"]] == ["git commit identifier available"] else "failing: " + "; ".join(esc(c["check"]) for c in repro if not c["passed"]) + ".")))
# ---------------------------------------------------------------- discussion etc
D_ = []
D_.append(r"""\section{Discussion}
Hiding updates from the server is not the same as checking them. Whatever protects an update from the server also hides it from any defence that has to inspect it, and none of the mechanisms we tested changes the aggregate itself, which is why backdoor success stayed the same in the ablation. For the same reason robust statistics and secure aggregation cannot just be stacked. A server cannot take a median or run Krum over masked or encrypted updates without further protocol design, and we did not attempt that.

A defence that works against a static attacker may say little about an adaptive one. Rules that stop a naive, boosted backdoor are bypassed by backdoors that stay inside the norm or distance envelope, and, under our omniscient attacker model, attackers that simulate the server's rule do measurably more damage. Our attackers are omniscient upper bounds. They show what a defence can promise at best, and nothing about how common such attackers are. The rule committee is a case in point: stacking four published rules and letting a random subset vote did not help against the attacker that knew the draw, and under label skew it cost more clean accuracy than any single member did. We read that as a caution against assuming that more rules, or more randomness, buy more security.

Benchmark conclusions also turned out to be fragile. Rankings from untuned baselines on synthetic Dirichlet splits weaken after tuning and mostly disappear on small real data. On the natural sites the federated models match pooled training and clearly beat local-only training, which is the practical argument for FL. The differences among the four algorithms are small next to the variation between sites. Natural heterogeneity (different prevalence, missing measurements) also produced transfer gaps between sites that the synthetic label-skew splits did not show.

Two more results qualify the picture. Federated standardisation through secure sums reproduces the pooled statistics, so the centralised preprocessing step that many FL benchmarks quietly rely on can be dropped for the price of two aggregation rounds. Whether global or per-site standardisation then gives better accuracy depends on the kind of heterogeneity: a measurement shift favoured local statistics, while natural cohorts with different prevalence did not. And robustness has a price that grows with label skew. With no attacker present, the coordinate median loses far more accuracy at low Dirichlet $\alpha$ than at high, and with few clients per round it offers no protection at all.

For practice, within the limits of a simulation, our results suggest choosing the integrity defence and the confidentiality mechanism separately, budgeting for cryptographic overhead that grows with the number of clients, evaluating on held-out sites and not only on held-out patients, and testing a defence against adaptive attackers before relying on it.""")
D_.append(r"""\section{Threats to Validity and Limitations}
Our experiments ran on two CPU cores in a single process, so wall-clock overheads are relative to a very cheap baseline. The SPDZ-style protocol (with a trusted dealer) and the secure aggregation (semi-honest, without a consistency round) are our own implementations. The hyper-parameter search is modest (7 to 42 grid points and 24 random configurations), and the compute of the random configurations is not matched. Backdoors use a single trigger family, and a model whose weights became non-finite is scored as a constant prediction of class 0 with AUC 0.5. The 24 random-search configurations were drawn once and shared by all algorithms, and the variance of that single draw was not estimated, so the grid-versus-random differences in Table~\ref{t:tun} (for example SCAFFOLD) cannot be attributed to the search strategy.

Most of our data are synthetic. The real data are small public research tables from the 1980s and 1990s (303 to 920 rows), the test sets hold between 91 and 277 samples, so intervals are wide, and Switzerland contributes only 8 negative cases. On three of the four real data sources, clients are synthetic partitions. The four heart-disease sites are natural cohorts but not a live federation, and the headline preprocessing statistics are fitted on pooled training rows (the federated alternative is evaluated separately and uses a global mean in place of the median, so the two are not interchangeable bit for bit). With only four sites, the leave-one-site-out results cannot support inference about new sites.

Attack success on tabular MLPs need not carry over to images, text or large models. Our AGR-tailored attacker searches a fixed set of candidates and is not a global optimum, and we did not implement a learning-based or loss-aware attacker. We had no credentialed clinical repository (e.g.\ PhysioNet) available, and we did not study large clinical datasets, real hospital federations, real networks, malicious servers against secure aggregation, formal security proofs, GPU performance, or regulatory compliance.""")
D_.append(r"\section{Ethics, Data and Legal Statement}\label{sec:ethics}" + "\n" + rf"""No patient was contacted and no non-public data were used. The data sets are de-identified research releases from the UCI repository; for some of them we used third-party GitHub mirrors, pinned by commit and SHA-256 hash, which we do not redistribute because we could not verify their licences. We read the legal texts only in part: GDPR Art.~9 (paragraphs 1 to 4, selected points) and Art.~32(1) from the legislation.gov.uk copy of the adopted text rather than the EUR-Lex Official Journal page, NIS2 Art.~21(1)--(2) and Annex~I from EUR-Lex, AI Act Art.~15(1)--(5) from an unofficial rendering, and German BDSG \S22(1). The remaining provisions, national transposition and later amendments were not read. Of {nref} reference entries, {nver} were checked against a source page for title, authors and date (we did not re-read the papers themselves), and {nunv} could not be verified and are flagged in the accompanying archive. Nothing in this paper is legal advice or a statement of compliance.""")
D_.append(r"""\section{Conclusion}
In a fully simulated and openly documented setting, the confidentiality mechanisms for federated aggregation kept accuracy intact, exceeded our communication budget, exceeded the training-time budget against a cheap baseline and provided no integrity protection against poisoning. Under our omniscient attacker model, robust aggregation helped against static attacks but not against attackers that respect its decision rule, and a committee built from four published rules did not change that. Conclusions about FL algorithms depended on tuning and on how heterogeneity is produced. The natural site splits showed both the benefit of federation over isolated training and large transfer gaps between sites. The statistics needed for preprocessing can be computed through the same secure aggregation without pooling rows, and calibration was moderate and similar across algorithms. The obvious next steps are a deployment across real institutions with approved data, learning-based adaptive attackers, protocols that combine secure aggregation with integrity checks, and a formal security analysis.""")
D_.append(r"""\section*{Availability}
Code, tests, executed notebooks, result tables, figures and a SHA-256 manifest are in the accompanying archive, and each experiment is re-run with one command.
\bibliographystyle{ieeetr}
\bibliography{references}
\end{document}""")
import shutil as _sh

_sh.copy(R / "reports/manuscript_support/references.bib", PUB / "references.bib")
(PUB / "manuscript_ieee.tex").write_text("\n".join(T + Rr + D_))
(PUB / "build.sh").write_text("""#!/usr/bin/env bash
# Builds manuscript_ieee.pdf. Needs a TeX Live with pdflatex, bibtex and the IEEEtran class (package `IEEEtran`;
# the class is NOT shipped here). If your TeX installation lacks it, put IEEEtran.cls next to this script or set TEXINPUTS.
set -euo pipefail
cd "$(dirname "$0")"
export TEXINPUTS="${TEXINPUTS:-}:$PWD:$PWD/vendor:"
pdflatex -interaction=nonstopmode manuscript_ieee.tex >/dev/null
bibtex manuscript_ieee >/dev/null || true
pdflatex -interaction=nonstopmode manuscript_ieee.tex >/dev/null
pdflatex -interaction=nonstopmode manuscript_ieee.tex | grep -E "Output written|Error" || true
""")
(PUB / "build.sh").chmod(0o755)
(PUB / "README.md").write_text("""# LaTeX manuscript (IEEEtran, conference style)

* `manuscript_ieee.tex` - full paper, generated by `scripts/generate_manuscript.py` from the stored result files (every table number comes from `experiments/results/`). Author: Guy M. Kaptue T.
* `references.bib` - 54 entries with per-entry verification status in `docs/references/references_master.md`.
* `figures/` - PNG figures copied from the result folders.
* `build.sh` - runs pdflatex / bibtex / pdflatex x2 to produce `manuscript_ieee.pdf`.
* The IEEEtran class is not shipped (LPPL, available in standard TeX Live / `tlmgr install ieeetran`). Bibliography style: `ieeetr`.

Status: DRAFT, not peer reviewed; all results are simulations (virtual clients); no compliance claim.
""")
print("manuscript_ieee.tex written", len("\n".join(T + Rr + D_)))
