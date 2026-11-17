"""Stage-2 manuscript blocks: four literature defences and architecture baselines. Every number comes from experiments/results/fl_stage2/."""
import pandas as pd

from manuscript_lib import O, figure, rd, table

ST = "fl_stage2"
NICE = {"mean": "mean", "median": "median", "trimmed_mean": "trimmed mean", "krum": "Krum", "multi_krum": "Multi-Krum", "clip_median_norm": "median-norm clip",
        "anomaly_filter": "anomaly filter", "flame": "FLAME ($\\lambda{=}0$)", "flame_noise": "FLAME ($\\lambda{=}0.001$)", "rlr": "RLR", "foolsgold": "FoolsGold",
        "signguard": "SignGuard"}
DEFS = list(NICE)
NEW = ["flame", "flame_noise", "rlr", "foolsgold", "signguard"]
UN = [("none", "no attack"), ("sign_flip", "sign flip"), ("alie", "ALIE"), ("minmax", "MIN-MAX"), ("agr_tailored", "AGR-tailored")]
BD = [("none_clean", "no attack"), ("static_boost", "static boost"), ("norm_matched", "norm-matched"), ("agr_backdoor", "AGR-backdoor")]


def nseed(fname):
    return int(pd.read_csv(O / ST / fname)["seed"].nunique())


def _pm(m, lo, hi, nd=3):
    return f"{m:.{nd}f}$\\pm${(hi - lo) / 2:.{nd}f}"


def methods() -> str:
    return r"""\subsection{Additional defences and baselines}
\label{sec:lit-method}
We re-implemented four published defences from the algorithm descriptions in the papers, not from the authors' code, and verified the bibliographic data on the venue pages. \emph{FLAME}~\cite{nguyen2022flame} clusters the clients' models with HDBSCAN on pairwise cosine distance (minimum cluster size $n/2+1$), clips admitted updates to the median update norm, and adds Gaussian noise with standard deviation $\lambda$ times that norm; we run it with $\lambda=0$ and with the paper's $\lambda=0.001$. \emph{Robust learning rate}~\cite{ozdayi2021rlr} flips the sign of the server step on every coordinate where fewer than $\theta$ net sign votes agree; the paper chose $\theta$ per data set, and our $\theta=\lceil 0.4\,m\rceil$ for $m$ clients per round is our own choice. \emph{FoolsGold}~\cite{fung2020foolsgold} down-weights clients whose accumulated update histories are unusually similar (pairwise cosine, pardoning, logit rescaling with $\kappa=1$); it needs client identities across rounds. \emph{SignGuard}~\cite{xu2022signguard} drops updates whose norm is outside $[0.1,3]$ times the median, clusters the sign statistics of a random 10\% of coordinates with Mean-Shift, keeps the largest cluster and averages the clipped survivors. Where the papers leave a choice open we state ours in the code documentation (for instance the Mean-Shift bandwidth); we do not implement FLAME's secure two-party filtering, FoolsGold's optional feature weighting or SignGuard's similarity variants. These are re-implementations on a small MLP and need not reproduce the published numbers.
For rules that reject updates we record, per round, the share of malicious rows rejected (TPR) and of honest rows rejected (FPR), averaged over rounds; rules that only re-weight or clip have no such rate. The attack suite is the one used before: sign flipping (scale 5), ALIE, MIN-MAX, the rule-simulating AGR-tailored attacker, and for backdoors a static boosted attacker, a norm-matched one and an AGR-backdoor attacker, with $f=4$ of $K=20$ colluding clients, 10 seeds, on SYN-TAB-40, and in addition label-skewed clients (Dirichlet $\alpha=0.3$) to measure what each rule costs without an attacker. As model baselines we compare a logistic regression (softmax regression), the one-hidden-layer MLP used so far, and a three-hidden-layer MLP, with identical learning rate, rounds and local epochs for all three and no per-model tuning."""


def results() -> list:
    R = []
    n1, n2, n3 = nseed("s2_1_runs.csv"), nseed("s2_2_runs.csv"), nseed("s2_3_runs.csv")
    u = rd(ST, "s2_1_summary.csv"); b = rd(ST, "s2_2_summary.csv")
    ut = rd(ST, "s2_1_tests.csv")
    gu = lambda d, sc, part="iid", col="acc": u[(u.defence == d) & (u.scenario == sc) & (u.partition == part)].iloc[0][col]  # noqa: E731
    gb = lambda d, sc, part="iid", col="asr": b[(b.defence == d) & (b.scenario == sc) & (b.partition == part)].iloc[0][col]  # noqa: E731
    R.append(r"\subsection{Literature defences and model baselines}\label{sec:lit}")
    R.append(r"\subsubsection{Untargeted attacks}")
    scs = [s for s, _ in UN]
    worst = {d: min(gu(d, s) for s in scs) for d in DEFS}
    best_old = max((d for d in DEFS if d not in NEW), key=lambda d: worst[d]); best_new = max(NEW, key=lambda d: worst[d])
    rows = []
    for d in DEFS:
        row = [NICE[d] + (" (new)" if d in NEW else "")]
        for s, _ in UN:
            r = u[(u.defence == d) & (u.scenario == s) & (u.partition == "iid")].iloc[0]; row.append(_pm(r.acc, r.acc_lo, r.acc_hi))
        rows.append(row)
    sf_txt = ", ".join(f"{NICE[d]} {gu(d, 'sign_flip'):.3f}" for d in NEW)
    ag_med = gu("median", "agr_tailored")
    hold = [d for d in NEW if gu(d, "agr_tailored") >= ag_med - 0.02]
    fall = [d for d in NEW if d not in hold]
    hold_txt = (f"{', '.join(NICE[d] for d in hold)} stay within 0.02 of the median ({ag_med:.3f}: " + ", ".join(f"{NICE[d]} {gu(d, 'agr_tailored'):.3f}" for d in hold) + ")" if hold else "no new rule stays within 0.02 of the median")
    hold_txt += ("; " + ", ".join(f"{NICE[d]} ({gu(d, 'agr_tailored'):.3f})" for d in fall) + " collapse towards the mean (" + f"{gu('mean', 'agr_tailored'):.3f}" + ")" if fall else "")
    nsig = int(((ut.holm_p < 0.05) & (ut.family == "vs mean") & (ut.arm.isin(NEW)) & (ut.mean_diff > 0)).sum())
    ntot = int(((ut.family == "vs mean") & (ut.arm.isin(NEW))).sum())
    clean_cost = {d: gu("mean", "none") - gu(d, "none") for d in DEFS}
    R.append(
        f"Table~\\ref{{t:s2u}} puts the four new rules next to the seven earlier ones under the same attacks ({n1} seeds). Judged by the worst accuracy over the five scenarios, the best earlier rule is {NICE[best_old]} ({worst[best_old]:.3f}) and the best of the new rules is {NICE[best_new]} ({worst[best_new]:.3f}); "
        f"the mean falls to {worst['mean']:.3f}. Without any attacker the new rules cost between {min(clean_cost[d] for d in NEW):+.3f} and {max(clean_cost[d] for d in NEW):+.3f} accuracy relative to the plain mean. "
        f"Against the rule-simulating attacker {hold_txt}; under plain sign flipping the new rules score {sf_txt} (median {gu('median', 'sign_flip'):.3f}, mean {gu('mean', 'sign_flip'):.3f}). "
        f"RLR and FoolsGold were designed for other threats (backdoors; sybils with similar updates) and are not expected to stop independent Byzantine updates, which is what we see. "
        f"Of {ntot} paired contrasts of a new rule against the mean (five scenarios, i.i.d.\\ and skewed), {nsig} show a significantly higher accuracy for the new rule after Holm correction. Each correction is applied within its own scenario, so {nsig} of {ntot} is a count of tests and not a family-wise result over all {ntot}. "
        f"All of this holds for one synthetic data set, one attacker fraction (20\\%) and an omniscient attacker, which is an upper bound; the numbers describe these rules in this setting, not their general quality.")
    R.append(table(pd.DataFrame(rows, columns=["Rule"] + [n for _, n in UN]),
                   f"Test accuracy (mean$\\pm$95\\% half-width over {n1} seeds), $K=20$, $f=4$ colluding attackers, i.i.d.\\ clients. New rules are those re-implemented in this stage.",
                   "t:s2u", 3, None, None, False, "lrrrrr", True, True))
    R.append(r"\subsubsection{Backdoor attacks}")
    wb = {d: max(gb(d, s) for s, _ in BD[1:]) for d in DEFS}
    low = [d for d in DEFS if wb[d] < 0.3]
    rows = []
    for d in DEFS:
        row = [NICE[d] + (" (new)" if d in NEW else "")]
        for s, _ in BD:
            r = b[(b.defence == d) & (b.scenario == s) & (b.partition == "iid")].iloc[0]; row.append(_pm(r.asr, r.asr_lo, r.asr_hi))
        rows.append(row)
    fg = "foolsgold"
    R.append(
        f"The picture for backdoors differs by rule (Table~\\ref{{t:s2b}}, {n2} seeds; ASR is the share of triggered test samples classified as the target class). "
        f"FoolsGold keeps the ASR at or below {max(gb(fg, s) for s, _ in BD):.3f} in every scenario, including the AGR-backdoor attacker, and rejects the colluding clients almost always (TPR {b[(b.defence == fg) & (b.scenario == 'agr_backdoor') & (b.partition == 'iid')].iloc[0].tpr:.2f}, FPR {b[(b.defence == fg) & (b.scenario == 'agr_backdoor') & (b.partition == 'iid')].iloc[0].fpr:.2f}). "
        f"That is the setting it was designed for: our four attackers train on the same kind of poisoned data and their updates are therefore similar, and our adaptive attackers do not try to decorrelate their updates. We did not test such an attacker, so this result should not be read as robustness to sybils that diversify. "
        f"FLAME stops the static and norm-matched backdoors (ASR {gb('flame', 'static_boost'):.3f} and {gb('flame', 'norm_matched'):.3f}) but not the AGR-backdoor attacker ({gb('flame', 'agr_backdoor'):.3f}); "
        f"RLR with our $\\theta$ does not stop the boosted backdoors on i.i.d.\\ clients ({gb('rlr', 'static_boost'):.3f}); SignGuard stops the static boost ({gb('signguard', 'static_boost'):.3f}) but not the norm-matched or AGR-backdoor attackers ({gb('signguard', 'norm_matched'):.3f}, {gb('signguard', 'agr_backdoor'):.3f}). "
        f"Rules with a worst-case ASR below 0.3 over the three attacks: {', '.join(NICE[d] for d in low) if low else 'none'}.")
    R.append(table(pd.DataFrame(rows, columns=["Rule"] + [n for _, n in BD]),
                   f"Backdoor attack success rate (mean$\\pm$95\\% half-width over {n2} seeds), $K=20$, $f=4$, i.i.d.\\ clients. Without attackers the plain mean has ASR {gb('mean', 'none_clean'):.3f}.",
                   "t:s2b", 3, None, None, False, "lrrrr", True, True))
    # detection
    det_rules = ["anomaly_filter", "multi_krum", "flame", "foolsgold", "signguard"]
    cols = [("sign_flip", "u"), ("alie", "u"), ("minmax", "u"), ("agr_tailored", "u"), ("static_boost", "b"), ("norm_matched", "b"), ("agr_backdoor", "b")]
    names = {"sign_flip": "sign flip", "alie": "ALIE", "minmax": "MIN-MAX", "agr_tailored": "AGR-tail.", "static_boost": "static", "norm_matched": "norm-m.", "agr_backdoor": "AGR-bd"}
    rows = []
    for d in det_rules:
        row = [NICE[d]]
        for sc, k in cols:
            df = (u if k == "u" else b)
            r = df[(df.defence == d) & (df.scenario == sc) & (df.partition == "iid")].iloc[0]
            row.append(f"{r.tpr:.2f}/{r.fpr:.2f}")
        rows.append(row)
    fl = u[(u.defence == "flame") & (u.partition == "iid") & (u.scenario != "none")]
    R.append(
        f"Detection rates explain part of this (Table~\\ref{{t:s2d}}). FLAME rejects on average {100 * fl.fpr.mean():.0f}\\% of the honest rows per round under attack; we did not isolate the cause, a plausible one being the minimum-cluster-size rule when honest updates are spread out. "
        f"SignGuard's FPR is {u[(u.defence == 'signguard') & (u.partition == 'iid') & (u.scenario != 'none')].fpr.mean():.2f} on average; its TPR is {gu('signguard', 'alie', 'iid', 'tpr'):.2f} against ALIE but {gu('signguard', 'agr_tailored', 'iid', 'tpr'):.2f} against the AGR-tailored attacker and {gb('signguard', 'agr_backdoor', 'iid', 'tpr'):.2f} against the AGR-backdoor attacker. A rate is an average over rounds and clients; it does not tell which rounds were fatal.")
    R.append(table(pd.DataFrame(rows, columns=["Rule"] + [names[c] for c, _ in cols]),
                   f"Share of malicious rows rejected / share of honest rows rejected (TPR/FPR, mean over rounds and {n1}--{n2} seeds), i.i.d.\\ clients, $f=4$ of 20. Untargeted attacks first, then backdoors.",
                   "t:s2d", 3, None, None, False, "lrrrrrrr", True, True))
    # non-IID cost
    rows = []
    for d in DEFS:
        rows.append([NICE[d] + (" (new)" if d in NEW else ""), f"{gu(d, 'none'):.3f}", f"{gu(d, 'none', 'dirichlet0.3'):.3f}", f"{gu(d, 'sign_flip', 'dirichlet0.3'):.3f}", f"{gb(d, 'static_boost', 'dirichlet0.3'):.3f}"])
    cost_h = {d: gu("mean", "none", "dirichlet0.3") - gu(d, "none", "dirichlet0.3") for d in DEFS}
    worst_h = max(DEFS, key=lambda d: cost_h[d])
    nlose = sum(1 for d in DEFS if d != 'mean' and cost_h[d] > 0)
    R.append(
        f"Label skew changes the ranking (Table~\\ref{{t:s2n}}). With Dirichlet $\\alpha=0.3$ and no attacker, {nlose} of the 11 other rules lose accuracy against the mean ({gu('mean', 'none', 'dirichlet0.3'):.3f}); the largest loss is {cost_h[worst_h]:.3f} ({NICE[worst_h]}). Under the static boosted backdoor on skewed clients the ASR of FLAME is {gb('flame', 'static_boost', 'dirichlet0.3'):.3f}, of RLR {gb('rlr', 'static_boost', 'dirichlet0.3'):.3f} and of FoolsGold {gb('foolsgold', 'static_boost', 'dirichlet0.3'):.3f}, "
        f"but their clean accuracy is {gb('flame', 'static_boost', 'dirichlet0.3', 'clean_acc'):.3f}, {gb('rlr', 'static_boost', 'dirichlet0.3', 'clean_acc'):.3f} and {gb('foolsgold', 'static_boost', 'dirichlet0.3', 'clean_acc'):.3f}. "
        f"A defence that lowers the ASR by lowering accuracy for everyone is not a good trade, so the two numbers must be read together.")
    R.append(table(pd.DataFrame(rows, columns=["Rule", "acc, i.i.d., none", "acc, skew, none", "acc, skew, sign flip", "ASR, skew, static"]),
                   f"Cost of each rule under label skew (Dirichlet $\\alpha=0.3$) ({n1}--{n2} seeds; i.i.d.\\ column for comparison).", "t:s2n", 3, None, None, False, "lrrrr", True, True))
    R.append(figure(f"{ST}/figures/s2_1_untargeted.png", "Accuracy of twelve rules under five untargeted scenarios (red: new rules; 95\\% $t$-intervals over seeds).", "f:s2u", star=True))
    R.append(figure(f"{ST}/figures/s2_2_backdoor.png", "Backdoor ASR of twelve rules (red: new rules; 95\\% $t$-intervals over seeds).", "f:s2b", star=True))
    # baselines
    s3 = rd(ST, "s2_3_summary.csv"); t3 = rd(ST, "s2_3_tests.csv")
    R.append(r"\subsubsection{Model baselines}")
    names3 = {"logreg": "logistic regression", "mlp": "MLP (1 hidden)", "deep_mlp": "MLP (3 hidden)"}
    rows = []
    for ds, dl in (("synthetic", "SYN-TAB-40"), ("heart-4sites", "heart, 4 sites")):
        for st in ("fedavg", "centralised"):
            for a in ("logreg", "mlp", "deep_mlp"):
                r = s3[(s3.dataset == ds) & (s3.setting == st) & (s3.arch == a)].iloc[0]
                tt = t3[(t3.dataset == ds) & (t3.setting == st) & (t3.arm == a)]
                d = "--" if a == "mlp" else f"{tt.iloc[0].mean_diff:+.3f} ({tt.iloc[0].dz:.1f})"
                rows.append([dl, "FedAvg" if st == "fedavg" else "pooled SGD", names3[a], f"{r.f1:.3f} [{r.f1_lo:.3f}, {r.f1_hi:.3f}]", f"{r.auc:.3f}", d])
    sy = lambda st, a, c: s3[(s3.dataset == "synthetic") & (s3.setting == st) & (s3.arch == a)].iloc[0][c]  # noqa: E731
    ht = lambda st, a, c: s3[(s3.dataset == "heart-4sites") & (s3.setting == st) & (s3.arch == a)].iloc[0][c]  # noqa: E731
    pr = lambda a, dn: s3[(s3.setting == f"{dn}+signflip20%") & (s3.arch == a)].iloc[0]["acc"]  # noqa: E731
    R.append(
        f"A linear model is a strong test of whether the MLP is needed. Under FedAvg on SYN-TAB-40 logistic regression reaches F1 {sy('fedavg', 'logreg', 'f1'):.3f}, the one-hidden-layer MLP {sy('fedavg', 'mlp', 'f1'):.3f} and the three-hidden-layer MLP {sy('fedavg', 'deep_mlp', 'f1'):.3f} (Table~\\ref{{t:s2a}}, {n3} seeds, same learning rate, rounds and local epochs for all three, no per-model tuning). "
        f"On the four heart-disease cohorts the corresponding values are {ht('fedavg', 'logreg', 'f1'):.3f}, {ht('fedavg', 'mlp', 'f1'):.3f} and {ht('fedavg', 'deep_mlp', 'f1'):.3f}. "
        f"Under 20\\% sign-flip clients the plain mean gives accuracy {pr('logreg', 'mean'):.3f} (logistic regression), {pr('mlp', 'mean'):.3f} (MLP) and {pr('deep_mlp', 'mean'):.3f} (deep MLP), and the median {pr('logreg', 'median'):.3f}, {pr('mlp', 'median'):.3f} and {pr('deep_mlp', 'median'):.3f} (the deep MLP's F1 under the median is {s3[(s3.setting == 'median+signflip20%') & (s3.arch == 'deep_mlp')].iloc[0].f1:.3f}, which suggests it mostly predicts one class). "
        f"The gaps therefore tell us about this synthetic generator, which has nonlinear structure by construction, and about small cohorts; they should not be transferred to other data.")
    R.append(table(pd.DataFrame(rows, columns=["Data", "Training", "Model", "F1 [95\\% CI]", "AUC", "$\\Delta$F1 vs MLP ($d_z$)"]),
                   f"Model baselines ({n3} seeds or splits). FedAvg: $K=10$ clients (Dirichlet $\\alpha=0.3$) on SYN-TAB-40, the four natural sites on the heart data; pooled SGD uses the same total number of local passes.",
                   "t:s2a", 3, None, None, False, "lllrrr", True, True))
    return R
