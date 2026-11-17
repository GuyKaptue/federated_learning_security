"""Stage-3 manuscript blocks: randomised rule committee under an attacker-knowledge ladder. Every number comes from experiments/results/fl_stage3/."""
import pandas as pd

from manuscript_lib import O, rd, table

ST = "fl_stage3"
NICE = {"median": "median", "multi_krum": "Multi-Krum", "anomaly_filter": "anomaly filter", "flame": "FLAME", "signguard": "SignGuard", "foolsgold": "FoolsGold",
        "rc_all_v1": "committee, all 4, vote$\\ge$1", "rc_all_v2": "committee, all 4, vote$\\ge$2", "rrc_k2_v1": "random committee, $k{=}2$, vote$\\ge$1",
        "rrc_k3_v2": "random committee, $k{=}3$, vote$\\ge$2"}
SINGLE = ["multi_krum", "flame", "signguard", "anomaly_filter"]
COMM = ["rc_all_v1", "rc_all_v2", "rrc_k2_v1", "rrc_k3_v2"]
RAND = ["rrc_k2_v1", "rrc_k3_v2"]


def nseed(f):
    return int(pd.read_csv(O / ST / f)["seed"].nunique())


def _pm(r, c, nd=3):
    return f"{r[c]:.{nd}f}$\\pm${(r[c + '_hi'] - r[c + '_lo']) / 2:.{nd}f}"


def methods() -> str:
    return r"""\subsection{Rule committee and attacker knowledge}
\label{sec:committee}
Randomised aggregation and hybrid (multi-rule) aggregation are not new~\cite{nabavirazavi2023randomization,yue2024hybrid}; we do not claim the idea. We study one concrete instance and ask how much an attacker's knowledge matters. A \emph{rule committee} draws $k$ rules from the pool \{anomaly filter, Multi-Krum, FLAME, SignGuard\} each round; a row is rejected if at least $v$ drawn members flag it, the survivors are combined by a clipped weighted mean, and if every row is rejected the coordinate-wise median is used. Four variants were fixed before the experiments and not tuned: all four members with $v{=}1$ or $v{=}2$ (deterministic), and a random draw of $k{=}2$ with $v{=}1$ or $k{=}3$ with $v{=}2$.
The rule-simulating attacker is given two levels of knowledge. \emph{Draw-aware}: it simulates the exact committee that will run, including the random draw (for a deterministic committee this is simply full knowledge). \emph{Pool-aware}: it knows the pool, $k$ and $v$, but every candidate it simulates sees an independent random draw. A pool-aware attacker cannot do better in expectation than a draw-aware one (Proposition~2 in \texttt{docs/threat\_analysis.md}; an elementary ordering, not a bound on the gap). We also add a \emph{jitter} option: the colluders add independent noise of norm equal to their update norm after choosing the attack vector, so that their rows are no longer identical. The jitter is not optimised against the defence and so is a weak decorrelation, not an adaptive attacker. The attackers remain omniscient upper bounds and all results are on one synthetic data set with $K{=}20$, $f{=}4$."""


def results() -> list:
    R = []
    n1, n2 = nseed("s3_1_runs.csv"), nseed("s3_2_runs.csv")
    u = rd(ST, "s3_1_summary.csv"); b = rd(ST, "s3_2_summary.csv")
    ut = rd(ST, "s3_1_tests.csv")
    gu = lambda d, sc, part="iid", col="acc": u[(u.defence == d) & (u.scenario == sc) & (u.partition == part)].iloc[0][col]  # noqa: E731
    gb = lambda d, sc, part="iid", col="asr": b[(b.defence == d) & (b.scenario == sc) & (b.partition == part)].iloc[0][col]  # noqa: E731
    R.append(r"\subsection{Rule committee under the knowledge ladder (exploratory)}\label{sec:s3}")
    R.append(f"This section is exploratory. It uses {n1} seeds for the untargeted and {n2} for the backdoor comparison (the twelve-rule comparison above used 10), none of the paired contrasts is significant after Holm correction, and the backdoor intervals overlap across almost all conditions. The main finding is therefore a null result: the committee did not beat its best member or the median, and we found no measurable benefit of randomisation. The point estimates follow.")
    # untargeted
    cols = [("none", "none"), ("agr", "AGR"), ("agr[pool]", "AGR, pool"), ("agr+jit", "AGR+jitter"), ("agr+jit[pool]", "AGR+jitter, pool")]
    rows = []
    for d in ["median"] + SINGLE + COMM:
        row = [NICE.get(d, d)]
        for s, _ in cols:
            m = u[(u.defence == d) & (u.scenario == s) & (u.partition == "iid")]
            row.append(_pm(m.iloc[0], "acc") if len(m) else "--")
        rows.append(row)
    best_single = max(SINGLE, key=lambda d: gu(d, "agr"))
    best_comm = max(COMM, key=lambda d: gu(d, "agr"))
    worst_single_agr = min(gu(d, "agr") for d in SINGLE)
    pool_gap = {d: gu(d, "agr[pool]") - gu(d, "agr") for d in RAND}
    nsig = int((ut.holm_p < 0.05).sum()); ntest = len(ut)
    cl = {d: gu("mean", "none", "dirichlet0.3") - gu(d, "none", "dirichlet0.3") for d in COMM + SINGLE}
    R.append(
        f"Table~\\ref{{t:s3u}} reports accuracy ({n1} seeds). Against the draw-aware AGR-tailored attacker the best committee variant ({NICE[best_comm]}) reaches {gu(best_comm, 'agr'):.3f}, the best single member ({NICE[best_single]}) {gu(best_single, 'agr'):.3f}, the worst single member {worst_single_agr:.3f} and the coordinate-wise median {gu('median', 'agr'):.3f}; the plain mean reaches {gu('mean', 'agr'):.3f}. "
        f"The differences among median, members and committees are small compared with the interval half-widths, and of {ntest} paired contrasts (against the median and FoolsGold) {nsig} remain significant after Holm correction; with {n1} seeds the exact Wilcoxon test cannot reach that level at all, so these tests are uninformative and we rely on the effect sizes in the stored test file. "
        f"Lowering the attacker's knowledge from draw-aware to pool-aware changed the accuracy of the two random committees by {pool_gap['rrc_k2_v1']:+.3f} and {pool_gap['rrc_k3_v2']:+.3f}, which is within the interval widths in this data; we therefore \\emph{{cannot}} claim that randomisation measurably helped here. "
        f"Under label skew (Dirichlet $\\alpha{{=}}0.3$, no attacker) the committees lose between {min(cl[d] for d in COMM):.3f} and {max(cl[d] for d in COMM):.3f} accuracy relative to the mean, against {min(cl[d] for d in SINGLE):.3f} to {max(cl[d] for d in SINGLE):.3f} for single members, and under attack with skew the best single member ({NICE[max(SINGLE, key=lambda d: gu(d, 'agr+jit', 'dirichlet0.3'))]}, {max(gu(d, 'agr+jit', 'dirichlet0.3') for d in SINGLE):.3f}) is ahead of every committee ({max(gu(d, 'agr+jit', 'dirichlet0.3') for d in COMM):.3f} at best).")
    R.append(table(pd.DataFrame(rows, columns=["Rule"] + [n for _, n in cols]),
                   f"Test accuracy (mean$\\pm$95\\% half-width over {n1} seeds), $K=20$, $f=4$ colluding omniscient attackers, i.i.d.\\ clients. ``pool'' rows: attacker does not know the random draw; only randomised committees are listed there (for deterministic rules the two knowledge levels coincide).",
                   "t:s3u", 3, None, None, False, "lrrrrr", True, True))
    # backdoor
    colsb = [("none_clean", "no attack"), ("agr_bd", "AGR-bd"), ("agr_bd[pool]", "AGR-bd, pool"), ("agr_bd+jit", "AGR-bd+jitter"), ("agr_bd+jit[pool]", "AGR-bd+jit., pool")]
    rows = []
    for d in ["median", "foolsgold"] + SINGLE + COMM:
        row = [NICE.get(d, d)]
        for s, _ in colsb:
            m = b[(b.defence == d) & (b.scenario == s) & (b.partition == "iid")]
            row.append(_pm(m.iloc[0], "asr") if len(m) else "--")
        rows.append(row)
    cb = min(COMM, key=lambda d: gb(d, "agr_bd"))
    R.append(
        f"For the backdoor (Table~\\ref{{t:s3b}}, {n2} seeds; ASR without any attacker is {gb('median', 'none_clean'):.3f} for the median) the idealised AGR-backdoor attacker with identical malicious rows gets ASR {gb('median', 'agr_bd'):.3f} against the median and {gb('foolsgold', 'agr_bd'):.3f} against FoolsGold, but {gb(cb, 'agr_bd'):.3f} to {max(gb(d, 'agr_bd') for d in COMM):.3f} against the committees: so the point estimates favour the median on this attack. The ASR intervals are very wide (the seed-to-seed spread of the backdoor outcome is large, and the no-attack ASR is already about 0.4 because the target class is common), so the intervals of median and committees overlap and this ordering is suggestive only. "
        f"Once the colluders' rows are jittered, Multi-Krum alone gives ASR {gb('multi_krum', 'agr_bd+jit'):.3f} and the committees {min(gb(d, 'agr_bd+jit') for d in COMM):.3f}--{max(gb(d, 'agr_bd+jit') for d in COMM):.3f}, i.e.\\ the committee matches but does not improve on its best member, while FoolsGold, which relies on similar updates, degrades to {gb('foolsgold', 'agr_bd+jit'):.3f}. "
        f"With the caveat that the intervals overlap, a pool-aware attacker obtains {gb('rrc_k2_v1', 'agr_bd[pool]'):.3f} and {gb('rrc_k3_v2', 'agr_bd[pool]'):.3f} against the two random committees, compared with {gb('rrc_k2_v1', 'agr_bd'):.3f} and {gb('rrc_k3_v2', 'agr_bd'):.3f} for a draw-aware one.")
    R.append(table(pd.DataFrame(rows, columns=["Rule"] + [n for _, n in colsb]),
                   f"Backdoor attack success rate (mean$\\pm$95\\% half-width over {n2} seeds), $K=20$, $f=4$, i.i.d.\\ clients.", "t:s3b", 3, None, None, False, "lrrrrr", True, True))
    # detection
    rows = []
    for d in SINGLE + COMM:
        row = [NICE[d]]
        for df, sc in ((u, "agr"), (u, "agr+jit"), (b, "agr_bd"), (b, "agr_bd+jit")):
            r = df[(df.defence == d) & (df.scenario == sc) & (df.partition == "iid")].iloc[0]
            row.append(f"{r.tpr:.2f}/{r.fpr:.2f}")
        rows.append(row)
    R.append(
        f"Detection rates (Table~\\ref{{t:s3d}}) show why the committee does not help against identical rows: all four members fail to flag the identical malicious rows (TPR $\\le$ {max(gu(d, 'agr', 'iid', 'tpr') for d in SINGLE + COMM):.2f} for the untargeted AGR attack) while the union rejects up to {100 * gu('rc_all_v1', 'agr', 'iid', 'fpr'):.0f}\\% of honest rows per round (all four members, $v{{\\ge}}1$). With jitter the union catches more malicious rows (TPR {gu('rc_all_v1', 'agr+jit', 'iid', 'tpr'):.2f}) at a higher honest-row cost (FPR {gu('rc_all_v1', 'agr+jit', 'iid', 'fpr'):.2f}); this is Proposition~1 at work, and the price is visible.")
    R.append(table(pd.DataFrame(rows, columns=["Rule", "AGR (untargeted)", "AGR+jitter", "AGR-bd", "AGR-bd+jitter"]),
                   "Share of malicious rows rejected / share of honest rows rejected (TPR/FPR, mean over rounds and seeds), i.i.d.\\ clients, $f=4$ of 20.", "t:s3d", 3, None, None, False, "lrrrr", True, True))
    R += ablations()
    R.append(
        r"\emph{Summary of Stage-3 evidence.} In this synthetic, omniscient-attacker setting a hand-built committee of four published rules did not beat the best of its members or the coordinate-wise median, and it cost accuracy under label skew; randomisation produced no measurable gain against a pool-aware attacker. The result we consider useful is the measurement itself: it shows where a union of rules adds false positives without adding detection, and how much the attacker's knowledge (and the identical-row signature of idealised attackers) changes the picture.")
    return R


def ablations() -> list:
    s = rd(ST, "s3_3_summary.csv")
    n = nseed("s3_3_runs.csv")
    R = []
    rows = []
    for ax, lab in (("k", "committee size $k$ (random draw, $v{=}1$)"), ("vote", "vote threshold $v$ (all four members)"), ("f", "attackers $f$ (defender assumes 4)")):
        for lvl in sorted(s[s.axis == ax].level.unique()):
            for atk in ("untargeted", "backdoor"):
                x = s[(s.axis == ax) & (s.level == lvl) & (s.attack == atk)]
                if ax == "f":
                    x = x[x.arm == "rrc_k2_v1"] if False else x
                key = "acc" if atk == "untargeted" else "asr"
                dr = x[x.knowledge == "draw"][key].mean(); po = x[x.knowledge == "pool"][key].mean() if (x.knowledge == "pool").any() else float("nan")
                rows.append([lab, f"{lvl}", atk, f"{dr:.3f}", "--" if pd.isna(po) else f"{po:.3f}"])
    R.append(
        f"Ablations ({n} seeds, averaged over the arms listed in the caption) are in Table~\\ref{{t:s3a}}. They are descriptive: with this few seeds we do not draw conclusions about small differences, and the attacker-fraction rows mix a deterministic committee (draw-aware) with a random one (pool-aware), so the two columns must not be read as a pure knowledge effect there.")
    R.append(table(pd.DataFrame(rows, columns=["Factor", "Level", "Attack", "draw-aware", "pool-aware"]),
                   f"Ablations. Untargeted rows report accuracy, backdoor rows report ASR (mean over {n} seeds). $k$ axis: random committee, $v{{=}}1$ (both knowledge levels). $v$ axis: all four members, deterministic (draw-aware only). $f$ axis: draw column = deterministic committee, pool column = random $k{{=}}2$ committee.",
                   "t:s3a", 3, None, None, False, "llrrr", True, True))
    return R
