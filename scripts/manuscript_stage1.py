"""Stage-1 manuscript blocks (federated preprocessing, calibration, sensitivity grids). Every number comes from experiments/results/fl_stage1/."""
import pandas as pd

from manuscript_lib import O, figure, rd, table

ST = "fl_stage1"


def nseed(fname):
    return int(pd.read_csv(O / ST / fname)["seed"].nunique())


def _pm(m, lo, hi, nd=3):
    return f"{m:.{nd}f}$\\pm${(hi - lo) / 2:.{nd}f}"


def _ci(m, lo, hi, nd=3):
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def methods() -> str:
    return r"""\subsection{Federated preprocessing}
\label{sec:fp-method}
The benchmark loaders originally standardised features with the mean and standard deviation of the \emph{pooled} training rows, and imputed missing values with the pooled median. That presupposes a party that sees every row. We replaced it with two rounds of secure aggregation (the pairwise-mask protocol described above, with real key agreement and masks). In round one each client sends, per feature, the sum and the count of its observed values, and the server obtains the global mean; in round two each client sends the sum of squared deviations from that mean, and the server obtains the global variance. The two-pass form avoids the cancellation of $E[x^2]-\mu^2$. Missing values are imputed with the global mean (a median is not a sum and cannot be obtained this way). The server only sees the sum over clients; it still learns the aggregate itself, we add no differential privacy, a malicious client could distort the statistics by misreporting its sums, and with very few clients the aggregate reveals more about each of them. We compare four regimes: \emph{pooled mean} (a centralised oracle), \emph{federated} (this method), \emph{local} (each site uses only its own training statistics, also for its own test rows) and \emph{none}; on the heart sites we add the original pooled-median loader. Test rows never enter any statistic. Because the federation is simulated, the claim we can test is numerical equivalence to the pooled oracle and the effect on downstream accuracy, not network or trust properties.
\subsection{Calibration and sensitivity analyses}
We report the Brier score and the expected calibration error (ECE; 15 equal-width bins on the predicted probability of the positive class) at the final round, without post-hoc recalibration. Four one-factor-at-a-time grids vary the malicious fraction (0--40\% of $K=20$ clients, sign-flip updates scaled by 5, against the mean, coordinate median, trimmed mean and Krum), the Dirichlet concentration $\alpha\in[0.1,1]$ ($K=10$; four FL algorithms plus FedAvg with a robust aggregator), the participation rate (10--100\% of $K=20$, with and without an attack), and the hidden width of the MLP. For each factor level we report the mean test accuracy over seeds with a 95\% $t$-interval, and paired contrasts against a stated reference with Cohen's $d_z$, Cliff's $\delta$ and Holm-adjusted Wilcoxon $p$-values."""


def results() -> list:
    R = []
    n1, n2, n3 = nseed("s1_1_runs.csv"), nseed("s1_2_runs.csv"), nseed("s1_3_runs.csv")
    # ------------------------------------------------------------ federated preprocessing
    s = rd(ST, "s1_1_summary.csv"); e = rd(ST, "s1_1_stat_error.csv"); t = rd(ST, "s1_1_tests.csv")
    err = max(e.max_abs_err_mean.max(), e.max_abs_err_std.max())
    syn_e = e[e.dataset == "synthetic-shifted"].iloc[0]; hrt_e = e[e.dataset == "heart-4sites"].iloc[0]
    tt = lambda ds, fam, arm, met="f1": t[(t.dataset == ds) & (t.family == fam) & (t.arm == arm) & (t.metric == met)].iloc[0]  # noqa: E731
    R.append(r"\subsection{Federated preprocessing and calibration}\label{sec:fp}")
    R.append(r"\subsubsection{Federated mean and variance}")
    sl = tt("synthetic-shifted", "vs pooled_mean", "local"); hl = tt("heart-4sites", "vs pooled_mean", "local")
    hm = tt("heart-4sites", "vs pooled_mean", "pooled_median"); fp_ = tt("heart-4sites", "vs pooled_mean", "federated")
    R.append(
        f"The securely aggregated statistics match the pooled ones to within fixed-point precision: over {n1} synthetic federations and {n1} heart-disease splits the largest absolute deviation of any mean or standard deviation was {err:.1e}, "
        f"and downstream F1 was identical to the pooled-mean oracle (mean difference {fp_.mean_diff:+.4f} on the heart sites). The price is two secure-aggregation rounds: "
        f"{(syn_e.up_bytes + syn_e.down_bytes) / 1024:.0f}\\,KiB in total for 40 features and 10 clients ({syn_e.time_s * 1e3:.0f}\\,ms in our process), and {(hrt_e.up_bytes + hrt_e.down_bytes) / 1024:.0f}\\,KiB for 10 features and 4 sites (Table~\\ref{{t:fp}}). "
        f"All three contrasts below use the pooled-mean oracle as reference, so imputation is matched across the compared arms; only the pooled-median row differs in imputation. Whether federated (global) standardisation is \\emph{{better}} than local standardisation depends on the kind of heterogeneity. On SYN-TAB-40 with a per-client affine measurement shift that we injected ourselves, local standardisation removes the shift and wins by {sl.mean_diff:+.3f} F1 "
        f"($d_z={sl.dz:.1f}$, Holm $p={sl.holm_p:.3f}$ against the pooled mean). On the four natural heart-disease cohorts the order reverses: local standardisation loses {abs(hl.mean_diff):.3f} F1 ($d_z={hl.dz:.1f}$, Holm $p={hl.holm_p:.3f}$). "
        f"We have not isolated why; plausible but untested reasons are that prevalence and missingness differ between the sites, so that per-site standardisation also removes label-relevant signal, and that one site has very few negative cases. "
        f"Replacing the pooled median by the global mean changed F1 by {hm.mean_diff:+.4f} on the heart sites (Holm $p={hm.holm_p:.3f}$, $d_z={hm.dz:.1f}$); the headline site results elsewhere in this paper still use the original pooled-median loader.")
    rows = []
    names = {"pooled_median": "pooled median (original)", "pooled_mean": "pooled mean (oracle)", "federated": "federated (secure sums)", "local": "local only", "none": "none"}
    for ds, lab in (("synthetic-shifted", "SYN-TAB-40, shifted clients"), ("heart-4sites", "heart, 4 sites")):
        for m in ("pooled_median", "pooled_mean", "federated", "local", "none"):
            r = s[(s.dataset == ds) & (s["mode"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            d = "--" if m == "pooled_mean" else f"{tt(ds, 'vs pooled_mean', m).mean_diff:+.3f} ({tt(ds, 'vs pooled_mean', m).dz:.1f})"
            rows.append([lab, names[m], _ci(r.f1, r.f1_lo, r.f1_hi), f"{r.auc:.3f}", d])
    R.append(table(pd.DataFrame(rows, columns=["Data", "Standardisation", "F1 [95\\% CI]", "AUC", "$\\Delta$F1 vs pooled mean ($d_z$)"]),
                   f"Effect of how features are standardised (FedAvg, {n1} seeds or splits). $\\Delta$F1 is the paired mean difference to the pooled-mean oracle; $d_z$ is the paired effect size.",
                   "t:fp", 3, None, None, False, "llrrr", True, True))
    # ------------------------------------------------------------ calibration
    c = rd(ST, "s1_2_summary.csv"); ct = rd(ST, "s1_2_tests.csv")
    rows = []
    for ds, lab in (("synthetic", "SYN-TAB-40, $\\alpha=0.3$"), ("heart-4sites", "heart, 4 sites")):
        for a, an in (("fedavg", "FedAvg"), ("fedprox", "FedProx"), ("scaffold", "SCAFFOLD"), ("fednova", "FedNova")):
            r = c[(c.dataset == ds) & (c.algo == a)].iloc[0]
            tb = ct[(ct.dataset == ds) & (ct.arm == a) & (ct.metric == "brier")]
            d = "--" if a == "fedavg" else f"{tb.iloc[0].mean_diff:+.4f} ({tb.iloc[0].dz:.1f})"
            rows.append([lab, an, _ci(r.brier, r.brier_lo, r.brier_hi), _ci(r.ece, r.ece_lo, r.ece_hi), d])
    nsig = int((ct.holm_p < 0.05).sum())
    def _overlap(ds):
        d = c[c.dataset == ds]
        return bool((d.ece_lo.max() <= d.ece_hi.min()))
    ovl = "the 95\\% intervals of ECE share a common range on both data sets" if all(_overlap(x) for x in ("synthetic", "heart-4sites")) else "the 95\\% intervals of ECE do not all overlap"
    R.append(r"\subsubsection{Calibration}")
    R.append(
        f"Final-round calibration is moderate for all four algorithms: ECE lies between {c.ece.min():.3f} and {c.ece.max():.3f} and the Brier score between {c.brier.min():.3f} and {c.brier.max():.3f} (Table~\\ref{{t:cal}}, {n2} seeds). "
        f"Against FedAvg, {nsig} of {len(ct)} Brier/ECE contrasts survive Holm correction within their family of three; {ovl}. "
        f"The largest Brier difference is small in absolute terms ({ct.loc[ct.metric == 'brier', 'mean_diff'].abs().max():.4f}), so accuracy and F1 differences between the algorithms should not be read as calibration differences. "
        f"On the heart data the test set holds roughly 280 rows, so ECE is noisy; we did not apply temperature scaling or any other recalibration, and a deployed risk model would need it validated on local data.")
    R.append(table(pd.DataFrame(rows, columns=["Data", "Algorithm", "Brier [95\\% CI]", "ECE [95\\% CI]", "$\\Delta$Brier vs FedAvg ($d_z$)"]),
                   f"Calibration at the final round ({n2} seeds, no recalibration). Lower is better for both scores.", "t:cal", 3, None, None, False, "llrrr", True, True))
    R.append(figure(f"{ST}/figures/s1_2_reliability.png", "Reliability diagrams pooled over seeds (bins with at least five predictions).", "f:rel"))
    # ------------------------------------------------------------ sensitivity grids
    sm = rd(ST, "s1_3_summary.csv"); st = rd(ST, "s1_3_tests.csv"); runs = rd(ST, "s1_3_runs.csv")
    R.append(r"\subsubsection{Sensitivity to malicious fraction, heterogeneity, participation and model size}")
    mf = sm[sm.axis == "malicious_fraction"]
    piv = lambda d, col: d.pivot_table(index="level", columns="arm", values=col)  # noqa: E731
    mean_acc = piv(mf, "acc")["mean"]
    collapse = mean_acc[mean_acc < 0.6].index.min()
    at40 = {a: piv(mf, "acc")[a].loc[0.4] for a in ("median", "trimmed_mean", "krum")}
    clean = {a: piv(mf, "acc")[a].loc[0.0] for a in ("mean", "median", "trimmed_mean", "krum")}
    rows = []
    for lv in sorted(mf.level.unique()):
        row = [f"{lv:.2f}"]
        for a in ("mean", "median", "trimmed_mean", "krum"):
            r = mf[(mf.level == lv) & (mf.arm == a)].iloc[0]; row.append(_pm(r.acc, r.acc_lo, r.acc_hi))
        rows.append(row)
    tmf = st[(st.axis == "malicious_fraction") & (st.family == "aggregator vs mean")]
    nh = int((tmf.holm_p < 0.05).sum())
    pr = sm[sm.axis == "participation"]
    low = pr[pr.level == pr.level.min()]
    low_att = low[low.arm == "median_20pct_signflip"].iloc[0].acc
    low_cl = low[low.arm == "mean_clean"].iloc[0].acc
    full_att = pr[(pr.level == 1.0) & (pr.arm == "median_20pct_signflip")].iloc[0].acc
    R.append(
        f"With sign-flip updates scaled by 5, the plain mean degrades from {clean['mean']:.3f} accuracy without attackers to below 0.6 once {int(round(100 * collapse))}\\% of the clients are malicious (Table~\\ref{{t:mf}}). "
        f"Median and trimmed mean stay within {max(clean['median'] - at40['median'], clean['trimmed_mean'] - at40['trimmed_mean']):.3f} of their clean accuracy even at 40\\% malicious clients; Krum is flat at about {clean['krum']:.3f}, which is {clean['mean'] - clean['krum']:.3f} below the mean when nobody attacks. "
        f"This is one attack behaviour against rules that suit it; the adaptive attackers of Section~\\ref{{sec:adapt}} break the same rules. "
        f"{nh} of {len(tmf)} aggregator-versus-mean contrasts are significant after Holm correction ({n3} seeds); from 10\\% malicious clients on, the smallest paired effect size is $d_z={tmf[tmf.level >= 0.1].dz.abs().min():.1f}$.")
    R.append(table(pd.DataFrame(rows, columns=["Malicious fraction", "Mean", "Median", "Trimmed mean", "Krum"]),
                   f"Test accuracy (mean$\\pm$95\\% half-width over {n3} seeds) under sign-flip updates scaled by 5, $K=20$.", "t:mf", 3, None, None, False, "lrrrr", True, True))
    al = sm[sm.axis == "dirichlet_alpha"]
    pa = al.pivot_table(index="level", columns="arm", values="acc")
    gap_lo = pa.loc[0.1, "fedavg"] - pa.loc[0.1, "fedavg+median"]; gap_hi = pa.loc[1.0, "fedavg"] - pa.loc[1.0, "fedavg+median"]
    sa = st[(st.axis == "dirichlet_alpha") & (st.family == "strategy vs fedavg")]
    best_gain = sa[sa.arm.isin(["scaffold", "fednova", "fedprox"])].sort_values("mean_diff").iloc[-1]
    rows = []
    for lv in sorted(al.level.unique()):
        row = [f"{lv:g}"]
        for a in ("fedavg", "fedprox", "scaffold", "fednova", "fedavg+median", "fedavg+trimmed_mean"):
            r = al[(al.level == lv) & (al.arm == a)].iloc[0]; row.append(_pm(r.acc, r.acc_lo, r.acc_hi))
        rows.append(row)
    nal = int((sa.holm_p < 0.05).sum())
    R.append(
        f"Label skew matters more than the choice of algorithm (Table~\\ref{{t:al}}). Between $\\alpha=1$ and $\\alpha=0.1$ FedAvg loses {pa.loc[1.0, 'fedavg'] - pa.loc[0.1, 'fedavg']:.3f} accuracy, while the largest gain of any other strategy over FedAvg at a given $\\alpha$ is {best_gain.mean_diff:+.3f} ({best_gain.arm}, $\\alpha={best_gain.level:g}$, $d_z={best_gain.dz:.1f}$). "
        f"Robust aggregation has a price that grows with heterogeneity even without any attacker: the coordinate median costs {gap_hi:.3f} accuracy at $\\alpha=1$ and {gap_lo:.3f} at $\\alpha=0.1$, because honest but dissimilar clients look like outliers. "
        f"{nal} of {len(sa)} strategy-or-robust-rule-versus-FedAvg contrasts are significant after Holm correction.")
    R.append(table(pd.DataFrame(rows, columns=["$\\alpha$", "FedAvg", "FedProx", "SCAFFOLD", "FedNova", "FedAvg+median", "FedAvg+trimmed"]),
                   f"Test accuracy (mean$\\pm$95\\% half-width over {n3} seeds) by Dirichlet concentration $\\alpha$ (smaller = more skew), clean data, $K=10$.", "t:al", 3, None, None, False, "lrrrrrr", True, True))
    hd = runs[runs.axis == "hidden_units"].groupby("level").agg(acc=("acc", "mean"), comm=("comm_mb", "mean"), p=("n_params", "mean"))
    R.append(
        f"Participation matters for robust rules because the number of clients in a round shrinks. At a 10\\% participation rate only two of 20 clients train per round, and the coordinate median of two updates is simply their mean: with 20\\% sign-flip clients accuracy falls to {low_att:.3f} (clean mean at the same rate: {low_cl:.3f}); at full participation the median holds {full_att:.3f}. "
        f"Model size changes accuracy modestly ({hd.acc.iloc[0]:.3f} at {int(hd.p.iloc[0])} parameters, {hd.acc.iloc[-1]:.3f} at {int(hd.p.iloc[-1])}) while per-experiment communication grows linearly from {hd.comm.iloc[0]:.1f} to {hd.comm.iloc[-1]:.1f}\\,MB. "
        f"All four grids use $K\\le 20$ virtual clients on synthetic data, so they show the direction and rough size of effects, not rates for real deployments.")
    R.append(figure(f"{ST}/figures/s1_3_ablation_grids.png", f"One-factor sensitivity grids ({n3} seeds; bars are 95\\% $t$-intervals).", "f:s13", star=True))
    return R
