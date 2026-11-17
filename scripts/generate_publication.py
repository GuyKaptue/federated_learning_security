"""Generates the scientific report (Markdown) and IEEE-style manuscript draft (LaTeX) from stored results."""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1]; O = R / "experiments/results"; PUB = R / "reports/manuscript_support"; PUB.mkdir(exist_ok=True)
refs = json.load(open(R / "docs/references/references.json")); rk = {r["key"]: r for r in refs}
def rd(s, f): return pd.read_csv(O / s / f)
def md(df, nd=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f": df[c] = df[c].round(nd)
    return "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values)
def tex(df, nd=3, cols=None, caption="", label=""):
    df = df.copy()
    if cols: df = df[cols]
    for c in df.columns:
        if df[c].dtype.kind == "f": df[c] = df[c].round(nd)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    body = " \\\\\n".join(" & ".join(esc(v) for v in r) for r in df.values)
    return ("\\begin{table}[t]\\centering\\caption{" + caption + "}\\label{" + label + "}\\scriptsize\n\\begin{tabular}{" + "l" * len(df.columns) + "}\\hline\n" +
            " & ".join(esc(c) for c in df.columns) + " \\\\\\hline\n" + body + " \\\\\\hline\\end{tabular}\\end{table}")

s15 = rd("fl_sec_001", "task_1_5_comparison_summary.csv")[["config", "acc", "train_overhead_pct", "comm_overhead_pct", "acc_loss_pp"]]
kpi_perf = rd("fl_perf_002", "task_2_5_kpi_check.csv")[["config", "n_clients", "training_overhead_pct", "target_lt_30", "comm_overhead_pct", "target_lt_40", "accuracy_loss_pp"]]
bd = rd("fl_attack_003", "task_3_1_summary.csv")[["level", "asr", "clean_acc"]]
mia = rd("fl_attack_003", "task_3_3_summary.csv")[["config", "auc", "utility_acc"]]
akpi = rd("fl_attack_003", "task_3_6_kpi.csv")[["kpi", "measured", "met"]]
lb = rd("fl_bench_004", "task_4_7_leaderboard_dir0.3.csv")[["algorithm", "utility_f1", "convergence_rounds", "communication_mb_to_target", "mean_rank"]]
ab = rd("fl_pub_005", "task_5_2_summary.csv")[["config", "acc", "train_overhead_pct", "acc_under_signflip", "backdoor_asr"]]
sig = rd("fl_bench_004", "task_4_6_pairwise_tests.csv"); nsig = int(sig["significant_holm_0.05"].sum())
repro = json.load(open(O / "fl_pub_005/task_5_1_reproducibility.json"))["checks"]
tun = rd("fl_bench_004", "task_4_8_summary.csv")[["algo", "setting", "lr", "mu", "f1", "f1_ci"]]
real = rd("fl_bench_004", "task_4_9_real_summary.csv"); real = real[real.setting != "default"][["dataset", "algo", "setting", "acc", "f1", "auc"]]
rank = rd("fl_bench_004", "task_4_9_real_ranking.csv"); rank = rank[rank.friedman_p.notna()][["dataset", "friedman_p"]]
adb = rd("fl_attack_003", "task_3_7_backdoor_summary.csv").pivot_table(index="defense", columns="attack", values="asr").reset_index()[["defense", "none_clean", "static_boost", "norm_matched", "minmax_backdoor"]]
adu = rd("fl_attack_003", "task_3_7_untargeted_summary.csv"); adu = adu[adu.dataset == "syn"].pivot_table(index="defense", columns="attack", values="acc").reset_index()[["defense", "none", "alie", "minmax"]]
rp = f"{sum(c['passed'] for c in repro)} of {len(repro)} reproducibility checks passed; failed: {[c['check'] for c in repro if not c['passed']]}"

report = f"""# Scientific report (DRAFT) — Secure Federated Learning Framework: cybersecurity validation in a simulated healthcare-motivated setting

**Status:** draft generated from stored results. Not peer reviewed; not claimed publication-ready. Reference status is per entry in REFERENCES_MASTER (41 of 48 metadata-verified on source pages; bodies not re-read; legal acts partially read). No patient data and no hospitals were involved: **virtual clients** on one machine, **synthetic** tabular data (SYN-IDS-v1). No regulatory compliance claim is made.

## 1. Summary
We implemented and measured three confidentiality mechanisms for federated aggregation (HE, secure aggregation, SMPC), four FL algorithms (FedAvg, FedProx, SCAFFOLD, FedNova), four attack families (backdoor, Byzantine, membership inference, gradient leakage) with defences, a scalability study at 10–500 virtual clients, a reproducibility audit and an ablation.

## 2. Key measured findings
1. **Accuracy is unaffected by confidentiality mechanisms; cost is not** (task 1.5):\n\n{md(s15, 2)}
2. **Overhead KPIs (<30% training, <40% communication) are mostly not met** in simulation (task 2.5):\n\n{md(kpi_perf, 1)}
3. **Backdoors succeed without defence; confidentiality tools do not stop them** (task 3.1 / 3.5):\n\n{md(bd)}
4. **Membership inference** is weak in the evaluated regime (task 3.3), reducing further with DP noise at utility cost:\n\n{md(mia)}
5. **KPI verification (attacks):**\n\n{md(akpi)}
6. **Algorithm benchmark under Dirichlet α=0.3** (task 4.7): SCAFFOLD and FedNova lead on F1 and rounds-to-target; FedProx (untuned μ) does not beat FedAvg. {nsig} pairwise comparisons survive Holm correction (task 4.6).\n\n{md(lb)}
7. **Ablation** (task 5.2):\n\n{md(ab)}
8. **Reproducibility:** {rp}.
9. **Baseline tuning (task 4.8)** — validation-selected lr/mu, synthetic Dirichlet 0.3, 10 seeds. Tuning narrows the gaps; no pairwise difference between tuned algorithms survives Holm correction, so the untuned ranking in item 6 is weaker than it looked (tuned FedProx gains about 1.8 pp F1; FedAvg's selected lr 0.8 is at the grid edge):\n\n{md(tun, 4)}
10. **Real public tables (task 4.9)** — WDBC, Pima diabetes, Cleveland heart; 5 virtual clients (synthetic partitions); nested tuning over 10 splits. Algorithms are statistically indistinguishable (Friedman p below), tuning did not help on average, CKKS/SecAgg reproduce plaintext parameters exactly:\n\n{md(real, 3)}\n\n{md(rank, 3)}
11. **Adaptive attacks (task 3.7; omniscient attacker)** — adaptive backdoors defeat defences that stopped the static one (ASR on synthetic data); untargeted ALIE/MIN-MAX did little damage in this setting:\n\n{md(adb)}\n\n{md(adu)}

## 3. Threats to validity
Mostly synthetic data (plus three small public tables with synthetic client partitions); virtual clients; two CPU cores; no network; adaptive attackers are omniscient upper bounds; a single trigger family; trusted-dealer SMPC; honest key-holder HE; semi-honest SecAgg without consistency round; only lr/mu tuned (boundary optimum for FedAvg); real-data test sets of 91-231 samples; percentage overheads are relative to a very cheap in-process baseline.

## 4. What is NOT shown
Behaviour on large real clinical data, real hospital heterogeneity or federations (only three small public tables with synthetic partitions were used), real networks, malicious-server SecAgg security, regulatory compliance, MP-SPDZ based MPC, GPU performance, complete-graph SecAgg at 500 clients.

## 5. Next steps
Larger real datasets with natural site splits (with approval); tuning beyond lr/mu; attacker adaptivity beyond ALIE/MIN-MAX; real transport and key management; formal security analysis.
"""
(PUB / "scientific_report.md").write_text(report)
(R / "docs/jira/fl_pub_005/final_report.md").write_text(report)

import re as _re


def bibauthors(a):
    parts = [x.strip() for x in _re.split(r"(?<=\.), ", a)]
    if len(parts) == 1 and not _re.search(r",\s*[A-Z]\.", a): return "{" + a + "}"
    return " and ".join("others" if x.startswith("et al") else x for x in parts)
cite = lambda *ks: "\\cite{" + ",".join(ks) + "}"
bib = []
_PROPER = {"scaffold": "SCAFFOLD", "scaffold:": "SCAFFOLD:", "idlg:": "iDLG:", "idlg": "iDLG", "adap": "ADAP", "eu": "EU", "nis2": "NIS2", "spdz": "SPDZ", "(spdz)": "(SPDZ)", "(nis2)": "(NIS2)", "(krum)": "(Krum)", "flame:": "FLAME:", "(signguard)": "(SignGuard)", "tenseal:": "TenSEAL:", "non-iid": "Non-IID", "(eu)": "(EU)", "(bdsg),": "(BDSG),", "dlg": "DLG", "ai": "AI", "(ai": "(AI"}
_KEEP = {"krum", "byzantine", "byzantine-robust", "byzantine:", "wisconsin", "pima", "indians", "cleveland", "(cleveland", "american", "detrano", "sybil", "sybils", "rényi", "bundesdatenschutzgesetz", "ieee", "uci", "(poly)logarithmic", "paillier", "shamir", "beaver", "gdpr", "(general", "regulation", "directive", "act)", "heart", "tails:", "white-box"}
def _title(t):
    out = []
    for i, w in enumerate(t.split()):
        lw = w.lower()
        if lw in _PROPER: w = _PROPER[lw]
        if lw in _PROPER or lw in _KEEP or (i > 0 and any(c.isupper() for c in _re.sub(r'-[A-Z]', '-x', w[1:]))):
            out.append("{" + w + "}")
        else:
            out.append(w)
    return " ".join(out)
for r in refs:
    bib.append("@misc{%s,\n  author={%s},\n  title={%s},\n  howpublished={%s},\n  note={%s}\n}" % (r["key"], bibauthors(r["authors"]), _title(r["title"]), r["venue"].replace("&", "\\&"), _re.sub(r"\s*\(page blocked by captcha\)", "", r["identifier"]).replace("_", "\\_")))
(PUB / "references.bib").write_text("\n\n".join(bib) + "\n")
print("publication files written (manuscript_ieee.tex is produced by generate_manuscript.py)")
