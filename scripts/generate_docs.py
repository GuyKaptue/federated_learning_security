"""Generates per-story documentation. All numbers come from experiments/results/<story>/ files at generation time."""
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(R / "src"))
from generate_notebooks import T

OUT = R / "experiments/results"

def tbl(df, n=40, nd=4):
    df = df.head(n).copy()
    for c in df.columns:
        if df[c].dtype.kind == "f": df[c] = df[c].round(nd)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): lines.append("| " + " | ".join(str(v).replace("|", "/") for v in r.values) + " |")
    return "\n".join(lines)
def csv(story, name, **k): return tbl(pd.read_csv(OUT / story / name), **k)

STORIES = {
 "fl_sec_001": dict(title="Cryptographic validation: HE, Secure Aggregation, SMPC", tables=[("Plaintext vs HE vs SecAgg vs SMPC (task 1.5, MEASURED, mean over seeds)", "task_1_5_comparison_summary.csv", ["config","acc","train_overhead_pct","comm_overhead_pct","acc_loss_pp","max_agg_err"]), ("SecAgg dropout (task 1.3)", "task_1_3_dropout.csv", None), ("SMPC by number of parties (task 1.4)", "task_1_4_parties.csv", None)],
   extra="""
### Acceptance verdicts (computed from the tables above)
* HE (CKKS/BFV/Paillier), SecAgg with dropout recovery and SMPC with MAC checks are **implemented and tested** (see `TESTS/`): MET.
* Accuracy loss < 1 pp: MET for every mechanism in the 1.5 table (`acc_loss_pp` column; the 95% CIs include 0).
* Training/communication overhead KPIs (< 30% / < 40%, from the Jira epic): **NOT MET** for HE and SMPC, and for SecAgg at the configuration measured (see `comm_overhead_pct`, `train_overhead_pct`). These are *measured simulation* costs with virtual clients.
* Security: correctness-to-plaintext is **not** a security proof. Server-view statistics (correlation, high-bit fraction) and abort/tamper tests are sanity checks only.
""",
   limits=["SecAgg: semi-honest server, no consistency round; SMPC: trusted-dealer offline phase, own implementation (MP-SPDZ unavailable); HE: honest key-holder; Flower secaggplus inspected not run end-to-end (NOT_RUN)."]),
 "fl_perf_002": dict(title="Performance and scalability at 10/50/100/500 virtual clients", tables=[("KPI check (task 2.5)", "task_2_5_kpi_check.csv", None), ("Scaling exponents (task 2.5)", "task_2_5_scaling_exponents.csv", None), ("Scalability table (task 2.5)", "task_2_5_scalability_table.csv", None)],
   extra="""
### Acceptance verdicts
* Monitoring of time, CPU, memory, communication with MLflow logging (SQLite backend): MET (GPU: not available, recorded as such).
* Benchmarks at 10, 50, 100, 500 **virtual** clients: MET, with the exception **complete-graph SecAgg at 500 clients = NOT_RUN** (a sparse k=14 graph was measured instead and is labelled as such).
* KPIs training overhead < 30%, communication overhead < 40%, accuracy loss < 1 pp: accuracy KPI MET everywhere; the other two are **NOT MET** in most rows of the KPI table above (the table is the verdict; rows with `True` are the exceptions).
* Cost estimates use labelled *illustrative* price assumptions (ESTIMATED), not quotes.
""",
   limits=["Virtual clients share one 2-core machine: wall time is serial; no network. Overheads are relative to a plain in-process baseline that is very cheap, which inflates percentage overheads."]),
 "fl_attack_003": dict(title="Attack simulation and defences", tables=[("Backdoor (task 3.1)", "task_3_1_summary.csv", ["level","asr","clean_acc","kpi_asr_lt_5pct"]), ("Membership inference (task 3.3)", "task_3_3_summary.csv", ["config","eps","auc","auc_loss_baseline","tpr_at_fpr_1pct","utility_acc"]), ("KPI verification (task 3.6)", "task_3_6_kpi.csv", None), ("Risk matrix (task 3.6)", "task_3_6_risk_matrix.csv", None),
   ("Adaptive untargeted attacks, synthetic data, K=20, f=4 (task 3.7; test accuracy)", "task_3_7_untargeted_summary.csv", ["defense","attack","acc","acc_ci","f1"], "dataset=='syn'"),
   ("Adaptive untargeted attacks, Cleveland heart disease, K=10, f=2 (task 3.7)", "task_3_7_untargeted_summary.csv", ["defense","attack","acc","acc_ci","f1"], "dataset=='heart'"),
   ("Static vs adaptive backdoor (task 3.7, synthetic)", "task_3_7_backdoor_summary.csv", ["defense","attack","asr","asr_ci","clean_acc"])],
   extra="""
### Acceptance verdicts
* Four attack families (backdoor, Byzantine, membership inference, gradient leakage) implemented with low/medium/high intensities, before/after defences: MET (Byzantine/MIA/DLG grids in `RESULTS/`).
* KPI *backdoor success < 5%*: **NOT MET** without defence (see 3.1 table); defended runs reach the clean-model trigger ASR with Multi-Krum/anomaly filtering/SMPC norm bound, but median/trimmed-mean/clipping do not in the high-intensity case (task 3.5 table).
* KPI *gradient leakage success < 10%*: **NOT MET** for plaintext small-batch updates; met only for aggregated/noised updates (task 3.4).
* KPI *membership inference AUC < 0.60*: met in the evaluated regime (synthetic, modest overfitting) – not a general claim.
* Secure aggregation and HE give **no** protection against poisoning (measured equal to the undefended ASR).

### Adaptive attacks (task 3.7; omniscient attacker, upper bound on knowledge)
* **Adaptive backdoors defeat the defences that stopped the static backdoor.** Static boosted backdoor under Multi-Krum / anomaly filter: ASR 0.106 (= clean-model level). Norm-matched backdoor: 0.557 / 0.498; MIN-MAX backdoor: 0.763 / 0.649; plain Krum: static 0.120 → MIN-MAX 0.996 (tables above). Median and trimmed mean hold ASR near 0.26-0.31 for all variants (above the 5% KPI, far below the undefended 0.99) at no measured clean-accuracy cost.
* **Untargeted ALIE and MIN-MAX did little damage in this small-model setting** (synthetic: ALIE ≈ 0 pp, MIN-MAX ≤ 2.5 pp; heart: up to 5.7 pp for Krum under MIN-MAX). The hypothesis "an adaptive untargeted attack breaks the robust aggregators" is **not supported** here; this may reflect an easy task and a small MLP, not general robustness.
* KPI *backdoor success < 5%* therefore remains **NOT MET** under adaptive attack for every defence tested.
""",
   limits=["Single trigger family, tabular data; adaptive attacks are omniscient (upper bound) and not optimised further per defence; DLG evaluated on sigmoid MLPs only."]),
 "fl_bench_004": dict(title="FedAvg / FedProx / SCAFFOLD / FedNova benchmark", tables=[("Leaderboard IID (task 4.7)", "task_4_7_leaderboard_iid.csv", ["algorithm","utility_f1","convergence_rounds","communication_mb_to_target","robustness_acc_under_signflip","training_cost_s","mean_rank"]), ("Leaderboard Dirichlet α=0.3 (task 4.7)", "task_4_7_leaderboard_dir0.3.csv", ["algorithm","utility_f1","convergence_rounds","communication_mb_to_target","robustness_acc_under_signflip","training_cost_s","mean_rank"]),
   ("Selected hyper-parameters (task 4.8; validation F1)", "task_4_8_selected_params.csv", ["algo","lr","mu","val_f1","at_lr_grid_edge"]),
   ("Tuned vs untuned, synthetic Dirichlet 0.3, 10 seeds (task 4.8)", "task_4_8_summary.csv", ["algo","setting","lr","mu","acc","f1","f1_ci","auc"]),
   ("Paired test tuned - default (task 4.8; uncorrected p)", "task_4_8_tuned_vs_default_tests.csv", ["algo","mean_diff","dz","wilcoxon_p","ttest_p"]),
   ("Pairwise tuned algorithms (task 4.8, Holm)", "task_4_8_tuned_pairwise.csv", ["a","b","mean_diff","wilcoxon_p","holm_p"]),
   ("REAL public tables: test performance, 10 outer splits, 5 virtual clients (task 4.9)", "task_4_9_real_summary.csv", ["dataset","algo","setting","acc","f1","f1_ci","auc"]),
   ("REAL data: tuned - default paired by split (task 4.9; uncorrected p)", "task_4_9_real_tuned_vs_default.csv", ["dataset","algo","mean_diff","wilcoxon_p"]),
   ("REAL data: Friedman across tuned algorithms (task 4.9)", "task_4_9_real_ranking.csv", ["dataset","friedman_stat","friedman_p","mean_f1_fedavg","mean_f1_fedprox","mean_f1_scaffold","mean_f1_fednova"], "friedman_p==friedman_p"),
   ("REAL data: secure-aggregation equivalence (task 4.9)", "task_4_9_real_secagg_summary.csv", None),
   ("REAL data: one sign-flip client of five (task 4.9)", "task_4_9_real_byzantine_summary.csv", None)],
   extra="""
### Acceptance verdicts
* Four algorithms implemented from the original papers and **validated** by identity tests (weighted aggregation, 1-client = local SGD, μ=0 ⇒ FedAvg, FD gradient check of the proximal term, SCAFFOLD invariant, FedNova equal-τ ⇒ FedAvg, closed-form quadratic toy): MET.
* Benchmark across IID/non-IID, scale, local epochs, partial participation, sign-flip attackers with 10 seeds (main conditions) or 5 seeds (others), Friedman + Wilcoxon + Holm + effect sizes: MET (task 4.6 table lists which differences survive Holm correction).
* Under IID data differences are tiny; under Dirichlet α=0.3 SCAFFOLD/FedNova lead FedAvg/FedProx on F1 and rounds-to-target in the **untuned** leaderboard (tasks 4.5-4.7).
* **Tuning (task 4.8)** shrinks that gap: tuned FedProx gains ≈ 1.8 pp F1 (paired Wilcoxon p = 0.006 uncorrected, d_z ≈ 1.0), FedAvg's selected lr is at the grid edge (0.8; unreliable), SCAFFOLD's default was already best; after Holm correction **no pairwise difference between tuned algorithms is significant** (10 seeds). The earlier "SCAFFOLD/FedNova beat FedAvg/FedProx" statement is therefore an untuned-baseline result and is **weaker than it looked**.
* **Real public tables (task 4.9; WDBC, Pima diabetes, Cleveland heart, nested tuning, 10 splits):** the four algorithms are statistically indistinguishable (Friedman p = 0.33 / 0.84 / 0.23); validation-based tuning did **not** improve test F1 on average (validation sets of only ~50-135 samples are too small to select reliably; mean tuned-minus-default F1 differences range from -0.029 to +0.012, and the single nominally significant one, FedNova on Pima, p = 0.039 uncorrected, is in the *wrong* direction). CKKS and SecAgg reproduce plaintext parameters exactly (max difference 0.0). One sign-flip client of five collapses plain averaging on WDBC (F1 0.38 vs 0.96 for median); Pima results under the attack are not informative (minority-class F1 unstable; see accuracy in the CSV).
* The "proposed framework" row is FedAvg + CKKS aggregation (not a novel algorithm): robustness **NOT_MEASURED**, hence unranked.
* Real-data caveat: clients are synthetic partitions of one small public table; Pima/Heart files come from hash-pinned third-party GitHub mirrors (licence of the mirrors not verified; UCI pages state CC BY 4.0 for WDBC and Heart). The heart table was de-duplicated (1 exact duplicate removed); Pima zeros in glucose/BP/skin/insulin/BMI were treated as missing and imputed with train medians.
""",
   limits=["Only lr and μ tuned; boundary optimum for FedAvg lr; real datasets tiny (test sets 91-231 samples); sign-flip robustness for SCAFFOLD/FedNova uses plain averaging of corrected updates and is not a defence claim."]),
 "fl_pub_005": dict(title="Reproducibility, ablation, threat model and publication draft", tables=[("Ablation (task 5.2)", "task_5_2_summary.csv", ["config","acc","train_overhead_pct","acc_under_signflip","backdoor_asr","server_sees_individual_updates"]), ("Incompatible combinations (task 5.2)", "task_5_2_incompatible.csv", None)],
   extra="""
### Acceptance verdicts
* Reproducibility audit executed (task 5.1): determinism, re-execution, provenance-hash and consistency checks pass; **git commit identifier unavailable** (check fails by design) → SHA-256 manifest used instead.
* Ablation with 9 feasible configurations plus documented incompatible combinations (NOT_RUN with reason): MET.
* Threat model and healthcare context documents written; **no compliance claim**; legal sources are `REFERENCE_NOT_VERIFIED`.
* Journal-style report and IEEE draft: **drafts**, not submission-ready (see `PUBLICATION/`).
""",
   limits=["Gradient-leakage and MIA columns in the ablation are reused from tasks 3.3/3.4."]),
}

STORIES["fl_attack_003"]["tables"] += [
   ("Stronger adaptive attacks, worst case per rule (task 3.8)", "task_3_8_worst_case.csv", None),
   ("Stronger untargeted attacks, synthetic K=20 f=4 (task 3.8; test accuracy)", "task_3_8_untargeted_summary.csv", ["defense","attack","acc","acc_ci","f1"], "dataset=='syn'"),
   ("AGR-aware backdoor (task 3.8, synthetic)", "task_3_8_backdoor_summary.csv", ["defense","attack","asr","asr_ci","clean_acc"]),
   ("AGR-tailored vs strongest fixed attack, paired by seed (task 3.8; uncorrected p)", "task_3_8_agr_vs_best_fixed.csv", ["dataset","defense","best_fixed_attack","mean_diff","ttest_p","wilcoxon_p"])]
STORIES["fl_attack_003"]["extra"] += """
### Task 3.8 verdicts (stronger, rule-aware attackers; computed from the tables above)
* An attacker that simulates the server rule (AGR-tailored) drops the **undefended mean** to 0.37 accuracy on synthetic data and 0.37 on WDBC (below the majority-class rate; 5 of 5 WDBC runs and 1 synthetic f=8 run ended with non-finite weights, scored as constant class-0 predictors).
* Against the robust rules at f=4 of 20 the worst-case accuracy drop is small (about 0.02-0.06); at f=8 of 20 it grows to 0.06 (median/trimmed mean) up to 0.27 (Krum): none of the tested rules is robust at 40% attackers.
* AGR-tailored is the strongest attack for most rules but not for median/trimmed mean at f=4, where Fang's directed deviation is as strong.
* AGR-aware backdoor: ASR >= 0.75 for mean, Krum, Multi-Krum, clipping and the anomaly filter; median/trimmed mean stay near 0.25-0.31 (clean-model ASR ~0.11).
* The attacker is an omniscient, rule-aware upper bound that searches a fixed candidate set; no learning-based attacker was implemented. 5 seeds only.
"""
STORIES["fl_bench_004"]["tables"] += [
   ("Natural sites, protocol A: pooled test, 10 splits (task 4.10)", "task_4_10_A_summary.csv", ["algo","setting","acc","f1","f1_ci","auc","bacc"]),
   ("Natural sites: baselines (task 4.10)", "task_4_10_A_baselines_summary.csv", ["method","acc","f1","auc","bacc"]),
   ("Natural sites, protocol A: per-site test metrics, tuned (task 4.10)", "task_4_10_A_site_summary.csv", ["algo","site","acc","f1","auc"]),
   ("Natural sites: tuned - default paired by split (task 4.10; uncorrected p)", "task_4_10_A_tuned_vs_default_tests.csv", ["algo","mean_diff","dz","wilcoxon_p","ttest_p"]),
   ("Natural sites: pairwise tuned algorithms (task 4.10, Holm)", "task_4_10_A_pairwise.csv", ["a","b","mean_diff","wilcoxon_p","holm_p"]),
   ("Natural sites, protocol B (leave-one-site-out): macro over held-out sites (task 4.10)", "task_4_10_B_macro.csv", None),
   ("Natural sites, protocol B: per held-out site (task 4.10)", "task_4_10_B_summary.csv", ["held_out","algo","setting","acc","f1","auc","bacc"]),
   ("Natural sites: secure-aggregation equivalence (task 4.10)", "task_4_10_secagg_equivalence.csv", None),
   ("Broader tuning: test metrics for default / grid / random search (task 4.11)", "task_4_11_summary.csv", ["dataset","algo","setting","acc","f1","f1_ci","auc"]),
   ("Broader tuning: paired tests (task 4.11; uncorrected p)", "task_4_11_tests.csv", ["dataset","algo","contrast","mean_diff","wilcoxon_p","ttest_p"])]
STORIES["fl_bench_004"]["extra"] += """
### Task 4.10 verdicts (natural sites: Cleveland / Hungary / Switzerland / VA, n=920; computed from the tables above)
* With the four natural cohorts as clients, federated training matches pooled centralised training (pooled-test F1 about 0.80-0.82 vs 0.814) and clearly beats local-only models (0.759).
* SCAFFOLD/FedNova are slightly ahead of FedAvg/FedProx (Friedman p = 0.014 over tuned algorithms) but **no pairwise difference survives Holm correction** (smallest adjusted p 0.16).
* Validation-based tuning did **not** improve test F1 for any algorithm (all mean differences -0.003 to -0.010, uncorrected p >= 0.09).
* Leave-one-site-out: macro AUC about 0.79-0.80 for all algorithms; transfer to Switzerland and VA (AUC about 0.70-0.72) is much worse than to Cleveland/Hungary (0.85-0.88). Seeds replicate only initialisation and only four sites exist: descriptive only.
* Secure aggregation reproduces plaintext parameters to float rounding (max abs difference about 1.5e-7, identical F1). Switzerland has 8 negatives (positive rate 0.93): F1/accuracy there are inflated; use AUC/balanced accuracy.
* Scope: four 1980s public research cohorts, virtual clients in one process, pooled preprocessing statistics; not a live federation; no compliance claim.
"""

STORIES["fl_bench_004"]["extra"] += """
### Task 4.11 verdicts (24-configuration random search over lr, mu, batch size, local epochs, hidden width, SCAFFOLD server step; computed from the tables above)
* Random search vs the lr/mu grid: mean F1 differences are between -0.011 and +0.018; only FedNova on synthetic data reaches uncorrected p = 0.049 (n = 10 seeds); with 24 comparisons about one such result is expected by chance, so **no reliable advantage of broader tuning is shown**.
* Tuning can hurt: SCAFFOLD's default (F1 0.873) beat both grid (0.801) and random (0.814) selections on synthetic data (differences not significant, n = 10); selection on a 25% validation split does not reliably transfer to test.
* Natural sites (heart, 5 splits): every default/grid/random difference is <= 0.011 F1 and none is significant (Wilcoxon cannot reach p < 0.05 with 5 pairs anyway).
* FedAvg's random-search learning rate (0.78) is again near the top of the search range: its optimum is not established.
* Budgets are matched in number of configurations, not compute; 24 configurations is a small search; no best-algorithm claim is made from this experiment.
"""

def derived():
    b = pd.read_csv(OUT / "fl_bench_004/task_4_9_real_byzantine.csv")
    g = b.groupby(["dataset", "aggregator"])[["acc", "f1"]].mean().reset_index(); g.to_csv(OUT / "fl_bench_004/task_4_9_real_byzantine_summary.csv", index=False)
    e = pd.read_csv(OUT / "fl_bench_004/task_4_9_real_secagg_equivalence.csv")
    g = e.groupby(["dataset", "mechanism"]).agg(f1=("f1", "mean"), acc=("acc", "mean"), max_param_diff_vs_plain=("max_abs_param_diff_vs_plain", "max"), n=("split", "count")).reset_index(); g.to_csv(OUT / "fl_bench_004/task_4_9_real_secagg_summary.csv", index=False)
    t = pd.read_csv(OUT / "fl_bench_004/task_4_8_tuning_grid.csv"); best = t.loc[t.groupby("algo")["val_f1"].idxmax()].copy(); best["at_lr_grid_edge"] = best["lr"] == t["lr"].max(); best.to_csv(OUT / "fl_bench_004/task_4_8_selected_params.csv", index=False)

def main():
    derived()
    for story, meta in STORIES.items():
        d = R / "docs/jira" / story; d.mkdir(parents=True, exist_ok=True)
        tasks = [t for t in T if t["story"] == story]
        refs = sorted({r for t in tasks for r in t["refs"]})
        (d / "README.md").write_text(f"# {story} — {meta['title']}\n\nPart of EPIC-001. Contents: `literature_review.md`, `mathematical_foundations.md`, `implementation.md`, `experimental_methodology.md`, `results_and_discussion.md`, `references.md`. Notebooks: `notebooks/sprint_tasks/{story}/`. Results: `experiments/results/{story}/`.\n\n**Scope labels:** all clients are virtual (single process); data is synthetic (SYN-IDS-v1); no patient data; no compliance claim.\n\n## Tasks\n" + "\n".join(f"- **{t['id']}** — {t['title']}" for t in tasks) + "\n")
        (d / "literature_review.md").write_text(f"# Literature review — {story}\n\nSource status is per entry in REFERENCES_MASTER.md (most are `REFERENCE_NOT_VERIFIED`; no access to publisher hosts); statements summarise well-known results in the author's words and must be checked against the papers.\n\n" + "\n\n".join(f"## {t['id']} {t['title']}\n{t['background']}" for t in tasks) + "\n\nKeys used: " + ", ".join(f"`{r}`" for r in refs) + "\n")
        (d / "mathematical_foundations.md").write_text(f"# Mathematical foundations — {story}\n\n" + "\n\n".join(f"## {t['id']} {t['title']}\n{t['math']}" for t in tasks) + "\n")
        (d / "implementation.md").write_text(f"# Implementation — {story}\n\n" + "\n\n".join(f"## {t['id']} {t['title']}\n**Objective.** {t['objective']}\n\n**Modules.** {t['modules']}\n\n**Notebook.** `notebooks/sprint_tasks/{story}/{t['id'].replace('-', '_').replace('.', '_')}.ipynb`\n\n**Experiment entry point.** `python -m federated_learning.pipelines.experiments.run_all --task {t['id']}`" for t in tasks) + "\n")
        (d / "experimental_methodology.md").write_text(f"# Experimental methodology — {story}\n\n* Data: synthetic SYN-IDS-v1 (40 features, binary, 70:30); IID and Dirichlet partitions; **virtual clients**.\n* Seeds fixed; unit of replication = seed; 95% CIs (t/bootstrap); non-parametric tests where applicable (Wilcoxon needs ≥6 seeds for p<0.05).\n* Provenance in every result JSON (experiment id, config SHA-256, environment, no git commit).\n* Labels: MEASURED / SIMULATED / ESTIMATED / THEORETICAL / NOT_RUN.\n* Hardware: 2 CPU cores, no GPU. Timing experiments were run alone on the machine.\n\n## Research questions and hypotheses\n" + "\n".join(f"### {t['id']}\n" + "\n".join(f"- RQ: {q}" for q in t["rqs"]) + "\n" + "\n".join(f"- H: {h}" for h in t["hyps"]) for t in tasks) + "\n")
        body = [f"# Results and discussion — {story}: {meta['title']}\n", "All numbers below are read from the stored result files (MEASURED unless stated; virtual clients; synthetic data).\n"]
        for name, f, cols, *q in [tuple(t) + (None,) * (4 - len(t)) if len(t) < 4 else t for t in meta["tables"]]:
            df = pd.read_csv(OUT / story / f)
            if q and q[0]: df = df.query(q[0])
            if cols: df = df[[c for c in cols if c in df.columns]]
            body += [f"## {name}\n", tbl(df), ""]
        body += [meta["extra"], "## Limitations\n" + "\n".join(f"- {l}" for l in meta["limits"] + ["Synthetic data, virtual clients, 2 CPU cores, no GPU, no real network."]), ""]
        (d / "results_and_discussion.md").write_text("\n".join(body))
    # story-specific extras
    (R/"docs/jira/fl_perf_002/metrics_definitions.md").write_text((R/"src/federated_learning/monitoring/resources.py").read_text().split('"""')[1].replace("see docs/fl_perf_002/metrics_definitions.md", "this file") + "\n")
    (R/"docs/jira/fl_attack_003/attack_methodology.md").write_text("""# Attack and defence methodology
Threat: malicious clients (backdoor/Byzantine), curious observers (MIA, gradient leakage). Intensity levels: **backdoor** low/medium/high = increasing attacker fraction, poison fraction and boosting (see `experiments/attack003.py` constants); **Byzantine** malicious fraction 10/20/30% for random, sign-flip, scale, drift; **MIA** none/clip/DP σ∈{0.1,0.3,1.0}; **DLG** batch sizes and hidden widths. Trigger selected among candidates by lowest clean-model ASR so that the KPI can be interpreted; both absolute and clean-model ASR are reported. Defences: median, trimmed mean, Krum/Multi-Krum (n ≥ 2f+3), clipping, Gaussian noise, anomaly filter, SMPC norm bound, client-side DP. Secure aggregation/HE are evaluated as 'defences' to show they give no poisoning robustness.
""")
    (R/"docs/jira/fl_bench_004/algorithm_implementations.md").write_text("""# Algorithm implementations
FedAvg, FedProx (proximal term), SCAFFOLD (option II control variates, 2x communication), FedNova (normalised averaging) are implemented in `src/federated_learning/fl.py` from the original papers (see per-entry reference status) and validated by the identity tests listed in `results_and_discussion.md` and `tests/unit/algorithms/`. No third-party FL library implements these here; Flower was not used for training.
""")
    for n, txt in {"statistical_methodology.md": "Paired comparisons across seeds (Wilcoxon signed-rank, paired t, Holm correction, Cohen d_z, bootstrap CIs, Friedman). Minimum attainable Wilcoxon p = 2/2^n ⇒ 10 seeds in main conditions. Seeds are the unit of replication; rounds within a run are not independent.",
                   "ablation_study.md": "See `docs/fl_pub_005/results_and_discussion.md` (ablation table) and incompatible-combination table. Confidentiality layers (HE/SecAgg/SMPC) leave robustness unchanged; clipping/robust aggregation/SMPC norm bound change it; combining SecAgg/HE with server-side robust statistics is infeasible without further protocol design (NOT_RUN).",
                   "reproducibility.md": "Run: `pip install -r requirements.txt; PYTHONPATH=src python -m federated_learning.pipelines.experiments.run_all --task all`; tests: `pytest tests`. Determinism verified on this stack (task 5.1). Wall-clock numbers vary by machine.",
                   "publication_methodology.md": "Report and manuscript are generated from stored results; references carry per-entry verification status and unverified ones are flagged. The draft is not claimed to be publication-ready."}.items():
        (R/"docs/jira/fl_pub_005"/n).write_text(f"# {n[:-3].replace('_',' ').title()}\n\n{txt}\n")
    print("docs written")
main()
