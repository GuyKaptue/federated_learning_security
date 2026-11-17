"""Result-traceability sheet: manuscript table/figure -> source CSV -> producing task -> config hash, seeds, rows, timestamp.
Writes docs/traceability.md and exits non-zero if any source file is missing or a caption states a seed count that differs from the stored runs.
Run: python scripts/generate_traceability.py"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1]
RES = R / "experiments" / "results"
TEX = (R / "latex" / "manuscript_ieee.tex").read_text()

# label -> (story, summary csv, runs csv or None, provenance json, producing task, module function)
T = {
    "t:sec": ("fl_sec_001", "task_1_5_comparison_summary.csv", "task_1_5_comparison_runs.csv", "task_1_5_comparison.json", "FL-SEC-001.5", "pipelines/experiments/sec001.py::task_1_5_comparison"),
    "t:kpi": ("fl_perf_002", "task_2_5_kpi_check.csv", None, "task_2_5_scalability_analysis.json", "FL-PERF-002.2-2.5", "pipelines/experiments/perf002.py::task_2_{2,3,4,5}"),
    "t:bd": ("fl_attack_003", "task_3_1_summary.csv", "task_3_1_runs.csv", "task_3_1_backdoor.json", "FL-ATTACK-003.1", "attack003.py::task_3_1_backdoor"),
    "t:ab": ("fl_pub_005", "task_5_2_summary.csv", "task_5_2_runs.csv", "task_5_2_ablation.json", "FL-PUB-005.2", "pub005.py::task_5_2_ablation"),
    "t:lb": ("fl_bench_004", "task_4_7_leaderboard_dir0.3.csv", "task_4_5_runs.csv", "task_4_7_leaderboard.json", "FL-BENCH-004.5/4.7", "bench004.py"),
    "t:tun": ("fl_bench_004", "task_4_8_summary.csv", "task_4_8_runs.csv", "task_4_8_tuning.json", "FL-BENCH-004.8", "bench004.py"),
    "t:real": ("fl_bench_004", "task_4_9_real_summary.csv", "task_4_9_real_runs.csv", "task_4_9_real_data.json", "FL-BENCH-004.9", "bench004.py"),
    "t:A10": ("fl_bench_004", "task_4_10_A_summary.csv", "task_4_10_A_runs.csv", "task_4_10_natural_sites.json", "FL-BENCH-004.10", "ext.py"),
    "t:B10": ("fl_bench_004", "task_4_10_B_summary.csv", "task_4_10_B_runs.csv", "task_4_10_natural_sites.json", "FL-BENCH-004.10", "ext.py"),
    "t:rs": ("fl_bench_004", "task_4_11_summary.csv", "task_4_11_runs.csv", "task_4_11_random_search.json", "FL-BENCH-004.11", "ext2.py"),
    "t:adb": ("fl_attack_003", "task_3_7_backdoor_summary.csv", "task_3_7_backdoor_runs.csv", "task_3_7_adaptive_attacks.json", "FL-ATTACK-003.7", "attack003.py"),
    "t:wc": ("fl_attack_003", "task_3_8_worst_case.csv", "task_3_8_untargeted_runs.csv", "task_3_8_stronger_adaptive.json", "FL-ATTACK-003.8", "ext.py"),
    "t:fp": ("fl_stage1", "s1_1_summary.csv", "s1_1_runs.csv", "s1_1_federated_preprocessing.json", "FL-S1-001", "stage1.py::task_s1_1_federated_preprocessing"),
    "t:cal": ("fl_stage1", "s1_2_summary.csv", "s1_2_runs.csv", "s1_2_calibration.json", "FL-S1-002", "stage1.py::task_s1_2_calibration"),
    "t:mf": ("fl_stage1", "s1_3_summary.csv", "s1_3_runs.csv", "s1_3_ablation_grids.json", "FL-S1-003", "stage1.py::task_s1_3_ablation_grids"),
    "t:s2u": ("fl_stage2", "s2_1_summary.csv", "s2_1_runs.csv", "s2_1_untargeted_defences.json", "FL-S2-001", "stage2.py::task_s2_1_untargeted_defences"),
    "t:s2d": ("fl_stage2", "s2_1_summary.csv", "s2_1_runs.csv", "s2_1_untargeted_defences.json", "FL-S2-001 + FL-S2-002", "stage2.py::task_s2_1_untargeted_defences / task_s2_2_backdoor_defences"),
    "t:s2n": ("fl_stage2", "s2_1_summary.csv", "s2_1_runs.csv", "s2_1_untargeted_defences.json", "FL-S2-001 + FL-S2-002", "stage2.py::task_s2_1_untargeted_defences / task_s2_2_backdoor_defences"),
    "t:s2b": ("fl_stage2", "s2_2_summary.csv", "s2_2_runs.csv", "s2_2_backdoor_defences.json", "FL-S2-002", "stage2.py::task_s2_2_backdoor_defences"),
    "t:s3u": ("fl_stage3", "s3_1_summary.csv", "s3_1_runs.csv", "s3_1_untargeted.json", "FL-S3-001", "stage3.py::task_s3_1_untargeted"),
    "t:s3d": ("fl_stage3", "s3_1_summary.csv", "s3_1_runs.csv", "s3_1_untargeted.json", "FL-S3-001", "stage3.py::task_s3_1_untargeted (also s3_2 for backdoor columns)"),
    "t:s3b": ("fl_stage3", "s3_2_summary.csv", "s3_2_runs.csv", "s3_2_backdoor.json", "FL-S3-002", "stage3.py::task_s3_2_backdoor"),
    "t:s3a": ("fl_stage3", "s3_3_summary.csv", "s3_3_runs.csv", "s3_3_ablations.json", "FL-S3-003", "stage3.py::task_s3_3_ablations"),
    "t:s2a": ("fl_stage2", "s2_3_summary.csv", "s2_3_runs.csv", "s2_3_baselines.json", "FL-S2-003", "stage2.py::task_s2_3_baselines"),
    "t:al": ("fl_stage1", "s1_3_summary.csv", "s1_3_runs.csv", "s1_3_ablation_grids.json", "FL-S1-003", "stage1.py::task_s1_3_ablation_grids"),
}
caps = {m.group(2): m.group(1) for m in re.finditer(r"\\caption\{(.*?)\}\\label\{(t:[A-Za-z0-9]+)\}", TEX, flags=re.S)}
rows, problems = [], []
for lab, (story, summ, runs, prov, task, fn) in T.items():
    ok = all((RES / story / f).exists() for f in (summ, prov)) and (runs is None or (RES / story / runs).exists())
    if not ok:
        problems.append(f"{lab}: missing source file"); continue
    pj = json.loads((RES / story / prov).read_text()).get("provenance", {})
    cfg = pj.get("config", {})
    seeds = "n/a"
    if runs:
        d = pd.read_csv(RES / story / runs)
        col = next((c for c in ("seed", "split", "rep") if c in d), None)
        seeds = f"{d[col].nunique()} ({col})" if col else "?"
        stated = re.findall(r"(\d+) (?:seeds|splits)", caps.get(lab, ""))
        if stated and col and int(stated[0]) != d[col].nunique():
            problems.append(f"{lab}: caption says {stated[0]} but runs contain {d[col].nunique()}")
    rows.append((lab, story + "/" + summ, task, fn, (pj.get("config_sha256") or "")[:12], seeds, json.dumps({k: v for k, v in cfg.items() if k not in ("configs",)})[:90], (pj.get("environment", {}) or {}).get("timestamp_utc", "")))
figs = re.findall(r"includegraphics\[[^\]]*\]\{figures/([^}]+)\}", TEX)
out = ["# Result traceability", "", "Every manuscript table is generated by `scripts/generate_manuscript.py` from the stored result files below; nothing is typed by hand. "
       "Seed counts in captions are derived from the runs files and checked here.", "",
       "| Table | Source file (experiments/results/) | Task | Code | Config SHA-256 | Seeds/splits | Key config | Produced (UTC) |", "|---|---|---|---|---|---|---|---|"]
out += [f"| {a} | `{b}` | {c} | `{d}` | `{e}` | {f} | `{g}` | {h} |" for a, b, c, d, e, f, g, h in rows]
out += ["", "## Figures", ""] + [f"* `latex/figures/{f}` (copied from the matching `experiments/results/<story>/figures/` file)" for f in figs]
out += ["", "## Reproduce", "", "`python -m federated_learning.pipelines.experiments.run_all --task <FL-...>` then `python scripts/generate_manuscript.py` and `cd latex && ./build.sh`.",
        "Timing experiments use a seed-outer design with randomised configuration order and a warm-up run, and must be run on an otherwise idle machine."]
(R / "docs" / "traceability.md").write_text("\n".join(out) + "\n")
print(f"{len(rows)} tables traced; {len(problems)} problems")
for p in problems:
    print("PROBLEM:", p)
sys.exit(1 if problems else 0)
