"""CLI: python -m federated_learning.pipelines.experiments.run_all --task <id|story|all> [--quick]"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback

REGISTRY = {
    "FL-SEC-001.1": ("sec001", "task_1_1_selection"), "FL-SEC-001.2": ("sec001", "task_1_2_he"),
    "FL-SEC-001.3": ("sec001", "task_1_3_secagg"), "FL-SEC-001.4": ("sec001", "task_1_4_smpc"),
    "FL-SEC-001.5": ("sec001", "task_1_5_comparison"),
    "FL-PERF-002.1": ("perf002", "task_2_1_monitoring"), "FL-PERF-002.2": ("perf002", "task_2_2_benchmark_10"),
    "FL-PERF-002.3": ("perf002", "task_2_3_benchmark_50"), "FL-PERF-002.4": ("perf002", "task_2_4_benchmark_100_500"),
    "FL-PERF-002.5": ("perf002", "task_2_5_scalability_analysis"),
    "FL-ATTACK-003.1": ("attack003", "task_3_1_backdoor"), "FL-ATTACK-003.2": ("attack003", "task_3_2_byzantine"),
    "FL-ATTACK-003.3": ("attack003", "task_3_3_membership_inference"), "FL-ATTACK-003.4": ("attack003", "task_3_4_gradient_leakage"),
    "FL-ATTACK-003.5": ("attack003", "task_3_5_countermeasures"), "FL-ATTACK-003.6": ("attack003", "task_3_6_risk_analysis"),
    "FL-BENCH-004.1": ("bench004", "task_4_1_fedavg"), "FL-BENCH-004.2": ("bench004", "task_4_2_fedprox"),
    "FL-BENCH-004.3": ("bench004", "task_4_3_scaffold"), "FL-BENCH-004.4": ("bench004", "task_4_4_fednova"),
    "FL-BENCH-004.5": ("bench004", "task_4_5_comparison"), "FL-BENCH-004.6": ("bench004", "task_4_6_statistics"),
    "FL-BENCH-004.7": ("bench004", "task_4_7_leaderboard"),
    "FL-BENCH-004.8": ("ext", "task_4_8_tuning"), "FL-BENCH-004.9": ("ext", "task_4_9_real_data"), "FL-ATTACK-003.7": ("ext", "task_3_7_adaptive_attacks"),
    "FL-BENCH-004.10": ("ext2", "task_4_10_natural_sites"), "FL-BENCH-004.11": ("ext2", "task_4_11_random_search"), "FL-ATTACK-003.8": ("ext2", "task_3_8_stronger_adaptive"),
    "FL-S1-001": ("stage1", "task_s1_1_federated_preprocessing"), "FL-S1-002": ("stage1", "task_s1_2_calibration"),
    "FL-S1-003": ("stage1", "task_s1_3_ablation_grids"),
    "FL-S2-001": ("stage2", "task_s2_1_untargeted_defences"), "FL-S2-002": ("stage2", "task_s2_2_backdoor_defences"),
    "FL-S2-003": ("stage2", "task_s2_3_baselines"),
    "FL-S3-001": ("stage3", "task_s3_1_untargeted"), "FL-S3-002": ("stage3", "task_s3_2_backdoor"), "FL-S3-003": ("stage3", "task_s3_3_ablations"),
    "FL-E3-BASE": ("e3", "task_e3_baseline"),
    "FL-E3-FGVAL": ("e3_fg", "task_e3_fg_validation"),
    "FL-PUB-005.1": ("pub005", "task_5_1_reproducibility"), "FL-PUB-005.2": ("pub005", "task_5_2_ablation"),
}

def run(task: str):
    mod, fn = REGISTRY[task]
    f = getattr(importlib.import_module(f"federated_learning.pipelines.experiments.{mod}"), fn)
    t = time.time()
    try:
        f()
        print(f"[{task}] OK in {time.time() - t:.1f}s", flush=True)
        return True
    except Exception:
        traceback.print_exc()
        print(f"[{task}] FAILED", flush=True)
        return False

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        os.environ["RS_QUICK"] = "1"
    keys = [k for k in REGISTRY if a.task in ("all", k) or k.startswith(a.task)]
    if not keys:
        sys.exit(f"unknown task {a.task}")
    sys.exit(0 if all(run(k) for k in keys) else 1)
