"""Stage-2 tasks end-to-end in quick mode, into a scratch results root (never touches experiments/results)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[3] / "src"


def _run(task: str, root: Path):
    env = dict(os.environ, RS_ROOT=str(root), RS_QUICK="1", PYTHONPATH=str(SRC))
    return subprocess.run([sys.executable, "-m", "federated_learning.pipelines.experiments.run_all", "--task", task, "--quick"],  # nosec B603
                          env=env, capture_output=True, text=True, timeout=1500, cwd=str(root))


@pytest.mark.parametrize("task,fname", [("FL-S2-001", "s2_1_summary.csv"), ("FL-S2-002", "s2_2_summary.csv")])
def test_defence_tasks_quick(tmp_path, task, fname):
    p = _run(task, tmp_path)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-1500:]
    d = pd.read_csv(tmp_path / "experiments/results/fl_stage2" / fname)
    assert {"flame", "flame_noise", "rlr", "foolsgold", "signguard", "mean", "median"} <= set(d.defence)
    key = "acc" if task == "FL-S2-001" else "asr"
    assert d[key].between(0, 1).all()
    runs = pd.read_csv(tmp_path / "experiments/results/fl_stage2" / fname.replace("summary", "runs"))
    det = runs[runs.defence.isin(["flame", "signguard", "foolsgold", "anomaly_filter"])]
    assert det.tpr.notna().all() and det.fpr.notna().all()            # rejecting rules report detection rates
    assert runs[runs.defence.isin(["mean", "median", "rlr"])].tpr.isna().all()


def test_baselines_quick(tmp_path):
    p = _run("FL-S2-003", tmp_path)
    if p.returncode != 0 and ("URLError" in p.stderr or "download" in p.stderr.lower() or "ConnectionError" in p.stderr):
        pytest.skip("heart-disease files could not be downloaded")
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-1500:]
    d = pd.read_csv(tmp_path / "experiments/results/fl_stage2/s2_3_summary.csv")
    assert set(d.arch) == {"logreg", "mlp", "deep_mlp"} and np.isfinite(d.f1).all()
