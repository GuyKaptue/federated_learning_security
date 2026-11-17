"""Stage-1 experiment tasks run end-to-end in quick mode into a temporary results root (never touches experiments/results)."""
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
                          env=env, capture_output=True, text=True, timeout=900, cwd=str(root))


def test_ablation_grid_quick(tmp_path):
    p = _run("FL-S1-003", tmp_path)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-1500:]
    d = pd.read_csv(tmp_path / "experiments/results/fl_stage1/s1_3_runs.csv")
    assert set(d.axis) == {"malicious_fraction", "dirichlet_alpha", "participation", "hidden_units"}
    assert d.seed.nunique() == 3 and d[["acc", "f1", "auc"]].notna().all().all()
    t = pd.read_csv(tmp_path / "experiments/results/fl_stage1/s1_3_tests.csv")
    assert {"dz", "cliffs_delta", "holm_p"} <= set(t.columns)


@pytest.mark.parametrize("task,out", [("FL-S1-001", "s1_1_stat_error.csv"), ("FL-S1-002", "s1_2_summary.csv")])
def test_heart_dependent_tasks_quick(tmp_path, task, out):
    p = _run(task, tmp_path)
    if p.returncode != 0 and ("URLError" in p.stderr or "download" in p.stderr.lower() or "ConnectionError" in p.stderr):
        pytest.skip("heart-disease files could not be downloaded")
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-1500:]
    d = pd.read_csv(tmp_path / "experiments/results/fl_stage1" / out)
    if task == "FL-S1-001":
        assert d.max_abs_err_mean.max() < 1e-6 and d.max_abs_err_std.max() < 1e-6
    else:
        assert ((d.ece >= 0) & (d.ece <= 1)).all() and ((d.brier >= 0) & (d.brier <= 1)).all()


def test_contrast_helper_pairs_by_seed():
    from federated_learning.pipelines.experiments.stage1 import _contrasts
    rows = [dict(c="a", arm=a, seed=s, acc=0.5 + 0.1 * (a == "x") + 0.001 * s) for a in ("ref", "x") for s in range(6)]
    out = _contrasts(rows, ("c",), "arm", "ref", "acc")
    assert len(out) == 1 and out[0]["mean_diff"] == pytest.approx(0.1) and out[0]["cliffs_delta"] == pytest.approx(1.0)
    assert np.isfinite(out[0]["holm_p"])
