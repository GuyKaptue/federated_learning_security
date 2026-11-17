"""Stage-3 tasks end-to-end in quick mode, into a scratch results root (never touches experiments/results)."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[3] / "src"


def _run(task: str, root: Path):
    env = dict(os.environ, RS_ROOT=str(root), RS_QUICK="1", PYTHONPATH=str(SRC))
    return subprocess.run([sys.executable, "-m", "federated_learning.pipelines.experiments.run_all", "--task", task, "--quick"],  # nosec B603
                          env=env, capture_output=True, text=True, timeout=2400, cwd=str(root))


@pytest.mark.parametrize("task,fname,key", [("FL-S3-001", "s3_1_summary.csv", "acc"), ("FL-S3-002", "s3_2_summary.csv", "asr")])
def test_committee_tasks_quick(tmp_path, task, fname, key):
    p = _run(task, tmp_path)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-1500:]
    d = pd.read_csv(tmp_path / "experiments/results/fl_stage3" / fname)
    assert {"rc_all_v1", "rc_all_v2", "rrc_k2_v1", "rrc_k3_v2", "median"} <= set(d.defence)
    assert d[key].between(0, 1).all()
    assert d.scenario.str.contains(r"\[pool\]").any()


def test_ablations_quick(tmp_path):
    p = _run("FL-S3-003", tmp_path)
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-1500:]
    d = pd.read_csv(tmp_path / "experiments/results/fl_stage3/s3_3_summary.csv")
    assert {"k", "vote", "f"} <= set(d.axis)
