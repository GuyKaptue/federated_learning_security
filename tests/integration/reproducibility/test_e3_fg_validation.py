"""E3 Phase 4A task in quick mode into a scratch root (never touches outputs/research)."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[3] / "src"


def test_e3_fg_validation_quick(tmp_path):
    env = dict(os.environ, RS_ROOT=str(tmp_path), RS_QUICK="1", PYTHONPATH=str(SRC))
    p = subprocess.run([sys.executable, "-m", "federated_learning.pipelines.experiments.run_all", "--task", "FL-E3-FGVAL", "--quick"],  # nosec B603
                       env=env, capture_output=True, text=True, timeout=2400, cwd=str(tmp_path))
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-1500:]
    d = pd.read_csv(next(tmp_path.rglob("e3_fg_summary.csv")))
    assert {"legacy", "paper_soft", "paper_hard", "allfeat_corrected"} == set(d.arm)
    assert d.asr_mean.between(0, 1).all()
    # the diagnostic arm differs from legacy only in the alpha=0 edge case, which does not occur in generic runs
    a = d[d.arm == "legacy"].set_index(["partition", "scenario"]).asr_mean
    b = d[d.arm == "allfeat_corrected"].set_index(["partition", "scenario"]).asr_mean
    assert (a - b).abs().max() < 1e-9
