"""E3 baseline reproduction gate in quick mode, into a scratch root (never touches outputs/research)."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[3] / "src"


def test_e3_baseline_quick(tmp_path):
    env = dict(os.environ, RS_ROOT=str(tmp_path), RS_QUICK="1", PYTHONPATH=str(SRC))
    p = subprocess.run([sys.executable, "-m", "federated_learning.pipelines.experiments.run_all", "--task", "FL-E3-BASE", "--quick"],  # nosec B603
                       env=env, capture_output=True, text=True, timeout=2400, cwd=str(tmp_path))
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-1500:]
    out = list(tmp_path.rglob("e3_baseline_summary.csv"))
    assert out, "summary missing"
    d = pd.read_csv(out[0])
    assert {"none_clean", "static_boost", "norm_matched", "agr_backdoor"} <= set(d.scenario)
    assert d.asr.between(0, 1).all()
