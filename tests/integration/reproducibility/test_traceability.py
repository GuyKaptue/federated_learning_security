"""Every manuscript table must trace to stored result files, and seed counts in captions must equal the stored runs."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_manuscript_tables_trace_to_results():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_traceability.py")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
