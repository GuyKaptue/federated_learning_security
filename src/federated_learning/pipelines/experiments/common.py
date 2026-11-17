"""Shared experiment infrastructure: output locations, provenance, saving, figures, markdown tables."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess  # nosec B404
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

ROOT = Path(os.environ.get("RS_ROOT", Path(__file__).resolve().parents[4]))

OUT = ROOT / "experiments" / "results"

QUICK = os.environ.get("RS_QUICK", "0") == "1"       # tiny sizes for CI / unit tests

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

def out_dir(story: str) -> Path:
    p = OUT / story
    (p / "figures").mkdir(parents=True, exist_ok=True)
    return p

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)

def save_json(story: str, name: str, obj) -> Path:
    p = out_dir(story) / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=_json_default))
    return p

def load_json(story: str, name: str):
    p = out_dir(story) / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

def save_csv(story: str, name: str, rows: list[dict]) -> Path:
    import csv
    p = out_dir(story) / f"{name}.csv"
    if not rows:
        p.write_text("")
        return p
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, default=_json_default) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    return p

def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],  # nosec B603 B607
                                       stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        return None

def env_info() -> dict:
    import importlib.metadata as md
    vers = {}
    for m in ("numpy", "scipy", "scikit-learn", "torch", "tenseal", "phe", "flwr", "mlflow", "opacus",
              "cryptography", "psutil", "matplotlib", "pandas"):
        try:
            vers[m] = md.version(m)
        except Exception:
            vers[m] = None
    return dict(python=sys.version.split()[0], platform=platform.platform(), cpu_count=os.cpu_count(),
                versions=vers, timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))

def provenance(task_id: str, config: dict) -> dict:
    h = hashlib.sha256(json.dumps(config, sort_keys=True, default=_json_default).encode()).hexdigest()
    return dict(experiment_id=f"{task_id}-{h[:10]}", task_id=task_id, config=config, config_sha256=h,
                git_commit=git_commit(), dataset_id="SYN-IDS-v1", environment=env_info(),
                execution_command=f"python -m federated_learning.pipelines.experiments.run_all --task {task_id}",
                result_labels="MEASURED unless stated; clients are virtual (single machine)")

def savefig(fig, story: str, name: str, prov: dict | None = None, note: str = "") -> Path:
    p = out_dir(story) / "figures" / f"{name}.png"
    foot = f"{prov['experiment_id'] if prov else ''}  data: SYN-IDS-v1 (synthetic)  {note}".strip()
    fig.text(0.01, 0.005, foot, fontsize=6, color="gray", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(p)
    plt.close(fig)
    return p

def md_table(rows: list[dict], cols: list[str] | None = None, floatfmt: str = "{:.4g}") -> str:
    if not rows:
        return "_no data_"
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))

    def f(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines)

def mean_ci(x):
    from federated_learning.evaluation.statistics import t_ci
    x = np.asarray(x, float)
    if x.size < 2:
        return float(x.mean()), float("nan"), float("nan")
    return t_ci(x)

def scale(full, quick):
    return quick if QUICK else full
