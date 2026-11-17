"""Repository paths. Override the root with the FL_ROOT environment variable (e.g. when the package is pip-installed)."""
from __future__ import annotations

import os
from pathlib import Path


def _find_root() -> Path:
    env = os.environ.get("FL_ROOT") or os.environ.get("RS_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parents[3]
    return here if (here / "pyproject.toml").exists() else Path.cwd()


ROOT = _find_root()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
EXTERNAL_DIR = DATA_DIR / "external"
CONFIG_DIR = ROOT / "configs"
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
DOCS_DIR = ROOT / "docs"
