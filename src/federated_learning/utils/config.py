"""YAML configuration loading (configs/*.yaml). Configs are versioned with the code; every run records the values it used."""
from __future__ import annotations

from pathlib import Path

import yaml

from federated_learning.utils.paths import CONFIG_DIR


def load_config(name: str, config_dir: Path | None = None) -> dict:
    """Load configs/<name>.yaml (name with or without suffix)."""
    p = (config_dir or CONFIG_DIR) / (name if name.endswith((".yaml", ".yml")) else f"{name}.yaml")
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{p}: top level must be a mapping")
    return cfg


def merge(base: dict, override: dict | None) -> dict:
    """Recursive dict merge (override wins) without mutating the inputs."""
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
