"""Model registry on disk: weights (.npz) + JSON card (config, metrics, SHA-256) under models/<name>/<version>/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from federated_learning.utils.hashing import sha256_arrays
from federated_learning.utils.paths import MODELS_DIR


def save_model(name: str, version: str, w: np.ndarray, config: dict, metrics: dict, root: Path | None = None) -> Path:
    d = Path(root or MODELS_DIR) / name / version
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "weights.npz", w=np.asarray(w, dtype=np.float32))
    card = {"name": name, "version": version, "n_parameters": int(np.size(w)), "weights_sha256": sha256_arrays(np.asarray(w, dtype=np.float32)),
            "config": config, "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))}}
    (d / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    return d


def load_model(name: str, version: str, root: Path | None = None) -> tuple[np.ndarray, dict]:
    d = Path(root or MODELS_DIR) / name / version
    w = np.load(d / "weights.npz")["w"]; card = json.loads((d / "model_card.json").read_text())
    if sha256_arrays(w) != card["weights_sha256"]:
        raise RuntimeError(f"{d}: weights do not match the recorded SHA-256")
    return w, card
