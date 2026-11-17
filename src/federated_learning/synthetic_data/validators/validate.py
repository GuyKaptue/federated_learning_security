from __future__ import annotations

import numpy as np

from federated_learning.synthetic_data.generators.syn_ids_v1 import SynIDSConfig


def validate_syn_ids(X: np.ndarray, y: np.ndarray, cfg: SynIDSConfig, tol: float = 0.03) -> dict:
    """Check the documented SYN-IDS-v1 properties. Returns {'passed': bool, 'checks': [{name, passed, detail}]}."""
    pos = float(np.mean(y == 1))
    checks = [
        ("shape", X.shape == (cfg.n_samples, cfg.n_features), f"got {X.shape}"),
        ("binary_labels", set(np.unique(y).tolist()) == {0, 1}, f"labels {sorted(set(np.unique(y).tolist()))}"),
        ("finite", bool(np.isfinite(X).all()), "no NaN/inf"),
        ("class_ratio_60_40", abs(pos - cfg.weights[1]) <= tol, f"positive rate {pos:.3f} vs target {cfg.weights[1]} (flip_y noise allowed, tol {tol})"),
        ("not_constant_features", bool((X.std(0) > 0).all()), "every feature varies"),
    ]
    rows = [{"name": n, "passed": bool(ok), "detail": d} for n, ok, d in checks]
    return {"passed": all(r["passed"] for r in rows), "checks": rows}
