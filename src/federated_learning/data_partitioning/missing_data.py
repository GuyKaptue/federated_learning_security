from __future__ import annotations

import numpy as np


def simulate_missing(X: np.ndarray, rate: float, mechanism: str = "mcar", seed: int = 0, y: np.ndarray | None = None) -> np.ndarray:
    """Return a copy of X with NaN injected. mcar: uniform; mar: probability driven by the first feature's rank; mnar: the largest
    values of each feature go missing preferentially. `rate` is the expected overall fraction of missing cells."""
    if not 0 <= rate < 1:
        raise ValueError("rate must be in [0, 1)")
    X = np.array(X, dtype=float, copy=True)
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if mechanism == "mcar":
        mask = rng.random((n, d)) < rate
    elif mechanism == "mar":
        r = np.argsort(np.argsort(X[:, 0])) / max(n - 1, 1)
        p = np.clip(2 * rate * r, 0, 1)[:, None] * np.ones((1, d)); p[:, 0] = 0
        mask = rng.random((n, d)) < p
    elif mechanism == "mnar":
        r = np.argsort(np.argsort(X, axis=0), axis=0) / max(n - 1, 1)
        mask = rng.random((n, d)) < np.clip(2 * rate * r, 0, 1)
    else:
        raise ValueError("mechanism must be mcar, mar or mnar")
    X[mask] = np.nan
    return X
