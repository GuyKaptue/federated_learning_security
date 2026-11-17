from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index with quantile bins of `expected`. Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 large."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + eps
    a = np.histogram(actual, edges)[0] / len(actual) + eps
    return float(np.sum((a - e) * np.log(a / e)))


def psi_per_feature(A: np.ndarray, B: np.ndarray, bins: int = 10) -> list[float]:
    return [psi(A[:, j], B[:, j], bins) for j in range(A.shape[1])]


def ks_per_feature(A: np.ndarray, B: np.ndarray) -> list[dict]:
    out = []
    for j in range(A.shape[1]):
        r = ks_2samp(A[:, j], B[:, j]); out.append({"statistic": float(r.statistic), "p_value": float(r.pvalue)})
    return out
