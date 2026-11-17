from __future__ import annotations

import numpy as np


def class_balance(y: np.ndarray) -> dict:
    y = np.asarray(y)
    vals, cnt = np.unique(y, return_counts=True)
    return {int(v): {"count": int(c), "fraction": float(c / len(y))} for v, c in zip(vals, cnt)}


def feature_summary(X: np.ndarray) -> dict:
    """Per-feature mean, std, min, max, skewness and excess kurtosis (population moments)."""
    X = np.asarray(X, dtype=float)
    mu, sd = X.mean(0), X.std(0)
    z = (X - mu) / np.where(sd == 0, 1.0, sd)
    return {"mean": mu.tolist(), "std": sd.tolist(), "min": X.min(0).tolist(), "max": X.max(0).tolist(),
            "skew": (z ** 3).mean(0).tolist(), "excess_kurtosis": ((z ** 4).mean(0) - 3).tolist()}
