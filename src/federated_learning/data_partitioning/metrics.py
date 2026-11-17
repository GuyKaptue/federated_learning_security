from __future__ import annotations

import numpy as np


def label_skew_stat(y: np.ndarray, parts: list[np.ndarray]) -> float:
    """Mean absolute deviation of client positive-rate from global positive-rate."""
    g = y.mean()
    return float(np.mean([abs(y[p].mean() - g) for p in parts]))
