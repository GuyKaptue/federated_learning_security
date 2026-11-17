from __future__ import annotations

import time

import numpy as np

from federated_learning.aggregators.base import Aggregator
from federated_learning.aggregators.mean import WeightedMean


class ClippedMean(Aggregator):
    """Norm clipping (L2 bound C) + optional Gaussian noise (DP-FedAvg style).

    noise ~ N(0, (sigma*C/m)^2) added to the mean of m clipped updates. This is the
    Gaussian mechanism on the mean; formal epsilon needs an accountant (see attacks/ notebooks).
    """
    name = "clipped_mean"

    def __init__(self, clip: float, noise_multiplier: float = 0.0, seed: int = 0):
        super().__init__()
        if clip <= 0:
            raise ValueError("clip must be > 0")
        self.clip, self.sigma = clip, noise_multiplier
        self.rng = np.random.default_rng(seed)

    def aggregate(self, deltas, weights, rnd=0):
        deltas, _ = self._check(deltas, weights)
        t = time.perf_counter()
        X = deltas.astype(np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X * np.minimum(1.0, self.clip / np.maximum(norms, 1e-12))
        m = X.shape[0]
        out = X.mean(0)
        if self.sigma > 0:
            out = out + self.rng.normal(0.0, self.sigma * self.clip / m, size=out.shape)
        self.last_stats = {"agg_time_s": time.perf_counter() - t,
                           "frac_clipped": float((norms.ravel() > self.clip).mean())}
        return out.astype(np.float32)

class AnomalyFiltered(Aggregator):
    """Drop updates whose cosine similarity to the coordinate-median update is low or whose
    norm is a robust outlier (median/MAD), then delegate to `base`. Heuristic, no guarantee."""
    name = "anomaly_filter"

    def __init__(self, base: Aggregator | None = None, cos_min: float = 0.0, norm_z: float = 4.0):
        super().__init__()
        self.base = base or WeightedMean()
        self.cos_min, self.norm_z = cos_min, norm_z

    def aggregate(self, deltas, weights, rnd=0):
        deltas, w = self._check(deltas, weights)
        t = time.perf_counter()
        X = deltas.astype(np.float64)
        med = np.median(X, axis=0)
        cos = X @ med / (np.linalg.norm(X, axis=1) * np.linalg.norm(med) + 1e-12)
        nr = np.linalg.norm(X, axis=1)
        mad = np.median(np.abs(nr - np.median(nr))) * 1.4826 + 1e-12
        keep = (cos >= self.cos_min) & (np.abs(nr - np.median(nr)) / mad <= self.norm_z)
        if keep.sum() == 0:
            keep[:] = True
        out = self.base.aggregate(deltas[keep], w[keep], rnd)
        self.last_stats = {"agg_time_s": time.perf_counter() - t, "n_dropped": int((~keep).sum()),
                           "dropped_idx": np.where(~keep)[0].tolist()}
        return out
