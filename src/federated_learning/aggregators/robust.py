from __future__ import annotations

import time

import numpy as np

from federated_learning.aggregators.base import Aggregator


class CoordinateMedian(Aggregator):
    name = "median"

    def aggregate(self, deltas, weights, rnd=0):
        deltas, _ = self._check(deltas, weights)
        t = time.perf_counter()
        out = np.median(deltas, axis=0).astype(np.float32)
        self.last_stats = {"agg_time_s": time.perf_counter() - t}
        return out

class TrimmedMean(Aggregator):
    """Coordinate-wise trimmed mean, removing the `beta` fraction at each tail."""
    name = "trimmed_mean"

    def __init__(self, beta: float = 0.2):
        super().__init__()
        if not 0 <= beta < 0.5:
            raise ValueError("beta must be in [0, 0.5)")
        self.beta = beta

    def aggregate(self, deltas, weights, rnd=0):
        deltas, _ = self._check(deltas, weights)
        t = time.perf_counter()
        m = deltas.shape[0]
        k = int(np.floor(self.beta * m))
        s = np.sort(deltas, axis=0)
        out = s[k:m - k].mean(0).astype(np.float32)
        self.last_stats = {"agg_time_s": time.perf_counter() - t, "trimmed_each_side": k}
        return out

class Krum(Aggregator):
    """Krum (m_select=1) / Multi-Krum (m_select>1); requires n >= 2f+3 (Blanchard et al.)."""
    name = "krum"

    def __init__(self, f: int = 1, m_select: int = 1):
        super().__init__()
        self.f, self.m_select = f, m_select

    def aggregate(self, deltas, weights, rnd=0):
        deltas, _ = self._check(deltas, weights)
        n = deltas.shape[0]
        if n < 2 * self.f + 3:
            raise ValueError(f"Krum needs n >= 2f+3 (n={n}, f={self.f})")
        t = time.perf_counter()
        X = deltas.astype(np.float64)
        sq = (X ** 2).sum(1)
        D = sq[:, None] + sq[None, :] - 2 * X @ X.T
        np.fill_diagonal(D, np.inf)
        k = n - self.f - 2
        scores = np.sort(D, axis=1)[:, :k].sum(1)
        sel = np.argsort(scores)[: self.m_select]
        self.last_stats = {"agg_time_s": time.perf_counter() - t, "selected": sel.tolist()}
        return X[sel].mean(0).astype(np.float32)
