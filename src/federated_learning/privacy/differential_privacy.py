from __future__ import annotations

import numpy as np

from federated_learning.aggregators.base import Aggregator


class ClientSideDP(Aggregator):
    """Distributed DP compatible with secure aggregation / HE / SMPC: each client clips its update to L2 norm C and adds
    N(0, sigma^2 C^2 / m) noise BEFORE it enters the secure protocol, so the sum carries N(0, sigma^2 C^2) noise
    (mean noise std sigma*C/m, identical to the central Gaussian mechanism in ClippedMean).
    The inner aggregator must be a weighted mean over equal-weight updates for the privacy statement to hold."""

    def __init__(self, inner: Aggregator, clip: float, noise_multiplier: float, seed: int = 0):
        super().__init__()
        self.inner, self.clip, self.sigma = inner, clip, noise_multiplier
        self.name = f"{inner.name}+dp"
        self.is_secure = getattr(inner, "is_secure", False)
        self.rng = np.random.default_rng(seed)

    def aggregate(self, deltas, weights, rnd=0):
        deltas, _ = self._check(deltas, weights)
        m = deltas.shape[0]
        X = deltas.astype(np.float64)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        X = X * np.minimum(1.0, self.clip / np.maximum(nr, 1e-12))
        X = X + self.rng.normal(0.0, self.sigma * self.clip / np.sqrt(m), size=X.shape)
        out = self.inner.aggregate(X.astype(np.float32), np.ones(m), rnd)   # equal weights (DP sensitivity)
        self.last_stats = dict(self.inner.last_stats)
        return out
