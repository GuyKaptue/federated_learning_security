from __future__ import annotations

import numpy as np

from federated_learning.aggregators.base import Aggregator
from federated_learning.aggregators.mean import WeightedMean


class Audited(Aggregator):
    """Runs `inner` (e.g. a secure aggregator) and, in parallel, the plaintext weighted mean, recording the
    numerical deviation every round. Returns the INNER result (the plaintext is only used for the audit)."""

    def __init__(self, inner: Aggregator):
        super().__init__()
        self.inner, self.name = inner, inner.name
        self.is_secure = getattr(inner, "is_secure", False)
        self.errors: list[float] = []
        self.rel_errors: list[float] = []
        self._plain = WeightedMean()

    def aggregate(self, deltas, weights, rnd=0):
        out = self.inner.aggregate(deltas, weights, rnd)
        ref = self._plain.aggregate(deltas, weights, rnd)
        self.errors.append(float(np.abs(out - ref).max()))
        self.rel_errors.append(float(np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-12)))
        self.last_stats = dict(self.inner.last_stats)
        return out
