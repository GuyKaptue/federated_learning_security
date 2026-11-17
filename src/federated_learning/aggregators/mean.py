from __future__ import annotations

import time

import numpy as np

from federated_learning.aggregators.base import Aggregator


class WeightedMean(Aggregator):
    name = "mean"

    def aggregate(self, deltas, weights, rnd=0):
        deltas, w = self._check(deltas, weights)
        t = time.perf_counter()
        out = (w[:, None] * deltas.astype(np.float64)).sum(0).astype(np.float32)
        self.last_stats = {"agg_time_s": time.perf_counter() - t}
        return out
