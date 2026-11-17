from __future__ import annotations

import numpy as np

from .base import Strategy


class FedNova(Strategy):
    """Wang et al. 2020: normalised averaging, w <- w + tau_eff * sum_k p_k (w_k - w) / tau_k."""
    name = "fednova"

    def server_step(self, agg, deltas, weights, taus, ids, rnd, cfg):
        W = np.asarray(weights, np.float64)
        p = W / W.sum()
        tau_arr = np.asarray(taus, np.float64)
        tau_eff = float((p * tau_arr).sum())
        normd = deltas / tau_arr[:, None].astype(np.float32)
        return tau_eff * agg.aggregate(normd, weights, rnd)
