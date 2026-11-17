from __future__ import annotations

import numpy as np

from .base import Strategy


class Scaffold(Strategy):
    """Karimireddy et al. 2020, option II control variates (doubles the payload in both directions)."""
    name = "scaffold"
    uplink_payloads = 2
    downlink_payloads = 2

    def begin(self, n_params, n_clients):
        super().begin(n_params, n_clients)
        self.c_global = np.zeros(n_params, np.float32)
        self.c_local: dict[int, np.ndarray] = {}
        self._dc: list[np.ndarray] = []

    def begin_round(self):
        self._dc = []

    def _ci(self, cid):
        return self.c_local.get(cid, np.zeros(self.n_params, np.float32))

    def client_correction(self, cid):
        return self.c_global - self._ci(cid)

    def after_client(self, cid, w, w_local, tau, cfg):
        ci = self._ci(cid)
        ci_new = ci - self.c_global + (w - w_local) / (tau * cfg.lr)
        self._dc.append(ci_new - ci)
        self.c_local[cid] = ci_new

    def server_step(self, agg, deltas, weights, taus, ids, rnd, cfg):
        step = cfg.server_lr * agg.aggregate(deltas, weights, rnd)
        self.c_global = self.c_global + (len(ids) / self.n_clients) * np.mean(np.stack(self._dc), axis=0).astype(np.float32)
        return step

    def extras(self):
        return dict(c_global=self.c_global, c_local=self.c_local)
