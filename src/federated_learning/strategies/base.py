"""Federated optimisation strategies (FedAvg, FedProx, SCAFFOLD, FedNova).

A strategy owns the algorithm-specific parts of a round so that clients, the server loop and the
aggregation rule stay independent:

* client side : which proximal coefficient / gradient correction a client uses (`client_mu`, `client_correction`)
                and what state it updates after local training (`after_client`);
* server side : how the stacked client deltas are turned into a global step (`server_step`);
* accounting  : extra payloads per round (`uplink_payloads`, `downlink_payloads`), e.g. SCAFFOLD's control variates.

Strategies never touch a network: the same object drives the in-process simulator and the networked server.
"""
from __future__ import annotations

import numpy as np


class Strategy:
    name = "base"
    uplink_payloads = 1       # model-sized vectors sent by each client per round
    downlink_payloads = 1     # model-sized vectors received by each client per round

    def begin(self, n_params: int, n_clients: int) -> None:
        self.n_params, self.n_clients = n_params, n_clients

    def begin_round(self) -> None:
        """Reset per-round state."""

    def client_mu(self, cfg) -> float:
        return 0.0

    def client_correction(self, cid: int):
        return None

    def after_client(self, cid: int, w: np.ndarray, w_local: np.ndarray, tau: int, cfg) -> None:
        """Update client-side state after local training."""

    def server_step(self, agg, deltas: np.ndarray, weights, taus, ids, rnd: int, cfg) -> np.ndarray:
        return agg.aggregate(deltas, weights, rnd)

    def extras(self) -> dict:
        return {}
