"""Interface adapters so the simulated FL components can be driven from external FL frameworks.

DUCK-TYPED ONLY: `NumPyClientAdapter` exposes the get_parameters / fit / evaluate / set_parameters surface that Flower's
`NumPyClient` expects (parameters as a list of NumPy arrays). It was NOT run against Flower, TensorFlow Federated or FedML in this
project, so compatibility is by interface shape, not demonstrated integration.
"""
from __future__ import annotations

import numpy as np

from federated_learning.clients.client import FederatedClient
from federated_learning.training.config import FLConfig
from federated_learning.training.models import build_model, get_flat, set_flat


class NumPyClientAdapter:
    def __init__(self, client: FederatedClient, cfg: FLConfig, n_features: int, X_val: np.ndarray, y_val: np.ndarray):
        self.client, self.cfg, self.X_val, self.y_val = client, cfg, X_val, y_val
        self.model = build_model(n_features, cfg.hidden, cfg.act, arch=cfg.arch)
        self._round = 0

    def get_parameters(self, config: dict | None = None) -> list[np.ndarray]:
        return [get_flat(self.model)]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        set_flat(self.model, parameters[0])

    def fit(self, parameters: list[np.ndarray], config: dict | None = None):
        self._round += 1
        upd = self.client.fit(self.model, parameters[0], self.cfg, self._round)
        w_new = upd[0] if isinstance(upd, tuple) else upd
        return [np.asarray(w_new, dtype=np.float32)], self.client.n_samples, {}

    def evaluate(self, parameters: list[np.ndarray], config: dict | None = None):
        from federated_learning.evaluation.metrics import evaluate
        self.set_parameters(parameters)
        r = evaluate(self.model, parameters[0], self.X_val, self.y_val)
        return float(1 - r["acc"]), int(len(self.y_val)), {"accuracy": float(r["acc"])}
