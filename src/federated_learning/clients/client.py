"""A virtual federated client: private data + local training. Nothing here touches a network."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from federated_learning.clients.local_training import local_train


@dataclass
class FederatedClient:
    cid: int
    X: torch.Tensor
    y: torch.Tensor
    epochs: int

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    def fit(self, model, w_global: np.ndarray, cfg, rnd: int, mu: float = 0.0, correction: np.ndarray | None = None):
        """One round of local SGD from the global weights. Returns (local weights, number of local steps)."""
        gen = torch.Generator().manual_seed(cfg.seed * 1_000_003 + rnd * 1009 + self.cid)
        return local_train(model, w_global, self.X, self.y, self.epochs, cfg.lr, cfg.batch_size, gen, mu=mu, corr=correction)
