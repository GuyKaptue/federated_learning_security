from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from federated_learning.training.models import set_flat


def local_train(model, w_global: np.ndarray, X: torch.Tensor, y: torch.Tensor, epochs: int,
                lr: float, bs: int, gen: torch.Generator, mu: float = 0.0,
                corr: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Plain SGD from w_global. Returns (w_local, tau = number of local SGD steps)."""
    set_flat(model, w_global)
    params = list(model.parameters())
    wg = torch.from_numpy(w_global.copy())
    corr_t = torch.from_numpy(corr.astype(np.float32)) if corr is not None else None
    n = X.shape[0]
    tau = 0
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            loss = F.cross_entropy(model(X[idx]), y[idx])
            model.zero_grad()
            loss.backward()
            g = torch.cat([p.grad.reshape(-1) for p in params])
            w = parameters_to_vector(params).detach()
            if mu > 0:
                g = g + mu * (w - wg)
            if corr_t is not None:
                g = g + corr_t
            vector_to_parameters(w - lr * g, params)
            tau += 1
    return parameters_to_vector(params).detach().numpy().astype(np.float32).copy(), tau
