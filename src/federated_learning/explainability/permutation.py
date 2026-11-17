from __future__ import annotations

import numpy as np
import torch

from federated_learning.training.models import set_flat


def permutation_importance(model, w: np.ndarray, X: np.ndarray, y: np.ndarray, n_repeats: int = 5, seed: int = 0) -> np.ndarray:
    """Mean accuracy drop when one feature column is shuffled (model-agnostic, applied to the global FL model `w`)."""
    set_flat(model, w); model.eval()
    rng = np.random.default_rng(seed)

    def acc(A):
        with torch.no_grad():
            logits = model(torch.as_tensor(A, dtype=torch.float32))
        return float((logits.argmax(1).numpy() == y).mean())

    base = acc(X)
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        d = []
        for _ in range(n_repeats):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); d.append(base - acc(Xp))
        imp[j] = np.mean(d)
    return imp
