from __future__ import annotations

import numpy as np


def partition_stratified(y: np.ndarray, n_clients: int, seed: int = 0) -> list[np.ndarray]:
    """Every client receives (as nearly as possible) the global class proportions."""
    if n_clients < 1 or n_clients > len(y):
        raise ValueError("invalid number of clients")
    rng = np.random.default_rng(seed)
    idx: list[list[int]] = [[] for _ in range(n_clients)]
    for c in np.unique(y):
        ids = rng.permutation(np.where(y == c)[0])
        for k, chunk in enumerate(np.array_split(ids, n_clients)):
            idx[k].extend(chunk.tolist())
    return [np.asarray(sorted(i)) for i in idx]
