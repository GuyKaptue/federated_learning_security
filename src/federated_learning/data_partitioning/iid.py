from __future__ import annotations

import numpy as np


def partition_iid(y: np.ndarray, n_clients: int, seed: int = 0) -> list[np.ndarray]:
    if n_clients < 1 or n_clients > len(y):
        raise ValueError("invalid number of clients")
    rng = np.random.default_rng(seed)
    return [np.asarray(p) for p in np.array_split(rng.permutation(len(y)), n_clients)]
