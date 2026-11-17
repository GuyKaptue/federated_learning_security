from __future__ import annotations

import numpy as np


def partition_dirichlet(y: np.ndarray, n_clients: int, alpha: float, seed: int = 0,
                        min_size: int = 5, max_tries: int = 200) -> list[np.ndarray]:
    """Label-skew partition: per class, proportions ~ Dirichlet(alpha) over clients."""
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    for _ in range(max_tries):
        idx = [[] for _ in range(n_clients)]
        for c in classes:
            ids = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
            for k, part in enumerate(np.split(ids, cuts)):
                idx[k].extend(part.tolist())
        if min(len(i) for i in idx) >= min_size:
            return [np.asarray(sorted(i)) for i in idx]
    raise RuntimeError("could not satisfy min_size; increase alpha or reduce clients")
