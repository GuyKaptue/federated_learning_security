from __future__ import annotations

import numpy as np


def partition_quantity_skew(n: int, n_clients: int, sigma: float = 1.0, seed: int = 0, min_size: int = 5) -> list[np.ndarray]:
    """Client-size imbalance: sizes ~ lognormal(0, sigma) normalised to n (labels stay IID). sigma=0 gives equal sizes."""
    if sigma < 0 or n < n_clients * min_size:
        raise ValueError("sigma must be >= 0 and n >= n_clients * min_size")
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0.0, sigma, n_clients) if sigma > 0 else np.ones(n_clients)
    spare = n - n_clients * min_size
    sizes = min_size + np.floor(w / w.sum() * spare).astype(int)
    sizes[np.argmax(sizes)] += n - sizes.sum()
    perm = rng.permutation(n)
    cuts = np.cumsum(sizes)[:-1]
    return [np.sort(p) for p in np.split(perm, cuts)]
