from __future__ import annotations

import numpy as np


def partition_label_shards(y: np.ndarray, n_clients: int, shards_per_client: int = 2, seed: int = 0) -> list[np.ndarray]:
    """Pathological non-IID (McMahan et al.): sort by label, cut into n_clients*shards_per_client shards, deal shards randomly."""
    if n_clients < 1 or shards_per_client < 1:
        raise ValueError("n_clients and shards_per_client must be >= 1")
    n_shards = n_clients * shards_per_client
    if n_shards > len(y):
        raise ValueError("more shards than samples")
    rng = np.random.default_rng(seed)
    order = np.argsort(y, kind="stable")
    shards = np.array_split(order, n_shards)
    pick = rng.permutation(n_shards)
    return [np.sort(np.concatenate([shards[j] for j in pick[i * shards_per_client:(i + 1) * shards_per_client]])) for i in range(n_clients)]
