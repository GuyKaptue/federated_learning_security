from __future__ import annotations

import numpy as np

from .dirichlet import partition_dirichlet
from .iid import partition_iid
from .imbalance import partition_quantity_skew
from .noniid import partition_label_shards
from .site_based import partition_by_site
from .stratified import partition_stratified

STRATEGIES = ("iid", "dirichlet", "label_shards", "stratified", "quantity_skew", "site")


def partition(strategy: str, y: np.ndarray, n_clients: int = 10, seed: int = 0, **kw) -> list[np.ndarray]:
    """Single entry point used by configs: strategy in STRATEGIES; extra args: alpha, shards_per_client, sigma, site_ids."""
    if strategy == "iid":
        return partition_iid(y, n_clients, seed)
    if strategy == "dirichlet":
        return partition_dirichlet(y, n_clients, kw.get("alpha", 0.5), seed)
    if strategy == "label_shards":
        return partition_label_shards(y, n_clients, kw.get("shards_per_client", 2), seed)
    if strategy == "stratified":
        return partition_stratified(y, n_clients, seed)
    if strategy == "quantity_skew":
        return partition_quantity_skew(len(y), n_clients, kw.get("sigma", 1.0), seed)
    if strategy == "site":
        if "site_ids" not in kw:
            raise ValueError("site strategy needs site_ids=")
        return partition_by_site(kw["site_ids"])
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
