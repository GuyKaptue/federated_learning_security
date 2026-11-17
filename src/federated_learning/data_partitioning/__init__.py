"""Federated dataset simulation: IID, Dirichlet, label-shard non-IID, stratified, quantity-skew, site-based, missing-data."""
from .dirichlet import partition_dirichlet
from .iid import partition_iid
from .imbalance import partition_quantity_skew
from .metrics import label_skew_stat
from .missing_data import simulate_missing
from .noniid import partition_label_shards
from .registry import STRATEGIES, partition
from .site_based import partition_by_site
from .stratified import partition_stratified

__all__ = ["STRATEGIES", "label_skew_stat", "partition", "partition_by_site", "partition_dirichlet", "partition_iid",
           "partition_label_shards", "partition_quantity_skew", "partition_stratified", "simulate_missing"]
