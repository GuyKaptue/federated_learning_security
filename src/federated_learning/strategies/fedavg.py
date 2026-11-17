from __future__ import annotations

from .base import Strategy


class FedAvg(Strategy):
    """McMahan et al. 2017: w <- w + sum_k p_k (w_k - w)."""
    name = "fedavg"
