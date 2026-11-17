from __future__ import annotations

from .base import Strategy


class FedProx(Strategy):
    """Li et al. 2020: local objective F_k(w) + mu/2 ||w - w_t||^2."""
    name = "fedprox"

    def client_mu(self, cfg) -> float:
        return cfg.mu
