"""Federated optimisation strategies."""
from .base import Strategy
from .fedavg import FedAvg
from .fednova import FedNova
from .fedprox import FedProx
from .scaffold import Scaffold

STRATEGIES = {"fedavg": FedAvg, "fedprox": FedProx, "scaffold": Scaffold, "fednova": FedNova}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"algo must be one of {tuple(STRATEGIES)}")
    return STRATEGIES[name]()


__all__ = ["Strategy", "FedAvg", "FedProx", "Scaffold", "FedNova", "STRATEGIES", "get_strategy"]
