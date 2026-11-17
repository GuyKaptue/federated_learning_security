from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

ALGOS = ("fedavg", "fedprox", "scaffold", "fednova")

@dataclass
class FLConfig:
    algo: str = "fedavg"
    rounds: int = 20
    local_epochs: int = 2
    lr: float = 0.05
    batch_size: int = 32
    participation: float = 1.0
    mu: float = 0.0              # FedProx proximal coefficient
    hidden: int = 32
    act: str = "relu"
    seed: int = 0
    hetero_epochs: bool = False  # per-client epochs drawn in [1, local_epochs] (FedNova setting)
    eval_every: int = 1
    server_lr: float = 1.0       # SCAFFOLD global step size
    arch: str = "mlp"            # "mlp" (default, 1 hidden layer) | "logreg" | "deep_mlp"

    def validate(self):
        if self.algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}")
        if not 0 < self.participation <= 1:
            raise ValueError("participation in (0,1]")
        if self.rounds < 1 or self.local_epochs < 1 or self.lr <= 0 or self.batch_size < 1:
            raise ValueError("invalid hyper-parameters")
        from federated_learning.training.models import ARCHS
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}")
        if self.algo == "fedprox" and self.mu < 0:
            raise ValueError("mu >= 0")

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def load_fl_config(overrides: dict | None = None, config_dir=None) -> FLConfig:
    """Build an FLConfig from configs/{training,federated,model}.yaml (later files win, then `overrides`). Unknown keys raise."""
    from dataclasses import fields

    from federated_learning.utils.config import load_config, merge
    cfg: dict = {}
    for name in ("training", "federated", "model"):
        cfg = merge(cfg, load_config(name, config_dir))
    cfg = merge(cfg, overrides)
    allowed = {f.name for f in fields(FLConfig)}
    flat = {k: v for k, v in cfg.items() if k not in ("version", "partitioning")}
    bad = set(flat) - allowed
    if bad:
        raise KeyError(f"unknown FLConfig keys in configs: {sorted(bad)}")
    c = FLConfig(**flat); c.validate()
    return c
