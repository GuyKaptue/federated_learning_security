from __future__ import annotations

import time

import torch

from federated_learning.clients.local_training import local_train
from federated_learning.evaluation.metrics import evaluate
from federated_learning.preprocessing.dataset import Dataset
from federated_learning.training.models import build_model, get_flat


def centralized_baseline(data: Dataset, epochs: int = 20, lr: float = 0.05, bs: int = 32,
                         hidden: int = 32, seed: int = 0, arch: str = "mlp") -> dict:
    """Pooled-data SGD with the same model and optimiser (reference upper bound)."""
    model = build_model(data.n_features, hidden, "relu", seed=seed, arch=arch)
    gen = torch.Generator().manual_seed(seed)
    X, y = torch.from_numpy(data.X_train), torch.from_numpy(data.y_train)
    t = time.perf_counter()
    w, _ = local_train(model, get_flat(model), X, y, epochs, lr, bs, gen)
    out = evaluate(model, w, data.X_test, data.y_test)
    out["train_wall_s"] = time.perf_counter() - t
    return out
