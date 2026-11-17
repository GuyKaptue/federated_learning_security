"""Small MLP and flat-parameter helpers shared by every experiment."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 32, n_out: int = 2, act: str = "relu"):
        super().__init__()
        acts = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        if act not in acts:
            raise ValueError(f"unknown activation {act}")
        self.net = nn.Sequential(nn.Linear(d_in, hidden), acts[act](), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)

class LogReg(nn.Module):
    """Softmax regression (binary logistic regression with two logits): the linear baseline."""
    def __init__(self, d_in: int, n_out: int = 2):
        super().__init__()
        self.net = nn.Linear(d_in, n_out)

    def forward(self, x):
        return self.net(x)


class DeepMLP(nn.Module):
    """Three hidden layers (hidden, hidden, hidden//2), no normalisation layers (so every state is a parameter and FL averaging is exact)."""
    def __init__(self, d_in: int, hidden: int = 32, n_out: int = 2, act: str = "relu"):
        super().__init__()
        acts = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        if act not in acts:
            raise ValueError(f"unknown activation {act}")
        h2 = max(2, hidden // 2)
        self.net = nn.Sequential(nn.Linear(d_in, hidden), acts[act](), nn.Linear(hidden, hidden), acts[act](),
                                 nn.Linear(hidden, h2), acts[act](), nn.Linear(h2, n_out))

    def forward(self, x):
        return self.net(x)


ARCHS = ("mlp", "logreg", "deep_mlp")


def build_model(d_in: int, hidden: int = 32, act: str = "relu", seed: int = 0, arch: str = "mlp") -> nn.Module:
    torch.manual_seed(seed)
    if arch == "mlp":
        return MLP(d_in, hidden, 2, act)
    if arch == "logreg":
        return LogReg(d_in, 2)
    if arch == "deep_mlp":
        return DeepMLP(d_in, hidden, 2, act)
    raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")

def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))

def get_flat(model: nn.Module) -> np.ndarray:
    return parameters_to_vector(model.parameters()).detach().cpu().numpy().astype(np.float32).copy()

def set_flat(model: nn.Module, vec: np.ndarray) -> None:
    vec = np.asarray(vec, dtype=np.float32)
    if vec.size != n_params(model):
        raise ValueError(f"vector size {vec.size} != model size {n_params(model)}")
    vector_to_parameters(torch.from_numpy(vec.copy()), model.parameters())
