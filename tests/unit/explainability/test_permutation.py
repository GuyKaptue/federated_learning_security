import numpy as np

from federated_learning.explainability import permutation_importance
from federated_learning.training.models import build_model, get_flat


def test_permutation_importance_shape_and_signal():
    import torch
    rng = np.random.default_rng(0); X = rng.normal(size=(400, 5)).astype(np.float32); y = (X[:, 0] > 0).astype(np.int64)
    m = build_model(5, 8, "relu"); opt = torch.optim.Adam(m.parameters(), 0.05)
    for _ in range(200):
        opt.zero_grad(); torch.nn.functional.cross_entropy(m(torch.from_numpy(X)), torch.from_numpy(y)).backward(); opt.step()
    imp = permutation_importance(m, get_flat(m), X, y, n_repeats=3)
    assert imp.shape == (5,) and imp.argmax() == 0
