import numpy as np
import torch

from federated_learning.clients.client import FederatedClient
from federated_learning.communication import NumPyClientAdapter
from federated_learning.synthetic_data import make_dataset
from federated_learning.training.config import FLConfig
from federated_learning.training.models import build_model, get_flat


def test_numpy_client_adapter_contract():
    d = make_dataset(n_samples=600, seed=0); cfg = FLConfig(local_epochs=1, hidden=8)
    cl = FederatedClient(0, torch.from_numpy(d.X_train[:100]), torch.from_numpy(d.y_train[:100]), 1)
    a = NumPyClientAdapter(cl, cfg, 40, d.X_test, d.y_test)
    params = [get_flat(build_model(40, 8, cfg.act))]
    new, n, _ = a.fit(params, {}); assert n == 100 and new[0].shape == params[0].shape and not np.array_equal(new[0], params[0])
    loss, m, metrics = a.evaluate(new, {}); assert m == len(d.y_test) and 0 <= metrics["accuracy"] <= 1
    assert a.get_parameters()[0].shape == params[0].shape
