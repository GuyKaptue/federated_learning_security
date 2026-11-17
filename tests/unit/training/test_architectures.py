import numpy as np
import pytest

from federated_learning.data_partitioning.iid import partition_iid
from federated_learning.servers.orchestrator import run_fl
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset
from federated_learning.training.config import FLConfig
from federated_learning.training.models import ARCHS, build_model, get_flat, n_params, set_flat


def test_parameter_counts():
    assert n_params(build_model(40, 32, arch="logreg")) == 40 * 2 + 2
    assert n_params(build_model(40, 32, arch="mlp")) == 40 * 32 + 32 + 32 * 2 + 2
    assert n_params(build_model(40, 32, arch="deep_mlp")) == (40 * 32 + 32) + (32 * 32 + 32) + (32 * 16 + 16) + (16 * 2 + 2)


@pytest.mark.parametrize("arch", ARCHS)
def test_flat_roundtrip_and_determinism(arch):
    a, b = build_model(10, 8, seed=3, arch=arch), build_model(10, 8, seed=3, arch=arch)
    assert np.array_equal(get_flat(a), get_flat(b))
    w = get_flat(a) * 0 + 0.5
    set_flat(a, w)
    assert np.array_equal(get_flat(a), w.astype(np.float32))


def test_unknown_arch_rejected():
    with pytest.raises(ValueError):
        build_model(5, arch="cnn")
    with pytest.raises(ValueError):
        FLConfig(arch="cnn").validate()


def test_default_arch_unchanged_so_golden_runs_are_unaffected():
    assert FLConfig().arch == "mlp"


@pytest.mark.parametrize("arch", ["logreg", "deep_mlp"])
def test_run_fl_learns_with_each_arch(arch):
    data = make_dataset(seed=0, n_samples=4000)
    r = run_fl(FLConfig(rounds=20, seed=0, arch=arch, lr=0.1, local_epochs=2), data, partition_iid(data.y_train, 5, 0))   # the deep MLP needs more training than the default 8 rounds at lr 0.05
    assert r.final["acc"] > 0.7 and r.totals["n_params"] == n_params(build_model(data.n_features, 32, arch=arch))
