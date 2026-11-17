import numpy as np
import pytest

from federated_learning import attacks as A
from federated_learning.data_partitioning.iid import partition_iid
from federated_learning.servers.orchestrator import run_fl
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset
from federated_learning.training.config import FLConfig
from federated_learning.training.models import build_model


@pytest.fixture(scope="module")
def data():
    return make_dataset(n_samples=3000, seed=0)


def test_trigger_and_poisoning(data):
    X = data.X_train[:50]; Xt = A.apply_trigger(X)
    assert np.all(Xt[:, list(A.TRIGGER_IDX)] == A.TRIGGER_VAL) and np.array_equal(Xt[:, 10:], X[:, 10:])
    Xp, yp, n = A.poison_client_data(X, data.y_train[:50], 0.4, 0, np.random.default_rng(0))
    assert n == 20 and (yp.numpy() == 0).sum() >= 20
    with pytest.raises(ValueError):
        A.poison_client_data(X, data.y_train[:50], 1.5, 0, np.random.default_rng(0))


def test_backdoor_succeeds_in_fl(data):
    parts = partition_iid(data.y_train, 10, 0); rng = np.random.default_rng(0)
    mal = {0, 1, 2}; cd = {k: A.poison_client_data(data.X_train[parts[k]], data.y_train[parts[k]], 0.6, 0, rng)[:2] for k in mal}
    ev = lambda m, w: A.backdoor_metrics(m, w, data.X_test, data.y_test, 0)
    r = run_fl(FLConfig(rounds=10, local_epochs=1), data, parts, client_data=cd, client_hook=A.boost_hook(mal, 3.0), eval_fn=ev)
    c = run_fl(FLConfig(rounds=10, local_epochs=1), data, parts, eval_fn=ev)
    assert r.final["asr"] > c.final["asr"] + 0.2


@pytest.mark.parametrize("bh", ["random", "sign_flip", "scale", "drift"])
def test_byzantine_hooks_change_only_malicious(bh):
    hook = A.byzantine_hook(bh, {1}, 5.0); d = np.ones(20, np.float32)
    assert np.array_equal(hook(1, 0, d, {}), d) and not np.array_equal(hook(1, 1, d, {}), d)
    assert np.all(np.isfinite(hook(1, 1, d, {})))


def test_byzantine_invalid():
    with pytest.raises(ValueError):
        A.byzantine_hook("nope", {0})


def test_mia_metrics_perfect_and_random():
    m = A.mia_metrics(np.ones(100), np.zeros(100)); assert m["auc"] == 1.0 and m["max_advantage"] == 1.0
    rng = np.random.default_rng(0); r = A.mia_metrics(rng.normal(size=2000), rng.normal(size=2000)); assert abs(r["auc"] - 0.5) < 0.05


def test_analytic_inversion_exact(data):
    m = build_model(40, 16, "sigmoid")
    from federated_learning.training.models import get_flat
    w = get_flat(m); X, y = data.X_test[:1], data.y_test[:1]
    g = A.model_grad(m, w, X, y)
    xa = A.analytic_first_layer_inversion(g, 40, 16)
    assert np.linalg.norm(xa - X[0]) / np.linalg.norm(X[0]) < 1e-4


def test_dlg_recovers_small_batch(data):
    from federated_learning.training.models import get_flat
    m = build_model(40, 32, "sigmoid"); w = get_flat(m); X, y = data.X_test[:2], data.y_test[:2]
    g = A.model_grad(m, w, X, y); Xh = A.dlg_attack(m, w, g, 2, 40, labels=y, iters=300, seed=0)
    assert A.match_and_score(X, Xh)["success_rate"] == 1.0


def test_dlg_degrades_with_noise(data):
    from federated_learning.training.models import get_flat
    m = build_model(40, 32, "sigmoid"); w = get_flat(m); X, y = data.X_test[:2], data.y_test[:2]
    g = A.model_grad(m, w, X, y); g = g + np.random.default_rng(0).normal(0, 1.0 * np.linalg.norm(g) / np.sqrt(g.size), g.shape).astype(np.float32)
    Xh = A.dlg_attack(m, w, g, 2, 40, labels=y, iters=200, seed=0)
    assert A.match_and_score(X, Xh)["success_rate"] < 1.0


def test_dlg_requires_labels_for_batches(data):
    m = build_model(40, 8, "sigmoid")
    from federated_learning.training.models import get_flat
    with pytest.raises(ValueError):
        A.dlg_attack(m, get_flat(m), np.zeros(m and sum(p.numel() for p in m.parameters()), np.float32), 3, 40)


def test_match_handles_nonfinite():
    assert A.match_and_score(np.ones((2, 4)), np.full((2, 4), np.nan))["success_rate"] == 0.0
