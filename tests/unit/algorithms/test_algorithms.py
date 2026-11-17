import numpy as np
import pytest
import torch

from federated_learning.aggregators.mean import WeightedMean
from federated_learning.clients.local_training import local_train
from federated_learning.data_partitioning.dirichlet import partition_dirichlet
from federated_learning.data_partitioning.iid import partition_iid
from federated_learning.data_partitioning.metrics import label_skew_stat
from federated_learning.evaluation.metrics import evaluate, rounds_to_target
from federated_learning.pipelines.experiments.bench004 import _quadratic_toy
from federated_learning.servers.orchestrator import run_fl
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset
from federated_learning.training.config import ALGOS, FLConfig
from federated_learning.training.models import build_model, get_flat, set_flat


@pytest.fixture(scope="module")
def data():
    return make_dataset(n_samples=4000, seed=0)


def test_partitions(data):
    iid = partition_iid(data.y_train, 8, 0); assert sorted(np.concatenate(iid).tolist()) == list(range(len(data.y_train)))
    d = partition_dirichlet(data.y_train, 8, 0.1, 0); assert sorted(np.concatenate(d).tolist()) == list(range(len(data.y_train)))
    assert label_skew_stat(data.y_train, d) > 3 * label_skew_stat(data.y_train, iid)
    with pytest.raises(ValueError):
        partition_dirichlet(data.y_train, 8, -1)
    with pytest.raises(ValueError):
        partition_iid(data.y_train, 0)


@pytest.mark.parametrize("algo", ALGOS)
def test_all_algorithms_learn(data, algo):
    parts = partition_iid(data.y_train, 6, 0)
    r = run_fl(FLConfig(algo=algo, rounds=12, local_epochs=2, mu=0.05), data, parts)
    assert r.final["acc"] > 0.75 and np.isfinite(r.w).all()


def test_fedavg_weighted_aggregation():
    D = np.random.default_rng(0).normal(size=(5, 9)).astype(np.float32); W = np.array([1., 2., 3., 4., 5.])
    assert np.allclose(WeightedMean().aggregate(D, W), (W / W.sum()) @ D, atol=1e-6)


def test_fedavg_single_client_equals_local_sgd(data):
    parts = [np.arange(len(data.y_train))]
    r = run_fl(FLConfig(rounds=1, local_epochs=1, seed=3), data, parts)
    m = build_model(40, 32, seed=3); gen = torch.Generator().manual_seed(3 * 1_000_003 + 1009)
    w, _ = local_train(m, get_flat(m), torch.from_numpy(data.X_train), torch.from_numpy(data.y_train), 1, 0.05, 32, gen)
    assert np.allclose(r.w, w, atol=1e-6)


def test_fedprox_mu0_equals_fedavg_and_mu_shrinks_drift(data):
    parts = partition_dirichlet(data.y_train, 6, 0.3, 0)
    a = run_fl(FLConfig(algo="fedavg", rounds=3, seed=1), data, parts); b = run_fl(FLConfig(algo="fedprox", mu=0.0, rounds=3, seed=1), data, parts)
    assert np.array_equal(a.w, b.w)
    m = build_model(40, 32); w0 = get_flat(m); X, y = torch.from_numpy(data.X_train[parts[0]]), torch.from_numpy(data.y_train[parts[0]])
    d = [np.linalg.norm(local_train(m, w0, X, y, 3, 0.05, 32, torch.Generator().manual_seed(0), mu=mu)[0] - w0) for mu in (0.0, 1.0, 20.0)]
    assert d[0] > d[1] > d[2]


def test_scaffold_control_variate_invariant(data):
    parts = partition_dirichlet(data.y_train, 6, 0.3, 0)
    r = run_fl(FLConfig(algo="scaffold", rounds=4, local_epochs=2), data, parts)
    c = np.stack([r.extras["c_local"][k] for k in range(6)]).mean(0)
    assert np.abs(r.extras["c_global"] - c).max() < 1e-5 and np.linalg.norm(c) > 0


def test_scaffold_doubles_communication(data):
    parts = partition_iid(data.y_train, 5, 0)
    a = run_fl(FLConfig(algo="fedavg", rounds=2), data, parts); b = run_fl(FLConfig(algo="scaffold", rounds=2), data, parts)
    assert b.totals["comm_mb"] == pytest.approx(2 * a.totals["comm_mb"])


def test_fednova_equals_fedavg_with_equal_tau(data):
    parts = partition_iid(data.y_train[:2800], 7, 0)
    sub = type(data)(data.X_train[:2800], data.y_train[:2800], data.X_test, data.y_test, data.meta)
    a = run_fl(FLConfig(algo="fedavg", rounds=3, seed=2), sub, parts); b = run_fl(FLConfig(algo="fednova", rounds=3, seed=2), sub, parts)
    assert np.abs(a.w - b.w).max() < 1e-5


def test_fednova_fixes_objective_inconsistency():
    rng = np.random.default_rng(0); a = np.array([1., 2., .5]); b = [rng.normal(size=3) for _ in range(3)]; p = np.array([.4, .3, .3])
    ws = sum(p[k] * a[k] * b[k] for k in range(3)) / sum(p[k] * a[k] for k in range(3))
    e_avg = np.linalg.norm(_quadratic_toy("fedavg", [1, 5, 40], a, b, p, rounds=1500) - ws)
    e_nova = np.linalg.norm(_quadratic_toy("fednova", [1, 5, 40], a, b, p, rounds=1500) - ws)
    assert e_nova < 0.1 * e_avg


def test_invalid_configs(data):
    for bad in (dict(algo="x"), dict(participation=0), dict(rounds=0), dict(lr=-1)):
        with pytest.raises(ValueError):
            FLConfig(**bad).validate()
    with pytest.raises(ValueError):
        set_flat(build_model(40), np.zeros(3))


def test_rounds_to_target():
    assert rounds_to_target([dict(round=1, acc=.5), dict(round=2, acc=.9)], "acc", .8) == 2 and rounds_to_target([dict(round=1, acc=.5)], "acc", .8) is None


def test_evaluate_handles_diverged_weights_without_crashing():
    d = make_dataset(n_samples=400, seed=0); m = build_model(d.n_features, 8, "relu", seed=0); w = get_flat(m)
    out = evaluate(m, np.full_like(w, np.nan), d.X_test, d.y_test)
    assert out["diverged"] == 1.0 and out["auc"] == 0.5 and 0.0 <= out["acc"] <= 1.0
    ok = evaluate(m, w, d.X_test, d.y_test)
    assert ok["diverged"] == 0.0 and 0.0 <= ok["auc"] <= 1.0


def _sites_or_skip(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # data files need network on first use (not shipped)
        pytest.skip(f"site data not reachable: {type(e).__name__}")


def test_natural_sites_structure_and_no_test_leakage():
    from federated_learning.preprocessing.multisite_heart import load_loso, load_sites, site_validation
    sd = _sites_or_skip(load_sites, 0)
    assert len(sd.parts) == 4 and sorted(np.concatenate(sd.parts).tolist()) == list(range(len(sd.y_train)))
    assert sd.X_train.shape[1] == 10 and np.isfinite(sd.X_train).all() and np.isfinite(sd.X_test).all()
    assert sum(len(y) for _, y in sd.site_tests) == len(sd.y_test)
    rates = sd.meta["site_positive_rate"]; assert max(rates) - min(rates) > 0.4          # natural label-prevalence shift
    v = site_validation(sd, 0.25, 0)
    assert len(v.X_train) + len(v.X_test) == len(sd.X_train) and len(v.parts) == 4
    lo = load_loso("switzerland"); assert len(lo.parts) == 3 and lo.X_test.shape[0] == 123 and "switzerland" not in lo.site_names


def test_natural_sites_missing_codes_handled():
    from federated_learning.preprocessing.multisite_heart import _site_raw
    X, y = _sites_or_skip(_site_raw, "switzerland")
    assert np.isnan(X[:, 4]).all()               # cholesterol == 0 for every Swiss patient is treated as missing
    assert set(np.unique(y)) <= {0, 1}
