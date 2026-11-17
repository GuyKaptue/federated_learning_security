import numpy as np
import pytest

from federated_learning.preprocessing.federated_scaler import FederatedScaler, pooled_stats, prepare_sites


def _clients(seed=0, K=4, d=5, nan=0.1, shift=True):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(K):
        n = int(rng.integers(40, 120))
        X = rng.normal(loc=(50.0 * k if shift else 0.0) + 100.0, scale=1.0 + k, size=(n, d))
        X[rng.random(X.shape) < nan] = np.nan
        out.append(X)
    return out


def test_secure_equals_pooled_reference():
    cl = _clients()
    fs = FederatedScaler.fit(cl, secure=True)
    mu, sd = pooled_stats(cl)
    assert np.abs(fs.mean_ - mu).max() < 1e-6
    assert np.abs(fs.scale_ - sd).max() < 1e-6         # + 1e-8 eps
    assert fs.stats["mode"] == "secagg" and fs.stats["rounds"] == 2 and fs.stats["up_bytes"] > 0


def test_plain_and_secure_agree():
    cl = _clients(1)
    a = FederatedScaler.fit(cl, secure=True); b = FederatedScaler.fit(cl, secure=False)
    assert np.abs(a.mean_ - b.mean_).max() < 1e-6 and np.abs(a.scale_ - b.scale_).max() < 1e-6


def test_large_mean_small_variance_is_stable():
    rng = np.random.default_rng(0)
    cl = [rng.normal(1e4, 0.5, size=(200, 3)) for _ in range(3)]
    fs = FederatedScaler.fit(cl)
    assert np.abs(fs.scale_ - pooled_stats(cl)[1]).max() < 1e-5      # E[x^2]-mu^2 would lose all digits here


def test_transform_imputes_with_global_mean_and_standardises():
    cl = _clients(2)
    fs = FederatedScaler.fit(cl)
    Z = fs.transform(np.vstack(cl))
    assert np.isfinite(Z).all()
    assert np.abs(Z.mean(0)).max() < 1e-3 and np.abs(Z.std(0) - 1).max() < 0.1


def test_validation_errors():
    with pytest.raises(ValueError):
        FederatedScaler.fit([np.zeros((5, 2))])
    with pytest.raises(ValueError):
        FederatedScaler.fit([np.zeros((5, 2)), np.zeros((5, 3))])
    a = np.full((5, 2), np.nan); b = np.full((5, 2), np.nan)
    with pytest.raises(ValueError):
        FederatedScaler.fit([a, b])
    with pytest.raises(OverflowError):
        FederatedScaler.fit([np.full((5, 2), 1e12), np.full((5, 2), 1e12)])


def test_prepare_sites_modes_do_not_use_test_rows():
    tr, te = _clients(3), _clients(4)
    base = prepare_sites(tr, te, "federated")
    te2 = [t * 1000 + 7 for t in te]                       # changing test rows must not change train transform
    alt = prepare_sites(tr, te2, "federated")
    assert all(np.array_equal(a, b) for a, b in zip(base[0], alt[0]))
    for mode in ("pooled_median", "pooled_mean", "local", "none"):
        a, b = prepare_sites(tr, te, mode)
        assert all(np.isfinite(x).all() for x in a + b)
    with pytest.raises(ValueError):
        prepare_sites(tr, te, "bogus")


def test_federated_matches_pooled_mean_downstream_inputs():
    tr, te = _clients(5), _clients(6)
    f, _ = prepare_sites(tr, te, "federated"); p, _ = prepare_sites(tr, te, "pooled_mean")
    assert max(np.abs(a - b).max() for a, b in zip(f, p)) < 1e-4
