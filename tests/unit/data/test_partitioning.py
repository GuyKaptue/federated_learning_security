import numpy as np
import pytest
from federated_learning.data_partitioning import STRATEGIES, partition, simulate_missing

Y = np.array([0] * 600 + [1] * 400)


@pytest.mark.parametrize("s", [s for s in STRATEGIES if s != "site"])
def test_partitions_cover_all_samples_once(s):
    parts = partition(s, Y, 10, seed=0)
    allidx = np.sort(np.concatenate(parts))
    assert np.array_equal(allidx, np.arange(len(Y))) and len(parts) == 10


def test_stratified_keeps_class_ratio():
    for p in partition("stratified", Y, 10):
        assert abs(Y[p].mean() - 0.4) < 0.02


def test_label_shards_are_skewed_and_quantity_skew_unequal():
    sk = partition("label_shards", Y, 10, shards_per_client=1)
    assert max(abs(Y[p].mean() - 0.4) for p in sk) > 0.3
    sizes = [len(p) for p in partition("quantity_skew", Y, 10, sigma=1.5)]
    assert max(sizes) > 3 * min(sizes)


def test_site_partition_and_errors():
    sites = np.array([0, 1, 1, 2, 2, 2])
    assert [len(p) for p in partition("site", Y[:6], site_ids=sites)] == [1, 2, 3]
    with pytest.raises(ValueError):
        partition("site", Y)
    with pytest.raises(ValueError):
        partition("nope", Y)


@pytest.mark.parametrize("mech", ["mcar", "mar", "mnar"])
def test_missing_simulation(mech):
    X = np.random.default_rng(0).normal(size=(2000, 6))
    Xm = simulate_missing(X, 0.2, mech, seed=1)
    assert not np.isnan(X).any() and abs(np.isnan(Xm).mean() - 0.2) < 0.06
    assert np.array_equal(simulate_missing(X, 0.2, mech, seed=1), Xm, equal_nan=True)


def test_missing_invalid():
    with pytest.raises(ValueError):
        simulate_missing(np.zeros((3, 3)), 1.2)
    with pytest.raises(ValueError):
        simulate_missing(np.zeros((3, 3)), 0.1, "x")
