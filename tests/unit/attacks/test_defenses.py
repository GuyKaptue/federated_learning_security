import numpy as np
import pytest

from federated_learning.aggregators.audit import Audited
from federated_learning.aggregators.defenses import AnomalyFiltered, ClippedMean
from federated_learning.aggregators.mean import WeightedMean
from federated_learning.aggregators.robust import CoordinateMedian, Krum, TrimmedMean
from federated_learning.privacy.differential_privacy import ClientSideDP


def honest_plus_bad(n=10, f=2, P=30, seed=0):
    r = np.random.default_rng(seed); D = r.normal(1.0, 0.05, (n, P)).astype(np.float32); D[:f] = -50.0; return D, np.ones(n)


@pytest.mark.parametrize("agg", [CoordinateMedian(), TrimmedMean(0.25), Krum(2, 1), Krum(2, 6), AnomalyFiltered()])
def test_robust_rules_ignore_outliers(agg):
    D, W = honest_plus_bad(); out = agg.aggregate(D, W)
    assert np.abs(out - 1.0).max() < 0.2


def test_mean_is_broken_by_outliers():
    D, W = honest_plus_bad(); assert np.abs(WeightedMean().aggregate(D, W) - 1.0).max() > 5


def test_krum_assumption_enforced():
    with pytest.raises(ValueError):
        Krum(4, 1).aggregate(np.ones((8, 5), np.float32), np.ones(8))


def test_clipping_bounds_influence_and_noise_scales():
    D, W = honest_plus_bad(); a = ClippedMean(1.0, 0.0); out = a.aggregate(D, W)
    assert np.linalg.norm(out) <= 1.0 + 1e-5 and a.last_stats["frac_clipped"] > 0
    n1 = np.std(ClippedMean(1.0, 1.0, 0).aggregate(np.zeros((10, 4000), np.float32), np.ones(10)))
    assert 0.05 < n1 < 0.15                                   # sigma*C/m = 0.1


@pytest.mark.parametrize("agg", [WeightedMean(), CoordinateMedian(), TrimmedMean(0.1), ClippedMean(1.0), AnomalyFiltered()])
def test_edge_cases(agg):
    with pytest.raises(ValueError):
        agg.aggregate(np.zeros((0, 3), np.float32), np.zeros(0))
    with pytest.raises(ValueError):
        agg.aggregate(np.array([[np.inf, 0.0]], np.float32), np.ones(1))


def test_trimmed_invalid():
    with pytest.raises(ValueError):
        TrimmedMean(0.5)


def test_audited_and_clientdp():
    D, W = honest_plus_bad(f=0); a = Audited(WeightedMean()); a.aggregate(D, W); assert a.errors[0] == 0.0
    out = ClientSideDP(WeightedMean(), 100.0, 0.0).aggregate(D, W); assert np.abs(out - D.mean(0)).max() < 1e-5
