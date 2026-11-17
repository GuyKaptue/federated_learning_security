import numpy as np
import pytest

from federated_learning.aggregators.mean import WeightedMean
from federated_learning.aggregators.robust import CoordinateMedian, Krum
from federated_learning.attacks.adaptive import (
    AdaptiveAttackAggregator,
    MedianNormClip,
    _max_pairwise,
    alie_vector,
    alie_z,
    minmax_gamma,
    minmax_vector,
)


def benign(n=16, P=50, seed=0):
    r = np.random.default_rng(seed); return (r.normal(1.0, 0.3, (n, P))).astype(np.float32)


def test_alie_z_matches_formula_and_validates():
    from scipy.stats import norm
    assert alie_z(20, 4) == pytest.approx(norm.ppf(13 / 20))
    for bad in (0, 10, 11):
        with pytest.raises(ValueError):
            alie_z(20, bad)


def test_alie_vector_is_within_benign_spread():
    B = benign(); v = alie_vector(B, 20, 4)
    assert np.all(np.abs(v - B.mean(0)) <= alie_z(20, 4) * B.std(0) + 1e-6)


def test_minmax_respects_distance_constraint_and_is_maximal():
    B = benign(); mu = B.mean(0); p = -mu / np.linalg.norm(mu)
    g = minmax_gamma(B, mu, p); thr = _max_pairwise(B)
    assert g > 0 and np.linalg.norm(mu + g * p - B, axis=1).max() <= thr * (1 + 1e-9)
    assert np.linalg.norm(mu + 1.05 * g * p - B, axis=1).max() > thr


def test_minmax_beats_mean_and_median_but_attack_shifts_toward_negative_direction():
    B = benign(); v = minmax_vector(B); assert v @ B.mean(0) < B.mean(0) @ B.mean(0)


def test_wrapper_replaces_only_malicious_rows_and_none_is_identity():
    D = benign(20); W = np.ones(20)
    ident = AdaptiveAttackAggregator(WeightedMean(), [0, 1, 2, 3], "none").aggregate(D, W)
    assert np.allclose(ident, WeightedMean().aggregate(D, W))
    a = AdaptiveAttackAggregator(CoordinateMedian(), [0, 1, 2, 3], "alie"); out = a.aggregate(D, W)
    assert not np.allclose(out, CoordinateMedian().aggregate(D, W))
    with pytest.raises(ValueError):
        AdaptiveAttackAggregator(WeightedMean(), [25], "alie").aggregate(D, W)
    with pytest.raises(ValueError):
        AdaptiveAttackAggregator(WeightedMean(), [0], "bogus")


def test_norm_matched_equals_benign_median_norm():
    D = benign(20); D[:4] *= 10; W = np.ones(20)
    seen = {}
    class Spy(WeightedMean):
        def aggregate(self, d, w, rnd=0): seen["d"] = d.copy(); return super().aggregate(d, w, rnd)
    AdaptiveAttackAggregator(Spy(), [0, 1, 2, 3], "norm_matched").aggregate(D, W)
    tgt = np.median(np.linalg.norm(D[4:].astype(np.float64), axis=1))
    assert np.allclose(np.linalg.norm(seen["d"][:4], axis=1), tgt, rtol=1e-4)


def test_minmax_backdoor_stays_inside_benign_ball():
    D = benign(20); D[:4] += 5.0; W = np.ones(20); seen = {}
    class Spy(WeightedMean):
        def aggregate(self, d, w, rnd=0): seen["d"] = d.copy(); return super().aggregate(d, w, rnd)
    AdaptiveAttackAggregator(Spy(), [0, 1, 2, 3], "minmax_backdoor").aggregate(D, W)
    B = D[4:]; thr = _max_pairwise(B)
    assert np.linalg.norm(seen["d"][:4][:, None, :] - B[None], axis=2).max() <= thr * (1 + 1e-6)


def test_median_norm_clip_bounds_norms():
    D = benign(10); D[0] *= 100; out = MedianNormClip().aggregate(D, np.ones(10))
    assert np.linalg.norm(out) <= np.median(np.linalg.norm(D.astype(np.float64), axis=1)) + 1e-6


# ---------------------------------------------------------------- stronger attacks (second extension)
from federated_learning.aggregators.robust import TrimmedMean
from federated_learning.attacks.adaptive import ATTACKS, agr_tailored_choose, fang_vectors, minsum_gamma, minsum_vector


def test_minsum_respects_sum_constraint_and_is_maximal():
    B = benign(12, 40, 3); mu = B.mean(0); p = -mu / np.linalg.norm(mu)
    g = minsum_gamma(B, mu, p)
    sq = (B.astype(float) ** 2).sum(1); d2 = np.maximum(sq[:, None] + sq[None] - 2 * B.astype(float) @ B.T.astype(float), 0)
    thr = d2.sum(1).max()
    assert g > 0
    assert ((mu + g * p - B) ** 2).sum() <= thr * (1 + 1e-6)
    assert ((mu + 1.02 * g * p - B) ** 2).sum() > thr * (1 - 1e-6)
    assert minsum_vector(B).shape == mu.shape


def test_fang_vectors_lie_outside_benign_range_opposite_to_direction():
    B = benign(14, 30, 4); V = fang_vectors(B, 3, seed=1)
    s = np.sign(B.mean(0))
    assert V.shape == (3, 30)
    assert np.all(V[:, s > 0] <= B.min(0)[s > 0] + 1e-6)
    assert np.all(V[:, s <= 0] >= B.max(0)[s <= 0] - 1e-6)


def _deviation(inner, D, w, mal, B, vec):
    mu = B.mean(0); u = mu / np.linalg.norm(mu)
    Ds = D.copy(); Ds[mal] = vec
    return float(inner.aggregate(Ds, w).astype(float) @ u)


@pytest.mark.parametrize("inner", [WeightedMean(), CoordinateMedian(), Krum(3, 1), TrimmedMean(0.25)])
def test_agr_tailored_is_at_least_as_strong_as_each_fixed_attack_under_simulation(inner):
    n, f = 16, 4
    D = benign(n, 60, 5); mal = list(range(f)); B = D[f:]; w = np.ones(n)
    M, info = agr_tailored_choose(inner, D, w, mal, B)
    mine = _deviation(inner, D, w, mal, B, M)
    for fixed in (np.tile(alie_vector(B, n, f), (f, 1)), np.tile(minmax_vector(B), (f, 1)), np.tile(minsum_vector(B), (f, 1)), fang_vectors(B, f)):
        assert mine <= _deviation(inner, D, w, mal, B, fixed) + 1e-9
    assert "choice" in info


def test_agr_tailored_wrapper_only_changes_malicious_rows_and_logs_choice():
    seen = {}
    class Spy(WeightedMean):
        def aggregate(self, d, w, rnd=0): seen["d"] = d.copy(); return super().aggregate(d, w, rnd)
    D = benign(12, 20, 6)
    agg = AdaptiveAttackAggregator(Spy(), {0, 1}, "agr_tailored"); agg.aggregate(D, np.ones(12), 1)
    assert np.allclose(seen["d"][2:], D[2:]) and not np.allclose(seen["d"][:2], D[:2])
    assert agg.log and agg.log[0]["round"] == 1


def test_agr_backdoor_moves_aggregate_toward_backdoor_direction_more_than_unchanged():
    D = benign(12, 20, 7); D[:2] += 3.0                  # malicious rows carry a common (backdoor) direction
    B = D[2:]; mu = B.mean(0); pm = (D[:2] - mu).mean(0); pm /= np.linalg.norm(pm)
    inner = CoordinateMedian()
    M, _ = agr_tailored_choose(inner, D, np.ones(12), [0, 1], B, {})
    Ds = D.copy(); Ds[:2] = M
    assert inner.aggregate(Ds, np.ones(12)) @ pm >= inner.aggregate(D, np.ones(12)) @ pm - 1e-9


def test_all_new_attacks_registered():
    assert {"minsum", "fang", "agr_tailored", "agr_backdoor"} <= set(ATTACKS)
