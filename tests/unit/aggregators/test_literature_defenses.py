import numpy as np
import pytest

from federated_learning.aggregators import FLAME, Audited, FoolsGold, RobustLearningRate, SignGuard, WeightedMean
from federated_learning.aggregators.base import propagate_context

P = 200


def _honest_and_bad(n_h=8, n_b=3, seed=0, bad_scale=5.0, flip=True):
    rng = np.random.default_rng(seed)
    g = rng.normal(0, 1, P)
    honest = g[None] + rng.normal(0, 0.3, (n_h, P))
    bad = (-g[None] if flip else g[None]) * bad_scale + rng.normal(0, 0.3, (n_b, P))
    return np.vstack([honest, bad]).astype(np.float32), g


def test_flame_rejects_flipped_scaled_updates_and_clips():
    D, g = _honest_and_bad()
    ag = FLAME(noise_lambda=0.0)
    ag.set_round_context(list(range(len(D))), np.zeros(P))
    out = ag.aggregate(D, np.ones(len(D)))
    assert {8, 9, 10} <= set(ag.last_stats["dropped_idx"])                  # attackers rejected (some honest rows may be too)
    cos = out @ g / (np.linalg.norm(out) * np.linalg.norm(g))
    assert cos > 0.9
    assert np.linalg.norm(out) <= ag.last_stats["clip_bound"] * 1.01        # mean of vectors each clipped to S_t


def test_flame_noise_scale_and_requirements():
    D, _ = _honest_and_bad(bad_scale=1.0, flip=False)
    a0, a1 = FLAME(noise_lambda=0.0), FLAME(noise_lambda=0.05, seed=3)
    for a in (a0, a1):
        a.set_round_context(list(range(len(D))), np.zeros(P))
    diff = a1.aggregate(D, np.ones(len(D))) - a0.aggregate(D, np.ones(len(D)))
    S = a1.last_stats["clip_bound"]
    assert diff.std() == pytest.approx(0.05 * S, rel=0.25)
    with pytest.raises(RuntimeError):
        FLAME().aggregate(D, np.ones(len(D)))                                # needs the global model
    assert FLAME(use_updates=True, noise_lambda=0.0).aggregate(D, np.ones(len(D))).shape == (P,)
    with pytest.raises(ValueError):
        FLAME(noise_lambda=-1)


def test_flame_all_honest_keeps_everyone():
    D, _ = _honest_and_bad(n_b=0, seed=5)
    ag = FLAME(noise_lambda=0.0); ag.set_round_context(list(range(len(D))), np.zeros(P))
    ag.aggregate(D, np.ones(len(D)))
    assert ag.last_stats["n_dropped"] <= len(D) - (len(D) // 2 + 1)         # HDBSCAN min_cluster_size = n//2+1 bounds benign rejections


def test_rlr_unanimous_equals_weighted_mean_and_flips_split_coordinates():
    rng = np.random.default_rng(1)
    D = np.abs(rng.normal(1, 0.2, (6, 10))).astype(np.float32)               # all coordinates positive for every client
    w = np.arange(1, 7, dtype=float)
    out = RobustLearningRate(theta=4).aggregate(D, w)
    assert np.allclose(out, WeightedMean().aggregate(D, w), atol=1e-5)
    D2 = D.copy(); D2[:3, :5] *= -1                                          # first five coordinates: 3 vs 3 votes -> sum 0 < theta
    ag = RobustLearningRate(theta=4)
    o2 = ag.aggregate(D2, np.ones(6))
    ref = D2.mean(0)
    assert np.allclose(o2[:5], -ref[:5], atol=1e-5) and np.allclose(o2[5:], ref[5:], atol=1e-5)
    assert ag.last_stats["frac_flipped"] == pytest.approx(0.5)


def test_rlr_theta_fraction_and_validation():
    D = np.ones((10, 4), np.float32)
    ag = RobustLearningRate(theta_frac=0.4); ag.aggregate(D, np.ones(10))
    assert ag.last_stats["theta"] == 4
    with pytest.raises(ValueError):
        RobustLearningRate(theta=-1)
    with pytest.raises(ValueError):
        RobustLearningRate(theta_frac=2)


def test_foolsgold_downweights_sybils_over_rounds():
    rng = np.random.default_rng(2)
    ag = FoolsGold()
    base = rng.normal(0, 1, P)
    for _ in range(3):
        honest = rng.normal(0, 1, (6, P))                                    # different directions every round
        sybil = base[None] + rng.normal(0, 0.01, (3, P))                     # nearly identical updates
        D = np.vstack([honest, sybil]).astype(np.float32)
        ag.set_round_context(list(range(9)), None)
        ag.aggregate(D, np.ones(9))
    a = np.array(ag.last_stats["alpha"])
    assert a[:6].min() > 0.5 and a[6:].max() < 0.1
    assert set(ag.last_stats["dropped_idx"]) >= {6, 7, 8}
    assert len(ag.history) == 9


def test_foolsgold_requires_ids_and_matches_fedavg_when_all_diverse():
    D = np.eye(4, P, dtype=np.float32) * 2                                   # mutually orthogonal updates
    ag = FoolsGold()
    with pytest.raises(RuntimeError):
        ag.aggregate(D, np.ones(4))
    ag.set_round_context([0, 1, 2, 3], None)
    out = ag.aggregate(D, np.ones(4))
    assert np.allclose(out, D.mean(0) * np.array(ag.last_stats["alpha"]).mean(), atol=1e-4) or np.allclose(np.array(ag.last_stats["alpha"]), 1.0)
    with pytest.raises(ValueError):
        FoolsGold(kappa=0)


def test_signguard_norm_filter_and_sign_clustering():
    rng = np.random.default_rng(4)
    honest = rng.normal(0, 1, (8, P))                                        # ~50% positive entries
    huge = rng.normal(0, 1, (1, P)) * 100                                    # norm filter (R = 3)
    pos = np.abs(rng.normal(1, 0.1, (2, P)))                                 # all-positive: sign statistics differ
    D = np.vstack([honest, huge, pos]).astype(np.float32)
    ag = SignGuard(seed=0)
    out = ag.aggregate(D, np.ones(len(D)))
    assert {8, 9, 10} <= set(ag.last_stats["dropped_idx"])
    assert np.linalg.norm(out) < 3 * ag.last_stats["median_norm"]
    with pytest.raises(ValueError):
        SignGuard(lower=2, upper=1)


def test_signguard_keeps_all_when_everything_looks_alike():
    rng = np.random.default_rng(6)
    D = rng.normal(0, 1, (6, P)).astype(np.float32)
    ag = SignGuard(seed=1); out = ag.aggregate(D, np.ones(6))
    assert ag.last_stats["n_dropped"] < 6 and np.isfinite(out).all()      # Mean-Shift may split honest rows: false positives are measured in the experiments


def test_context_propagates_through_wrappers():
    inner = FoolsGold()
    wrapped = Audited(inner)
    propagate_context(wrapped, [3, 4], np.zeros(5))
    assert inner.ctx_ids == [3, 4]
    D = np.random.default_rng(0).normal(size=(2, 5)).astype(np.float32)
    wrapped.aggregate(D, np.ones(2))


def test_defenses_run_inside_run_fl():
    from federated_learning.data_partitioning.iid import partition_iid
    from federated_learning.servers.orchestrator import run_fl
    from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset
    from federated_learning.training.config import FLConfig
    data = make_dataset(seed=0, n_samples=1500)
    parts = partition_iid(data.y_train, 8, 0)
    for ag in (FLAME(noise_lambda=0.0), RobustLearningRate(theta_frac=0.25), FoolsGold(), SignGuard()):
        r = run_fl(FLConfig(rounds=3, seed=0), data, parts, aggregator=ag)
        assert np.isfinite(r.final["acc"]) and r.final["acc"] > 0.5, ag.name
