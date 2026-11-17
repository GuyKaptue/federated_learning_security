import copy

import numpy as np
import pytest

from federated_learning.aggregators import FLAME, Audited, Krum, RuleCommittee, SignGuard, WeightedMean
from federated_learning.aggregators.defenses import AnomalyFiltered
from federated_learning.attacks.adaptive import AdaptiveAttackAggregator

P = 120


def _data(seed=0, n_h=12, n_b=4, scale=6.0):
    rng = np.random.default_rng(seed)
    g = rng.normal(0, 1, P)
    honest = g[None] + rng.normal(0, 0.3, (n_h, P))
    bad = -g[None] * scale + rng.normal(0, 0.3, (n_b, P))
    return np.vstack([honest, bad]).astype(np.float32), g


def _pool(n=16, f=4):
    return [AnomalyFiltered(), Krum(f, n - f), FLAME(noise_lambda=0.0), SignGuard()]


def test_committee_rejects_flipped_scaled_rows_and_recovers_direction():
    D, g = _data()
    c = RuleCommittee(_pool(), k=4, vote=2, randomise=False)
    c.set_round_context(list(range(16)), np.zeros(P))
    out = c.aggregate(D, np.ones(16))
    assert {12, 13, 14, 15} <= set(c.last_stats["dropped_idx"]) and not c.last_stats["fail_open"]
    assert out @ g / (np.linalg.norm(out) * np.linalg.norm(g)) > 0.95


def test_vote_threshold_monotone_union_vs_majority():
    D, _ = _data(1)
    c1 = RuleCommittee(_pool(), k=4, vote=1, randomise=False); c3 = RuleCommittee(_pool(), k=4, vote=3, randomise=False)
    for c in (c1, c3):
        c.set_round_context(list(range(16)), np.zeros(P)); c.aggregate(D, np.ones(16))
    assert set(c3.last_stats["dropped_idx"]) <= set(c1.last_stats["dropped_idx"])


def test_random_draw_is_private_seeded_and_varies():
    c = RuleCommittee(_pool(), k=2, vote=1, randomise=True, seed=5)
    D, _ = _data(2)
    c.set_round_context(list(range(16)), np.zeros(P))
    seq = []
    for r in range(8):
        c.aggregate(D, np.ones(16), r); seq.append(tuple(c.last_stats["members"]))
    assert len(set(seq)) > 1
    c2 = RuleCommittee(_pool(), k=2, vote=1, randomise=True, seed=5); c2.set_round_context(list(range(16)), np.zeros(P))
    seq2 = []
    for r in range(8):
        c2.aggregate(D, np.ones(16), r); seq2.append(tuple(c2.last_stats["members"]))
    assert seq == seq2                                              # same seed => same draws
    c2.reseed(99)
    assert c2.rng.bit_generator.state != c.rng.bit_generator.state


def test_deepcopy_predicts_next_draw_but_reseed_decouples():
    c = RuleCommittee(_pool(), k=1, randomise=True, seed=3)
    sim = copy.deepcopy(c)
    assert (sim.rng.choice(4, size=1, replace=False) == c.rng.choice(4, size=1, replace=False)).all()   # draw-aware attacker sees the real draw
    c = RuleCommittee(_pool(), k=1, randomise=True, seed=3)
    draws = []
    for s in range(40):
        sim = copy.deepcopy(c); sim.reseed(1000 + s); draws.append(int(sim.rng.choice(4)))
    assert len(set(draws)) > 1                                      # pool-aware simulation uses independent draws


def test_fail_open_to_median_and_validation():
    D = np.random.default_rng(0).normal(size=(6, 10)).astype(np.float32)

    class RejectAll(WeightedMean):
        name = "reject_all"

        def aggregate(self, d, w, rnd=0):
            out = super().aggregate(d, w, rnd)
            self.last_stats = dict(dropped_idx=list(range(len(d))))
            return out
    c = RuleCommittee([RejectAll()], vote=1, randomise=False)
    out = c.aggregate(D, np.ones(6))
    assert c.last_stats["fail_open"] and np.allclose(out, np.median(D, axis=0), atol=1e-6)
    for bad in (dict(members=[]), dict(members=[WeightedMean()], k=2), dict(members=[WeightedMean()], vote=0)):
        with pytest.raises(ValueError):
            RuleCommittee(**bad)


def test_context_forwarded_through_wrapper_to_members():
    from federated_learning.aggregators.base import propagate_context
    c = RuleCommittee(_pool(), k=2)
    propagate_context(Audited(c), [0, 1, 2], np.ones(P))
    assert all(m.ctx_ids == [0, 1, 2] for m in c.members)


def test_adaptive_jitter_decorrelates_and_knowledge_validated():
    D, _ = _data(3, n_h=12, n_b=4, scale=1.0)
    ids = [12, 13, 14, 15]
    cos = lambda A: np.mean([A[i] @ A[j] / (np.linalg.norm(A[i]) * np.linalg.norm(A[j])) for i in range(len(A)) for j in range(i)])  # noqa: E731
    seen = {}
    for jit in (0.0, 1.0):
        class Spy(WeightedMean):
            def aggregate(self, d, w, rnd=0):
                seen[jit] = np.array(d)[ids]
                return super().aggregate(d, w, rnd)
        AdaptiveAttackAggregator(Spy(), ids, "alie", jitter=jit).aggregate(D, np.ones(16))
    assert cos(seen[0.0]) > 0.99 and cos(seen[1.0]) < 0.8
    with pytest.raises(ValueError):
        AdaptiveAttackAggregator(WeightedMean(), ids, "alie", knowledge="oracle")
    with pytest.raises(ValueError):
        AdaptiveAttackAggregator(WeightedMean(), ids, "alie", jitter=-1)
