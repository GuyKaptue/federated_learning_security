"""Rule committee: several rejecting rules vote on every update; the survivors are averaged with norm clipping.

This is a deliberately simple *combination* of existing detectors, with optional per-round random choice of which members vote.
It is NOT claimed to be new in concept: randomising the aggregation rule has been studied (Nabavirazavi et al., TrustCom 2023, alternating Krum / trimmed
mean) and adaptive hybrids of aggregation rules exist (Yue et al., arXiv:2409.06474, reference-set selection or cascades). What this module is used for
is a measured comparison under an explicit attacker-knowledge ladder (see docs/threat_analysis.md):
  * draw-aware attacker: simulates exactly the committee that will run (same random draw) -> reduces to a deterministic committee;
  * pool-aware attacker: knows the pool, k and the voting rule but not the private draw -> each of its simulations sees an independent random draw.

Voting: a row is rejected when at least ``vote`` of the drawn members flag it (``vote`` is capped at k). If every row would be rejected,
the committee fails over to the coordinate-wise median of all rows (so it never aggregates nothing).
Final aggregation: weighted mean of the accepted rows, each clipped to the median norm of the accepted rows.
"""
from __future__ import annotations

import time

import numpy as np

from federated_learning.aggregators.base import Aggregator


def flagged_rows(stats: dict, n: int) -> set | None:
    """Rows a rejecting rule discarded in its last call, or None if the rule does not reject (re-weights/clips only, or selects a single row)."""
    if "dropped_idx" in stats:
        return {int(i) for i in stats["dropped_idx"]}
    if "selected" in stats and len(stats["selected"]) > 1:
        return set(range(n)) - {int(i) for i in stats["selected"]}
    return None


class RuleCommittee(Aggregator):
    name = "rule_committee"

    def __init__(self, members: list[Aggregator], k: int | None = None, vote: int = 1, randomise: bool = True, seed: int = 0):
        super().__init__()
        if not members:
            raise ValueError("need at least one member")
        k = len(members) if k is None else k
        if not 1 <= k <= len(members) or vote < 1:
            raise ValueError("need 1 <= k <= len(members) and vote >= 1")
        self.members, self.k, self.vote, self.randomise = members, k, vote, randomise
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def set_round_context(self, ids, w) -> None:
        super().set_round_context(ids, w)
        for m in self.members:
            m.set_round_context(ids, w)

    def aggregate(self, deltas, weights, rnd=0):
        D, w = self._check(deltas, weights)
        t0 = time.perf_counter()
        n = D.shape[0]
        idx = self.rng.choice(len(self.members), size=self.k, replace=False) if (self.randomise and self.k < len(self.members)) else np.arange(self.k)
        votes = np.zeros(n)
        voters = 0
        for j in idx:
            m = self.members[int(j)]
            m.aggregate(D, w, rnd)
            fl = flagged_rows(m.last_stats or {}, n)
            if fl is not None:
                voters += 1
                votes[list(fl)] += 1
        thr = min(self.vote, max(1, voters))
        rej = votes >= thr if voters else np.zeros(n, bool)
        fail_open = bool(rej.all())
        X = D.astype(np.float64)
        if fail_open:
            out = np.median(X, axis=0)
            rej = np.zeros(n, bool)
        else:
            A = np.where(~rej)[0]
            nr = np.linalg.norm(X[A], axis=1)
            c = float(np.median(nr))
            Xc = X[A] * np.minimum(1.0, c / np.maximum(nr, 1e-12))[:, None]
            ww = w[A] / w[A].sum()
            out = (ww[:, None] * Xc).sum(0)
        dropped = np.where(rej)[0].tolist()
        self.last_stats = dict(agg_time_s=time.perf_counter() - t0, dropped_idx=dropped, n_dropped=len(dropped), members=[self.members[int(j)].name for j in idx],
                               voters=voters, fail_open=fail_open)
        return out.astype(np.float32)
