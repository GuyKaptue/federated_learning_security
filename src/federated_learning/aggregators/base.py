"""Aggregation rules / defences with a common interface.

Aggregator.aggregate(deltas[m,P], weights[m], rnd) -> delta[P]
`last_stats` may carry measured cost/diagnostics of the last call.
Every robust rule is only meaningful inside its own threat model (see docs).
"""
from __future__ import annotations

import numpy as np


class Aggregator:
    name = "base"

    def __init__(self):
        self.last_stats: dict = {}
        self.ctx_ids: list[int] | None = None   # client ids of the rows of the next `aggregate` call
        self.ctx_w: np.ndarray | None = None    # global model before this round (needed by FLAME)

    def set_round_context(self, ids, w) -> None:
        """Server-side side information for identity- or model-aware rules (FoolsGold, FLAME). Plain rules ignore it."""
        self.ctx_ids = None if ids is None else [int(i) for i in ids]
        self.ctx_w = None if w is None else np.asarray(w, dtype=np.float64)

    def aggregate(self, deltas: np.ndarray, weights: np.ndarray, rnd: int = 0) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _check(deltas, weights):
        deltas = np.asarray(deltas)
        if deltas.ndim != 2 or deltas.shape[0] == 0:
            raise ValueError("empty aggregation: need a [m>=1, P] array of updates")
        if not np.all(np.isfinite(deltas)):
            raise ValueError("non-finite update received")
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != deltas.shape[0] or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("invalid weights")
        return deltas, w / w.sum()


def propagate_context(agg, ids, w) -> None:
    """Hand the round context to `agg` and to every aggregator it wraps (Audited.inner, AnomalyFiltered.base, ...)."""
    seen = set()
    while agg is not None and id(agg) not in seen:
        seen.add(id(agg))
        if hasattr(agg, "set_round_context"):
            agg.set_round_context(ids, w)
        agg = getattr(agg, "inner", None) or getattr(agg, "base", None)
