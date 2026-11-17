"""Adaptive (defence-aware, omniscient) model-poisoning attacks for the simulator.

Threat model (STRONG, an upper bound on what a real attacker can do): the f malicious clients collude and, in every round, see
the benign clients' updates that the aggregator receives (omniscient) and may replace their own updates before aggregation.
Implemented as an aggregator wrapper because the per-client hook of `run_fl` cannot see other clients' updates.
Requires full participation so that row index == client id (malicious = rows in `mal_rows`).

Attacks
* ALIE ("A Little Is Enough", Baruch et al. 2019): mean_benign - z * std_benign, z from the number of clients n and attackers f.
* MIN-MAX (Shejwalkar & Houmansadr 2021 style): mu + gamma * p with the largest gamma such that the malicious vector stays no farther
  from any benign update than the benign updates are from each other -> hard for distance-based defences (Krum, filters).
* NORM-MATCHED BACKDOOR: malicious clients do real backdoor training; the wrapper rescales their updates to the benign median norm
  (passes norm clipping / norm filters).
* MIN-MAX BACKDOOR: backdoor direction p = d_m - mu pushed as far as the distance constraint allows (passes distance filters).

Stronger attacks added in the second extension (same omniscient threat model, additionally the attacker KNOWS the server's aggregation rule):
* MIN-SUM (Shejwalkar & Houmansadr 2021): like MIN-MAX but the constraint is on the SUM of squared distances to the benign updates.
* FANG (Fang et al. 2020, full-knowledge directed deviation for coordinate-wise median / trimmed mean): per coordinate, push beyond the benign
  extreme opposite to the estimated benign update direction.
* AGR-TAILORED: the attacker SIMULATES the server rule on every candidate malicious vector (candidates = scaled directions with a gamma grid plus the
  ALIE / MIN-MAX / MIN-SUM / FANG vectors) and submits the one that minimises <aggregate, benign mean direction>, i.e. the strongest measured
  deviation against the actual defence. Unconstrained in gamma for the grid candidates -> an upper bound, not a realistic attacker.
* AGR-BACKDOOR: same simulation, but candidates are mu + gamma * (d_m - mu) (genuine backdoor direction) and the score is <aggregate, unit(d_m - mu)>.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.stats import norm

from federated_learning.aggregators.base import Aggregator

ATTACKS = ("none", "alie", "minmax", "norm_matched", "minmax_backdoor", "minsum", "fang", "agr_tailored", "agr_backdoor")

def alie_z(n: int, f: int) -> float:
    """z_max of Baruch et al.: s = floor(n/2 + 1) - f supporters needed; z = Phi^{-1}((n - s)/n)."""
    if not 0 < f < n / 2:
        raise ValueError("need 0 < f < n/2")
    s = int(np.floor(n / 2 + 1)) - f
    return float(norm.ppf((n - s) / n))

def alie_vector(benign: np.ndarray, n: int, f: int) -> np.ndarray:
    return (benign.mean(0) - alie_z(n, f) * benign.std(0)).astype(benign.dtype)

def _max_pairwise(benign: np.ndarray) -> float:
    X = benign.astype(np.float64)
    sq = (X ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
    return float(np.sqrt(d2.max()))

def minmax_gamma(benign: np.ndarray, mu: np.ndarray, p: np.ndarray, iters: int = 40) -> float:
    """Largest gamma >= 0 with max_i ||mu + gamma p - b_i|| <= max_{i,j} ||b_i - b_j||  (bisection)."""
    thr = _max_pairwise(benign)
    B = benign.astype(np.float64)
    mu, p = mu.astype(np.float64), p.astype(np.float64)
    ok = lambda g: np.linalg.norm(mu + g * p - B, axis=1).max() <= thr
    lo, hi = 0.0, 1.0
    while ok(hi) and hi < 1e6:
        lo, hi = hi, hi * 2
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return float(lo)

def minmax_vector(benign: np.ndarray, direction: str = "inverse_unit") -> np.ndarray:
    mu = benign.mean(0)
    if direction == "inverse_unit":
        p = -mu / (np.linalg.norm(mu) + 1e-12)
    elif direction == "inverse_std":
        p = -benign.std(0)
    else:
        raise ValueError("direction must be inverse_unit|inverse_std")
    g = minmax_gamma(benign, mu, p)
    return (mu + g * p).astype(benign.dtype)

def minsum_gamma(benign: np.ndarray, mu: np.ndarray, p: np.ndarray, iters: int = 40) -> float:
    """Largest gamma >= 0 with sum_i ||mu + gamma p - b_i||^2 <= max_i sum_j ||b_i - b_j||^2 (bisection)."""
    B = benign.astype(np.float64)
    sq = (B ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * B @ B.T, 0)
    thr = float(d2.sum(1).max())
    mu, p = mu.astype(np.float64), p.astype(np.float64)
    ok = lambda g: ((mu + g * p - B) ** 2).sum() <= thr
    lo, hi = 0.0, 1.0
    while ok(hi) and hi < 1e6:
        lo, hi = hi, hi * 2
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return float(lo)

def minsum_vector(benign: np.ndarray) -> np.ndarray:
    mu = benign.mean(0)
    p = -mu / (np.linalg.norm(mu) + 1e-12)
    return (mu + minsum_gamma(benign, mu, p) * p).astype(benign.dtype)

def fang_vectors(benign: np.ndarray, f: int, b: float = 2.0, seed: int = 0) -> np.ndarray:
    """Directed-deviation attack (Fang et al. 2020, full knowledge, median / trimmed mean): [f, P] malicious vectors."""
    rng = np.random.default_rng(seed)
    s = np.sign(benign.mean(0))
    lo, hi = benign.min(0), benign.max(0)
    out = np.empty((f, benign.shape[1]), dtype=np.float64)
    for j in range(f):
        u = rng.uniform(1.0, b, size=benign.shape[1])
        below = np.where(lo > 0, lo / u, lo * u)        # just below the benign minimum
        above = np.where(hi > 0, hi * u, hi / u)        # just above the benign maximum
        out[j] = np.where(s > 0, below, above)          # move opposite to the benign direction
    return out.astype(benign.dtype)

GAMMA_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)

def agr_tailored_choose(inner: Aggregator, D: np.ndarray, weights, mal, benign: np.ndarray, backdoor_dirs: dict | None = None, seed: int = 0,
                        reseed_base: int | None = None):
    """Return ([f,P] malicious rows, info) maximising the simulated deviation of inner.aggregate against the benign direction."""
    mal = list(mal)
    f = len(mal)
    mu = benign.mean(0)
    nm = float(np.linalg.norm(mu)) + 1e-12
    u = mu / nm
    cands: dict[str, np.ndarray] = {}
    if backdoor_dirs is None:
        std = benign.std(0); sg = np.sign(mu)
        dirs = {"inv_unit": -u, "inv_std": -std / (np.linalg.norm(std) + 1e-12), "inv_sign": -sg / np.sqrt(len(sg))}
        for dn, d in dirs.items():
            for g in GAMMA_GRID:
                cands[f"{dn}*{g}"] = np.tile(mu + g * nm * d, (f, 1))
        n = D.shape[0]
        cands["alie"] = np.tile(alie_vector(benign, n, f), (f, 1)); cands["minmax"] = np.tile(minmax_vector(benign), (f, 1))
        cands["minsum"] = np.tile(minsum_vector(benign), (f, 1)); cands["fang"] = fang_vectors(benign, f, seed=seed)
        score_dir = u
    else:
        P_ = np.stack([D[i] - mu for i in mal])           # genuine backdoor directions
        pm = P_.mean(0); score_dir = pm / (np.linalg.norm(pm) + 1e-12)
        for g in GAMMA_GRID:
            cands[f"bd*{g}"] = mu + g * P_
        cands["bd*minmax"] = np.stack([mu + minmax_gamma(benign, mu, P_[k]) * P_[k] for k in range(f)])
        cands["bd*1(unchanged)"] = np.stack([D[i] for i in mal])
    best, best_s, best_name = None, np.inf, ""
    for ci, (name, M) in enumerate(cands.items()):
        sim = copy.deepcopy(inner)
        if reseed_base is not None and hasattr(sim, "reseed"):
            sim.reseed(reseed_base + ci)      # pool-aware attacker: its simulation of a randomised rule uses an independent draw, not the server's
        Ds = np.array(D, copy=True); Ds[mal] = M.astype(D.dtype)
        agg = sim.aggregate(Ds, weights, 0).astype(np.float64)
        sc = float(agg @ score_dir) if backdoor_dirs is None else -float(agg @ score_dir)
        if sc < best_s:
            best, best_s, best_name = M, sc, name
    return best.astype(D.dtype), dict(choice=best_name, score=float(best_s))

class AdaptiveAttackAggregator(Aggregator):
    """Wrap a defence; replace the malicious rows (omniscient) before delegating to it."""

    def __init__(self, inner: Aggregator, mal_rows, attack: str, n_total: int | None = None, knowledge: str = "draw", jitter: float = 0.0, seed: int = 0):
        super().__init__()
        if attack not in ATTACKS:
            raise ValueError(f"attack must be in {ATTACKS}")
        if knowledge not in ("draw", "pool") or jitter < 0:
            raise ValueError("knowledge in {draw, pool}; jitter >= 0")
        self.knowledge, self.jitter, self.seed = knowledge, jitter, seed
        self.inner, self.mal, self.attack, self.n_total = inner, sorted(set(mal_rows)), attack, n_total
        self.name = f"{inner.name}|{attack}"
        self.log: list[dict] = []

    def _rb(self, rnd: int):
        return (self.seed * 7919 + rnd) * 1000 + 17 if self.knowledge == "pool" else None

    def aggregate(self, deltas, weights, rnd=0):
        D, _ = self._check(deltas, weights)
        D = np.array(D, copy=True)
        n = D.shape[0]
        if self.mal and max(self.mal) >= n:
            raise ValueError("malicious row index out of range (requires full participation)")
        ben_idx = [i for i in range(n) if i not in set(self.mal)]
        if self.attack != "none" and self.mal:
            B = D[ben_idx]
            if self.attack == "alie":
                v = alie_vector(B, n, len(self.mal)); D[self.mal] = v
            elif self.attack == "minmax":
                v = minmax_vector(B); D[self.mal] = v
            elif self.attack == "norm_matched":
                tgt = float(np.median(np.linalg.norm(B.astype(np.float64), axis=1)))
                for i in self.mal:
                    D[i] = D[i] * (tgt / (np.linalg.norm(D[i]) + 1e-12))
            elif self.attack == "minmax_backdoor":
                mu = B.mean(0)
                for i in self.mal:
                    p = D[i] - mu
                    g = minmax_gamma(B, mu, p)
                    D[i] = (mu + g * p).astype(D.dtype)
                    self.log.append(dict(round=rnd, row=i, gamma=g))
            elif self.attack == "minsum":
                v = minsum_vector(B); D[self.mal] = v
            elif self.attack == "fang":
                D[self.mal] = fang_vectors(B, len(self.mal), seed=rnd)
            elif self.attack == "agr_tailored":
                M, info = agr_tailored_choose(self.inner, D, weights, self.mal, B, None, seed=rnd, reseed_base=self._rb(rnd))
                D[self.mal] = M; self.log.append(dict(round=rnd, **info))
            elif self.attack == "agr_backdoor":
                M, info = agr_tailored_choose(self.inner, D, weights, self.mal, B, {}, seed=rnd, reseed_base=self._rb(rnd))
                D[self.mal] = M; self.log.append(dict(round=rnd, **info))
        if self.jitter > 0 and self.attack != "none" and self.mal:
            # decorrelate the colluders: independent noise of relative size `jitter` per malicious row (pairwise cosine ~ 1/(1+jitter^2))
            rg = np.random.default_rng(self.seed * 100003 + rnd)
            for i in self.mal:
                z = rg.normal(size=D.shape[1])
                D[i] = (D[i] + self.jitter * np.linalg.norm(D[i]) * z / np.linalg.norm(z)).astype(D.dtype)
        out = self.inner.aggregate(D, weights, rnd)
        self.last_stats = dict(self.inner.last_stats)
        return out

class MedianNormClip(Aggregator):
    """Per-round adaptive clipping: clip every update to the median update norm, then average (a common 'norm-bound' defence)."""
    name = "clip_median_norm"

    def aggregate(self, deltas, weights, rnd=0):
        D, w = self._check(deltas, weights)
        X = D.astype(np.float64)
        nr = np.linalg.norm(X, axis=1)
        c = float(np.median(nr))
        X = X * np.minimum(1.0, c / np.maximum(nr, 1e-12))[:, None]
        self.last_stats = dict(clip=c, frac_clipped=float((nr > c).mean()))
        return (w[:, None] * X).sum(0).astype(np.float32)
