"""Federated feature standardisation: global mean / standard deviation without pooling any raw row.

Why this exists. The benchmark loaders in this package used to standardise with statistics of the *pooled* training set
(``make_dataset``, ``multisite_heart._prep``). In a real federation nobody holds the pooled set, so that step quietly
assumed a trusted centre that sees every row. Here the same statistics are obtained from per-client sums, and the
server only ever sees the *sum* over clients, because the sums travel through the Bonawitz-style secure aggregation
protocol of :mod:`federated_learning.privacy.secure_aggregation` (real masks, real key agreement; not a mock).

Two secure rounds, both over vectors of length 2d:
  round 1: per feature j   [ sum of observed x_j , number of observed x_j ]            -> global mean  mu_j
  round 2: per feature j   [ sum of (x_j - mu_j)^2 over observed , 0 ]                 -> global variance
The two-pass form (instead of E[x^2] - mu^2) avoids catastrophic cancellation when |mu| >> sigma.
Missing values (NaN) are handled per feature (observed-only moments) and imputed with the *global mean*.
(A global median, which the old pooled loader used for imputation, is not a sum and therefore not available
from one secure-sum round; the experiment reports the effect of this change.)

What it does NOT give you: the aggregate itself (n, mean, variance) is revealed to the server and, through the
broadcast, to every client. No differential privacy is added; with very few clients the aggregate can leak about
individuals (e.g. K=2). Malicious clients can distort the statistics by lying about their sums; no robust variant
is implemented here.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from federated_learning.privacy.secure_aggregation import SecureAggregationProtocol

_EPS = 1e-8


def _moments1(X: np.ndarray) -> np.ndarray:
    obs = ~np.isnan(X)
    return np.concatenate([np.where(obs, X, 0.0).sum(0), obs.sum(0).astype(np.float64)])


def _moments2(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    obs = ~np.isnan(X)
    d = np.where(obs, X - mu, 0.0)
    return np.concatenate([(d * d).sum(0), np.zeros(X.shape[1])])


def pooled_stats(client_X: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Reference (insecure, centralised) NaN-aware mean and population std of the pooled rows."""
    A = np.vstack([np.asarray(X, np.float64) for X in client_X])
    return np.nanmean(A, 0), np.nanstd(A, 0)


@dataclass
class FederatedScaler:
    mean_: np.ndarray
    scale_: np.ndarray
    stats: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, client_X: list[np.ndarray], secure: bool = True, threshold_frac: float = 0.67,
            scale_bits: int = 24, graph_seed: int = 0) -> FederatedScaler:
        """Fit from the clients' local TRAIN matrices. ``secure=False`` sums in the clear (reference/ablation)."""
        client_X = [np.asarray(X, np.float64) for X in client_X]
        K = len(client_X)
        d = client_X[0].shape[1]
        if K < 2 or any(X.ndim != 2 or X.shape[1] != d for X in client_X):
            raise ValueError("need >= 2 clients with the same number of features")
        up = down = 0
        t = 0.0

        def total(vecs: list[np.ndarray]) -> np.ndarray:
            nonlocal up, down, t
            bound = max(float(np.abs(v).max()) for v in vecs) * K * 2.0 ** scale_bits
            if bound >= 2.0 ** 62:
                raise OverflowError("fixed-point overflow: lower scale_bits or rescale the features")
            if not secure:
                return np.sum(vecs, 0)
            proto = SecureAggregationProtocol(K, 2 * d, threshold_frac, None, scale_bits, graph_seed=graph_seed)
            mean_vec, wtot = proto.run({i: vecs[i] for i in range(K)}, {i: 1.0 for i in range(K)}, out_dtype=np.float64)
            up += proto.stats["up_bytes_total"]; down += proto.stats["down_bytes_extra"]; t += proto.stats["total_time_s"]
            return mean_vec.astype(np.float64) * wtot

        # fixed-point (2^-scale_bits) is the only error source: the protocol returns float64 here
        s1 = total([_moments1(X) for X in client_X])
        n_obs = s1[d:]
        if np.any(n_obs < 1):
            raise ValueError("a feature has no observed value in any client")
        mu = s1[:d] / n_obs
        s2 = total([_moments2(X, mu) for X in client_X])
        var = s2[:d] / n_obs
        out = cls(mu, np.sqrt(var) + _EPS)
        out.stats = dict(mode="secagg" if secure else "plain", rounds=2, n_clients=K, n_features=d,
                         up_bytes=int(up), down_bytes=int(down), time_s=float(t), n_obs_total=int(n_obs.max()))
        return out

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64)
        Z = np.where(np.isnan(X), self.mean_, X)
        return ((Z - self.mean_) / self.scale_).astype(np.float32)


def prepare_sites(Xtr_list: list[np.ndarray], Xte_list: list[np.ndarray], mode: str, **kw) -> tuple[list, list]:
    """Standardise per-site train/test matrices under one of five regimes (test rows never enter any statistic).

    pooled_median : the original loader (pooled median imputation + pooled mean/std)           [centralised]
    pooled_mean   : pooled NaN-aware mean imputation + pooled mean/std                          [centralised oracle]
    federated     : FederatedScaler through secure aggregation                                  [no raw row leaves a site]
    local         : every site uses only its own train statistics (also for its own test rows)  [no communication]
    none          : NaN -> 0 and no scaling
    """
    if mode == "pooled_median":
        A = np.vstack(Xtr_list)
        med = np.nanmedian(A, 0); med = np.where(np.isnan(med), 0.0, med)
        fill = lambda X: np.where(np.isnan(X), med, X)  # noqa: E731
        A = fill(A); mu, sd = A.mean(0), A.std(0) + _EPS
        z = lambda X: ((fill(X) - mu) / sd).astype(np.float32)  # noqa: E731
        return [z(X) for X in Xtr_list], [z(X) for X in Xte_list]
    if mode in ("pooled_mean", "federated"):
        if mode == "pooled_mean":
            mu, sd = pooled_stats(Xtr_list)
            sc = FederatedScaler(mu, sd + _EPS)
        else:
            sc = FederatedScaler.fit(Xtr_list, **kw)
        return [sc.transform(X) for X in Xtr_list], [sc.transform(X) for X in Xte_list]
    if mode == "local":
        tr, te = [], []
        for A, B in zip(Xtr_list, Xte_list):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)       # an all-missing column (e.g. Switzerland cholesterol) -> 0 / 1
                mu = np.nanmean(A, 0); sd = np.nanstd(A, 0)
            mu = np.where(np.isnan(mu), 0.0, mu); sd = np.where(np.isnan(sd) | (sd < 1e-12), 1.0, sd)   # constant/all-missing column: no scaling
            f = lambda X, mu=mu, sd=sd: ((np.where(np.isnan(X), mu, X) - mu) / sd).astype(np.float32)  # noqa: E731
            tr.append(f(A)); te.append(f(B))
        return tr, te
    if mode == "none":
        f = lambda X: np.nan_to_num(np.asarray(X, np.float64), nan=0.0).astype(np.float32)  # noqa: E731
        return [f(X) for X in Xtr_list], [f(X) for X in Xte_list]
    raise ValueError(f"unknown preprocessing mode {mode!r}")
