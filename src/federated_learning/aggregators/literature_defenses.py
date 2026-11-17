"""Four defences re-implemented from their original papers (no code copied from the authors' repositories).

Each class states what is taken from the paper, what is our adaptation to this package's interface
(``aggregate(deltas[m,P], weights[m]) -> delta[P]`` plus the round context set by the server), and what is NOT implemented.
References (verified against the venue pages / arXiv versions):

* FLAME   - T. D. Nguyen et al., "FLAME: Taming Backdoors in Federated Learning", USENIX Security 2022, pp. 1415-1432 (arXiv:2101.02281).
* RLR     - M. S. Ozdayi, M. Kantarcioglu, Y. R. Gel, "Defending against Backdoors in Federated Learning with Robust Learning Rate",
            AAAI 2021, vol. 35(10), pp. 9268-9276 (arXiv:2007.03767).
* FoolsGold - C. Fung, C. J. M. Yoon, I. Beschastnikh, "Mitigating Sybils in Federated Learning Poisoning" (arXiv:1808.04866); venue version
            "The Limitations of Federated Learning in Sybil Settings", RAID 2020, pp. 301-316.
* SignGuard - J. Xu, S.-L. Huang, L. Song, T. Lan, "Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering",
            ICDCS 2022, pp. 1223-1235 (arXiv:2109.05872). The method is named SignGuard in the paper's text.

These are research re-implementations on a small MLP with virtual clients. They are not guaranteed to reproduce the published numbers and
none of them is a security guarantee; the paper's own caveats are repeated in each docstring.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.cluster import HDBSCAN, MeanShift, estimate_bandwidth

from federated_learning.aggregators.base import Aggregator

_TINY = 1e-12


def _cos_matrix(A: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(A, axis=1, keepdims=True)
    U = A / np.maximum(n, _TINY)
    return U @ U.T


class FLAME(Aggregator):
    """Filtering (HDBSCAN on pairwise cosine distance of the *models*), clipping to the median update norm, adaptive Gaussian noise.

    From the paper: c_ij = 1 - cos(W_i, W_j) with W_i the client MODEL; HDBSCAN with min_cluster_size = n//2 + 1 and min_samples = 1 (appendix);
    S_t = median_i ||G_{t-1} - W_i|| over ALL n clients (also rejected ones); admitted updates are scaled by min(1, S_t / ||G_{t-1} - W_i||);
    the clipped admitted models are averaged with equal weight; noise N(0, (lambda * S_t)^2) is added to the global model, lambda = 0.001
    for the paper's image/NLP settings and 0.01 for its network-intrusion setting.
    Adaptation: we work on deltas and add the model-space noise to the delta (identical result). Cosine is taken on W_i = G + delta_i
    (needs the round context ``w``); ``use_updates=True`` switches to cosine on deltas (a common variant, NOT the paper's).
    Our choices: HDBSCAN ``allow_single_cluster=True`` (otherwise a fully honest round has no cluster); if no cluster is found every client is admitted;
    if several clusters exist the largest is used. Not implemented: the paper's secure two-party-computation version of the filtering.
    Caveat from the paper: the filter assumes fewer than 50% malicious clients.
    """
    name = "flame"

    def __init__(self, noise_lambda: float = 0.001, use_updates: bool = False, seed: int = 0):
        super().__init__()
        if noise_lambda < 0:
            raise ValueError("noise_lambda must be >= 0")
        self.lam, self.use_updates = noise_lambda, use_updates
        self.rng = np.random.default_rng(seed)

    def aggregate(self, deltas, weights, rnd=0):
        deltas, _ = self._check(deltas, weights)
        t0 = time.perf_counter()
        D = deltas.astype(np.float64)
        n = D.shape[0]
        if not self.use_updates and self.ctx_w is None:
            raise RuntimeError("FLAME needs the global model (round context); use the orchestrator or set use_updates=True")
        M = D if self.use_updates else D + self.ctx_w[None, :]
        norms = np.linalg.norm(D, axis=1)
        admitted = np.arange(n)
        n_clusters = 1
        if n >= 3:
            dist = np.clip(1.0 - _cos_matrix(M), 0.0, 2.0)
            np.fill_diagonal(dist, 0.0)
            lab = HDBSCAN(min_cluster_size=max(2, n // 2 + 1), min_samples=1, metric="precomputed", allow_single_cluster=True, copy=True).fit_predict(dist)
            ks = [k for k in set(lab.tolist()) if k >= 0]
            n_clusters = len(ks)
            if ks:
                big = max(ks, key=lambda k: int((lab == k).sum()))
                admitted = np.where(lab == big)[0]
        S = float(np.median(norms))
        gam = np.minimum(1.0, S / np.maximum(norms[admitted], _TINY))
        out = (D[admitted] * gam[:, None]).mean(0)
        if self.lam > 0:
            out = out + self.rng.normal(0.0, self.lam * S, size=out.shape)
        rej = sorted(set(range(n)) - set(admitted.tolist()))
        self.last_stats = dict(agg_time_s=time.perf_counter() - t0, n_dropped=len(rej), dropped_idx=rej, clip_bound=S, n_clusters=n_clusters)
        return out.astype(np.float32)


class RobustLearningRate(Aggregator):
    """Robust learning rate (Ozdayi et al.): per coordinate, move with +eta if |sum_k sgn(delta_k,i)| >= theta, else with -eta.

    Update = eta_theta (elementwise) * weighted mean of the updates (FedAvg weights n_k / n), exactly the paper's w_{t+1} = w_t + eta_theta * sum n_k d_k / n.
    The paper says theta may lie between K*F+1 and K-K*F (K agents per round, F corrupt fraction) and chose theta empirically per data set
    (e.g. 4, 7, 8); it calls the choice non-trivial. We expose ``theta`` (absolute) or ``theta_frac`` (theta = ceil(frac * m)); the default
    frac=0.4 is OUR choice, not a value from the paper. Not implemented: the paper's optional combination with clipping/noise and delayed activation.
    Caveats from the paper: RLR slows convergence, assumes a corrupt minority (F=0.1 in the paper), and gives no guarantee against every adaptive attack.
    """
    name = "rlr"

    def __init__(self, theta: int | None = None, theta_frac: float = 0.4, eta: float = 1.0):
        super().__init__()
        if theta is not None and theta < 0:
            raise ValueError("theta must be >= 0")
        if not 0 <= theta_frac <= 1 or eta <= 0:
            raise ValueError("theta_frac in [0,1], eta > 0")
        self.theta, self.theta_frac, self.eta = theta, theta_frac, eta

    def aggregate(self, deltas, weights, rnd=0):
        deltas, w = self._check(deltas, weights)
        t0 = time.perf_counter()
        m = deltas.shape[0]
        theta = self.theta if self.theta is not None else int(np.ceil(self.theta_frac * m))
        votes = np.abs(np.sign(deltas.astype(np.float64)).sum(0))
        lr = np.where(votes >= theta, self.eta, -self.eta)
        out = lr * (w[:, None] * deltas.astype(np.float64)).sum(0)
        self.last_stats = dict(agg_time_s=time.perf_counter() - t0, theta=int(theta), frac_flipped=float((lr < 0).mean()))
        return out.astype(np.float32)


class FoolsGold(Aggregator):
    """FoolsGold (Fung et al.): down-weight clients whose accumulated update history is suspiciously similar to another client's.

    Paper (Algorithm 1): H_i = sum of client i's past updates; cs_ij = cosine(H_i, H_j); v_i = max_j cs_ij; pardoning: if v_j > v_i then
    cs_ij *= v_i / v_j; alpha_i = 1 - max_j cs_ij, clipped to [0,1]; alpha_i /= max alpha; alpha_i = kappa * (ln(alpha_i / (1 - alpha_i)) + 0.5);
    values outside [0,1] are clipped (kappa = 1 in the paper's experiments). Learning rates alpha_i then scale the clients' updates.
    Adaptation: the paper sums scaled SGD updates; here the output is sum_i alpha_i p_i delta_i with p_i the FedAvg weights, so alpha = 1 for all
    reproduces FedAvg. History is kept per client id across rounds (needs the round context ``ids``); a client's first round has no history
    beyond the current update. A value alpha_i = 1 is set to 0.99 before the logit to avoid infinity (our handling: the paper's formula is undefined at 1).
    NOT implemented: the optional indicative-feature weighting of the output layer.
    Caveat from the paper: it targets sybils whose updates are similar; it does not defend against a single attacker or against sybils that
    deliberately diversify their updates.
    """
    name = "foolsgold"

    def __init__(self, kappa: float = 1.0):
        super().__init__()
        if kappa <= 0:
            raise ValueError("kappa must be > 0")
        self.kappa = kappa
        self.history: dict[int, np.ndarray] = {}

    def aggregate(self, deltas, weights, rnd=0):
        deltas, w = self._check(deltas, weights)
        t0 = time.perf_counter()
        D = deltas.astype(np.float64)
        n = D.shape[0]
        if self.ctx_ids is None or len(self.ctx_ids) != n:
            raise RuntimeError("FoolsGold needs the client ids of this round (round context)")
        H = []
        for i, cid in enumerate(self.ctx_ids):
            self.history[cid] = self.history.get(cid, 0.0) + D[i]
            H.append(self.history[cid])
        alpha = self._alpha(np.stack(H))
        out = ((alpha * w)[:, None] * D).sum(0)
        self.last_stats = dict(agg_time_s=time.perf_counter() - t0, alpha=alpha.tolist(), n_dropped=int((alpha <= 0).sum()),
                               dropped_idx=np.where(alpha <= 0)[0].tolist())
        return out.astype(np.float32)

    def _alpha(self, H: np.ndarray) -> np.ndarray:
        n = H.shape[0]
        if n < 2:
            return np.ones(n)
        cs = _cos_matrix(H) - np.eye(n)
        v = cs.max(1)
        for i in range(n):
            for j in range(n):
                if i != j and v[j] > v[i]:
                    cs[i, j] *= v[i] / v[j]
        a = np.clip(1.0 - cs.max(1), 0.0, 1.0)
        mx = a.max()
        a = a / mx if mx > 0 else np.ones(n)
        a[a >= 1.0] = 0.99
        with np.errstate(divide="ignore", invalid="ignore"):
            a = self.kappa * (np.log(a / (1.0 - a)) + 0.5)
        a[~np.isfinite(a)] = 1.0
        return np.clip(a, 0.0, 1.0)


class SignGuard(Aggregator):
    """SignGuard (Xu et al.): norm filter, sign-statistics clustering, clipped mean over the trusted set.

    Paper: S1 = {i : L <= ||g_i|| / M <= R} with M the median norm (L=0.1, R=3.0); on a random 10% of the coordinates compute each client's
    fractions of positive / zero / negative entries; fit Mean-Shift clustering on these features and take the largest cluster as S2;
    trusted set S = S1 intersect S2; output = mean over S of g_i * min(1, M / ||g_i||).
    Our choices (the text we verified does not fix them): Mean-Shift bandwidth by sklearn.estimate_bandwidth(quantile=0.5) with bin seeding
    and cluster_all=True; one fresh random coordinate subset per round shared by all clients; equal weights in the final mean (as in the paper);
    if S is empty we fall back to S1, then to all clients. Not implemented: the SignGuard-Sim / SignGuard-Dist feature variants.
    Our updates are model deltas (negative-gradient direction scaled by the local learning rate), which changes the sign statistics' meaning
    only by a global sign.
    Caveat from the paper: it assumes a malicious minority, and the clustering step relies on malicious updates having different sign statistics
    from honest ones; attackers that match honest sign statistics are not detected by that step.
    """
    name = "signguard"

    def __init__(self, lower: float = 0.1, upper: float = 3.0, coord_frac: float = 0.1, seed: int = 0):
        super().__init__()
        if not 0 < lower < upper or not 0 < coord_frac <= 1:
            raise ValueError("need 0 < lower < upper and coord_frac in (0, 1]")
        self.L, self.R, self.cf = lower, upper, coord_frac
        self.rng = np.random.default_rng(seed)

    def aggregate(self, deltas, weights, rnd=0):
        deltas, _ = self._check(deltas, weights)
        t0 = time.perf_counter()
        D = deltas.astype(np.float64)
        n, P = D.shape
        norms = np.linalg.norm(D, axis=1)
        M = float(np.median(norms))
        s1 = np.where((norms / max(M, _TINY) >= self.L) & (norms / max(M, _TINY) <= self.R))[0]
        idx = self.rng.choice(P, size=max(1, int(round(self.cf * P))), replace=False)
        sub = D[:, idx]
        feats = np.stack([(sub > 0).mean(1), (sub == 0).mean(1), (sub < 0).mean(1)], 1)
        s2 = np.arange(n)
        if n >= 3:
            bw = estimate_bandwidth(feats, quantile=0.5)
            if bw > 1e-9:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)        # sklearn: 'binning failed, using data points as seeds'
                    lab = MeanShift(bandwidth=bw, bin_seeding=True, cluster_all=True).fit_predict(feats)
                big = np.bincount(lab).argmax()
                s2 = np.where(lab == big)[0]
        trusted = np.intersect1d(s1, s2)
        if trusted.size == 0:
            trusted = s1 if s1.size else np.arange(n)
        sc = np.minimum(1.0, M / np.maximum(norms[trusted], _TINY))
        out = (D[trusted] * sc[:, None]).mean(0)
        rej = sorted(set(range(n)) - set(trusted.tolist()))
        self.last_stats = dict(agg_time_s=time.perf_counter() - t0, n_dropped=len(rej), dropped_idx=rej, median_norm=M)
        return out.astype(np.float32)
