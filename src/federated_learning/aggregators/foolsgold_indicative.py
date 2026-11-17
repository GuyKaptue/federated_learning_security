"""FoolsGold with indicative-feature weighting (separate from the legacy `FoolsGold`, which is kept unchanged).

Source: Fung, Yoon, Beschastnikh, "Mitigating Sybils in Federated Learning Poisoning", arXiv:1808.04866 v5 (RAID 2020 version:
"The Limitations of Federated Learning in Sybil Settings"). What the paper states explicitly (Alg. 1, Sec. 5/6.2/6.4):
  * the history H_i is the sum of client i's updates; cs_ij is the cosine similarity of H_i and H_j *on the indicative features*, weighted by S_t,
    "the weight of indicative features at iteration t";
  * indicative features are found "by measuring the magnitude of model parameters in the output layer of the global model";
  * they "can be filtered (hard) or re-weighed (soft) based on their influence on the model, and are normalized across all classes";
  * v_i = max_j cs_ij; pardoning cs_ij *= v_i/v_j if v_j > v_i; alpha_i = 1 - max_j cs_ij; alpha /= max alpha;
    alpha = kappa (ln(alpha/(1-alpha)) + 0.5); values outside [0,1] are clipped; kappa = 1 in the paper's experiments.
What the paper does NOT give (our documented implementation choices, i.e. adaptations, not reproductions):
  * no formula for S_t; we use |W_out| of the global model before the round (weights only, not biases), divided per class (per output row) by the
    row maximum; 'soft' uses these values as per-feature weights on the histories before the cosine; 'hard' keeps the top `proportion` of the
    normalised values (weight 1) and drops the rest (weight 0). The paper reports (MNIST, Fig. 10) that hard filtering below 10% defeats its
    noise attack; it states no default, so 0.1 is our choice.
  * `mode='all'` (S_t = 1 on every parameter) is a diagnostic arm that isolates the edge-case handling below from the indicative features.
Differences from the legacy class (legacy unchanged): (i) similarity on the weighted output-layer slice; (ii) alpha = 0 is clipped to 0 (the limit of the
logit), whereas legacy sets any non-finite value, including ln(0) = -inf, to 1.0; (iii) if all alphas are 0 (max undefined) all ones are used, as in legacy.
Adaptation to this package: the model is a single-head 2-class MLP (flat parameter vector); multi-task heads are not supported and raise.
"""
from __future__ import annotations

import time

import numpy as np

from federated_learning.aggregators.literature_defenses import FoolsGold, _cos_matrix


def output_layer_weight_index(n_params: int, n_out: int, n_in: int, with_bias: bool = True) -> np.ndarray:
    """Flat indices of the final Linear layer's weight matrix (row-major [n_out, n_in]) in a PyTorch parameter vector whose last tensors are (W, b)."""
    tail = n_out * n_in + (n_out if with_bias else 0)
    if tail > n_params:
        raise ValueError("output layer larger than the parameter vector")
    start = n_params - tail
    return np.arange(start, start + n_out * n_in)


def output_layer_from_model(model) -> tuple[np.ndarray, int]:
    """(indices of the last Linear weight in parameters_to_vector order, n_out). Raises if the last two tensors are not (weight, bias) of a Linear."""
    import torch.nn as nn
    lin = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not lin:
        raise ValueError("model has no Linear output layer")
    last = lin[-1]
    n_params = int(sum(p.numel() for p in model.parameters()))
    if last.weight is not list(model.parameters())[-2] or last.bias is None:
        raise ValueError("last parameters are not (weight, bias) of the output Linear layer")
    return output_layer_weight_index(n_params, last.out_features, last.in_features), last.out_features


class FoolsGoldIndicative(FoolsGold):
    name = "foolsgold_indicative"

    def __init__(self, out_idx: np.ndarray | None, n_out: int = 2, mode: str = "soft", proportion: float = 0.1, kappa: float = 1.0):
        super().__init__(kappa)
        if mode not in ("soft", "hard", "all"):
            raise ValueError("mode in {soft, hard, all}")
        if mode == "hard" and not 0 < proportion <= 1:
            raise ValueError("proportion in (0, 1]")
        if mode != "all" and out_idx is None:
            raise ValueError("out_idx required unless mode='all'")
        if n_out < 1:
            raise ValueError("n_out >= 1")
        self.out_idx = None if out_idx is None else np.asarray(out_idx, dtype=np.int64)
        if self.out_idx is not None and self.out_idx.size % n_out:
            raise ValueError("output weight count not divisible by n_out (multi-head layouts are not supported)")
        self.n_out, self.mode, self.proportion = n_out, mode, proportion
        self.name = f"foolsgold_{mode}"

    def importance(self, w_global: np.ndarray) -> np.ndarray:
        """S_t over the output-layer weights (flattened row-major [n_out, n_in]); see module docstring for the documented choices."""
        W = np.abs(np.asarray(w_global, dtype=np.float64)[self.out_idx]).reshape(self.n_out, -1)
        mx = W.max(axis=1, keepdims=True)
        S = np.divide(W, mx, out=np.zeros_like(W), where=mx > 0)       # normalised across classes: each class row scaled to max 1
        if self.mode == "hard":
            k = max(1, int(np.ceil(self.proportion * S.size)))
            thr = np.sort(S, axis=None)[-k]
            S = (S >= thr).astype(np.float64)
        s = S.reshape(-1)
        return s if s.any() else np.ones_like(s)                       # all-zero output layer: no information, fall back to uniform

    def aggregate(self, deltas, weights, rnd=0):
        deltas, w = self._check(deltas, weights)
        t0 = time.perf_counter()
        D = deltas.astype(np.float64)
        n = D.shape[0]
        if self.ctx_ids is None or len(self.ctx_ids) != n:
            raise RuntimeError("FoolsGold needs the client ids of this round (round context)")
        if self.mode == "all":
            sl, s = slice(None), 1.0
        else:
            if self.ctx_w is None:
                raise RuntimeError("indicative features need the global model of this round (round context)")
            if self.out_idx.max() >= D.shape[1]:
                raise ValueError("output-layer index outside the update vector")
            sl, s = self.out_idx, self.importance(self.ctx_w)
        for i, cid in enumerate(self.ctx_ids):
            self.history[cid] = self.history.get(cid, 0.0) + D[i][sl]
        Hs = np.stack([self.history[cid] for cid in self.ctx_ids]) * s
        alpha = self._alpha_paper(Hs)
        out = ((alpha * w)[:, None] * D).sum(0)
        self.last_stats = dict(agg_time_s=time.perf_counter() - t0, alpha=alpha.tolist(), n_dropped=int((alpha <= 0).sum()),
                               dropped_idx=np.where(alpha <= 0)[0].tolist())
        return out.astype(np.float32)

    def _alpha_paper(self, H: np.ndarray) -> np.ndarray:
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
        if mx <= 0:
            return np.ones(n)
        a = a / mx
        out = np.empty(n)
        for i, x in enumerate(a):
            out[i] = 1.0 if x >= 1.0 else 0.0 if x <= 0.0 else float(np.clip(self.kappa * (np.log(x / (1.0 - x)) + 0.5), 0.0, 1.0))
        return out


def make_foolsgold(variant: str, out_idx: np.ndarray | None = None, n_out: int = 2, proportion: float = 0.1, kappa: float = 1.0) -> FoolsGold:
    """Config-selectable variants: legacy | paper_soft | paper_hard | allfeat_corrected."""
    if variant == "legacy":
        return FoolsGold(kappa)
    if variant == "paper_soft":
        return FoolsGoldIndicative(out_idx, n_out, "soft", kappa=kappa)
    if variant == "paper_hard":
        return FoolsGoldIndicative(out_idx, n_out, "hard", proportion=proportion, kappa=kappa)
    if variant == "allfeat_corrected":
        return FoolsGoldIndicative(None, n_out, "all", kappa=kappa)
    raise ValueError(f"unknown FoolsGold variant {variant!r}")
