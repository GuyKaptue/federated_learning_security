from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from federated_learning.training.models import MLP, set_flat


def model_grad(model: MLP, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean-CE gradient of the model at w on batch (X, y), flattened (what a client would upload as a gradient)."""
    set_flat(model, w); model.zero_grad()
    F.cross_entropy(model(torch.from_numpy(X)), torch.from_numpy(y)).backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()]).numpy().copy()

def analytic_first_layer_inversion(g: np.ndarray, d_in: int, hidden: int) -> np.ndarray:
    """Exact reconstruction for a single sample: dL/dW1[i,:] = delta_i x, dL/db1[i] = delta_i  =>  x = dW1[i]/db1[i]."""
    gW1 = g[: hidden * d_in].reshape(hidden, d_in)
    gb1 = g[hidden * d_in: hidden * d_in + hidden]
    i = int(np.argmax(np.abs(gb1)))
    if abs(gb1[i]) < 1e-12:
        return np.full(d_in, np.nan)
    return gW1[i] / gb1[i]

def dlg_attack(model: MLP, w: np.ndarray, g_obs: np.ndarray, B: int, d_in: int, labels: np.ndarray | None = None,
               iters: int = 300, seed: int = 0, restarts: int = 1) -> np.ndarray:
    """Gradient matching (DLG/iDLG): min_{X_hat} || grad L(X_hat, y) - g_obs ||^2 via L-BFGS. Needs a smooth activation.
    If labels is None and B == 1, the label is recovered from the sign of the last-layer bias gradient (iDLG)."""
    set_flat(model, w)
    params = list(model.parameters())
    g_t = torch.from_numpy(g_obs.astype(np.float32))
    if labels is None:
        if B != 1:
            raise ValueError("labels must be supplied for B > 1")
        n_out = params[-1].numel()
        labels = np.array([int(np.argmin(g_obs[-n_out:]))])          # dL/dlogit_c = p_c - 1[c=y] < 0 only at y
    y_t = torch.from_numpy(np.asarray(labels, np.int64))
    best, best_loss = None, float("inf")
    for r in range(restarts):
        torch.manual_seed(seed * 101 + r)
        xh = torch.randn(B, d_in, requires_grad=True)
        opt = torch.optim.LBFGS([xh], lr=1.0, max_iter=iters, history_size=50, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()  # noqa: B023 (closure is consumed within the same iteration)
            loss = F.cross_entropy(model(xh), y_t)  # noqa: B023
            gr = torch.autograd.grad(loss, params, create_graph=True)
            gv = torch.cat([x.reshape(-1) for x in gr])
            obj = ((gv - g_t) ** 2).sum()
            obj.backward()
            return obj
        try:
            opt.step(closure)
        except Exception:  # nosec B110
            pass
        gr_loss = float(closure().detach())
        if np.isfinite(gr_loss) and gr_loss < best_loss:
            best_loss, best = gr_loss, xh.detach().numpy().copy()
    return best if best is not None else np.full((B, d_in), np.nan)

def match_and_score(X_true: np.ndarray, X_hat: np.ndarray, success_rel_err: float = 0.1) -> dict:
    """Optimal one-to-one matching (Hungarian) then per-sample relative L2 error.
    success := relative L2 error <= success_rel_err. feat_acc := fraction of features within 0.1 abs error."""
    if np.any(~np.isfinite(X_hat)):
        return dict(success_rate=0.0, mean_rel_err=float("inf"), mse=float("inf"), feat_acc=0.0)
    C = ((X_true[:, None, :] - X_hat[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(C)
    xt, xh = X_true[r], X_hat[c]
    rel = np.linalg.norm(xt - xh, axis=1) / (np.linalg.norm(xt, axis=1) + 1e-12)
    return dict(success_rate=float((rel <= success_rel_err).mean()), mean_rel_err=float(rel.mean()),
                mse=float(((xt - xh) ** 2).mean()), feat_acc=float((np.abs(xt - xh) <= 0.1).mean()))
