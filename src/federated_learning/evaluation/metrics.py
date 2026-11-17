from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from federated_learning.training.models import set_flat


def evaluate(model, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict:
    set_flat(model, w)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        loss = float(F.cross_entropy(logits, torch.from_numpy(y)))
        prob = torch.softmax(logits, 1)[:, 1].numpy()
    diverged = not bool(np.isfinite(prob).all())          # non-finite weights (e.g. a successful model-destroying attack)
    if diverged:
        prob = np.zeros_like(prob)                        # convention: constant prediction of class 0, AUC = 0.5
    pred = (prob >= 0.5).astype(int)
    out = dict(acc=float((pred == y).mean()), loss=loss,
               precision=float(precision_score(y, pred, zero_division=0)),
               recall=float(recall_score(y, pred, zero_division=0)),
               f1=float(f1_score(y, pred, zero_division=0)))
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    out["specificity"] = tn / max(tn + fp, 1)
    out["auc"] = (0.5 if diverged else float(roc_auc_score(y, prob))) if len(np.unique(y)) > 1 else float("nan")
    out["diverged"] = float(diverged)
    return out

def rounds_to_target(history: list, key: str = "acc", target: float = 0.9) -> int | None:
    for h in history:
        if key in h and h[key] >= target:
            return int(h["round"])
    return None
