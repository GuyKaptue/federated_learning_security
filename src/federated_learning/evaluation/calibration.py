"""Probability calibration metrics (Brier score, expected calibration error, reliability curve).

Conventions (fixed, so numbers are comparable across runs): binary task, probability of the positive class,
ECE with 15 equal-width bins and the usual count-weighted |accuracy - confidence| form on the *positive-class
probability* (i.e. bin by p, compare with the observed positive rate). Diverged models (non-finite outputs)
return the constant-0 prediction convention used by :func:`federated_learning.evaluation.metrics.evaluate`.
"""
from __future__ import annotations

import numpy as np
import torch

from federated_learning.training.models import set_flat


def predict_proba(model, w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """P(class 1) for each row; zeros if the weights are non-finite."""
    set_flat(model, w)
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(np.asarray(X, np.float32))), 1)[:, 1].numpy().astype(np.float64)
    return p if np.isfinite(p).all() else np.zeros_like(p)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("y and p must be non-empty and the same shape")
    if p.min() < 0 or p.max() > 1:
        raise ValueError("p must lie in [0, 1]")
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    conf = np.bincount(idx, weights=p, minlength=n_bins)
    pos = np.bincount(idx, weights=y, minlength=n_bins)
    ok = count > 0
    return dict(count=count.tolist(), mean_p=np.where(ok, conf / np.maximum(count, 1), np.nan).tolist(),
                pos_rate=np.where(ok, pos / np.maximum(count, 1), np.nan).tolist())


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    r = reliability_curve(y, p, n_bins)
    c = np.asarray(r["count"], float)
    gap = np.abs(np.nan_to_num(np.asarray(r["mean_p"]) - np.asarray(r["pos_rate"])))
    return float((c * gap).sum() / c.sum())


def calibration_report(model, w: np.ndarray, X: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    p = predict_proba(model, w, X)
    return dict(brier=brier(y, p), ece=ece(y, p, n_bins))
