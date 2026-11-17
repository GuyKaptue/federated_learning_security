from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from federated_learning.training.models import set_flat


def per_sample_features(model, w, X, y) -> np.ndarray:
    set_flat(model, w); model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X)); yt = torch.from_numpy(y)
        loss = F.cross_entropy(logits, yt, reduction="none").numpy()
        p = torch.softmax(logits, 1).numpy()
    ent = -(p * np.log(p + 1e-12)).sum(1)
    ptrue = p[np.arange(len(y)), y]
    return np.stack([loss, p.max(1), ent, ptrue, (p.argmax(1) == y).astype(float)], 1)

def mia_metrics(scores_in: np.ndarray, scores_out: np.ndarray, threshold: float | None = None) -> dict:
    """Higher score => 'member'. Reports AUC, TPR@FPR, advantage, precision/recall at `threshold`."""
    y = np.r_[np.ones(len(scores_in)), np.zeros(len(scores_out))]
    s = np.r_[scores_in, scores_out]
    auc = float(roc_auc_score(y, s))
    fpr, tpr, thr = roc_curve(y, s)
    out = dict(auc=auc, tpr_at_fpr_1pct=float(np.interp(0.01, fpr, tpr)), tpr_at_fpr_10pct=float(np.interp(0.10, fpr, tpr)),
               max_advantage=float((tpr - fpr).max()))
    if threshold is None:
        threshold = float(thr[np.argmax(tpr - fpr)])
    pred = s >= threshold
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum())
    out.update(threshold=threshold, precision=tp / max(tp + fp, 1), recall=tp / max((y == 1).sum(), 1),
               tpr=tp / max((y == 1).sum(), 1), fpr=fp / max((y == 0).sum(), 1))
    out["_roc"] = (fpr.tolist(), tpr.tolist())
    return out

def shadow_attack_scores(shadow_feats_in, shadow_feats_out, target_feats, seed: int = 0):
    """Train the attack model on shadow member/non-member features; return membership scores for target features."""
    Xs = np.r_[shadow_feats_in, shadow_feats_out]
    ys = np.r_[np.ones(len(shadow_feats_in)), np.zeros(len(shadow_feats_out))]
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed).fit(Xs, ys)
    return clf.predict_proba(target_feats)[:, 1]
