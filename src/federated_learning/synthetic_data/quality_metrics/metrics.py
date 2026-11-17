from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def quality_report(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Sanity/quality indicators: finiteness, duplicate rows, max |feature correlation|, and the held-out AUC of a
    logistic regression (a learnability indicator -- not a claim about realism)."""
    X = np.asarray(X, float); y = np.asarray(y)
    corr = np.corrcoef(X, rowvar=False); np.fill_diagonal(corr, 0)
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); k = int(0.7 * len(y))
    tr, te = idx[:k], idx[k:]
    auc = float(roc_auc_score(y[te], LogisticRegression(max_iter=500).fit(X[tr], y[tr]).decision_function(X[te])))
    return {"n_samples": int(len(y)), "n_features": int(X.shape[1]), "all_finite": bool(np.isfinite(X).all()),
            "duplicate_rows": int(len(X) - len(np.unique(X, axis=0))), "positive_rate": float(y.mean()),
            "max_abs_feature_correlation": float(np.abs(corr).max()), "linear_probe_auc": auc}
