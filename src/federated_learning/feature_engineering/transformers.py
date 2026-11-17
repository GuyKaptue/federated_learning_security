"""Leakage-safe feature transformers: fit on TRAIN rows only, then apply to any split."""
from __future__ import annotations

import numpy as np


class MedianImputer:
    """Replace NaN by the per-feature median of the fitting data (all-NaN columns -> 0)."""

    def fit(self, X: np.ndarray) -> "MedianImputer":
        med = np.nanmedian(X, axis=0)
        self.median_ = np.where(np.isnan(med), 0.0, med)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(X), self.median_, X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Standardizer:
    """z-score with train mean/std (+1e-8 for constant columns), returning float32."""

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_, self.std_ = X.mean(0), X.std(0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def preprocess_split(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute then standardise; statistics come from `X_train` only."""
    imp = MedianImputer().fit(X_train)
    a, b = imp.transform(X_train), imp.transform(X_test)
    sc = Standardizer().fit(a)
    return sc.transform(a), sc.transform(b)
