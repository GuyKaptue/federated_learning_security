"""Synthetic intrusion-detection-style tabular data and federated partitioning.

DATASET_ID = "SYN-IDS-v1": a deterministic synthetic binary task (benign vs attack)
a COMPLETELY SYNTHETIC benchmark (scikit-learn make_classification) despite the "IDS" in its name. It is NOT real
intrusion data, NOT CIC-IDS-2017 (which was not available and is not used) and contains no patient data. Results on it must never be compared numerically to the source chapter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from federated_learning.preprocessing.dataset import Dataset

DATASET_ID = "SYN-IDS-v1"

@dataclass(frozen=True)
class SynIDSConfig:
    """All generator parameters of SYN-IDS-v1 (fully synthetic; scikit-learn make_classification)."""
    n_samples: int = 12000
    n_features: int = 40
    seed: int = 0
    class_sep: float = 0.9
    test_frac: float = 0.3
    weights: tuple = (0.6, 0.4)
    n_clusters_per_class: int = 3
    flip_y: float = 0.02


def generate_raw(cfg: SynIDSConfig) -> tuple[np.ndarray, np.ndarray]:
    """Unstandardised features and labels (float64 / int)."""
    if cfg.n_samples < 100 or cfg.n_features < 4:
        raise ValueError("n_samples >= 100 and n_features >= 4 required")
    n = cfg.n_features
    return make_classification(
        n_samples=cfg.n_samples, n_features=n, n_informative=max(4, n // 2), n_redundant=max(0, n // 5),
        n_clusters_per_class=cfg.n_clusters_per_class, weights=list(cfg.weights), class_sep=cfg.class_sep,
        flip_y=cfg.flip_y, random_state=cfg.seed)


def make_dataset(n_samples: int = 12000, n_features: int = 40, seed: int = 0,
                 class_sep: float = 0.9, test_frac: float = 0.3) -> Dataset:
    """Generate and standardise the synthetic dataset (70:30 split like the chapter)."""
    cfg = SynIDSConfig(n_samples=n_samples, n_features=n_features, seed=seed, class_sep=class_sep, test_frac=test_frac)
    X, y = generate_raw(cfg)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_frac, stratify=y, random_state=seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xte = ((Xte - mu) / sd).astype(np.float32)
    meta = dict(dataset_id=DATASET_ID, n_samples=n_samples, n_features=n_features, seed=seed,
                class_sep=class_sep, test_frac=test_frac, n_train=len(ytr), n_test=len(yte),
                positive_rate_train=float(ytr.mean()))
    return Dataset(Xtr, ytr.astype(np.int64), Xte, yte.astype(np.int64), meta)
