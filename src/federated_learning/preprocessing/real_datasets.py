"""Real, public, de-identified tabular medical datasets for the FL benchmark (small, published research datasets).

* wdbc  : Wisconsin Diagnostic Breast Cancer (569 x 30), bundled with scikit-learn (no download).
* pima  : Pima Indians Diabetes (768 x 8). Downloaded from a public GitHub mirror; zeros in physiologically impossible
          columns are treated as missing and imputed with the TRAIN median.
* heart : Cleveland heart-disease table (303 x 13 + target, 1 exact duplicate removed). Downloaded from a public GitHub mirror.

These are NOT hospital federations: partitions into 'clients' are synthetic (IID or Dirichlet label skew). Downloaded files
are stored under data/raw/<dataset>/ and verified against pinned SHA-256 hashes; they are NOT redistributed in the sprint ZIP.
Licence/terms of the mirrors were not verified -> do not redistribute; the original sources are UCI/Kaggle-style public releases.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from federated_learning.feature_engineering.transformers import preprocess_split
from federated_learning.preprocessing.dataset import Dataset
from federated_learning.preprocessing.download import fetch_file
from federated_learning.preprocessing.registry import get_spec

REAL_NAMES = ("wdbc", "pima", "heart")


def _raw(name: str):
    """Return (X, y, info) of the raw dataset: features with NaN for missing-value codes, binary labels, provenance info."""
    spec = get_spec(name)
    if name == "wdbc":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        return d.data.astype(float), d.target.astype(int), dict(source="sklearn.datasets.load_breast_cancer (bundled)", sha256=None)
    f = spec.files[0]
    b = fetch_file(spec.key, f).read_bytes()
    info = dict(source_url=f.url, sha256=f.sha256)
    if name == "pima":
        df = pd.read_csv(io.BytesIO(b), header=None)
        X = df.iloc[:, :8].to_numpy(float)
        X[:, 1:6][X[:, 1:6] == 0] = np.nan          # missing-value encoding (glucose, BP, skin, insulin, BMI)
        return X, df.iloc[:, 8].to_numpy(int), info
    df = pd.read_csv(io.BytesIO(b), encoding="utf-8-sig").drop_duplicates()
    return df.drop(columns="target").to_numpy(float), df["target"].to_numpy(int), info


def load_real(name: str, seed: int = 0, test_frac: float = 0.3) -> Dataset:
    if name not in REAL_NAMES:
        raise ValueError(f"name must be one of {REAL_NAMES}")
    X, y, info = _raw(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_frac, stratify=y, random_state=seed)
    Xtr, Xte = preprocess_split(Xtr, Xte)                            # train-only statistics (no leakage)
    meta = dict(dataset_id=f"REAL-{name.upper()}", n_samples=int(len(y)), n_features=int(X.shape[1]), seed=seed, test_frac=test_frac,
                n_train=int(len(ytr)), n_test=int(len(yte)), positive_rate_train=float(ytr.mean()), real_data=True, **info)
    return Dataset(Xtr, ytr.astype(np.int64), Xte, yte.astype(np.int64), meta)
