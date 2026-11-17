"""UCI heart-disease data with NATURAL site structure: four collection sites (Cleveland, Hungary, Switzerland, VA Long Beach).

The four site files of the UCI "Heart Disease" database (Janosi, Steinbrunn, Pfisterer, Detrano) are fetched from a public GitHub
mirror pinned to a commit and verified against SHA-256 hashes; they are NOT redistributed in the sprint ZIP (mirror licence not verified).
The sites differ naturally in label prevalence (positive rate ~0.46 / 0.36 / 0.93 / 0.74), missingness and measurement practice
(Switzerland: cholesterol recorded as 0 for every patient -> treated as missing). Each site = one federated client.

CAVEATS (stated, not hidden): these are four public research cohorts from the 1980s (Detrano et al.), not live hospital federations;
all 'clients' still run in one process; n=920 patients in total, so confidence intervals are wide; Switzerland has only 8 negatives.
Features used (10): age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak. slope/ca/thal are dropped (>=50% missing at 3 of 4 sites).
Target: num > 0 (any angiographic disease).
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from federated_learning.preprocessing.dataset import Dataset
from federated_learning.preprocessing.download import fetch_file
from federated_learning.preprocessing.registry import BASE_SITES, COMMIT_SITES, SPECS

COMMIT = COMMIT_SITES
BASE = BASE_SITES
_SPEC = SPECS["uci_heart_disease_multisite"]
_FILES = {f.filename: f for f in _SPEC.files}
SITES = {  # site name -> (file, sha256)
    "cleveland": ("cleveland14.csv", _FILES["cleveland14.csv"].sha256),
    "hungary": ("hungarian14r.csv", _FILES["hungarian14r.csv"].sha256),
    "switzerland": ("switzerland14.csv", _FILES["switzerland14.csv"].sha256),
    "va_long_beach": ("long_beach_va14.csv", _FILES["long_beach_va14.csv"].sha256),
}

SITE_NAMES = tuple(SITES)

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"]

@dataclass
class SiteData(Dataset):
    parts: list = field(default_factory=list)          # per-site index arrays into X_train
    site_names: tuple = SITE_NAMES
    site_tests: list = field(default_factory=list)     # per-site (X_test, y_test), same preprocessing

def _site_raw(name: str) -> tuple[np.ndarray, np.ndarray]:
    fn, _ = SITES[name]
    b = fetch_file(_SPEC.key, _FILES[fn]).read_bytes()      # downloads into data/raw/uci_heart_disease_multisite/ and verifies SHA-256
    df = pd.read_csv(io.BytesIO(b), header=None, na_values=["?", "-9", "-9.0"])
    X = df.iloc[:, :10].to_numpy(float)
    X[X[:, 4] == 0, 4] = np.nan                        # chol == 0 is a missing-value code
    y = (df.iloc[:, 13].to_numpy(float) > 0).astype(int)
    return X, y

def _prep(Xtr_list, Xte_list):
    A = np.vstack(Xtr_list)
    med = np.nanmedian(A, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    fill = lambda X: np.where(np.isnan(X), med, X)
    A = fill(A); mu, sd = A.mean(0), A.std(0) + 1e-8
    z = lambda X: ((fill(X) - mu) / sd).astype(np.float32)
    return [z(X) for X in Xtr_list], [z(X) for X in Xte_list]

def load_sites(seed: int = 0, test_frac: float = 0.3, prep: str = "pooled_median") -> SiteData:
    """Protocol A: every site split 70/30 (stratified); the 4 sites are the 4 clients; test = union of site test sets (+ per-site tests)."""
    tr, te = [], []
    for n in SITE_NAMES:
        X, y = _site_raw(n)
        a, b, c, d = train_test_split(X, y, test_size=test_frac, stratify=y, random_state=seed)
        tr.append((a, c)); te.append((b, d))
    return _build2(tr, te, dict(protocol="A within-site holdout", seed=seed, train_sites=SITE_NAMES, test_frac=test_frac, preprocessing=prep), prep=prep)

def _build2(tr, te, meta, prep: str = "pooled_median") -> SiteData:
    if prep == "pooled_median":
        Xtr_z, Xte_z = _prep([t[0] for t in tr], [t[0] for t in te])
    else:
        from federated_learning.preprocessing.federated_scaler import prepare_sites
        Xtr_z, Xte_z = prepare_sites([t[0] for t in tr], [t[0] for t in te], prep)
    ys = [t[1] for t in tr]
    sizes = np.cumsum([0] + [len(y) for y in ys])
    parts = [np.arange(sizes[i], sizes[i + 1]) for i in range(len(ys))]
    yt = np.concatenate([t[1] for t in te])
    m = dict(meta, dataset_id="REAL-HEART-4SITES", real_data=True, features=FEATURES, n_train=int(sum(len(y) for y in ys)), n_test=int(len(yt)),
             site_train_sizes=[int(len(y)) for y in ys], site_positive_rate=[float(y.mean()) for y in ys], source=BASE,
             sha256={k: v[1] for k, v in SITES.items()})
    return SiteData(np.vstack(Xtr_z), np.concatenate(ys).astype(np.int64), np.vstack(Xte_z), yt.astype(np.int64), m, parts=parts,
                    site_names=tuple(meta["train_sites"]), site_tests=[(x, t[1].astype(np.int64)) for x, t in zip(Xte_z, te)])

def load_loso(held_out: str) -> SiteData:
    """Protocol B (external validation): train on the 3 other sites (all their rows), test on ALL rows of the held-out site."""
    if held_out not in SITES:
        raise ValueError(f"held_out must be one of {SITE_NAMES}")
    tr = [_site_raw(n) for n in SITE_NAMES if n != held_out]
    te = [_site_raw(held_out)]
    return _build2(tr, te, dict(protocol="B leave-one-site-out", held_out=held_out, train_sites=tuple(n for n in SITE_NAMES if n != held_out)))

def site_validation(sd: SiteData, frac: float = 0.25, seed: int = 0) -> SiteData:
    """Validation split carved out of every training site (test untouched); same client structure."""
    tr, va = [], []
    for p in sd.parts:
        X, y = sd.X_train[p], sd.y_train[p]
        strat = y if min(np.bincount(y, minlength=2)) >= 2 else None
        a, b, c, d = train_test_split(X, y, test_size=frac, stratify=strat, random_state=seed)
        tr.append((a, c)); va.append((b, d))
    ys = [t[1] for t in tr]
    sizes = np.cumsum([0] + [len(y) for y in ys])
    parts = [np.arange(sizes[i], sizes[i + 1]) for i in range(len(ys))]
    return SiteData(np.vstack([t[0] for t in tr]), np.concatenate(ys).astype(np.int64), np.vstack([v[0] for v in va]),
                    np.concatenate([v[1] for v in va]).astype(np.int64), dict(sd.meta, role="validation in place of test"), parts=parts,
                    site_names=sd.site_names, site_tests=[(v[0], v[1].astype(np.int64)) for v in va])
