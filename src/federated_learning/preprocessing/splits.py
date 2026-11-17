from __future__ import annotations

from sklearn.model_selection import train_test_split

from federated_learning.preprocessing.dataset import Dataset


def split_validation(data: Dataset, frac: float = 0.25, seed: int = 0) -> Dataset:
    """Carve a validation set out of TRAIN (the test set is untouched) for hyper-parameter selection."""
    Xa, Xv, ya, yv = train_test_split(data.X_train, data.y_train, test_size=frac, stratify=data.y_train, random_state=seed)
    return Dataset(Xa, ya, Xv, yv, dict(data.meta, role="tuning (validation in place of test)", n_train=len(ya), n_test=len(yv)))
