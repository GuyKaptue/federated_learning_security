from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])
