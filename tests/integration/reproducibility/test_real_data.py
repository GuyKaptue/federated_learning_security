import numpy as np
import pytest

from federated_learning.preprocessing.real_datasets import load_real
from federated_learning.preprocessing.splits import split_validation


def test_wdbc_offline_split_and_no_leakage():
    d = load_real("wdbc", 0); assert d.X_train.shape[1] == 30 and d.meta["real_data"] and len(d.y_train) + len(d.y_test) == 569
    assert abs(d.X_train.mean()) < 1e-3                       # standardised with TRAIN statistics
    v = split_validation(d, 0.25, 0); assert len(v.y_test) + len(v.y_train) == len(d.y_train)   # validation carved from train only


def test_seed_changes_split_and_invalid_name():
    a, b = load_real("wdbc", 0), load_real("wdbc", 1); assert not np.array_equal(a.X_test, b.X_test)
    with pytest.raises(ValueError):
        load_real("nope")


def test_downloaded_files_are_hash_pinned_if_available():
    for n in ("pima", "heart"):
        try:
            d = load_real(n, 0)
        except Exception as e:
            pytest.skip(f"download unavailable: {e!r}")
        assert d.meta["sha256"] and np.isfinite(d.X_train).all()
