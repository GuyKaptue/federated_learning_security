import numpy as np
import pytest
from federated_learning.feature_engineering import preprocess_split
from federated_learning.preprocessing.registry import SPECS, get_spec
from federated_learning.utils.hashing import sha256_arrays, sha256_bytes, sha256_file


def test_registry_documents_all_four_datasets():
    assert set(SPECS) == {"wisconsin_breast_cancer", "pima_diabetes", "cleveland_heart_disease", "uci_heart_disease_multisite"}
    assert get_spec("pima").n_samples == 768 and get_spec("wdbc").n_features == 30
    with pytest.raises(KeyError):
        get_spec("cic_ids_2017")
    for s in SPECS.values():
        assert s.license and s.citation and all(len(f.sha256) == 64 for f in s.files)


def test_hashing(tmp_path):
    p = tmp_path / "a.bin"; p.write_bytes(b"abc")
    assert sha256_file(p) == sha256_bytes(b"abc")
    assert sha256_arrays(np.arange(3)) == sha256_arrays(np.arange(3)) != sha256_arrays(np.arange(4))


def test_preprocess_uses_train_statistics_only():
    Xtr = np.array([[1., np.nan], [3., 2.], [5., 4.]]); Xte = np.array([[100., np.nan]])
    a, b = preprocess_split(Xtr, Xte)
    assert np.isfinite(a).all() and np.isfinite(b).all() and abs(a.mean(0)).max() < 1e-5 and b[0, 0] > 10


def test_raw_files_are_hash_verified(monkeypatch, tmp_path):
    from federated_learning.preprocessing import download
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    f = get_spec("pima").files[0]
    (tmp_path / "pima_diabetes").mkdir(); (tmp_path / "pima_diabetes" / f.filename).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download.fetch_file("pima_diabetes", f)
