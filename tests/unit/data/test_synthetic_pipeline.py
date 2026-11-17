import json
import numpy as np
import pytest
from federated_learning.synthetic_data import SynIDSConfig, generate_raw
from federated_learning.synthetic_data.drift_analysis import ks_per_feature, psi
from federated_learning.synthetic_data.pipelines import generate_and_validate, load_synthetic
from federated_learning.synthetic_data.quality_metrics import quality_report
from federated_learning.synthetic_data.validators import validate_syn_ids


def test_generation_is_deterministic_and_matches_spec():
    cfg = SynIDSConfig(n_samples=1000, seed=3)
    X1, y1 = generate_raw(cfg); X2, y2 = generate_raw(cfg)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2) and X1.shape == (1000, 40)
    assert validate_syn_ids(X1, y1, cfg)["passed"]


def test_validator_detects_violations():
    cfg = SynIDSConfig(n_samples=1000)
    X, y = generate_raw(cfg)
    assert not validate_syn_ids(X[:, :10], y, cfg)["passed"]
    assert not validate_syn_ids(X, np.zeros_like(y), cfg)["passed"]


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        generate_raw(SynIDSConfig(n_samples=10))


def test_pipeline_writes_outputs(tmp_path):
    r = generate_and_validate(SynIDSConfig(n_samples=800, seed=1), tmp_path)
    assert r["validation"]["passed"]
    X, y = load_synthetic(1, tmp_path)
    assert X.shape == (800, 40)
    card = json.loads((tmp_path / "metadata" / "syn_ids_v1_seed1.json").read_text())
    assert "COMPLETELY SYNTHETIC" in card["statement"] and card["npz_sha256"] == r["arrays_sha256"]
    with pytest.raises(FileNotFoundError):
        load_synthetic(99, tmp_path)


def test_quality_and_drift():
    X, y = generate_raw(SynIDSConfig(n_samples=1000))
    q = quality_report(X, y); assert q["all_finite"] and q["linear_probe_auc"] > 0.6
    assert psi(X[:, 0], X[:, 0]) < 1e-6 and psi(X[:, 0], X[:, 0] + 2 * X[:, 0].std()) > 0.25
    assert ks_per_feature(X, X)[0]["statistic"] == 0.0
