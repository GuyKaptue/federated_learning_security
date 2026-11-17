import numpy as np
import pytest

from federated_learning.training import load_model, save_model


def test_roundtrip_and_tamper_detection(tmp_path):
    w = np.random.default_rng(0).normal(size=100).astype(np.float32)
    d = save_model("m", "v1", w, {"algo": "fedavg"}, {"acc": 0.9}, tmp_path)
    w2, card = load_model("m", "v1", tmp_path)
    assert np.array_equal(w, w2) and card["metrics"]["acc"] == 0.9
    np.savez_compressed(d / "weights.npz", w=w + 1)
    with pytest.raises(RuntimeError):
        load_model("m", "v1", tmp_path)
