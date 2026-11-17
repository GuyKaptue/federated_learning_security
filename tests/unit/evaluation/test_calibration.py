import numpy as np
import pytest

from federated_learning.evaluation.calibration import brier, ece, reliability_curve


def test_perfect_and_worst():
    y = np.array([0, 1, 1, 0])
    assert brier(y, y.astype(float)) == 0 and ece(y, y.astype(float)) == 0
    assert brier(y, 1 - y.astype(float)) == 1


def test_constant_prediction_known_values():
    y = np.array([1] * 30 + [0] * 70)
    p = np.full(100, 0.3)
    assert ece(y, p) == pytest.approx(0.0, abs=1e-12)          # 0.3 predicted, 0.3 observed
    assert brier(y, p) == pytest.approx(0.3 * 0.49 + 0.7 * 0.09)


def test_overconfident_has_large_ece():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000); p = np.where(y == 1, 0.99, 0.01)
    flip = rng.random(5000) < 0.3
    p = np.where(flip, 1 - p, p)                                 # 30% of confident predictions are wrong
    assert ece(y, p) > 0.2


def test_reliability_bins_and_validation():
    r = reliability_curve(np.array([0, 1]), np.array([0.0, 1.0]), 10)
    assert sum(r["count"]) == 2 and r["count"][0] == 1 and r["count"][-1] == 1
    with pytest.raises(ValueError):
        reliability_curve(np.array([0]), np.array([1.5]))
    with pytest.raises(ValueError):
        reliability_curve(np.array([]), np.array([]))
