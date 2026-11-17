import numpy as np
import pytest

from federated_learning.privacy.smpc import MACCheckFailure, P, SMPCAggregator, SMPCSession, _centered, _to_field


def upd(n=6, L=40):
    r = np.random.default_rng(0)
    return r.normal(0, .1, (n, L)).astype(np.float32), r.integers(5, 50, n).astype(float)


@pytest.mark.parametrize("m", [2, 3, 5])
def test_sum_correct(m):
    D, W = upd()
    assert np.abs(SMPCAggregator(m).aggregate(D, W) - (W / W.sum()) @ D).max() < 1e-5


def test_beaver_multiplication():
    s = SMPCSession(3); a, b = np.array([3, -4, 7, 100]), np.array([5, 6, -9, 2])
    z = s.multiply(s.input(_to_field(a)), s.input(_to_field(b)))
    assert [int(v) for v in _centered(s.open(z))] == (a * b).tolist()


def test_mac_detects_tampered_share():
    s = SMPCSession(3); x = s.input(_to_field(np.arange(6)))
    def bad(v):
        v = v.copy(); v[1, 2] = (v[1, 2] + 1) % P; return v
    s.tamper_hook = bad
    with pytest.raises(MACCheckFailure):
        s.open(x)


def test_mac_detects_tampered_mac_share():
    s = SMPCSession(3); x = s.input(_to_field(np.arange(6)))
    x.mac[0, 2, 1] = (x.mac[0, 2, 1] + 5) % P
    with pytest.raises(MACCheckFailure):
        s.open(x)


def test_single_party_share_hides_secret():
    s = SMPCSession(3); x = s.input(_to_field(np.full(3000, 7)))
    top = np.array([int(v) >> 58 for v in x.val[1]]); counts = np.bincount(top, minlength=8)
    assert counts.min() > 150                       # roughly uniform over 8 buckets (expected 375)


def test_norm_bound_rejects_outlier():
    D, W = upd(); D[2] *= 100
    a = SMPCAggregator(3, norm_bound=3.0); a.aggregate(D, W)
    assert a.last_rejected == [2]


def test_invalid_configs_and_inputs():
    with pytest.raises(ValueError):
        SMPCSession(1)
    with pytest.raises(ValueError):
        SMPCAggregator(3).aggregate(np.ones((2, 4), np.float32), np.array([0.2, 0.3]))
    with pytest.raises(ValueError):
        SMPCAggregator(3).aggregate(np.zeros((0, 4), np.float32), np.zeros(0))
