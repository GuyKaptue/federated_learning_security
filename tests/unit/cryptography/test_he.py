import importlib.util

import numpy as np
import pytest

from federated_learning.privacy.homomorphic import BFVAggregator, CKKSAggregator, PaillierAggregator, batching_prime

needs_tenseal = pytest.mark.skipif(importlib.util.find_spec("tenseal") is None, reason="optional dependency tenseal not installed")
needs_phe = pytest.mark.skipif(importlib.util.find_spec("phe") is None, reason="optional dependency phe not installed")


@pytest.fixture(scope="module")
def upd():
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.1, (4, 300)).astype(np.float32), np.array([10., 20., 30., 40.])


def ref(D, W):
    return (W / W.sum()) @ D


@needs_tenseal
def test_ckks_matches_plaintext(upd):
    D, W = upd
    out = CKKSAggregator().aggregate(D, W)
    assert np.abs(out - ref(D, W)).max() < 1e-5


@needs_tenseal
def test_ckks_server_context_has_no_secret_key():
    a = CKKSAggregator()
    assert a.keygen_info()["secret_in_server_ctx"] is False
    assert a.secret_ctx.has_secret_key()


@needs_tenseal
def test_ckks_multi_chunk(upd):
    D = np.random.default_rng(1).normal(0, .1, (3, 9000)).astype(np.float32)   # > 4096 slots => 3 chunks
    W = np.ones(3)
    assert np.abs(CKKSAggregator().aggregate(D, W) - ref(D, W)).max() < 1e-5


@needs_tenseal
def test_bfv_matches_plaintext_to_quantisation(upd):
    D, W = upd
    out = BFVAggregator().aggregate(D, W)
    assert np.abs(out - ref(D, W)).max() < 2e-5


@needs_tenseal
def test_bfv_overflow_detected():
    D = np.full((2, 8), 5000.0, np.float32)
    with pytest.raises(OverflowError):
        BFVAggregator().aggregate(D, np.ones(2))


@needs_tenseal
def test_batching_prime_valid():
    t = batching_prime(8192, 32)
    assert (t - 1) % (2 * 8192) == 0 and t > 2 ** 31


@needs_phe
def test_paillier_small():
    D = np.random.default_rng(2).normal(0, .1, (3, 4)).astype(np.float32)
    W = np.array([1., 2., 3.])
    assert np.abs(PaillierAggregator(1024).aggregate(D, W) - ref(D, W)).max() < 1e-6


@needs_phe
def test_wrong_key_does_not_decrypt():
    a, b = CKKSAggregator(), CKKSAggregator()
    x = np.linspace(-.1, .1, 16)
    ct = a.encrypt(x)
    try:
        y = b.decrypt(ct, 16)
    except Exception:
        return                                   # rejecting is also acceptable
    assert not np.allclose(y, x, atol=1e-3)


@needs_phe
def test_malformed_ciphertext_rejected():
    a = CKKSAggregator(); ct = a.encrypt(np.ones(8))
    with pytest.raises(Exception):
        a.decrypt([ct[0][: len(ct[0]) // 2]], 8)


@needs_tenseal
@pytest.mark.parametrize("bad", ["empty", "nan", "weights"])
def test_invalid_inputs(bad):
    a = CKKSAggregator()
    with pytest.raises(ValueError):
        if bad == "empty":
            a.aggregate(np.zeros((0, 4), np.float32), np.zeros(0))
        elif bad == "nan":
            a.aggregate(np.array([[np.nan, 1.0]], np.float32), np.ones(1))
        else:
            a.aggregate(np.ones((2, 4), np.float32), np.array([1.0, -1.0]))
