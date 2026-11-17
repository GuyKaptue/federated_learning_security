import numpy as np
import pytest
from cryptography.exceptions import InvalidTag

from federated_learning.privacy.secure_aggregation import (
    SecAggAbort,
    SecAggAggregator,
    SecureAggregationProtocol,
    dequantize,
    prg,
    quantize,
    shamir_reconstruct,
    shamir_split,
)


def data(n, P=64, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(0, .1, (n, P)), r.integers(5, 50, n).astype(float)


def run(n, P=64, **kw):
    D, W = data(n, P)
    deg = kw.pop("degree", None)
    pr = SecureAggregationProtocol(n, P, degree=deg)
    out, w = pr.run({i: D[i] for i in range(n)}, {i: W[i] for i in range(n)}, **kw)
    return pr, out, D, W, w


def test_correct_sum_no_dropout():
    _, out, D, W, w = run(8)
    assert np.abs(out - (W / W.sum()) @ D).max() < 1e-6 and abs(w - W.sum()) < 1e-6


def test_dropout_before_masking_recovered():
    drop = {1, 4}
    _, out, D, W, _ = run(10, drop_before_masking=drop)
    keep = [i for i in range(10) if i not in drop]
    assert np.abs(out - (W[keep] / W[keep].sum()) @ D[keep]).max() < 1e-6


def test_dropout_before_unmask_recovered():
    _, out, D, W, _ = run(10, drop_before_unmask={2, 3})
    assert np.abs(out - (W / W.sum()) @ D).max() < 1e-6


def test_abort_below_threshold():
    with pytest.raises(SecAggAbort):
        run(10, drop_before_masking=set(range(7)))


def test_sparse_graph_correct():
    _, out, D, W, _ = run(30, degree=8, drop_before_masking={3, 9})
    keep = [i for i in range(30) if i not in {3, 9}]
    assert np.abs(out - (W[keep] / W[keep].sum()) @ D[keep]).max() < 1e-6


def test_server_view_is_masked():
    # masks are fresh randomness each run, so use a long vector: with 65 values |corr| > 0.2 happens ~11% of the time by chance,
    # with 4097 values the threshold is >12 standard deviations away
    pr, *_ , = run(6, P=4096)
    D, W = data(6, P=4096)
    x = quantize(np.r_[D[0] * W[0], W[0]], 24)
    y = pr.server_view[0]
    assert not np.array_equal(x, y)
    assert abs(np.corrcoef(y.view(np.int64).astype(float), x.view(np.int64).astype(float))[0, 1]) < 0.2
    assert 0.3 < ((y >> np.uint64(63)).astype(float)).mean() < 0.7


def test_tampered_share_detected():
    with pytest.raises(InvalidTag):
        run(6, tamper=(0, 1))


def test_shamir_threshold_property():
    s = 2 ** 200 + 12345; sh = shamir_split(s, 4, list(range(1, 8)))
    assert shamir_reconstruct({x: sh[x] for x in (2, 4, 5, 7)}) == s
    assert shamir_reconstruct({x: sh[x] for x in (1, 2, 3)}) != s


@pytest.mark.parametrize("n,thr", [(2, .67), (5, .4), (5, 1.5)])
def test_invalid_config(n, thr):
    with pytest.raises(ValueError):
        SecureAggregationProtocol(n, 8, threshold_frac=thr)


def test_prg_deterministic_and_seed_dependent():
    assert np.array_equal(prg(b"a" * 32, 10), prg(b"a" * 32, 10)) and not np.array_equal(prg(b"a" * 32, 10), prg(b"b" * 32, 10))


def test_quantize_roundtrip_negative():
    x = np.array([-0.5, 0.25, 1e-3]); assert np.abs(dequantize(quantize(x, 24), 24) - x).max() < 1e-7


def test_aggregator_interface_with_dropout():
    D, W = data(10, 32)
    a = SecAggAggregator(dropout_rate=0.2)
    out = a.aggregate(D.astype(np.float32), W)
    assert out.shape == (32,) and a.last_stats["n_dropped"] == 2
