"""Homomorphic-encryption aggregation: CKKS and BFV (TenSEAL/Microsoft SEAL) and Paillier (python-paillier).

Threat model implemented here (documented in docs/fl_sec_001):
  * a *key holder* (not the aggregation server) owns the secret key; clients get the public/encryption context
  * the aggregation server only sees a public context + ciphertexts and adds ciphertexts
  * semi-honest server. A malicious server/key-holder collusion, threshold HE and key rotation are NOT implemented.

Weighted FedAvg: each client encrypts p_i * delta_i with p_i = n_i / sum(n_j) (weights are public).
"""
from __future__ import annotations

import time

import numpy as np

from federated_learning.aggregators.base import Aggregator


class _LazyTenSEAL:
    """Imports TenSEAL on first use, so `import federated_learning` works without the optional HE dependency."""

    _mod = None

    def __getattr__(self, name):
        if _LazyTenSEAL._mod is None:
            try:
                import tenseal
            except ImportError as e:
                raise ImportError("CKKS/BFV aggregation needs the optional dependency TenSEAL: pip install 'federated-learning-security[he]'") from e
            _LazyTenSEAL._mod = tenseal
        return getattr(_LazyTenSEAL._mod, name)


ts = _LazyTenSEAL()


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def batching_prime(poly_degree: int, bits: int = 32) -> int:
    """Smallest prime t > 2^(bits-1) with t = 1 (mod 2N) so BFV batching (SIMD slots) works."""
    step = 2 * poly_degree
    t = (1 << (bits - 1)) // step * step + 1
    while not _is_prime(t):
        t += step
    return t

class _HEBase(Aggregator):
    is_secure = True

    def __init__(self):
        super().__init__()
        self.enc_s = self.agg_s = self.dec_s = 0.0

    def _finish(self, up_bytes_total, n_clients, P, enc_t, agg_t, dec_t, ct_bytes_client):
        self.last_stats = dict(up_bytes_total=int(up_bytes_total), enc_time_s=enc_t, he_agg_time_s=agg_t,
                               dec_time_s=dec_t, ct_bytes_per_client=int(ct_bytes_client),
                               expansion=ct_bytes_client / (4 * P), crypto_time_s=enc_t + agg_t + dec_t,
                               enc_time_per_client_s=enc_t / n_clients)

class CKKSAggregator(_HEBase):
    """Approximate real-number encrypted aggregation (CKKS)."""
    name = "he_ckks"

    def __init__(self, poly_degree: int = 8192, coeff_bits=(60, 40, 40, 60), scale_bits: int = 40):
        super().__init__()
        self.poly_degree, self.coeff_bits, self.scale_bits = poly_degree, list(coeff_bits), scale_bits
        self.secret_ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_degree, coeff_mod_bit_sizes=self.coeff_bits)
        self.secret_ctx.global_scale = 2.0 ** scale_bits
        # no Galois keys: ciphertext addition needs no rotations (keeps the public context small)
        pub = self.secret_ctx.copy()
        pub.make_context_public()
        self.public_bytes = pub.serialize()
        self.server_ctx = ts.context_from(self.public_bytes)   # what the server holds: no secret key
        self.slots = poly_degree // 2

    def keygen_info(self) -> dict:
        return dict(scheme="CKKS", poly_degree=self.poly_degree, coeff_mod_bits=self.coeff_bits,
                    scale_bits=self.scale_bits, slots=self.slots, public_context_bytes=len(self.public_bytes),
                    secret_in_server_ctx=self.server_ctx.has_secret_key())

    def encrypt(self, vec: np.ndarray) -> list[bytes]:
        out = []
        for s in range(0, vec.size, self.slots):
            out.append(ts.ckks_vector(self.secret_ctx, vec[s:s + self.slots].astype(np.float64).tolist()).serialize())
        return out

    def server_sum(self, cts_per_client: list[list[bytes]]) -> list[bytes]:
        n_chunks = len(cts_per_client[0])
        res = []
        for c in range(n_chunks):
            acc = ts.ckks_vector_from(self.server_ctx, cts_per_client[0][c])
            for cl in cts_per_client[1:]:
                acc += ts.ckks_vector_from(self.server_ctx, cl[c])
            res.append(acc.serialize())
        return res

    def decrypt(self, cts: list[bytes], length: int) -> np.ndarray:
        parts = [np.asarray(ts.ckks_vector_from(self.secret_ctx, b).decrypt()) for b in cts]
        return np.concatenate(parts)[:length].astype(np.float32)

    def aggregate(self, deltas, weights, rnd=0):
        deltas, w = self._check(deltas, weights)
        m, P = deltas.shape
        t0 = time.perf_counter()
        cts = [self.encrypt((w[i] * deltas[i].astype(np.float64))) for i in range(m)]
        enc_t = time.perf_counter() - t0
        ct_client = sum(len(b) for b in cts[0])
        t1 = time.perf_counter(); summed = self.server_sum(cts); agg_t = time.perf_counter() - t1
        t2 = time.perf_counter(); out = self.decrypt(summed, P); dec_t = time.perf_counter() - t2
        self._finish(ct_client * m, m, P, enc_t, agg_t, dec_t, ct_client)
        return out

class BFVAggregator(_HEBase):
    """Exact integer encrypted aggregation (BFV) over fixed-point quantised updates."""
    name = "he_bfv"

    def __init__(self, poly_degree: int = 8192, plain_bits: int = 32, frac_bits: int = 20):
        super().__init__()
        self.poly_degree, self.frac_bits = poly_degree, frac_bits
        self.t = batching_prime(poly_degree, plain_bits)
        self.secret_ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_degree, plain_modulus=self.t)
        pub = self.secret_ctx.copy(); pub.make_context_public()
        self.public_bytes = pub.serialize()
        self.server_ctx = ts.context_from(self.public_bytes)
        self.slots = poly_degree

    def keygen_info(self) -> dict:
        return dict(scheme="BFV", poly_degree=self.poly_degree, plain_modulus=self.t, frac_bits=self.frac_bits,
                    slots=self.slots, public_context_bytes=len(self.public_bytes),
                    secret_in_server_ctx=self.server_ctx.has_secret_key())

    def _quant(self, x: np.ndarray) -> list[int]:
        return np.rint(x.astype(np.float64) * (1 << self.frac_bits)).astype(np.int64).tolist()

    def aggregate(self, deltas, weights, rnd=0):
        deltas, w = self._check(deltas, weights)
        m, P = deltas.shape
        bound = np.abs(w[:, None] * deltas).sum(0).max() * (1 << self.frac_bits)
        if bound >= self.t / 2:
            raise OverflowError(f"BFV plaintext modulus too small: |sum|={bound:.3g} >= t/2={self.t / 2:.3g}")
        t0 = time.perf_counter()
        cts = []
        for i in range(m):
            q = self._quant(w[i] * deltas[i].astype(np.float64))
            cts.append([ts.bfv_vector(self.secret_ctx, q[s:s + self.slots]).serialize()
                        for s in range(0, P, self.slots)])
        enc_t = time.perf_counter() - t0
        ct_client = sum(len(b) for b in cts[0])
        t1 = time.perf_counter()
        res = []
        for c in range(len(cts[0])):
            acc = ts.bfv_vector_from(self.server_ctx, cts[0][c])
            for cl in cts[1:]:
                acc += ts.bfv_vector_from(self.server_ctx, cl[c])
            res.append(acc.serialize())
        agg_t = time.perf_counter() - t1
        t2 = time.perf_counter()
        dec = np.concatenate([np.asarray(ts.bfv_vector_from(self.secret_ctx, b).decrypt(), dtype=np.int64) for b in res])[:P]
        dec = np.where(dec > self.t // 2, dec - self.t, dec)          # centred representative
        out = (dec.astype(np.float64) / (1 << self.frac_bits)).astype(np.float32)
        dec_t = time.perf_counter() - t2
        self._finish(ct_client * m, m, P, enc_t, agg_t, dec_t, ct_client)
        return out

class PaillierAggregator(_HEBase):
    """Additively homomorphic (Paillier) element-wise encryption. Very slow & large: ~one ciphertext per coordinate."""
    name = "he_paillier"

    def __init__(self, key_bits: int = 2048):
        super().__init__()
        from phe import paillier
        self._paillier = paillier
        self.key_bits = key_bits
        self.pub, self.priv = paillier.generate_paillier_keypair(n_length=key_bits)

    def keygen_info(self) -> dict:
        return dict(scheme="Paillier", key_bits=self.key_bits)

    def aggregate(self, deltas, weights, rnd=0):
        deltas, w = self._check(deltas, weights)
        m, P = deltas.shape
        t0 = time.perf_counter()
        enc = [[self.pub.encrypt(float(v)) for v in (w[i] * deltas[i].astype(np.float64))] for i in range(m)]
        enc_t = time.perf_counter() - t0
        ct_bytes = P * (2 * self.key_bits // 8)                      # one element of Z_{n^2} per coordinate (+ small exponent)
        t1 = time.perf_counter()
        acc = enc[0]
        for cl in enc[1:]:
            acc = [a + b for a, b in zip(acc, cl)]
        agg_t = time.perf_counter() - t1
        t2 = time.perf_counter()
        out = np.asarray([self.priv.decrypt(c) for c in acc], dtype=np.float32)
        dec_t = time.perf_counter() - t2
        self._finish(ct_bytes * m, m, P, enc_t, agg_t, dec_t, ct_bytes)
        return out
