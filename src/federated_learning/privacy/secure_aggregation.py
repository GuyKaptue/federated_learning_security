"""Secure Aggregation with pairwise masking, Shamir-shared dropout recovery (Bonawitz et al., CCS 2017 design).

Implemented primitives (real, not simulated): X25519 key agreement, HKDF-SHA256, ChaCha20-Poly1305 authenticated
encryption of shares, SHAKE-256 as the mask PRG, Shamir secret sharing over GF(2^521-1).

Protocol (semi-honest server, honest-but-curious/dropping clients):
  R0 AdvertiseKeys : each client i publishes (c_pk_i, s_pk_i)
  R1 ShareKeys     : i Shamir-shares its mask-key s_sk_i and self-mask seed b_i among its neighbours
                     (threshold t), each share AEAD-encrypted under the c-key shared with the recipient
  R2 MaskedInput   : y_i = q(x_i) + PRG(b_i) + sum_{j in N(i)} sign(i,j) PRG(s_ij)   (mod 2^64)
  R3 (survivor set U3 announced by the server)
  R4 Unmask        : for j in U2\\U3 survivors reveal shares of s_sk_j, for j in U3 shares of b_j
                     (a client NEVER reveals both kinds for the same j)
  Server: sum y_i, remove PRG(b_j) for j in U3 and the dangling pair masks of dropped clients.

`degree=None` -> complete graph (original protocol). `degree=k` -> sparse random k-regular-ish neighbour graph
(SecAgg+-style communication reduction); then threshold and dropout tolerance apply per neighbourhood.

NOT implemented: the signature/consistency round that stops a *malicious* server from lying about U3
(Bonawitz et al. active-server variant). Without it, a malicious server can break privacy.
"""
from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from federated_learning.aggregators.base import Aggregator

PRIME = 2 ** 521 - 1

SHARE_BYTES = 66 + 2          # y (521 bits) + x index

RAW = serialization.Encoding.Raw

PUB = serialization.PublicFormat.Raw

class SecAggAbort(Exception):
    """Raised when fewer than t clients survive (privacy-preserving abort)."""

def shamir_split(secret: int, t: int, xs: list[int]) -> dict[int, int]:
    if not 0 <= secret < PRIME:
        raise ValueError("secret out of field")
    if t < 1 or t > len(xs) or len(set(xs)) != len(xs) or 0 in xs:
        raise ValueError("invalid sharing parameters")
    coeffs = [secret] + [secrets.randbelow(PRIME) for _ in range(t - 1)]
    out = {}
    for x in xs:
        y = 0
        for c in reversed(coeffs):
            y = (y * x + c) % PRIME
        out[x] = y
    return out

def shamir_reconstruct(points: dict[int, int]) -> int:
    xs = list(points)
    acc = 0
    for i in xs:
        num = den = 1
        for j in xs:
            if j != i:
                num = num * (-j) % PRIME
                den = den * (i - j) % PRIME
        acc = (acc + points[i] * num * pow(den, -1, PRIME)) % PRIME
    return acc

def prg(seed: bytes, length: int, tag: bytes = b"") -> np.ndarray:
    """SHAKE-256 expanded to `length` uint64 words."""
    import hashlib
    return np.frombuffer(hashlib.shake_256(seed + tag).digest(8 * length), dtype=np.uint64).copy()

def _kdf(shared: bytes, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(shared)

def quantize(x: np.ndarray, scale_bits: int) -> np.ndarray:
    q = np.rint(np.asarray(x, np.float64) * float(1 << scale_bits)).astype(np.int64)
    return q.view(np.uint64)

def dequantize(q: np.ndarray, scale_bits: int) -> np.ndarray:
    return q.view(np.int64).astype(np.float64) / float(1 << scale_bits)

@dataclass
class _Client:
    cid: int
    c_sk: X25519PrivateKey
    s_sk: X25519PrivateKey
    b_seed: bytes
    nbrs: list[int]
    inbox: dict | None = None      # sender -> (share_b, share_s)

    @property
    def c_pk(self) -> bytes: return self.c_sk.public_key().public_bytes(RAW, PUB)
    @property
    def s_pk(self) -> bytes: return self.s_sk.public_key().public_bytes(RAW, PUB)

def _neighbours(n: int, degree: int | None, seed: int) -> list[list[int]]:
    if degree is None or degree >= n - 1:
        return [[j for j in range(n) if j != i] for i in range(n)]
    if degree < 2 or degree % 2:
        raise ValueError("degree must be an even integer >= 2 (ring lattice on a random permutation)")
    perm = np.random.default_rng(seed).permutation(n)
    pos = np.empty(n, int); pos[perm] = np.arange(n)
    nb = []
    for i in range(n):
        p = pos[i]
        nb.append(sorted({int(perm[(p + d) % n]) for d in range(-degree // 2, degree // 2 + 1) if d != 0}))
    return nb

class SecureAggregationProtocol:
    def __init__(self, n_clients: int, vector_len: int, threshold_frac: float = 0.67,
                 degree: int | None = None, scale_bits: int = 24, graph_seed: int = 0):
        if n_clients < 3:
            raise ValueError("secure aggregation needs at least 3 clients")
        if not 0.5 < threshold_frac <= 1.0:
            raise ValueError("threshold_frac must be in (0.5, 1]")
        self.n, self.L, self.scale_bits = n_clients, vector_len + 1, scale_bits   # +1: weight slot
        self.nbrs = _neighbours(n_clients, degree, graph_seed)
        self.thr = {i: max(2, int(np.ceil(threshold_frac * (len(self.nbrs[i]) + 1)))) for i in range(n_clients)}
        self.server_view: dict[int, np.ndarray] = {}
        self.stats: dict = {}

    # -- the protocol ---------------------------------------------------------------
    def run(self, vectors: dict[int, np.ndarray], weights: dict[int, float],
            drop_before_masking: set[int] = frozenset(), drop_before_unmask: set[int] = frozenset(),
            tamper: tuple[int, int] | None = None, out_dtype=np.float32) -> tuple[np.ndarray, float]:
        """Returns (weighted-mean vector, total weight). Raises SecAggAbort / InvalidTag."""
        t0 = time.perf_counter()
        up = down = 0
        T = {}
        # R0
        cl = {i: _Client(i, X25519PrivateKey.generate(), X25519PrivateKey.generate(), os.urandom(32), self.nbrs[i])
              for i in range(self.n)}
        up += self.n * 64; down += sum(len(self.nbrs[i]) * 64 for i in range(self.n))
        T["r0_keygen"] = time.perf_counter() - t0
        # R1: shares (AEAD under pairwise c-key)
        t1 = time.perf_counter()
        mailbox: dict[int, dict[int, bytes]] = {i: {} for i in range(self.n)}
        for i in range(self.n):
            xs = [j + 1 for j in cl[i].nbrs] + [i + 1]
            sb = shamir_split(int.from_bytes(cl[i].b_seed, "big"), self.thr[i], xs)
            ss = shamir_split(int.from_bytes(cl[i].s_sk.private_bytes(RAW, serialization.PrivateFormat.Raw,
                                                                       serialization.NoEncryption()), "big"), self.thr[i], xs)
            for j in cl[i].nbrs:
                key = _kdf(cl[i].c_sk.exchange(X25519PublicKey.from_public_bytes(cl[j].c_pk)), b"sa-share")
                pt = i.to_bytes(4, "big") + j.to_bytes(4, "big") + sb[j + 1].to_bytes(66, "big") + ss[j + 1].to_bytes(66, "big")
                nonce = os.urandom(12)
                ct = nonce + ChaCha20Poly1305(key).encrypt(nonce, pt, i.to_bytes(4, "big") + j.to_bytes(4, "big"))
                if tamper == (i, j):
                    ct = ct[:-1] + bytes([ct[-1] ^ 1])
                mailbox[j][i] = ct
                up += len(ct); down += len(ct)
            cl[i].own_b = sb[i + 1]; cl[i].own_s = ss[i + 1]
        T["r1_share"] = time.perf_counter() - t1
        # R2: masked input
        t2 = time.perf_counter()
        for j in range(self.n):
            cl[j].inbox = {}
            for i, ct in mailbox[j].items():
                key = _kdf(cl[j].c_sk.exchange(X25519PublicKey.from_public_bytes(cl[i].c_pk)), b"sa-share")
                pt = ChaCha20Poly1305(key).decrypt(ct[:12], ct[12:], i.to_bytes(4, "big") + j.to_bytes(4, "big"))
                if not (int.from_bytes(pt[:4], "big") == i and int.from_bytes(pt[4:8], "big") == j):
                    raise SecAggAbort("share routing mismatch")
                cl[j].inbox[i] = (int.from_bytes(pt[8:74], "big"), int.from_bytes(pt[74:140], "big"))
        U2 = [i for i in range(self.n) if i not in drop_before_masking]
        masked: dict[int, np.ndarray] = {}
        for i in U2:
            x = np.concatenate([np.asarray(vectors[i], np.float64) * weights[i], [weights[i]]])
            y = quantize(x, self.scale_bits) + prg(cl[i].b_seed, self.L, b"self")
            for j in cl[i].nbrs:
                seed = _kdf(cl[i].s_sk.exchange(X25519PublicKey.from_public_bytes(cl[j].s_pk)), b"sa-mask")
                m = prg(seed, self.L, b"pair")
                y = y + m if i < j else y - m
            masked[i] = y
            up += 8 * self.L
        self.server_view = {i: v.copy() for i, v in masked.items()}
        T["r2_mask"] = time.perf_counter() - t2
        # R3/R4
        t3 = time.perf_counter()
        U3 = [i for i in U2]
        dropped = [i for i in range(self.n) if i not in U3]
        responders = [i for i in U3 if i not in drop_before_unmask]
        survivors_set = set(U3)
        sum_y = np.zeros(self.L, np.uint64)
        for i in U3:
            sum_y = sum_y + masked[i]
        for j in range(self.n):                                     # every client j must keep >= thr[j] responders
            need_b = j in survivors_set
            avail = [k for k in cl[j].nbrs if k in responders] + ([j] if j in responders and (j in survivors_set) else [])
            if len(avail) < self.thr[j] and (need_b or any(nb in survivors_set for nb in cl[j].nbrs)):
                raise SecAggAbort(f"client {j}: only {len(avail)} shares available, need {self.thr[j]}")
        for j in range(self.n):
            if j in survivors_set:                                  # remove self-mask of survivors
                pts = {k + 1: cl[k].inbox[j][0] for k in cl[j].nbrs if k in responders}
                if j in responders:
                    pts[j + 1] = cl[j].own_b
                b = shamir_reconstruct(dict(list(pts.items())[: self.thr[j]])).to_bytes(32, "big")
                sum_y = sum_y - prg(b, self.L, b"self")
                down += 0
                up += SHARE_BYTES * len(pts)
            else:                                                   # dropped: remove its dangling pair masks
                pts = {k + 1: cl[k].inbox[j][1] for k in cl[j].nbrs if k in responders}
                up += SHARE_BYTES * len(pts)
                if not any(nb in survivors_set for nb in cl[j].nbrs):
                    continue
                if len(pts) < self.thr[j]:
                    raise SecAggAbort(f"dropped client {j}: cannot reconstruct mask key")
                sk = X25519PrivateKey.from_private_bytes(
                    shamir_reconstruct(dict(list(pts.items())[: self.thr[j]])).to_bytes(32, "big"))
                for k in cl[j].nbrs:
                    if k in survivors_set:
                        seed = _kdf(sk.exchange(X25519PublicKey.from_public_bytes(cl[k].s_pk)), b"sa-mask")
                        m = prg(seed, self.L, b"pair")
                        sum_y = sum_y - m if k < j else sum_y + m   # undo k's contribution
        T["r4_unmask"] = time.perf_counter() - t3
        total = dequantize(sum_y, self.scale_bits)
        wtot = float(total[-1])
        if wtot <= 0:
            raise SecAggAbort("no surviving weight")
        self.stats = dict(up_bytes_total=int(up), down_bytes_extra=int(down), n_survivors=len(U3),
                          n_dropped=len(dropped), total_time_s=time.perf_counter() - t0, phase_times_s=T,
                          n_neighbours_mean=float(np.mean([len(v) for v in self.nbrs])),
                          threshold_mean=float(np.mean(list(self.thr.values()))))
        return (total[:-1] / wtot).astype(out_dtype), wtot

class SecAggAggregator(Aggregator):
    """Drop-in FL aggregator running the full protocol every round among the participating clients."""
    name = "secagg"
    is_secure = True

    def __init__(self, threshold_frac: float = 0.67, degree: int | None = None, dropout_rate: float = 0.0,
                 seed: int = 0, scale_bits: int = 24):
        super().__init__()
        self.threshold_frac, self.degree, self.dropout_rate, self.scale_bits = threshold_frac, degree, dropout_rate, scale_bits
        self.rng = np.random.default_rng(seed)

    def aggregate(self, deltas, weights, rnd=0):
        deltas, _ = self._check(deltas, weights)
        m, P = deltas.shape
        proto = SecureAggregationProtocol(m, P, self.threshold_frac, self.degree, self.scale_bits, graph_seed=rnd)
        drop = set(self.rng.choice(m, size=int(round(self.dropout_rate * m)), replace=False).tolist()) if self.dropout_rate else set()
        vec, _ = proto.run({i: deltas[i] for i in range(m)}, {i: float(weights[i]) for i in range(m)},
                           drop_before_masking=drop)
        self.last_stats = dict(proto.stats)
        self.last_stats["crypto_time_s"] = proto.stats["total_time_s"]
        return vec
