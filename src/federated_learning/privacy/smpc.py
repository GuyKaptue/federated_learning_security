"""SPDZ-style SMPC for federated aggregation: m computing parties, additive secret sharing over GF(2^61-1)
with information-theoretic MACs, Beaver-triple multiplication, authenticated opening with MAC check.

What is real:  additive sharing, MAC-authenticated shares (K independent global keys => ~61*K bit forging
               resistance), input via authenticated random masks, linear ops, Beaver multiplication, MAC-checked
               openings (tampering by a party is detected => abort with security-with-abort for the checked outputs).
What is NOT:   the *offline/preprocessing phase* is a TRUSTED DEALER here (real SPDZ generates masks/triples with
               HE or OT). Commitments before MAC-check opening are omitted (no rushing-adversary protection).
               This is not MP-SPDZ (its binaries could not be obtained in this environment); it follows the SPDZ
               design principle only. Dishonest majority: tolerates up to m-1 colluding computing parties.

Roles: FL clients are *input parties*; m aggregation servers are the computing parties. Compare with
secure aggregation, where the clients themselves cooperate via pairwise masks and one server learns the sum.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from federated_learning.aggregators.base import Aggregator

P = 2 ** 61 - 1

class MACCheckFailure(Exception):
    """A party's share/MAC was inconsistent: abort (active-adversary detection)."""

def _rand(n: int) -> np.ndarray:
    raw = os.urandom(16 * n)
    return np.array([int.from_bytes(raw[16 * i:16 * i + 16], "big") % P for i in range(n)], dtype=object)

def _to_field(q: np.ndarray) -> np.ndarray:
    return np.array([int(v) % P for v in q], dtype=object)

def _centered(v: np.ndarray) -> np.ndarray:
    return np.array([int(x) - P if int(x) > P // 2 else int(x) for x in v], dtype=object)

@dataclass
class Shared:
    """Authenticated additively shared vector: val[j] is party j's share; mac[k][j] its MAC share for key k."""
    val: np.ndarray        # (m, L)
    mac: np.ndarray        # (K, m, L)

    def __add__(self, o: "Shared") -> "Shared":
        return Shared((self.val + o.val) % P, (self.mac + o.mac) % P)

    def __sub__(self, o: "Shared") -> "Shared":
        return Shared((self.val - o.val) % P, (self.mac - o.mac) % P)

    def mul_public(self, c: int) -> "Shared":
        return Shared((self.val * c) % P, (self.mac * c) % P)

class SMPCSession:
    def __init__(self, m_parties: int = 3, n_mac: int = 2, frac_bits: int = 20):
        if m_parties < 2:
            raise ValueError("SMPC needs >= 2 computing parties")
        self.m, self.K, self.frac_bits = m_parties, n_mac, frac_bits
        self.alpha = [[int(a) for a in _rand(m_parties)] for _ in range(n_mac)]        # alpha[k][j]
        self.alpha_tot = [sum(a) % P for a in self.alpha]
        self.bytes_offline = 0      # dealer -> parties/clients (preprocessing)
        self.bytes_parties = 0      # party <-> party openings
        self.bytes_input = 0        # client -> parties (online input)
        self.tamper_hook = None     # optional: callable(shares[m,L]) -> shares, simulates a corrupted party at open()

    # ---- dealer (trusted, offline) ------------------------------------------------------------
    def _auth_from_plain(self, v: np.ndarray) -> Shared:
        L = v.size
        val = np.empty((self.m, L), dtype=object)
        for j in range(self.m - 1):
            val[j] = _rand(L)
        val[self.m - 1] = (v - val[:self.m - 1].sum(0)) % P
        mac = np.empty((self.K, self.m, L), dtype=object)
        for k in range(self.K):
            tgt = (v * self.alpha_tot[k]) % P
            for j in range(self.m - 1):
                mac[k, j] = _rand(L)
            mac[k, self.m - 1] = (tgt - mac[k, :self.m - 1].sum(0)) % P
        self.bytes_offline += (1 + self.K) * self.m * L * 8
        return Shared(val % P, mac % P)

    def auth_random(self, L: int) -> tuple[np.ndarray, Shared]:
        r = _rand(L)
        return r, self._auth_from_plain(r)

    def beaver_triple(self, L: int) -> tuple[Shared, Shared, Shared]:
        a, b = _rand(L), _rand(L)
        return self._auth_from_plain(a), self._auth_from_plain(b), self._auth_from_plain((a * b) % P)

    # ---- online ------------------------------------------------------------------------------------
    def add_public(self, s: Shared, c: np.ndarray) -> Shared:
        val = s.val.copy(); mac = s.mac.copy()
        val[0] = (val[0] + c) % P
        for k in range(self.K):
            for j in range(self.m):
                mac[k, j] = (mac[k, j] + c * self.alpha[k][j]) % P
        return Shared(val, mac)

    def input(self, x_field: np.ndarray) -> Shared:
        """Client input via an authenticated random mask r: broadcast eps = x - r; [x] = [r] + eps."""
        r, sr = self.auth_random(x_field.size)
        eps = (x_field - r) % P
        self.bytes_input += self.m * x_field.size * 8
        self.bytes_offline += x_field.size * 8            # mask r handed to the client
        return self.add_public(sr, eps)

    def open(self, s: Shared, check: bool = True) -> np.ndarray:
        vals = s.val.copy()
        if self.tamper_hook is not None:
            vals = self.tamper_hook(vals)
        S = vals.sum(0) % P
        self.bytes_parties += self.m * (self.m - 1) * S.size * 8
        if check:
            for k in range(self.K):
                sigma = (s.mac[k].sum(0) - np.array([(self.alpha_tot[k] * int(v)) % P for v in S], dtype=object)) % P
                # each party j holds mac_j - alpha_j*S ; sum must be 0 (done via totals here)
                if any(int(z) != 0 for z in sigma):
                    raise MACCheckFailure("MAC check failed: a party deviated from the protocol")
                self.bytes_parties += self.m * 8
        return S

    def multiply(self, x: Shared, y: Shared) -> Shared:
        """Element-wise [x*y] with one Beaver triple per element."""
        L = x.val.shape[1]
        a, b, c = self.beaver_triple(L)
        d = self.open(x - a); e = self.open(y - b)
        z = c + _mul_pub_vec(b, d) + _mul_pub_vec(a, e)
        return self.add_public(z, (d * e) % P)

    def sum_elements(self, s: Shared) -> Shared:
        val = s.val.sum(1, keepdims=True) % P
        mac = s.mac.sum(2, keepdims=True) % P
        return Shared(val, mac)

def _mul_pub_vec(s: Shared, c: np.ndarray) -> Shared:
    return Shared((s.val * c[None, :]) % P, (s.mac * c[None, None, :]) % P)

class SMPCAggregator(Aggregator):
    """Weighted FedAvg computed by m SPDZ-style computing parties; optional secure L2-norm bound filtering
    (squared norm computed with Beaver multiplication, then revealed: leaks one scalar ||delta_i||^2 per client)."""
    name = "smpc"
    is_secure = True

    def __init__(self, m_parties: int = 3, n_mac: int = 2, frac_bits: int = 20, norm_bound: float | None = None):
        super().__init__()
        self.m, self.K, self.frac_bits, self.norm_bound = m_parties, n_mac, frac_bits, norm_bound
        self.last_rejected: list[int] = []

    def aggregate(self, deltas, weights, rnd=0):
        deltas, _ = self._check(deltas, weights)
        n, Pn = deltas.shape
        iw = np.rint(np.asarray(weights, np.float64)).astype(np.int64)
        if np.any(iw < 1):
            raise ValueError("SMPC aggregator needs positive integer weights")
        sess = SMPCSession(self.m, self.K, self.frac_bits)
        t0 = time.perf_counter()
        scale = float(1 << self.frac_bits)
        acc, rejected = None, []
        for i in range(n):
            q = np.rint(deltas[i].astype(np.float64) * scale).astype(np.int64)
            xs = sess.input(_to_field(q))
            if self.norm_bound is not None:
                sq = sess.sum_elements(sess.multiply(xs, xs))
                n2 = float(int(_centered(sess.open(sq))[0])) / scale ** 2
                if np.sqrt(max(n2, 0.0)) > self.norm_bound:
                    rejected.append(i); continue
            term = xs.mul_public(int(iw[i]))
            acc = term if acc is None else acc + term
        if acc is None:
            raise ValueError("all updates rejected by the norm bound")
        S = _centered(sess.open(acc))
        kept = [i for i in range(n) if i not in rejected]
        out = (np.array([float(v) for v in S]) / scale / float(iw[kept].sum())).astype(np.float32)
        self.last_rejected = rejected
        crypto = time.perf_counter() - t0
        self.last_stats = dict(up_bytes_total=int(sess.bytes_input), down_bytes_extra=0, crypto_time_s=crypto,
                               bytes_offline=int(sess.bytes_offline), bytes_party_to_party=int(sess.bytes_parties),
                               n_parties=self.m, n_rejected=len(rejected), agg_time_s=crypto,
                               online_bytes_per_client=int(sess.bytes_input // n))
        return out
