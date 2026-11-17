"""Networked federated server: a real TCP (optionally mutual-TLS) server that coordinates remote client processes.

It reuses the same Strategy / Aggregator / FLConfig objects as the in-process simulator (`run_fl`), so the algorithm is identical;
only the transport differs. With plain FedAvg and no dropouts it reproduces `run_fl` exactly (tested).

Security properties (mTLS mode, the default):
  * clients must present a certificate signed by the federation CA; the certificate CN must be "client-<cid>" and equal the cid they claim;
  * TLS >= 1.2, frames are size-limited before allocation, updates with wrong shape or non-finite values are rejected;
  * a client that is slow (round timeout), disconnects or misbehaves is dropped and the round continues if >= min_clients responded.
Limits: aggregators such as SecAggAggregator still run their protocol *inside the server process* on updates that arrive over TLS;
a distributed secure-aggregation message flow between clients is not implemented here. Hetero local epochs are not supported over the network.
"""
from __future__ import annotations

import argparse
import json
import socket
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from federated_learning.aggregators.base import Aggregator, propagate_context
from federated_learning.aggregators.mean import WeightedMean
from federated_learning.communication import tls
from federated_learning.communication.protocol import ProtocolError, recv_message, send_message
from federated_learning.evaluation.metrics import evaluate
from federated_learning.strategies import get_strategy
from federated_learning.training.config import FLConfig
from federated_learning.training.models import build_model, get_flat
from federated_learning.utils.hashing import sha256_arrays
from federated_learning.utils.logging import get_logger

log = get_logger("fl.server")


class _Conn:
    def __init__(self, cid: int, sock, n_samples: int):
        self.cid, self.sock, self.n_samples = cid, sock, n_samples
        self.tx = self.rx = 0
        self.alive = True

    def close(self):
        self.alive = False
        try:
            self.sock.close()
        except OSError:
            pass


class FederatedServer:
    def __init__(self, cfg: FLConfig, n_features: int, n_clients: int, host: str = "127.0.0.1", port: int = 0,
                 aggregator: Aggregator | None = None, ssl_context: ssl.SSLContext | None = None, allow_insecure: bool = False,
                 min_clients: int | None = None, join_timeout: float = 60.0, round_timeout: float = 120.0,
                 eval_data: tuple | None = None, max_payload: int = 256 * 1024 * 1024):
        cfg.validate()
        if ssl_context is None and not allow_insecure:
            raise ValueError("refusing to start without TLS; pass an ssl_context or allow_insecure=True (tests only)")
        if cfg.hetero_epochs:
            raise ValueError("hetero_epochs is not supported over the network")
        self.cfg, self.K, self.host = cfg, n_clients, host
        self.agg = aggregator or WeightedMean()
        self.strategy = get_strategy(cfg.algo)
        self.ctx, self.min_clients = ssl_context, min_clients or max(1, int(round(cfg.participation * n_clients)))
        self.join_timeout, self.round_timeout, self.eval_data, self.max_payload = join_timeout, round_timeout, eval_data, max_payload
        self.model = build_model(n_features, cfg.hidden, cfg.act, seed=cfg.seed, arch=cfg.arch)
        self.w = get_flat(self.model)
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port)); self._srv.listen(max(8, n_clients))
        self.port = self._srv.getsockname()[1]
        self.conns: dict[int, _Conn] = {}
        self.rejected: list[str] = []
        self.history: list[dict] = []

    # ---- connection handling -------------------------------------------------------------------
    def _handshake(self, raw: socket.socket) -> None:
        try:
            raw.settimeout(10)
            s = self.ctx.wrap_socket(raw, server_side=True) if self.ctx else raw
            cn = tls.peer_common_name(s) if self.ctx else None
            hdr, _, _ = recv_message(s, 4096)
            if hdr["type"] != "hello" or not isinstance(hdr.get("cid"), int) or not isinstance(hdr.get("n_samples"), int) or hdr["n_samples"] <= 0:
                raise ProtocolError("bad hello")
            cid = hdr["cid"]
            if self.ctx and cn != f"client-{cid}":
                raise ProtocolError(f"certificate CN {cn!r} does not match claimed client id {cid}")
            if not 0 <= cid < self.K or cid in self.conns:
                raise ProtocolError(f"client id {cid} out of range or already connected")
            send_message(s, {"type": "welcome", "config": asdict(self.cfg), "n_params": int(self.w.size)})
            s.settimeout(self.round_timeout)
            self.conns[cid] = _Conn(cid, s, hdr["n_samples"])
            log.info("client %s joined (%s samples)", cid, hdr["n_samples"])
        except (ssl.SSLError, ProtocolError, OSError, ValueError) as e:
            self.rejected.append(f"{type(e).__name__}: {e}")
            log.warning("connection rejected: %s: %s", type(e).__name__, e)
            try:
                raw.close()
            except OSError:
                pass

    def wait_for_clients(self) -> None:
        deadline = time.monotonic() + self.join_timeout
        threads = []
        while len(self.conns) < self.K and time.monotonic() < deadline:
            self._srv.settimeout(0.5)
            try:
                raw, _ = self._srv.accept()
            except TimeoutError:
                continue
            t = threading.Thread(target=self._handshake, args=(raw,), daemon=True); t.start(); threads.append(t)
            for th in threads:
                th.join(timeout=0.05)
        for th in threads:
            th.join(timeout=12)
        if len(self.conns) < self.min_clients:
            raise RuntimeError(f"only {len(self.conns)} of the required {self.min_clients} clients joined")

    # ---- one round ----------------------------------------------------------------------------
    def _ask(self, c: _Conn, rnd: int):
        corr = self.strategy.client_correction(c.cid)
        arrays = [self.w] + ([np.asarray(corr, dtype=np.float32)] if corr is not None else [])
        try:
            c.tx += send_message(c.sock, {"type": "train", "round": rnd, "mu": float(self.strategy.client_mu(self.cfg)), "has_corr": corr is not None}, arrays)
            hdr, arr, n = recv_message(c.sock, self.max_payload)
            c.rx += n
            if hdr["type"] != "update" or hdr.get("round") != rnd or len(arr) != 1 or arr[0].size != self.w.size or not np.all(np.isfinite(arr[0])):
                raise ProtocolError("invalid update")
            return c.cid, arr[0], int(hdr["tau"])
        except (ProtocolError, OSError, KeyError, TypeError, ValueError) as e:
            log.warning("dropping client %s in round %s: %s: %s", c.cid, rnd, type(e).__name__, e)
            c.close()
            return c.cid, None, 0

    def run(self) -> dict:
        rng = np.random.default_rng(self.cfg.seed + 12345)
        self.strategy.begin(self.w.size, self.K)
        t0 = time.perf_counter()
        for r in range(1, self.cfg.rounds + 1):
            tr = time.perf_counter()
            alive = sorted(k for k, c in self.conns.items() if c.alive)
            m = max(1, int(round(self.cfg.participation * self.K)))
            ids = sorted(rng.choice(self.K, size=m, replace=False).tolist()) if m < self.K else list(range(self.K))
            ids = [k for k in ids if k in alive]
            if len(ids) < self.min_clients:
                raise RuntimeError(f"round {r}: only {len(ids)} selected clients alive, need {self.min_clients}")
            self.strategy.begin_round()
            with ThreadPoolExecutor(max_workers=len(ids)) as ex:
                got = list(ex.map(lambda k, rnd=r: self._ask(self.conns[k], rnd), ids))
            ok = [(k, wk, tau) for k, wk, tau in got if wk is not None]
            if len(ok) < self.min_clients:
                raise RuntimeError(f"round {r}: only {len(ok)} valid updates, need {self.min_clients}")
            ids_ok = [k for k, _, _ in ok]
            deltas, taus, wts = [], [], []
            for k, wk, tau in ok:
                self.strategy.after_client(k, self.w, wk, tau, self.cfg)
                deltas.append(wk - self.w); taus.append(tau); wts.append(self.conns[k].n_samples)
            D = np.stack(deltas).astype(np.float32)
            propagate_context(self.agg, ids_ok, self.w)
            step = self.strategy.server_step(self.agg, D, wts, taus, ids_ok, r, self.cfg)
            self.w = (self.w + step).astype(np.float32)
            rec = dict(round=r, n_clients=len(ids_ok), dropped=len(ids) - len(ids_ok), round_time_s=time.perf_counter() - tr)
            if self.eval_data is not None and (r % self.cfg.eval_every == 0 or r == self.cfg.rounds):
                rec.update(evaluate(self.model, self.w, *self.eval_data))
            self.history.append(rec)
        wall = time.perf_counter() - t0
        for c in self.conns.values():
            if c.alive:
                try:
                    c.tx += send_message(c.sock, {"type": "done"}, [self.w])
                except OSError:
                    pass
        tx = sum(c.tx for c in self.conns.values()); rx = sum(c.rx for c in self.conns.values())
        out = dict(config=asdict(self.cfg), history=self.history, final=self.history[-1], weights_sha256=sha256_arrays(self.w),
                   wire_bytes=dict(server_sent=tx, server_received=rx), wall_s=wall, rejected_connections=self.rejected,
                   transport="mutual TLS" if self.ctx else "PLAINTEXT (insecure)")
        self.close()
        return out

    def close(self) -> None:
        for c in self.conns.values():
            c.close()
        try:
            self._srv.close()
        except OSError:
            pass


def main() -> None:
    from federated_learning.pipelines.network_runtime import build_dataset_and_parts
    ap = argparse.ArgumentParser(description="Run the federated server (see pipelines/network_demo.py for a full local demo)")
    ap.add_argument("--run-config", required=True, help="JSON written by the launcher"); ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=0); ap.add_argument("--port-file", default=None)
    a = ap.parse_args()
    rc = json.loads(Path(a.run_config).read_text())
    data, _ = build_dataset_and_parts(rc)
    cfg = FLConfig(**rc["fl"])
    ctx = tls.server_context(Path(rc["pki"]["ca"]), Path(rc["pki"]["server_cert"]), Path(rc["pki"]["server_key"])) if rc.get("pki") else None
    srv = FederatedServer(cfg, data.n_features, rc["n_clients"], rc.get("host", "127.0.0.1"), a.port, ssl_context=ctx, allow_insecure=ctx is None,
                          eval_data=(data.X_test, data.y_test), join_timeout=rc.get("join_timeout", 120), round_timeout=rc.get("round_timeout", 120))
    if a.port_file:
        Path(a.port_file).write_text(str(srv.port))
    srv.wait_for_clients()
    Path(a.out).write_text(json.dumps(srv.run(), indent=1, default=float))


if __name__ == "__main__":
    main()
