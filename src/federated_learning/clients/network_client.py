"""Networked federated client: connects to the server over TCP/mTLS, trains on its LOCAL data and returns only model weights.

The raw data never leave this process. The training code is the same `FederatedClient.fit` as in the simulator.
`update_hook(round, cid, delta, ctx) -> delta` lets tests/experiments model a malicious client that tampers with its own update
before sending (same signature as the simulator's client_hook).
"""
from __future__ import annotations

import argparse
import json
import socket
import ssl
import time
from pathlib import Path

import numpy as np
import torch

from federated_learning.clients.client import FederatedClient
from federated_learning.communication import tls
from federated_learning.communication.protocol import ProtocolError, recv_message, send_message
from federated_learning.training.config import FLConfig
from federated_learning.training.models import build_model
from federated_learning.utils.logging import get_logger

log = get_logger("fl.client")


class NetworkClient:
    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray, host: str, port: int, ssl_context: ssl.SSLContext | None = None,
                 allow_insecure: bool = False, update_hook=None, connect_timeout: float = 60.0, io_timeout: float = 300.0):
        if ssl_context is None and not allow_insecure:
            raise ValueError("refusing to connect without TLS; pass an ssl_context or allow_insecure=True (tests only)")
        self.cid, self.host, self.port, self.ctx = cid, host, port, ssl_context
        self.X, self.y = torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(np.ascontiguousarray(y))
        self.hook, self.connect_timeout, self.io_timeout = update_hook, connect_timeout, io_timeout
        self.rounds_done = 0
        self.final_w: np.ndarray | None = None

    def _connect(self) -> socket.socket:
        deadline = time.monotonic() + self.connect_timeout
        while True:
            try:
                raw = socket.create_connection((self.host, self.port), timeout=10)
                return self.ctx.wrap_socket(raw, server_hostname=self.host) if self.ctx else raw
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                if isinstance(e, ssl.SSLError) or time.monotonic() > deadline:
                    raise
                time.sleep(0.2)

    def run(self) -> None:
        s = self._connect()
        s.settimeout(self.io_timeout)
        try:
            send_message(s, {"type": "hello", "cid": self.cid, "n_samples": int(self.X.shape[0])})
            hdr, _, _ = recv_message(s, 1 << 20)
            if hdr["type"] != "welcome":
                raise ProtocolError("expected welcome")
            cfg = FLConfig(**hdr["config"]); cfg.validate()
            model = build_model(self.X.shape[1], cfg.hidden, cfg.act, seed=cfg.seed, arch=cfg.arch)
            fc = FederatedClient(self.cid, self.X, self.y, cfg.local_epochs)
            while True:
                hdr, arr, _ = recv_message(s)
                if hdr["type"] == "done":
                    self.final_w = arr[0] if arr else None
                    return
                if hdr["type"] != "train":
                    raise ProtocolError(f"unexpected message {hdr['type']}")
                w = arr[0]; corr = arr[1] if hdr.get("has_corr") else None
                wk, tau = fc.fit(model, w, cfg, int(hdr["round"]), mu=float(hdr["mu"]), correction=corr)
                if self.hook is not None:
                    wk = (w + self.hook(int(hdr["round"]), self.cid, wk - w, {"w": w, "tau": tau, "n": fc.n_samples})).astype(np.float32)
                send_message(s, {"type": "update", "round": int(hdr["round"]), "cid": self.cid, "tau": int(tau), "n": fc.n_samples}, [np.asarray(wk, dtype=np.float32)])
                self.rounds_done += 1
        finally:
            try:
                s.close()
            except OSError:
                pass


def main() -> None:
    from federated_learning.pipelines.network_runtime import build_dataset_and_parts
    ap = argparse.ArgumentParser(description="Run one federated client process")
    ap.add_argument("--run-config", required=True); ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    rc = json.loads(Path(a.run_config).read_text())
    data, parts = build_dataset_and_parts(rc)       # each client derives ONLY its own partition from the shared, deterministic recipe
    p = parts[a.cid]
    ctx = None
    if rc.get("pki"):
        c, k = rc["pki"]["clients"][str(a.cid)]
        ctx = tls.client_context(Path(rc["pki"]["ca"]), Path(c), Path(k))
    NetworkClient(a.cid, data.X_train[p], data.y_train[p], rc.get("host", "127.0.0.1"), a.port, ctx, allow_insecure=ctx is None).run()


if __name__ == "__main__":
    main()
