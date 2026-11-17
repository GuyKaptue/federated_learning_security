"""Shared helpers for the networked demo: deterministic data recipe + local launcher of server and client OS processes."""
from __future__ import annotations

import json
import subprocess  # nosec B404
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from federated_learning.communication import tls
from federated_learning.data_partitioning import partition
from federated_learning.synthetic_data import make_dataset
from federated_learning.training.config import FLConfig


def build_dataset_and_parts(rc: dict):
    d = rc["data"]
    data = make_dataset(n_samples=d["n_samples"], seed=d["seed"])
    return data, partition(rc["partition"]["strategy"], data.y_train, rc["n_clients"], d["seed"], **rc["partition"].get("kwargs", {}))


def launch_local_federation(cfg: FLConfig, n_clients: int = 5, n_samples: int = 3000, data_seed: int = 0, strategy: str = "dirichlet",
                            kwargs: dict | None = None, secure: bool = True, workdir: Path | None = None, timeout: float = 600) -> dict:
    """Start 1 server + n_clients SEPARATE OS processes on localhost (mTLS unless secure=False) and return the server's result JSON."""
    wd = Path(workdir or tempfile.mkdtemp(prefix="fl_net_"))
    wd.mkdir(parents=True, exist_ok=True)
    rc = {"fl": asdict(cfg), "n_clients": n_clients, "host": "127.0.0.1", "data": {"n_samples": n_samples, "seed": data_seed},
          "partition": {"strategy": strategy, "kwargs": kwargs or {"alpha": 0.5}}, "join_timeout": 120, "round_timeout": 120}
    if secure:
        pki = tls.generate_pki(wd / "pki", list(range(n_clients)))
        rc["pki"] = {"ca": str(pki.ca_cert), "server_cert": str(pki.server_cert), "server_key": str(pki.server_key),
                     "clients": {str(k): [str(v[0]), str(v[1])] for k, v in pki.client_certs.items()}}
    (wd / "run.json").write_text(json.dumps(rc))
    out, portf = wd / "server_result.json", wd / "port.txt"
    py = [sys.executable, "-m"]
    srv = subprocess.Popen(py + ["federated_learning.servers.network_server", "--run-config", str(wd / "run.json"), "--out", str(out), "--port-file", str(portf)],  # nosec B603
                           stdout=(wd / "server.log").open("w"), stderr=subprocess.STDOUT)
    t0 = time.monotonic()
    while not portf.exists():
        if srv.poll() is not None or time.monotonic() - t0 > 120:
            raise RuntimeError("server failed to start; log tail:\n" + (wd / "server.log").read_text()[-1500:])
        time.sleep(0.1)
    time.sleep(0.2)
    port = portf.read_text().strip()
    cl = [subprocess.Popen(py + ["federated_learning.clients.network_client", "--run-config", str(wd / "run.json"), "--cid", str(k), "--port", port],  # nosec B603
                           stdout=(wd / f"client_{k}.log").open("w"), stderr=subprocess.STDOUT) for k in range(n_clients)]
    try:
        srv.wait(timeout=timeout)
        for p in cl:
            p.wait(timeout=30)
    finally:
        for p in [srv, *cl]:
            if p.poll() is None:
                p.kill()
    if srv.returncode != 0 or not out.exists():
        raise RuntimeError("server failed; log tail:\n" + (wd / "server.log").read_text()[-1500:])
    res = json.loads(out.read_text())
    res["client_exit_codes"] = [p.returncode for p in cl]; res["workdir"] = str(wd); res["processes"] = 1 + n_clients
    return res
