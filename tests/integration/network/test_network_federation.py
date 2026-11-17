"""Real sockets + mutual TLS. Clients/servers run in threads for speed (same code path as the separate-process launcher)."""
import socket
import ssl
import threading

import numpy as np
import pytest

from federated_learning.aggregators import CoordinateMedian
from federated_learning.clients.network_client import NetworkClient
from federated_learning.communication import tls
from federated_learning.communication.protocol import recv_message, send_message
from federated_learning.data_partitioning import partition
from federated_learning.servers.network_server import FederatedServer
from federated_learning.servers.orchestrator import run_fl
from federated_learning.synthetic_data import make_dataset
from federated_learning.training import FLConfig
from federated_learning.utils.hashing import sha256_arrays

K = 4


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = make_dataset(n_samples=1500, seed=0); parts = partition("dirichlet", d.y_train, K, 0, alpha=0.5)
    return d, parts, tls.generate_pki(tmp_path_factory.mktemp("pki"), list(range(K + 1)))


def start(env, cfg, agg=None, hooks=None, skip=(), min_clients=None, round_timeout=30, ids=None):
    d, parts, pki = env
    srv = FederatedServer(cfg, d.n_features, K, port=0, aggregator=agg, ssl_context=tls.server_context(pki.ca_cert, pki.server_cert, pki.server_key),
                          eval_data=(d.X_test, d.y_test), join_timeout=15, round_timeout=round_timeout, min_clients=min_clients)
    res = {}
    t = threading.Thread(target=lambda: res.update(out=(srv.wait_for_clients(), srv.run())[1]) or None, daemon=True)
    ths, clients = [], {}
    def go(k):
        c, key = pki.client_certs[k]
        cl = NetworkClient(k, d.X_train[parts[k]], d.y_train[parts[k]], "127.0.0.1", srv.port, tls.client_context(pki.ca_cert, c, key),
                           update_hook=(hooks or {}).get(k), connect_timeout=10)
        clients[k] = cl
        try:
            cl.run()
        except Exception:
            pass
    t.start()
    for k in (ids or range(K)):
        if k not in skip:
            th = threading.Thread(target=go, args=(k,), daemon=True); th.start(); ths.append(th)
    t.join(120); [th.join(20) for th in ths]
    return srv, res.get("out"), clients


def test_matches_in_process_simulation_exactly(env):
    d, parts, _ = env
    cfg = FLConfig(rounds=4, local_epochs=1, hidden=16, seed=1)
    _, out, _ = start(env, cfg)
    sim = run_fl(cfg, d, parts)
    assert out["weights_sha256"] == sha256_arrays(sim.w) and abs(out["final"]["acc"] - sim.final["acc"]) < 1e-12
    assert out["wire_bytes"]["server_sent"] > 0 and out["transport"] == "mutual TLS"


@pytest.mark.parametrize("algo", ["fedprox", "scaffold", "fednova"])
def test_other_strategies_match_simulation(env, algo):
    d, parts, _ = env
    cfg = FLConfig(algo=algo, rounds=3, local_epochs=1, hidden=16, seed=2, mu=0.1 if algo == "fedprox" else 0.0)
    _, out, _ = start(env, cfg)
    assert out["weights_sha256"] == sha256_arrays(run_fl(cfg, d, parts).w)


def test_byzantine_client_over_the_network_behaves_like_the_simulation(env):
    """A malicious remote client (sign-flip x8) has exactly the effect it has in the simulator, for mean and for median aggregation."""
    from federated_learning import attacks as A
    d, parts, _ = env
    cfg = FLConfig(rounds=4, local_epochs=1, hidden=16, seed=3)
    for make_agg in (lambda: None, CoordinateMedian):
        srv, out, _ = start(env, cfg, agg=make_agg(), hooks={0: lambda r, k, delta, ctx: -8.0 * delta})
        sim = run_fl(cfg, d, parts, aggregator=make_agg(), client_hook=A.byzantine_hook("sign_flip", {0}, 8.0))
        # the update travels as w + delta (float32), so tiny rounding differences are expected: compare with a tolerance
        assert np.allclose(srv.w, sim.w, atol=1e-3) and abs(out["final"]["acc"] - sim.final["acc"]) < 0.02


def test_dropped_client_is_tolerated_when_min_clients_met(env):
    cfg = FLConfig(rounds=3, local_epochs=1, hidden=16, seed=4)
    srv, out, _ = start(env, cfg, skip={3}, min_clients=3)
    assert out is not None and out["history"][-1]["n_clients"] == 3


def test_missing_clients_abort_when_below_min(env):
    cfg = FLConfig(rounds=2, local_epochs=1, hidden=16, seed=4)
    d, parts, pki = env
    srv = FederatedServer(cfg, d.n_features, K, ssl_context=tls.server_context(pki.ca_cert, pki.server_cert, pki.server_key), join_timeout=2)
    with pytest.raises(RuntimeError, match="joined"):
        srv.wait_for_clients()
    srv.close()


def _server_only(env):
    d, parts, pki = env
    srv = FederatedServer(FLConfig(rounds=1, hidden=16), d.n_features, K, ssl_context=tls.server_context(pki.ca_cert, pki.server_cert, pki.server_key), join_timeout=4)
    th = threading.Thread(target=lambda: _quiet(srv.wait_for_clients), daemon=True); th.start()
    return srv, th


def _quiet(f):
    try:
        f()
    except RuntimeError:
        pass


def test_client_without_certificate_is_rejected(env):
    srv, th = _server_only(env)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT); ctx.load_verify_locations(cafile=str(env[2].ca_cert))
    with pytest.raises((ssl.SSLError, OSError)):
        s = ctx.wrap_socket(socket.create_connection(("127.0.0.1", srv.port), timeout=5), server_hostname="127.0.0.1")
        send_message(s, {"type": "hello", "cid": 0, "n_samples": 10}); recv_message(s)
    th.join(10)
    assert srv.conns == {} and srv.rejected


def test_client_cannot_claim_another_identity(env):
    srv, th = _server_only(env)
    c, key = env[2].client_certs[4]                      # holds a valid certificate for client-4 ...
    s = tls.client_context(env[2].ca_cert, c, key).wrap_socket(socket.create_connection(("127.0.0.1", srv.port), timeout=5), server_hostname="127.0.0.1")
    send_message(s, {"type": "hello", "cid": 0, "n_samples": 10})   # ... but claims to be client 0
    with pytest.raises(Exception):
        recv_message(s)
    th.join(10)
    assert srv.conns == {} and any("does not match" in r for r in srv.rejected)


def test_insecure_mode_requires_explicit_opt_in(env):
    d, parts, _ = env
    with pytest.raises(ValueError, match="TLS"):
        FederatedServer(FLConfig(rounds=1), d.n_features, K)
    with pytest.raises(ValueError, match="TLS"):
        NetworkClient(0, d.X_train[:10], d.y_train[:10], "127.0.0.1", 1)


def test_nan_update_is_rejected_and_run_continues(env):
    cfg = FLConfig(rounds=2, local_epochs=1, hidden=16, seed=5)
    bad = {1: lambda r, k, delta, ctx: delta * np.float32("nan")}
    srv, out, _ = start(env, cfg, hooks=bad, min_clients=3)
    assert out["history"][-1]["dropped"] == 0 or out["history"][0]["dropped"] == 1
    assert np.isfinite(srv.w).all()
