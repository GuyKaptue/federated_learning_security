"""End to end with SEPARATE OS processes (1 server + 4 clients) over localhost mutual TLS; result must equal the in-process simulation."""
import numpy as np

from federated_learning.data_partitioning import partition
from federated_learning.pipelines.network_runtime import launch_local_federation
from federated_learning.servers.orchestrator import run_fl
from federated_learning.synthetic_data import make_dataset
from federated_learning.training import FLConfig
from federated_learning.utils.hashing import sha256_arrays


def test_separate_processes_match_simulation_and_use_tls(tmp_path):
    cfg = FLConfig(algo="fedprox", mu=0.05, rounds=3, local_epochs=1, hidden=16, seed=7)
    res = launch_local_federation(cfg, n_clients=4, n_samples=1500, data_seed=0, workdir=tmp_path)
    d = make_dataset(n_samples=1500, seed=0)
    sim = run_fl(cfg, d, partition("dirichlet", d.y_train, 4, 0, alpha=0.5))
    assert res["processes"] == 5 and res["client_exit_codes"] == [0, 0, 0, 0] and res["transport"] == "mutual TLS"
    assert res["weights_sha256"] == sha256_arrays(sim.w)
    assert res["wire_bytes"]["server_sent"] > 4 * 3 * 4 * sim.w.size * 0.9 and not res["rejected_connections"]
    assert np.isfinite(res["final"]["acc"])


def test_plaintext_mode_is_labelled_insecure(tmp_path):
    res = launch_local_federation(FLConfig(rounds=1, local_epochs=1, hidden=8), n_clients=2, n_samples=1000, secure=False, workdir=tmp_path)
    assert "insecure" in res["transport"].lower()
