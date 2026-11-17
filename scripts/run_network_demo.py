"""Runs a real networked federation on localhost: 1 server process + N client processes, mutual TLS, then compares with the simulator.
Usage: PYTHONPATH=src python scripts/run_network_demo.py [--clients 5] [--rounds 10] [--algo fedavg]
Writes reports/network_demo.json."""
import argparse
import json
from pathlib import Path

from federated_learning.data_partitioning import partition
from federated_learning.pipelines.network_runtime import launch_local_federation
from federated_learning.servers.orchestrator import run_fl
from federated_learning.synthetic_data import make_dataset
from federated_learning.training import FLConfig
from federated_learning.utils.hashing import sha256_arrays

ap = argparse.ArgumentParser()
ap.add_argument("--clients", type=int, default=5); ap.add_argument("--rounds", type=int, default=10); ap.add_argument("--algo", default="fedavg")
ap.add_argument("--samples", type=int, default=3000); ap.add_argument("--insecure", action="store_true", help="plaintext TCP (never for real use)")
a = ap.parse_args()
cfg = FLConfig(algo=a.algo, rounds=a.rounds, local_epochs=1, hidden=16, seed=0, mu=0.05 if a.algo == "fedprox" else 0.0)
net = launch_local_federation(cfg, n_clients=a.clients, n_samples=a.samples, secure=not a.insecure)
d = make_dataset(n_samples=a.samples, seed=0)
sim = run_fl(cfg, d, partition("dirichlet", d.y_train, a.clients, 0, alpha=0.5))
out = {"transport": net["transport"], "processes": net["processes"], "network_final_acc": net["final"]["acc"], "simulation_final_acc": sim.final["acc"],
       "identical_weights": net["weights_sha256"] == sha256_arrays(sim.w), "wire_bytes": net["wire_bytes"], "accounted_bytes_sim": {"up": sim.totals["up_bytes"], "down": sim.totals["down_bytes"]},
       "wall_s_network": net["wall_s"], "wall_s_simulation": sim.totals["train_wall_s"]}
Path("reports").mkdir(exist_ok=True); Path("reports/network_demo.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
