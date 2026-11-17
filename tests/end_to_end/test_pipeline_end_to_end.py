"""Full path: config -> synthetic data -> partition -> federated training (+ secure aggregation / robust defence) -> evaluation -> report."""
import numpy as np

from federated_learning.aggregators import CoordinateMedian
from federated_learning.data_partitioning import partition
from federated_learning.servers.orchestrator import run_fl
from federated_learning.synthetic_data import make_dataset
from federated_learning.training import centralized_baseline, load_fl_config


def test_config_driven_federated_run_beats_chance_and_is_reproducible():
    d = make_dataset(n_samples=3000, seed=0)
    cfg = load_fl_config({"rounds": 8, "local_epochs": 1})
    parts = partition("dirichlet", d.y_train, 8, seed=0, alpha=1.0)
    r1 = run_fl(cfg, d, parts); r2 = run_fl(cfg, d, parts)
    assert r1.final["acc"] > 0.65 and np.array_equal(r1.w, r2.w)
    rob = run_fl(cfg, d, parts, aggregator=CoordinateMedian())
    assert rob.final["acc"] > 0.6
    assert centralized_baseline(d, epochs=8)["acc"] >= r1.final["acc"] - 0.1


def test_real_dataset_pipeline_from_local_raw_cache():
    from federated_learning.preprocessing import load_real
    d = load_real("wdbc"); assert d.X_train.shape[1] == 30 and d.meta["real_data"]
