import numpy as np
import pytest

from federated_learning.data_partitioning.iid import partition_iid
from federated_learning.evaluation import statistics as S
from federated_learning.pipelines.experiments.common import md_table, provenance
from federated_learning.servers.orchestrator import run_fl
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset
from federated_learning.training.config import FLConfig


def test_same_seed_bit_identical():
    d = make_dataset(n_samples=2000, seed=0); p = partition_iid(d.y_train, 5, 0)
    a = run_fl(FLConfig(rounds=3, seed=7), d, p); b = run_fl(FLConfig(rounds=3, seed=7), d, p); c = run_fl(FLConfig(rounds=3, seed=8), d, p)
    assert np.array_equal(a.w, b.w) and not np.array_equal(a.w, c.w)


def test_provenance_hash_stable():
    a, b = provenance("T", dict(x=1, y=[1, 2])), provenance("T", dict(y=[1, 2], x=1))
    assert a["config_sha256"] == b["config_sha256"] and a["experiment_id"] == b["experiment_id"]


def test_stats_functions():
    a = np.array([.90, .91, .92, .93, .94, .95, .96, .97]); b = a - 0.02
    r = S.paired_tests(a, b); assert r["wilcoxon_p"] < 0.05 and r["mean_diff"] > 0
    adj = S.holm([0.01, 0.02, 0.5]); assert adj[0] <= adj[1] <= adj[2] and adj[0] == 0.03
    assert S.min_wilcoxon_p(5) == 2 / 32
    m, lo, hi = S.bootstrap_ci(a); assert lo < m < hi
    assert S.cliffs_delta(a, b) == pytest.approx(30 / 64 )  # (#a>b - #a<b)/64 with overlapping values
    assert S.cliffs_delta(a, a - 1.0) == 1.0
    assert S.friedman({"a": a, "b": b, "c": b - .01})["p"] < 0.05


def test_md_table():
    assert "| a |" in md_table([dict(a=1)])
