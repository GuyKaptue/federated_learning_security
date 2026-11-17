import pytest
import yaml

from federated_learning.data_partitioning import STRATEGIES
from federated_learning.synthetic_data import SynIDSConfig
from federated_learning.training import load_fl_config
from federated_learning.utils.paths import CONFIG_DIR


def test_all_config_files_parse():
    for n in ("training", "federated", "model", "synthetic_data", "privacy"):
        assert isinstance(yaml.safe_load((CONFIG_DIR / f"{n}.yaml").read_text()), dict)


def test_fl_config_loads_and_overrides():
    c = load_fl_config(); assert c.algo == "fedavg" and c.hidden == 32
    assert load_fl_config({"algo": "fedprox", "mu": 0.1}).mu == 0.1
    with pytest.raises(KeyError):
        load_fl_config({"bogus": 1})
    with pytest.raises(ValueError):
        load_fl_config({"algo": "nope"})


def test_synthetic_config_keys_match_dataclass():
    d = yaml.safe_load((CONFIG_DIR / "synthetic_data.yaml").read_text())
    d = {k: v for k, v in d.items() if k not in ("version", "dataset_id")}; d["weights"] = tuple(d["weights"])
    SynIDSConfig(**d)


def test_partitioning_strategy_is_known():
    assert yaml.safe_load((CONFIG_DIR / "federated.yaml").read_text())["partitioning"]["strategy"] in STRATEGIES
