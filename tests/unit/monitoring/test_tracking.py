from federated_learning.monitoring.tracking import log_mlflow


def test_mlflow_logging_roundtrip(tmp_path):
    rid = log_mlflow("t", {"a": 1, "b": [1, 2]}, {"acc": 0.9, "bad": float("nan")}, str(tmp_path), tags={"k": "v"})
    assert rid is None or isinstance(rid, str)
    if rid:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///" + str(tmp_path / "mlflow.db"))
        r = mlflow.get_run(rid); assert r.data.metrics["acc"] == 0.9 and "bad" not in r.data.metrics
