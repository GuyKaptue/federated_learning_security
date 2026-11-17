from __future__ import annotations

import json
import os

import numpy as np


def log_mlflow(run_name: str, params: dict, metrics: dict, tracking_dir: str | None = None,
               experiment: str = "federated_learning", tags: dict | None = None,
               artifacts: list[str] | None = None) -> str | None:
    """Log one run to a local SQLite-backed MLflow store. Returns run_id or None if unavailable."""
    try:
        import mlflow
        if tracking_dir is None:
            from federated_learning.utils.paths import EXPERIMENTS_DIR
            tracking_dir = str(EXPERIMENTS_DIR / "mlruns")
        os.makedirs(tracking_dir, exist_ok=True)
        db = os.path.abspath(os.path.join(tracking_dir, "mlflow.db"))
        mlflow.set_tracking_uri("sqlite:///" + db)
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location="file:" + os.path.abspath(os.path.join(tracking_dir, "artifacts")))
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({k: (v if isinstance(v, (int, float, str, bool)) else json.dumps(v))
                               for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float, np.floating)) and np.isfinite(v)})
            if tags:
                mlflow.set_tags(tags)
            for a in artifacts or []:
                if os.path.exists(a):
                    mlflow.log_artifact(a)
            return run.info.run_id
    except Exception as e:  # pragma: no cover
        print("MLflow logging skipped:", repr(e))
        return None
