from __future__ import annotations

from dataclasses import asdict

from federated_learning.synthetic_data.generators.syn_ids_v1 import DATASET_ID, SynIDSConfig

STATEMENT = ("SYN-IDS-v1 is COMPLETELY SYNTHETIC despite its name: it is generated with scikit-learn make_classification, is not real "
             "intrusion-detection data, contains no patient data and no personal data. CIC-IDS-2017 was not available and is not used.")


def dataset_card(cfg: SynIDSConfig, npz_sha256: str | None = None) -> dict:
    return {"dataset_id": DATASET_ID, "statement": STATEMENT, "generator": "sklearn.datasets.make_classification",
            "parameters": asdict(cfg), "task": "binary classification (label 1 = minority 'attack-like' class)",
            "intended_use": ["federated-learning benchmarks", "cryptographic overhead evaluation", "secure aggregation experiments",
                             "adversarial attack simulations", "defence evaluation", "scalability experiments", "hyper-parameter optimisation"],
            "not_for": ["any claim about real intrusion detection or real patients", "numerical comparison with published CIC-IDS-2017 results"],
            "npz_sha256": npz_sha256}
