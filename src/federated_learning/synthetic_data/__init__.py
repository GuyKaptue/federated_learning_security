"""SYN-IDS-v1: fully synthetic federated-learning benchmark data (generation, validation, quality, drift, metadata)."""
from .generators import DATASET_ID, SynIDSConfig, generate_raw, make_dataset

__all__ = ["DATASET_ID", "SynIDSConfig", "generate_raw", "make_dataset"]
