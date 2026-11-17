"""SHA-256 helpers used for data integrity and reproducibility records."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    """Hash of the raw bytes of arrays (C-contiguous), independent of file formats."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
