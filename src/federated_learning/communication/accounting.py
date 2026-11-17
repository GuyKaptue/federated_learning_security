"""Logical communication accounting (payload bytes, not network traffic)."""
from __future__ import annotations

FLOAT32_BYTES = 4


def payload_bytes(vec_len: int) -> int:
    """Size in bytes of a float32 parameter/update vector of length `vec_len`."""
    return int(vec_len) * FLOAT32_BYTES


_bytes = payload_bytes  # backwards-compatible alias used by older experiment code
