from __future__ import annotations

import numpy as np


def byzantine_hook(behavior: str, mal_ids: set, scale: float = 5.0, seed: int = 0):
    """Return a client_hook implementing the behaviour for clients in mal_ids."""
    if behavior not in ("random", "sign_flip", "scale", "drift"):
        raise ValueError("behavior must be random|sign_flip|scale|drift")
    rng = np.random.default_rng(seed)

    def hook(rnd, cid, delta, ctx):
        if cid not in mal_ids:
            return delta
        nrm = float(np.linalg.norm(delta)) + 1e-12
        if behavior == "random":
            v = rng.normal(size=delta.shape); return (scale * nrm * v / np.linalg.norm(v)).astype(np.float32)
        if behavior == "sign_flip":
            return (-scale * delta).astype(np.float32)
        if behavior == "scale":
            return (scale * delta).astype(np.float32)
        v = np.random.default_rng(seed + 777).normal(size=delta.shape)            # fixed direction for all rounds/clients
        return (delta + scale * nrm * v / np.linalg.norm(v)).astype(np.float32)
    return hook
