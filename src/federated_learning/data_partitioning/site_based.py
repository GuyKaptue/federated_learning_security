from __future__ import annotations

import numpy as np


def partition_by_site(site_ids: np.ndarray) -> list[np.ndarray]:
    """Natural partition: one client per distinct site label (ordered by sorted site value)."""
    site_ids = np.asarray(site_ids)
    return [np.where(site_ids == s)[0] for s in np.unique(site_ids)]
