"""Pipeline: generate SYN-IDS-v1 -> validate -> quality + drift reports -> dataset card.

CLI: python -m federated_learning.synthetic_data.pipelines --seed 0 --n-samples 12000
Outputs (fully synthetic, so they may be committed):
    data/synthetic/syn_ids_v1/syn_ids_v1_seed<S>.npz      raw arrays X, y
    data/synthetic/validation/syn_ids_v1_seed<S>.json     validation + quality + train/test drift
    data/synthetic/metadata/syn_ids_v1_seed<S>.json       dataset card incl. SHA-256 of the arrays
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from federated_learning.synthetic_data.distributions import class_balance
from federated_learning.synthetic_data.drift_analysis import psi_per_feature
from federated_learning.synthetic_data.generators.syn_ids_v1 import SynIDSConfig, generate_raw
from federated_learning.synthetic_data.metadata import dataset_card
from federated_learning.synthetic_data.quality_metrics import quality_report
from federated_learning.synthetic_data.validators import validate_syn_ids
from federated_learning.utils.hashing import sha256_arrays
from federated_learning.utils.paths import SYNTHETIC_DIR


def generate_and_validate(cfg: SynIDSConfig, out_dir: Path | None = None) -> dict:
    out = Path(out_dir) if out_dir else SYNTHETIC_DIR
    X, y = generate_raw(cfg)
    stem = f"syn_ids_v1_seed{cfg.seed}"
    for sub in ("syn_ids_v1", "validation", "metadata", "generated"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    npz = out / "syn_ids_v1" / f"{stem}.npz"
    np.savez_compressed(npz, X=X, y=y)
    digest = sha256_arrays(X, y)
    Xtr, Xte = train_test_split(X, test_size=cfg.test_frac, stratify=y, random_state=cfg.seed)
    report = {"validation": validate_syn_ids(X, y, cfg), "quality": quality_report(X, y, cfg.seed), "class_balance": class_balance(y),
              "train_test_psi_max": float(max(psi_per_feature(Xtr, Xte))), "arrays_sha256": digest, "config": asdict(cfg)}
    (out / "validation" / f"{stem}.json").write_text(json.dumps(report, indent=2))
    (out / "metadata" / f"{stem}.json").write_text(json.dumps(dataset_card(cfg, digest), indent=2))
    return report


def load_synthetic(seed: int = 0, root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    p = Path(root or SYNTHETIC_DIR) / "syn_ids_v1" / f"syn_ids_v1_seed{seed}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `python -m federated_learning.synthetic_data.pipelines --seed {seed}`")
    d = np.load(p); return d["X"], d["y"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-samples", type=int, default=12000)
    ap.add_argument("--n-features", type=int, default=40); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = generate_and_validate(SynIDSConfig(n_samples=a.n_samples, n_features=a.n_features, seed=a.seed), a.out)
    print(json.dumps({"passed": r["validation"]["passed"], "sha256": r["arrays_sha256"]}))


if __name__ == "__main__":
    main()
