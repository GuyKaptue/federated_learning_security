"""Attack implementations for the controlled, synthetic-data security benchmark (FL-ATTACK-003).

All attacks run only against locally simulated models trained on synthetic data (SYN-IDS-v1).

  * Backdoor            : data poisoning with a feature trigger (+ optional update boosting / model replacement)
                          -- Bagdasaryan et al., AISTATS 2020
  * Byzantine           : random / sign-flip / scaling / constant-direction drift update manipulation
  * Membership inference: shadow-model attack (Shokri et al., S&P 2017) and a loss-threshold baseline
  * Gradient leakage    : DLG-style optimisation (Zhu et al., NeurIPS 2019) with known labels (iDLG-style label
                          recovery for B=1), plus an exact analytic first-layer inversion for B=1
"""
from __future__ import annotations

import numpy as np
import torch

from federated_learning.training.models import set_flat

TRIGGER_IDX = (0, 1, 2, 3)

TRIGGER_VAL = 4.0

def apply_trigger(X: np.ndarray, idx=TRIGGER_IDX, val: float = TRIGGER_VAL) -> np.ndarray:
    X2 = X.copy()
    X2[:, list(idx)] = val
    return X2

def poison_client_data(X: np.ndarray, y: np.ndarray, poison_frac: float, target: int, rng,
                       idx=TRIGGER_IDX, val: float = TRIGGER_VAL):
    """Stamp the trigger on `poison_frac` of the client's samples (preferring non-target class) and relabel."""
    if not 0 <= poison_frac <= 1:
        raise ValueError("poison_frac in [0,1]")
    X2, y2 = X.copy(), y.copy()
    cand = np.where(y != target)[0]
    k = int(round(poison_frac * len(y)))
    sel = rng.choice(cand, size=min(k, len(cand)), replace=False) if k and len(cand) else np.array([], int)
    if len(sel):
        X2[sel] = apply_trigger(X[sel], idx, val)
        y2[sel] = target
    return torch.from_numpy(X2), torch.from_numpy(y2), int(len(sel))

def backdoor_metrics(model, w, Xte, yte, target: int = 0, idx=TRIGGER_IDX, val: float = TRIGGER_VAL) -> dict:
    """clean accuracy; ASR = fraction of non-target-class test samples classified as `target` once triggered."""
    set_flat(model, w); model.eval()
    with torch.no_grad():
        clean = (model(torch.from_numpy(Xte)).argmax(1).numpy() == yte).mean()
        sel = yte != target
        trig = model(torch.from_numpy(apply_trigger(Xte[sel], idx, val))).argmax(1).numpy()
        base = model(torch.from_numpy(Xte[sel])).argmax(1).numpy()
    return dict(clean_acc=float(clean), asr=float((trig == target).mean()),
                asr_untriggered=float((base == target).mean()))

def boost_hook(mal_ids: set, gamma: float):
    """Model-replacement style scaling of malicious updates by gamma (Bagdasaryan et al.)."""
    def hook(rnd, cid, delta, ctx):
        return delta * gamma if cid in mal_ids else delta
    return hook
