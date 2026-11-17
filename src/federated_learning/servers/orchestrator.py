"""In-process federated learning simulator (virtual clients on one machine; the real networked server is in network_server.py).

Algorithms implemented from the original papers (references in docs/fl_bench_004):
  fedavg   : McMahan et al.  w <- w + sum_k p_k (w_k - w)
  fedprox  : Li et al.       local objective F_k(w) + mu/2 ||w - w_t||^2
  scaffold : Karimireddy et al. option II control variates, gradient corrected by (c - c_i)
  fednova  : Wang et al.     normalised averaging, w <- w + tau_eff * sum_k p_k (w_k - w)/tau_k

Clients are *virtual*: they are not independent hospitals or machines.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from federated_learning.aggregators.base import Aggregator, propagate_context
from federated_learning.aggregators.mean import WeightedMean
from federated_learning.clients.client import FederatedClient
from federated_learning.communication.accounting import payload_bytes
from federated_learning.evaluation.metrics import evaluate
from federated_learning.preprocessing.dataset import Dataset
from federated_learning.strategies import get_strategy
from federated_learning.training.config import FLConfig
from federated_learning.training.models import build_model, get_flat

torch.set_num_threads(1)

@dataclass
class FLResult:
    config: dict
    history: list
    final: dict
    w: np.ndarray
    totals: dict
    extras: dict = field(default_factory=dict)

def run_fl(cfg: FLConfig, data: Dataset, parts: list[np.ndarray], aggregator: Aggregator | None = None,
           client_hook=None, client_data: dict | None = None, round_callback=None,
           malicious_ids: set | None = None, eval_fn=None) -> FLResult:
    """Run one federated experiment (server-side orchestration of virtual clients).

    client_hook(rnd, cid, delta, ctx) -> delta     : model-update level adversary (e.g. Byzantine)
    client_data[cid] = (X, y) torch tensors        : data-level adversary (e.g. poisoning)
    round_callback(rnd, w_before, ids, deltas, taus) : observation hook (privacy attacks)
    eval_fn(model, w) -> dict                      : extra per-round metrics (e.g. backdoor ASR)
    """
    cfg.validate()
    K = len(parts)
    agg = aggregator or WeightedMean()
    strategy = get_strategy(cfg.algo)
    model = build_model(data.n_features, cfg.hidden, cfg.act, seed=cfg.seed, arch=cfg.arch)
    w = get_flat(model)
    P = w.size
    Xs, ys = {}, {}
    for k, p in enumerate(parts):
        Xs[k] = torch.from_numpy(data.X_train[p]); ys[k] = torch.from_numpy(data.y_train[p])
    if client_data:
        Xs.update({k: v[0] for k, v in client_data.items()}); ys.update({k: v[1] for k, v in client_data.items()})
    rng = np.random.default_rng(cfg.seed + 12345)
    epochs_i = {k: (int(rng.integers(1, cfg.local_epochs + 1)) if cfg.hetero_epochs else cfg.local_epochs) for k in range(K)}
    clients = {k: FederatedClient(k, Xs[k], ys[k], epochs_i[k]) for k in range(K)}
    strategy.begin(P, K)
    m = max(1, int(round(cfg.participation * K)))
    history, up_total, down_total = [], 0, 0
    client_time_total = agg_time_total = 0.0
    samples_processed = 0
    t_start = time.perf_counter()
    for r in range(1, cfg.rounds + 1):
        t_round = time.perf_counter()
        ids = sorted(rng.choice(K, size=m, replace=False).tolist()) if m < K else list(range(K))
        strategy.begin_round()
        deltas, taus, wts = [], [], []
        t_c = time.perf_counter()
        client_durs = []
        for k in ids:
            tc = time.perf_counter()
            client = clients[k]
            wk, tau = client.fit(model, w, cfg, r, mu=strategy.client_mu(cfg), correction=strategy.client_correction(k))
            strategy.after_client(k, w, wk, tau, cfg)
            delta = wk - w
            if client_hook is not None:
                delta = client_hook(r, k, delta, {"w": w, "tau": tau, "n": client.n_samples})
            deltas.append(delta); taus.append(tau); wts.append(client.n_samples)
            samples_processed += client.n_samples * client.epochs
            client_durs.append(time.perf_counter() - tc)
        client_time_total += time.perf_counter() - t_c
        D = np.stack(deltas).astype(np.float32)
        if round_callback is not None:
            round_callback(r, w.copy(), ids, D, taus)
        t_a = time.perf_counter()
        propagate_context(agg, ids, w)
        step = strategy.server_step(agg, D, wts, taus, ids, r, cfg)
        w = (w + step).astype(np.float32)
        agg_t = time.perf_counter() - t_a
        agg_time_total += agg_t
        # communication accounting (logical payload bytes)
        st = getattr(agg, "last_stats", {}) or {}
        up = st.get("up_bytes_total", len(ids) * payload_bytes(P)) * 1 + (strategy.uplink_payloads - 1) * len(ids) * payload_bytes(P)
        down = len(ids) * payload_bytes(P) * strategy.downlink_payloads + st.get("down_bytes_extra", 0)
        up_total += up; down_total += down
        rec = dict(round=r, n_clients=len(ids), up_bytes=up, down_bytes=down,
                   round_time_s=time.perf_counter() - t_round, agg_time_s=agg_t,
                   client_time_s=float(np.sum(client_durs)), client_time_max_s=float(np.max(client_durs)))
        if r % cfg.eval_every == 0 or r == cfg.rounds:
            rec.update(evaluate(model, w, data.X_test, data.y_test))
            if eval_fn is not None:
                rec.update(eval_fn(model, w))
        history.append(rec)
    train_wall = time.perf_counter() - t_start
    final = {k: v for k, v in history[-1].items()}
    totals = dict(train_wall_s=train_wall, client_time_s=client_time_total, agg_time_s=agg_time_total,
                  up_bytes=up_total, down_bytes=down_total, comm_mb=(up_total + down_total) / 2 ** 20,
                  n_params=P, n_clients_total=K, samples_processed=samples_processed,
                  throughput_samples_per_s=samples_processed / max(client_time_total, 1e-9),
                  mean_round_s=train_wall / cfg.rounds)
    return FLResult(asdict(cfg), history, final, w, totals, extras=dict(model=model, **strategy.extras()))


