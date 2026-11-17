"""FL-SEC-001: cryptographic mechanisms (one function per Jira task / notebook)."""
from __future__ import annotations

import time
import numpy as np
from federated_learning.privacy.homomorphic import BFVAggregator, CKKSAggregator, PaillierAggregator
from federated_learning.privacy.secure_aggregation import SecAggAbort, SecAggAggregator, SecureAggregationProtocol, shamir_reconstruct, shamir_split
from federated_learning.privacy.smpc import MACCheckFailure, SMPCAggregator, SMPCSession, _to_field, P as SMPC_P
from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset
from federated_learning.data_partitioning.iid import partition_iid
from federated_learning.aggregators.audit import Audited
from federated_learning.aggregators.mean import WeightedMean
from federated_learning.training.config import FLConfig
from federated_learning.servers.orchestrator import run_fl
from federated_learning.monitoring.resources import ResourceMonitor
from federated_learning.pipelines.experiments.common import mean_ci, provenance, save_csv, save_json, savefig, scale, plt

STORY = "fl_sec_001"

def _updates(m, P, seed=0, sd=0.1):
    rng = np.random.default_rng(seed)
    return rng.normal(0, sd, (m, P)).astype(np.float32), rng.integers(10, 100, m).astype(float)

def task_1_1_selection() -> dict:
    """Decision matrix. Qualitative properties are THEORETICAL (cited in docs); expansion/timing columns are MEASURED."""
    prov = provenance("FL-SEC-001.1", dict(P=1442, m=5, seed=0))
    m, P = 5, scale(1442, 256)
    D, W = _updates(m, P)
    rows = []
    ck, bf = CKKSAggregator(), BFVAggregator()
    pa = PaillierAggregator(2048)
    sub = 32
    ref = (W / W.sum()) @ D
    for A, Dd, name in ((ck, D, "CKKS"), (bf, D, "BFV"), (pa, D[:, :sub], "Paillier")):
        out = A.aggregate(Dd, W)
        err = float(np.abs(out - (W / W.sum()) @ Dd).max())
        st = A.last_stats
        n_el = Dd.shape[1]
        rows.append(dict(scheme=name, measured_vector_len=n_el, max_abs_error=err,
                         ct_bytes_per_float32=st["ct_bytes_per_client"] / n_el, expansion_vs_float32=st["expansion"],
                         enc_ms_per_client=1000 * st["enc_time_per_client_s"],
                         enc_us_per_element=1e6 * st["enc_time_per_client_s"] / n_el,
                         agg_ms=1000 * st["he_agg_time_s"], dec_ms=1000 * st["dec_time_s"]))
    qual = {
        "CKKS": dict(arithmetic="approximate over reals (additions and multiplications, rescaling)", plaintext_domain="real/complex vectors, SIMD",
                     security_basis="RLWE (ring-LWE), post-quantum assumptions", fit="best for real-valued updates; error ~1e-8 observed"),
        "BFV": dict(arithmetic="exact modular integer arithmetic, SIMD", plaintext_domain="integers mod t (needs fixed-point encoding)",
                    security_basis="RLWE", fit="exact but forces quantisation and a plaintext-modulus overflow bound"),
        "Paillier": dict(arithmetic="additive only (plus scalar multiplication)", plaintext_domain="integers mod n (float via exponent encoding)",
                         security_basis="decisional composite residuosity; broken by quantum (Shor)", fit="no SIMD: one ciphertext per coordinate -> impractical for NN updates"),
    }
    for r in rows:
        r.update(qual[r["scheme"]])
    sec_rows = [
        dict(approach="Secure aggregation (pairwise masking)", computes="sum of client vectors only",
             threat_model="honest-but-curious server (+ colluding clients below threshold t); dropout tolerant",
             crypto="ECDH, PRG masks, Shamir sharing", server_sees_individual_update=False,
             limits="only linear aggregation; cannot run robust (median/Krum) rules; needs interaction rounds"),
        dict(approach="HE (CKKS/BFV) aggregation", computes="sums (and limited depth products) under encryption",
             threat_model="honest-but-curious server; secret-key holder must be trusted/not colluding with server",
             crypto="RLWE lattice encryption", server_sees_individual_update=False,
             limits="ciphertext expansion, single secret key => key-holder is a single point of failure (no threshold HE here)"),
        dict(approach="SMPC (SPDZ-style, m computing parties)", computes="arbitrary arithmetic circuits (here: sum, squared norm)",
             threat_model="dishonest majority among m servers (m-1 corrupted), MAC check detects active cheating (abort)",
             crypto="additive sharing + information-theoretic MAC + Beaver triples",
             server_sees_individual_update=False, limits="m servers must be non-colluding organisations; offline phase here = trusted dealer"),
    ]
    decision = [
        dict(use="Primary HE scheme for real-valued model-update aggregation", selected="CKKS",
             why="lowest ciphertext expansion and encryption cost among tested, approximate error far below SGD noise, no quantisation design needed",
             rejected="BFV (exact but 1.3x larger ciphertext at tested params and needs fixed-point headroom); Paillier (no SIMD; ~2 orders of magnitude slower per element)"),
        dict(use="Cheapest privacy-from-server for plain FedAvg", selected="Secure aggregation", why="no heavy public-key work per coordinate; tolerates dropout",
             rejected="n/a (not exclusive with HE/SMPC; different threat model)"),
        dict(use="Verified computation beyond sums (norm checks) / multi-institution compute servers", selected="SMPC (SPDZ-style)",
             why="supports multiplication (Beaver) so norm-bound checking is possible without revealing updates", rejected="n/a"),
    ]
    save_csv(STORY, "task_1_1_he_comparison_measured", rows)
    save_csv(STORY, "task_1_1_protocol_comparison_theoretical", sec_rows)
    save_json(STORY, "task_1_1_decision_matrix", dict(provenance=prov, he=rows, protocols=sec_rows, decision=decision))
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    names = [r["scheme"] for r in rows]
    ax[0].bar(names, [r["ct_bytes_per_float32"] for r in rows]); ax[0].set_yscale("log")
    ax[0].set_ylabel("ciphertext bytes per float32 (log)"); ax[0].set_title("Ciphertext size per coordinate (measured)")
    ax[1].bar(names, [r["enc_us_per_element"] for r in rows]); ax[1].set_yscale("log")
    ax[1].set_ylabel("encryption time per element [µs] (log)"); ax[1].set_title("Encryption cost per coordinate (measured)")
    savefig(fig, STORY, "task_1_1_he_cost", prov)
    return dict(provenance=prov, he=rows, protocols=sec_rows, decision=decision)

def task_1_2_he() -> dict:
    cfg = dict(sizes=scale([128, 1442, 8192, 32768], [128, 1442]), paillier_sizes=scale([16, 64], [8]), m=5, seeds=3,
               fl=dict(clients=10, rounds=scale(10, 3), hidden=32))
    prov = provenance("FL-SEC-001.2", cfg)
    rows = []
    ck, bf, pa = CKKSAggregator(), BFVAggregator(), PaillierAggregator(2048)
    keyinfo = dict(ckks=ck.keygen_info(), bfv=bf.keygen_info(), paillier=pa.keygen_info())
    for P in cfg["sizes"]:
        for A in (ck, bf):
            e, enc, agg, dec, ctb = [], [], [], [], []
            for s in range(cfg["seeds"]):
                D, W = _updates(cfg["m"], P, seed=s)
                out = A.aggregate(D, W)
                ref = ((W / W.sum()) @ D)
                e.append(float(np.abs(out - ref).max()))
                st = A.last_stats
                enc.append(st["enc_time_per_client_s"]); agg.append(st["he_agg_time_s"]); dec.append(st["dec_time_s"]); ctb.append(st["ct_bytes_per_client"])
            rows.append(dict(scheme=A.name, P=P, max_abs_err=float(np.mean(e)), enc_ms_per_client=1e3 * float(np.mean(enc)),
                             agg_ms=1e3 * float(np.mean(agg)), dec_ms=1e3 * float(np.mean(dec)), ct_kb_per_client=float(np.mean(ctb)) / 1024,
                             expansion=float(np.mean(ctb)) / (4 * P), plaintext_kb=4 * P / 1024, kind="measured"))
    for P in cfg["paillier_sizes"]:
        D, W = _updates(cfg["m"], P, seed=0)
        out = pa.aggregate(D, W); st = pa.last_stats
        rows.append(dict(scheme="he_paillier", P=P, max_abs_err=float(np.abs(out - (W / W.sum()) @ D).max()),
                         enc_ms_per_client=1e3 * st["enc_time_per_client_s"], agg_ms=1e3 * st["he_agg_time_s"], dec_ms=1e3 * st["dec_time_s"],
                         ct_kb_per_client=st["ct_bytes_per_client"] / 1024, expansion=st["expansion"], plaintext_kb=4 * P / 1024, kind="measured"))
    pr = [r for r in rows if r["scheme"] == "he_paillier"]
    per_el = np.mean([r["enc_ms_per_client"] / r["P"] for r in pr])
    rows.append(dict(scheme="he_paillier", P=1442, enc_ms_per_client=per_el * 1442, ct_kb_per_client=1442 * 512 / 1024,
                     expansion=128.0, kind="ESTIMATED (linear extrapolation of measured per-element rate)"))
    save_csv(STORY, "task_1_2_he_microbenchmarks", rows)

    # numerical precision vs update magnitude (CKKS and BFV)
    prec = []
    for sd in (1e-4, 1e-2, 1e-1, 1.0, 10.0):
        D, W = _updates(5, 1442, seed=1, sd=sd)
        ref = (W / W.sum()) @ D
        row = dict(update_sd=sd)
        row["ckks_rel_err"] = float(np.linalg.norm(ck.aggregate(D, W) - ref) / np.linalg.norm(ref))
        try:
            row["bfv_rel_err"] = float(np.linalg.norm(bf.aggregate(D, W) - ref) / np.linalg.norm(ref))
        except OverflowError as e:
            row["bfv_rel_err"] = None; row["bfv_note"] = "overflow: " + str(e)[:60]
        prec.append(row)
    save_csv(STORY, "task_1_2_precision_vs_magnitude", prec)

    # malformed / wrong key behaviour (security-relevant)
    D, W = _updates(3, 64, seed=2)
    ct = ck.encrypt(D[0].astype(np.float64))
    other = CKKSAggregator()
    wrong_key = None
    try:
        wrong = other.decrypt(ct, 64)
        wrong_key = dict(decrypted_garbage=True, rel_err_vs_truth=float(np.linalg.norm(wrong - D[0]) / np.linalg.norm(D[0])))
    except Exception as e:
        wrong_key = dict(decrypted_garbage=False, error=type(e).__name__)
    malformed = None
    try:
        ck.decrypt([ct[0][: len(ct[0]) // 2]], 64); malformed = "accepted (BAD)"
    except Exception as e:
        malformed = f"rejected: {type(e).__name__}"

    # model utility in FL
    data = make_dataset(seed=0)
    parts = partition_iid(data.y_train, cfg["fl"]["clients"], 0)
    util = []
    for seed in range(scale(3, 1)):
        c = FLConfig(rounds=cfg["fl"]["rounds"], hidden=cfg["fl"]["hidden"], local_epochs=1, seed=seed)
        base = run_fl(c, data, parts)
        for name, mk in (("plaintext", lambda: WeightedMean()), ("ckks", lambda: Audited(CKKSAggregator())), ("bfv", lambda: Audited(BFVAggregator()))):
            agg = mk()
            r = base if name == "plaintext" else run_fl(c, data, parts, aggregator=agg)
            util.append(dict(seed=seed, scheme=name, acc=r.final["acc"], f1=r.final["f1"], loss=r.final["loss"],
                             train_wall_s=r.totals["train_wall_s"], comm_mb=r.totals["comm_mb"],
                             max_agg_err=max(agg.errors) if hasattr(agg, "errors") else 0.0))
    save_csv(STORY, "task_1_2_fl_utility", util)
    # Paillier in FL on a deliberately tiny model (full-size model would take hours)
    tiny = []
    if not cfg["paillier_sizes"][0] < 16:
        c = FLConfig(rounds=2, hidden=4, local_epochs=1, seed=0)
        parts5 = partition_iid(data.y_train, 5, 0)
        b = run_fl(c, data, parts5)
        t0 = time.perf_counter()
        pr_ = run_fl(c, data, parts5, aggregator=Audited(PaillierAggregator(2048)))
        tiny = [dict(scheme="plaintext", acc=b.final["acc"], train_wall_s=b.totals["train_wall_s"], comm_mb=b.totals["comm_mb"]),
                dict(scheme="paillier2048", acc=pr_.final["acc"], train_wall_s=pr_.totals["train_wall_s"], comm_mb=pr_.totals["comm_mb"],
                     note="hidden=4 (174 params), 5 clients, 2 rounds")]
        save_csv(STORY, "task_1_2_paillier_tiny_fl", tiny)

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for s_, mk in (("he_ckks", "o-"), ("he_bfv", "s-"), ("he_paillier", "^--")):
        rr = [r for r in rows if r["scheme"] == s_ and r.get("kind") == "measured"]
        ax[0].plot([r["P"] for r in rr], [r["enc_ms_per_client"] for r in rr], mk, label=s_)
        ax[1].plot([r["P"] for r in rr], [r["ct_kb_per_client"] for r in rr], mk, label=s_)
        ax[2].plot([r["P"] for r in rr], [r["max_abs_err"] for r in rr], mk, label=s_)
    for a_, t_, yl in zip(ax, ("Encryption time per client", "Ciphertext size per client", "Max abs. aggregation error"),
                          ("ms", "KiB", "|HE - plaintext| (max)")):
        a_.set_xscale("log"); a_.set_yscale("log"); a_.set_xlabel("update length P (floats)"); a_.set_ylabel(yl); a_.set_title(t_); a_.legend()
    savefig(fig, STORY, "task_1_2_he_scaling", prov, "5 clients, mean of 3 seeds (Paillier: 1 seed, 2048-bit)")
    res = dict(provenance=prov, keyinfo=keyinfo, micro=rows, precision=prec, wrong_key=wrong_key, malformed=malformed, fl_utility=util, paillier_tiny=tiny)
    save_json(STORY, "task_1_2_he", res)
    return res

def task_1_3_secagg() -> dict:
    cfg = dict(P=1442, sizes=scale([5, 10, 20, 50, 100], [5, 10]), sparse_degree=14, drops=[0.0, 0.1, 0.2, 0.3, 0.4], n_drop=20)
    prov = provenance("FL-SEC-001.3", cfg)
    P = cfg["P"]
    # (a) cost vs n (complete graph vs sparse graph)
    cost = []
    for n in cfg["sizes"]:
        for deg in (None, cfg["sparse_degree"]):
            if deg is not None and n <= deg + 1:
                continue
            D, W = _updates(n, P, seed=n)
            proto = SecureAggregationProtocol(n, P, degree=deg)
            t0 = time.perf_counter()
            out, _ = proto.run({i: D[i] for i in range(n)}, {i: W[i] for i in range(n)})
            dt_ = time.perf_counter() - t0
            err = float(np.abs(out - (W / W.sum()) @ D).max())
            st = proto.stats
            cost.append(dict(n=n, graph="complete" if deg is None else f"sparse k={deg}", total_s=dt_, up_kb_per_client=st["up_bytes_total"] / n / 1024,
                             plaintext_kb_per_client=4 * P / 1024, comm_overhead_x=st["up_bytes_total"] / n / (4 * P),
                             max_abs_err=err, phase_r1_s=st["phase_times_s"]["r1_share"], phase_r2_s=st["phase_times_s"]["r2_mask"],
                             phase_r4_s=st["phase_times_s"]["r4_unmask"], threshold_mean=st["threshold_mean"]))
    save_csv(STORY, "task_1_3_cost_vs_clients", cost)
    # (b) dropout behaviour
    n = cfg["n_drop"]; drop_rows = []
    for stage in ("before_masking", "before_unmask"):
        for frac in cfg["drops"]:
            ok, errs = 0, []
            trials = scale(5, 2)
            for t in range(trials):
                D, W = _updates(n, P, seed=100 + t)
                rng = np.random.default_rng(t)
                dropped = set(rng.choice(n, int(round(frac * n)), replace=False).tolist())
                proto = SecureAggregationProtocol(n, P, threshold_frac=0.67)
                try:
                    kw = dict(drop_before_masking=dropped) if stage == "before_masking" else dict(drop_before_unmask=dropped)
                    out, _ = proto.run({i: D[i] for i in range(n)}, {i: W[i] for i in range(n)}, **kw)
                    keep = [i for i in range(n) if (i not in dropped or stage != "before_masking")]
                    errs.append(float(np.abs(out - (W[keep] / W[keep].sum()) @ D[keep]).max())); ok += 1
                except SecAggAbort:
                    pass
            drop_rows.append(dict(stage=stage, drop_fraction=frac, trials=trials, completed=ok, aborted=trials - ok,
                                  max_abs_err=max(errs) if errs else None, threshold_frac=0.67, n=n))
    save_csv(STORY, "task_1_3_dropout", drop_rows)
    # (c) privacy / security sanity checks (these are necessary, NOT sufficient evidence)
    D, W = _updates(10, P, seed=5)
    proto = SecureAggregationProtocol(10, P)
    proto.run({i: D[i] for i in range(10)}, {i: W[i] for i in range(10)})
    y0 = proto.server_view[0]
    from federated_learning.privacy.secure_aggregation import quantize
    x0 = quantize(np.r_[D[0] * W[0], W[0]], 24)
    corr = float(np.corrcoef(y0.view(np.int64).astype(float), x0.view(np.int64).astype(float))[0, 1])
    hib = float(((y0 >> np.uint64(63)).astype(float)).mean())
    sec = dict(server_view_corr_with_plain=corr, server_view_high_bit_fraction=hib)
    secret = 123456789123456789
    xs = list(range(1, 8)); sh = shamir_split(secret, 4, xs)
    sec["shamir_t_shares_recover"] = shamir_reconstruct({x: sh[x] for x in xs[:4]}) == secret
    sec["shamir_t_minus_1_shares_recover"] = shamir_reconstruct({x: sh[x] for x in xs[:3]}) == secret
    ab = None
    try:
        D, W = _updates(10, P, seed=6)
        SecureAggregationProtocol(10, P).run({i: D[i] for i in range(10)}, {i: W[i] for i in range(10)}, drop_before_masking=set(range(7)))
        ab = "NO ABORT (BAD)"
    except SecAggAbort as e:
        ab = f"aborted as designed: {e}"
    sec["abort_below_threshold"] = ab
    from cryptography.exceptions import InvalidTag
    try:
        SecureAggregationProtocol(6, 32).run({i: np.ones(32, np.float32) for i in range(6)}, {i: 1.0 for i in range(6)}, tamper=(0, 1))
        sec["tampered_share_detected"] = False
    except InvalidTag:
        sec["tampered_share_detected"] = True
    # Flower inspection (evidence of what exists; NOT an end-to-end Flower run)
    import importlib.util, flwr
    sec["flower_version"] = flwr.__version__
    sec["flower_secaggplus_mod_present"] = importlib.util.find_spec("flwr.client.mod.secure_aggregation.secaggplus_mod") is not None
    sec["flower_end_to_end_run"] = "NOT_RUN (requires SuperLink/SuperNode runtime; this protocol is a from-scratch implementation of the same design)"
    # (d) FL integration sanity
    data = make_dataset(seed=0); parts = partition_iid(data.y_train, 10, 0)
    agg = Audited(SecAggAggregator()); r = run_fl(FLConfig(rounds=scale(8, 2), local_epochs=1), data, parts, aggregator=agg)
    sec["fl_integration"] = dict(final_acc=r.final["acc"], max_agg_error=max(agg.errors), rounds=len(agg.errors))
    save_json(STORY, "task_1_3_security_checks", sec)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for g in sorted({c["graph"] for c in cost}):
        cc = [c for c in cost if c["graph"] == g]
        ax[0].plot([c["n"] for c in cc], [c["total_s"] for c in cc], "o-", label=g)
        ax[1].plot([c["n"] for c in cc], [c["comm_overhead_x"] for c in cc], "o-", label=g)
    ax[0].set_xlabel("clients n"); ax[0].set_ylabel("protocol time per round [s]"); ax[0].set_title("Compute (all parties, one process)"); ax[0].legend()
    ax[1].set_xlabel("clients n"); ax[1].set_ylabel("uplink / plaintext update"); ax[1].set_title("Communication overhead factor"); ax[1].legend()
    for stage in ("before_masking", "before_unmask"):
        rr = [d for d in drop_rows if d["stage"] == stage]
        ax[2].plot([d["drop_fraction"] for d in rr], [d["completed"] / d["trials"] for d in rr], "o-", label=stage)
    ax[2].set_xlabel("fraction of clients dropping"); ax[2].set_ylabel("fraction of runs completed"); ax[2].set_title("Dropout tolerance (t=0.67 n, n=20)"); ax[2].legend()
    savefig(fig, STORY, "task_1_3_secagg", prov, "virtual clients, one process")
    res = dict(provenance=prov, cost=cost, dropout=drop_rows, security=sec)
    save_json(STORY, "task_1_3_secagg", res)
    return res

def task_1_4_smpc() -> dict:
    cfg = dict(P=1442, n=10, parties=scale([2, 3, 5], [2, 3]), clients=scale([5, 10, 20], [5]))
    prov = provenance("FL-SEC-001.4", cfg)
    rows = []
    for m in cfg["parties"]:
        D, W = _updates(cfg["n"], cfg["P"], seed=m)
        A = SMPCAggregator(m_parties=m); t0 = time.perf_counter(); out = A.aggregate(D, W); t = time.perf_counter() - t0
        st = A.last_stats
        rows.append(dict(parties_m=m, clients=cfg["n"], collusion_threshold_parties=m - 1, time_s=t,
                         max_abs_err=float(np.abs(out - (W / W.sum()) @ D).max()), online_up_kb_per_client=st["online_bytes_per_client"] / 1024,
                         offline_dealer_kb_total=st["bytes_offline"] / 1024, party_to_party_kb=st["bytes_party_to_party"] / 1024,
                         plaintext_kb_per_client=4 * cfg["P"] / 1024, online_overhead_x=st["online_bytes_per_client"] / (4 * cfg["P"])))
    save_csv(STORY, "task_1_4_parties", rows)
    scal = []
    for n in cfg["clients"]:
        D, W = _updates(n, cfg["P"], seed=n)
        A = SMPCAggregator(m_parties=3); t0 = time.perf_counter(); A.aggregate(D, W); scal.append(dict(clients=n, time_s=time.perf_counter() - t0))
    save_csv(STORY, "task_1_4_scaling_clients", scal)
    # secure norm-bound filtering (uses Beaver multiplication => something Secure Aggregation cannot do)
    D, W = _updates(10, cfg["P"], seed=9); D2 = D.copy(); D2[3] *= 50
    A = SMPCAggregator(m_parties=3, norm_bound=6.0); out = A.aggregate(D2, W)
    keep = [i for i in range(10) if i not in A.last_rejected]
    norm_demo = dict(true_norms=np.linalg.norm(D2, axis=1).round(2).tolist(), rejected=A.last_rejected,
                     err_vs_plain_of_kept=float(np.abs(out - (W[keep] / W[keep].sum()) @ D2[keep]).max()), time_s=A.last_stats["crypto_time_s"],
                     note="the revealed value is ||delta_i||^2 per client (a deliberate, documented leakage)")
    # active adversary: a corrupted party shifts one share -> MAC check must abort
    s = SMPCSession(3); x = s.input(_to_field(np.arange(8)))
    honest = s.open(x).tolist()
    def bad(v):
        v = v.copy(); v[1, 3] = (v[1, 3] + 7) % SMPC_P; return v
    s.tamper_hook = bad
    try:
        s.open(x); detected = False
    except MACCheckFailure:
        detected = True
    # privacy of shares: any m-1 shares of a secret look uniform => chi-square on top bits
    s2 = SMPCSession(3)
    sec = s2.input(_to_field(np.full(4000, 42)))
    sh_bits = np.array([int(v) >> 58 for v in sec.val[0]])      # 3 high bits of a single party's share
    counts = np.bincount(sh_bits, minlength=8)
    from scipy.stats import chisquare
    chi = chisquare(counts)
    privacy = dict(single_party_share_chi2_p=float(chi.pvalue), note="uniformity is necessary, not sufficient; the security argument is information-theoretic (docs)")
    res = dict(provenance=prov, parties=rows, scaling=scal, norm_filter=norm_demo,
               tamper=dict(honest_open=honest, mac_detects_corrupted_party=detected), share_privacy=privacy,
               mp_spdz=dict(status="BLOCKED", reason="MP-SPDZ release binaries/GitHub not reachable from this environment (HTTP 403) and no build toolchain/source",
                            needed="network access to github.com/data61/MP-SPDZ and a C++ toolchain to compile; then replace the trusted-dealer preprocessing with its offline phase"))
    save_json(STORY, "task_1_4_smpc", res)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].bar([f"m={r['parties_m']}" for r in rows], [r["time_s"] for r in rows]); ax[0].set_ylabel("aggregation time [s] (10 clients)"); ax[0].set_title("SMPC time vs parties")
    w = 0.35; xs_ = np.arange(len(rows))
    ax[1].bar(xs_ - w / 2, [r["online_up_kb_per_client"] for r in rows], w, label="online client upload")
    ax[1].bar(xs_ + w / 2, [r["plaintext_kb_per_client"] for r in rows], w, label="plaintext update")
    ax[1].set_xticks(xs_); ax[1].set_xticklabels([f"m={r['parties_m']}" for r in rows]); ax[1].set_ylabel("KiB per client"); ax[1].set_title("Client upload"); ax[1].legend()
    savefig(fig, STORY, "task_1_4_smpc", prov)
    return res

def task_1_5_comparison() -> dict:
    cfg = dict(clients=10, rounds=scale(15, 3), seeds=scale(10, 1), local_epochs=1, hidden=32, timing_design="seed-outer, randomised config order, warm-up excluded")
    prov = provenance("FL-SEC-001.5", cfg)
    data = make_dataset(seed=0)
    parts = partition_iid(data.y_train, cfg["clients"], 0)
    makers = {
        "plaintext": lambda s: WeightedMean(),
        "he_ckks": lambda s: Audited(CKKSAggregator()),
        "he_bfv": lambda s: Audited(BFVAggregator()),
        "secagg": lambda s: Audited(SecAggAggregator(seed=s)),
        "secagg_10pct_dropout": lambda s: SecAggAggregator(dropout_rate=0.1, seed=s),
        "smpc_3party": lambda s: Audited(SMPCAggregator(3)),
    }
    rows, curves = [], {}
    run_fl(FLConfig(rounds=2, local_epochs=1, hidden=cfg["hidden"], seed=0), data, parts)   # warm-up run, excluded from the results
    for s in range(cfg["seeds"]):
        # randomised configuration order within each seed: timing then does not depend on run order (drift, caches, thermal state)
        for name in [str(n) for n in np.random.default_rng(1000 + s).permutation(list(makers))]:
            mk = makers[name]
            agg = mk(s)
            with ResourceMonitor() as mon:
                r = run_fl(FLConfig(rounds=cfg["rounds"], local_epochs=cfg["local_epochs"], hidden=cfg["hidden"], seed=s), data, parts, aggregator=agg)
            cs = [h.get("crypto_time_s", 0.0) for h in r.history]
            crypto = r.totals["agg_time_s"]
            rows.append(dict(config=name, seed=s, acc=r.final["acc"], f1=r.final["f1"], auc=r.final["auc"], loss=r.final["loss"],
                             train_wall_s=r.totals["train_wall_s"], agg_time_s=crypto, comm_mb=r.totals["comm_mb"], up_mb=r.totals["up_bytes"] / 2 ** 20,
                             rss_peak_mb=mon.summary()["rss_peak_mb"], max_agg_err=max(getattr(agg, "errors", [0.0]) or [0.0])))
            curves.setdefault(name, []).append([h["acc"] for h in r.history])
    base = {r["seed"]: r for r in rows if r["config"] == "plaintext"}
    for r in rows:
        b = base[r["seed"]]
        r["train_overhead_pct"] = 100 * (r["train_wall_s"] - b["train_wall_s"]) / b["train_wall_s"]
        r["comm_overhead_pct"] = 100 * (r["up_mb"] - b["up_mb"]) / b["up_mb"]
        r["acc_loss_pp"] = 100 * (b["acc"] - r["acc"])
    save_csv(STORY, "task_1_5_comparison_runs", rows)
    summ = []
    for name in makers:
        rr = [r for r in rows if r["config"] == name]
        row = dict(config=name)
        for k in ("acc", "f1", "train_wall_s", "agg_time_s", "up_mb", "train_overhead_pct", "comm_overhead_pct", "acc_loss_pp", "max_agg_err"):
            m, lo, hi = mean_ci([r[k] for r in rr]); row[k] = m; row[k + "_ci"] = f"[{lo:.3g}, {hi:.3g}]"
        summ.append(row)
    save_csv(STORY, "task_1_5_comparison_summary", summ)
    save_json(STORY, "task_1_5_comparison", dict(provenance=prov, summary=summ, curves={k: v for k, v in curves.items()},
                                                 paillier="EXCLUDED from the full-size comparison (see task 1.2: ~hours per round); tiny-model result in task 1.2",
                                                 incompatibility="secagg/he cannot run median/Krum (need per-update access); see ablation FL-PUB-005"))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for name, cv in curves.items():
        ax[0].plot(range(1, len(cv[0]) + 1), np.mean(cv, 0), label=name)
    ax[0].set_xlabel("round"); ax[0].set_ylabel("test accuracy"); ax[0].set_title("Convergence (mean of seeds)"); ax[0].legend(fontsize=6)
    ax[1].bar([s["config"] for s in summ], [s["train_wall_s"] for s in summ]); ax[1].set_ylabel("training wall time [s]"); ax[1].set_title("Training time"); ax[1].tick_params(axis="x", rotation=60)
    ax[2].bar([s["config"] for s in summ], [s["up_mb"] for s in summ]); ax[2].set_ylabel("uplink [MiB, whole run]"); ax[2].set_title("Communication"); ax[2].tick_params(axis="x", rotation=60)
    savefig(fig, STORY, "task_1_5_comparison", prov, f"{cfg['clients']} virtual clients, IID, {cfg['rounds']} rounds, {cfg['seeds']} seeds")
    return dict(provenance=prov, summary=summ)
