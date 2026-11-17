"""Generates one Jupyter notebook per Jira task with the mandatory 18-section structure.
Notebooks: (a) load the stored MEASURED results of the task (produced by `run_all`), (b) re-run a small, fast, live
check of the implementation, (c) compute findings from the stored data. Heavy experiments are NOT re-run inside
notebooks (CLI: python -m federated_learning.pipelines.experiments.run_all --task <id>)."""
from pathlib import Path

import nbformat as nbf

R = Path(__file__).resolve().parents[1]
NB = R / "notebooks" / "sprint_tasks"

SETUP = '''import os, sys, json, platform, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
def _find_root():
    p = Path(os.environ.get("FL_ROOT") or os.environ.get("RS_ROOT") or Path.cwd()).resolve()
    for q in [p, *p.parents]:
        if (q / "src" / "federated_learning").exists():
            return q
    raise RuntimeError("repository root not found; set RS_ROOT")
ROOT = _find_root()
sys.path.insert(0, str(ROOT / "src"))
STORY = "{story}"
_cands = [ROOT / "experiments" / "results" / STORY]
RES = next(c for c in _cands if c.exists())
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
SEED = 0; np.random.seed(SEED)
def show_csv(name, n=12):
    p = RES / name
    if p.exists(): display(pd.read_csv(p).round(4).head(n))
    else: print("MISSING", name)
def show_fig(name):
    for base in (RES / "figures", RES):
        if (base / name).exists(): display(Image(filename=str(base / name))); return
    print("MISSING figure", name)
print("root:", ROOT, "| results:", RES)'''

ENV = '''import importlib.metadata as md
vers = {}
for p in ["numpy","scipy","scikit-learn","torch","tenseal","phe","pandas","matplotlib","cryptography","psutil","mlflow","opacus"]:
    try: vers[p] = md.version(p)
    except Exception: vers[p] = "not installed"
print(json.dumps(dict(python=platform.python_version(), platform=platform.platform(), versions=vers, seed=SEED, gpu="none (CPU only; GPU metrics NOT_MEASURED)"), indent=1))'''

DATA = '''from federated_learning.synthetic_data.generators.syn_ids_v1 import make_dataset, DATASET_ID
from federated_learning.data_partitioning.iid import partition_iid
from federated_learning.data_partitioning.dirichlet import partition_dirichlet
from federated_learning.data_partitioning.metrics import label_skew_stat
data = make_dataset(n_samples=6000, seed=SEED)
print(DATASET_ID, "train", data.X_train.shape, "test", data.X_test.shape, "positive rate", float(data.y_train.mean()))
print("NOTE: SYNTHETIC tabular data (not CIC-IDS-2017, not patient data); clients are VIRTUAL, not hospitals.")'''

def cell_md(s): return nbf.v4.new_markdown_cell(s)
def cell_code(s): return nbf.v4.new_code_cell(s)

def build(t):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    c = nb.cells
    c.append(cell_md(f"# {t['id']} — {t['title']}\n\n*Epic EPIC-001 · story {t['story']} · all results below are loaded from files produced by the task's experiment function and labelled MEASURED/SIMULATED; clients are virtual; data is synthetic.*"))
    c.append(cell_md("## 1. Research objective\n" + t["objective"]))
    c.append(cell_md("## 2. Scientific background\n" + t["background"] + "\n\nReferences use keys of `docs/references/references_master.md` (status per entry)."))
    c.append(cell_md("## 3. Research questions\n" + "\n".join(f"- **RQ{i+1}.** {q}" for i, q in enumerate(t["rqs"]))))
    c.append(cell_md("## 4. Hypotheses\n" + "\n".join(f"- **H{i+1}.** {h}" for i, h in enumerate(t["hyps"]))))
    c.append(cell_md("## 5. Mathematical formulation\n" + t["math"]))
    c.append(cell_md("## 6. Experimental environment"))
    c.append(cell_code(SETUP.format(story=t["story"])))
    c.append(cell_code(ENV))
    c.append(cell_md("## 7. Dataset description\nSynthetic binary tabular dataset **SYN-IDS-v1** (scikit-learn `make_classification`, 40 features, 70:30 split). Chosen because no real IDS/clinical dataset was available in the sandbox. It has no clinical meaning; conclusions about absolute accuracy do not transfer."))
    c.append(cell_code(DATA))
    c.append(cell_md("## 8. Configuration and reproducibility settings\nSeeds are fixed; each stored result carries `provenance` (experiment id, SHA-256 of the configuration, environment versions; `git_commit` is `null` because the workspace is not a git repository)."))
    c.append(cell_code(t["config_code"]))
    c.append(cell_md("## 9. Data preparation\nPartitioning into virtual clients (IID or Dirichlet label skew) is performed by `federated_learning.data_partitioning`."))
    c.append(cell_code(t.get("prep_code", "parts_iid = partition_iid(data.y_train, 10, SEED)\nparts_dir = partition_dirichlet(data.y_train, 10, 0.3, SEED)\nprint('label-skew statistic IID:', round(label_skew_stat(data.y_train, parts_iid),3), ' Dirichlet(0.3):', round(label_skew_stat(data.y_train, parts_dir),3))")))
    c.append(cell_md("## 10. Implementation\nReusable code lives in `src/federated_learning/` (" + t["modules"] + "). The cell below exercises it live (small, fast, deterministic) and asserts correctness."))
    c.append(cell_code(t["live_code"]))
    c.append(cell_md("## 11. Experiment execution\nThe full experiment was executed by `python -m federated_learning.pipelines.experiments.run_all --task " + t["id"] + "` (see `logs/`). Its raw per-run results are loaded here; they are not re-computed inside the notebook."))
    c.append(cell_code("\n".join(f"show_csv({f!r})" for f in t["runs"])))
    c.append(cell_md("## 12. Metrics collection\nMetric definitions: " + t["metrics"]))
    c.append(cell_code(t["metrics_code"]))
    c.append(cell_md("## 13. Statistical analysis\n" + t["stats_text"]))
    c.append(cell_code(t["stats_code"]))
    c.append(cell_md("## 14. Visualizations"))
    c.append(cell_code("\n".join(f"show_fig({f!r})" for f in t["figs"]) or "print('no figure for this task (tabular result)')"))
    c.append(cell_md("## 15. Interpretation of results\nThe numbers are computed in the next cell from the stored data; the text interprets only what that cell prints.\n\n" + t["interp"]))
    c.append(cell_code(t["findings_code"]))
    c.append(cell_md("## 16. Limitations\n" + "\n".join(f"- {l}" for l in t["limits"] + [
        "Synthetic data, virtual clients on one machine, 2 CPU cores, no GPU; no real network. Comm. bytes are logical payload sizes.",
        "No regulatory/compliance conclusion can be drawn from this notebook."])))
    c.append(cell_md("## 17. Conclusions\n" + t["conclusions"] + "\n\n**Jira acceptance:** see `docs/" + t["story"] + "/results_and_discussion.md` for the criterion-by-criterion verdict (MET / PARTIALLY MET / NOT MET)."))
    c.append(cell_md("## 18. References\n" + ", ".join(f"`{k}`" for k in t["refs"]) + " — details in `docs/" + t["story"] + "/references.md`."))
    return nb

CONFIG_FL = '''from federated_learning.training.config import FLConfig
cfg = FLConfig(rounds=5, local_epochs=1, seed=SEED)
print(cfg)'''
CONFIG_NONE = 'print("No FL configuration is needed to read stored results; see the stored JSON `provenance` blocks.")\nprint(json.dumps(json.load(open(next(RES.glob("*.json"))))["provenance"]["config"], indent=1)[:800])'

def csvs(*a): return list(a)

def rj(name):  # snippet to load a json
    return f"J = json.load(open(RES / {name!r}))"

T = []
def add(**k): T.append(k)

# ------------------------------------------------------------------ FL-SEC-001
S = "fl_sec_001"
live_he = '''from federated_learning.privacy.homomorphic import CKKSAggregator, BFVAggregator
rng = np.random.default_rng(SEED); D = rng.normal(0, .1, (4, 300)).astype(np.float32); W = np.array([10., 20., 30., 40.])
ref = (W / W.sum()) @ D
for A in (CKKSAggregator(), BFVAggregator()):
    out = A.aggregate(D, W); err = float(np.abs(out - ref).max()); print(type(A).__name__, "max abs error vs plaintext:", err, "| keygen:", {k: A.keygen_info()[k] for k in A.keygen_info() if "secret" in k})
    assert err < 1e-4
print("server context holds no secret key (key-holder model) -> asserted in tests/")'''
live_sa = '''from federated_learning.privacy.secure_aggregation import SecureAggregationProtocol, SecAggAbort
n, P = 10, 64; r = np.random.default_rng(SEED); D = r.normal(0, .1, (n, P)); W = r.integers(5, 50, n).astype(float)
pr = SecureAggregationProtocol(n, P)
out, w = pr.run({i: D[i] for i in range(n)}, {i: W[i] for i in range(n)}, drop_before_unmask={2, 3})
print("dropout (2 clients after masking) -> max error vs plaintext:", float(np.abs(out - (W / W.sum()) @ D).max())); assert np.abs(out - (W / W.sum()) @ D).max() < 1e-6
try:
    SecureAggregationProtocol(n, P).run({i: D[i] for i in range(n)}, {i: W[i] for i in range(n)}, drop_before_masking=set(range(7)))
except SecAggAbort as e: print("abort below threshold:", e)'''
live_smpc = '''from federated_learning.privacy.smpc import SMPCSession, SMPCAggregator, MACCheckFailure, _to_field, _centered
s = SMPCSession(3); a, b = np.array([3, -4, 7]), np.array([5, 6, -9])
z = s.multiply(s.input(_to_field(a)), s.input(_to_field(b))); print("Beaver product:", [int(v) for v in _centered(s.open(z))], "expected", (a * b).tolist())
s2 = SMPCSession(3); x = s2.input(_to_field(np.arange(6)))
def bad(v):
    v = v.copy(); v[1, 2] = (v[1, 2] + 1) % (2**61 - 1); return v
s2.tamper_hook = bad
try: s2.open(x); print("NOT DETECTED")
except MACCheckFailure: print("tampered share detected by MAC check -> abort")'''

add(id="FL-SEC-001.1", story=S, title="Cryptographic approach selection (HE / SecAgg / SMPC)",
    objective="Select, with explicit criteria and evidence, the cryptographic aggregation mechanisms to implement for healthcare FL, and document assumptions. Output: a decision matrix with theoretical properties separated from measured costs.",
    background="Three families protect model updates from the aggregator: homomorphic encryption (CKKS [cheon2017], BFV [fv2012], Paillier [paillier1999]), pairwise-masking secure aggregation [bonawitz2017, bell2020] and secret-sharing MPC (SPDZ-style [damgard2012, beaver1991]). They differ in trust assumptions, dropout handling and cost; none protects against poisoning.",
    rqs=["Which mechanisms satisfy the story's requirements (server learns only the aggregate; correctness; dropout tolerance)?", "What are the measured per-update costs of the HE schemes at the model size used?"],
    hyps=["CKKS gives the lowest ciphertext expansion among HE schemes for float vectors, BFV is exact up to quantisation, Paillier is the most expensive.", "SecAgg is cheapest in bandwidth but requires dropout machinery."],
    math="CKKS encodes a real vector into a polynomial ring element and supports additive aggregation with error ε ≈ 2^{-scale}; BFV works over Z_t with fixed-point quantisation q(x)=round(x·2^f); Paillier: Enc(m)=g^m r^n mod n², Enc(a)·Enc(b)=Enc(a+b). Shamir t-of-n sharing over a prime field. Weighted mean: w̄ = Σ p_k Δ_k, p_k = n_k/Σn.",
    config_code=CONFIG_NONE, modules="`crypto/he.py`", live_code=live_he,
    runs=csvs("task_1_1_he_comparison_measured.csv", "task_1_1_protocol_comparison_theoretical.csv"),
    metrics="ciphertext bytes/client, expansion factor, enc/agg/dec time (MEASURED); security properties (THEORETICAL, from the cited papers).",
    metrics_code="show_csv('task_1_1_he_comparison_measured.csv')",
    stats_text="Single-configuration microbenchmark with repeated timing; no hypothesis test. Cost-vs-size scaling is analysed in task 1.2.",
    stats_code="J = json.load(open(RES / 'task_1_1_decision_matrix.json')); print(list(J.keys()))",
    figs=["task_1_1_he_cost.png"], interp="The decision matrix distinguishes *measured* rows from *theoretical* rows; no scheme is declared 'secure' based on timings.",
    findings_code="J = json.load(open(RES / 'task_1_1_decision_matrix.json')); print(json.dumps({k: J[k] for k in J if k != 'provenance'}, indent=1)[:2500])",
    limits=["Security claims are taken from the literature, not verified here.", "Only TenSEAL/phe backends were measured."],
    conclusions="All three families were implemented in this sprint; the choice is use-case dependent (see decision matrix).", refs=["cheon2017", "fv2012", "paillier1999", "bonawitz2017", "damgard2012", "kairouz2021"])
add(id="FL-SEC-001.2", story=S, title="Homomorphic-encryption aggregation (CKKS, BFV, Paillier)",
    objective="Implement and validate encrypted weighted aggregation in a key-holder model, measuring precision, cost and FL utility.",
    background="CKKS is approximate, BFV exact over integers with quantisation, Paillier additively homomorphic with one ciphertext per element [cheon2017, fv2012, paillier1999, benaloh_tenseal].",
    rqs=["Does encrypted aggregation reproduce plaintext FedAvg accuracy?", "How do error and cost scale with vector length and magnitude?"],
    hyps=["Max aggregation error of CKKS < 1e-5 for update magnitudes used.", "Final accuracy equals plaintext FedAvg (loss ≈ 0 pp)."],
    math="Dec(Σ_k Enc(p_k Δ_k)) ≈ Σ_k p_k Δ_k; CKKS error bound ~ N·2^{-scale}; BFV fixed-point: Δ̂ = round(2^f Δ), overflow if |Σ| ≥ t/2.",
    config_code=CONFIG_NONE, modules="`crypto/he.py`, `defenses.Audited`", live_code=live_he,
    runs=csvs("task_1_2_fl_utility.csv", "task_1_2_he_microbenchmarks.csv", "task_1_2_precision_vs_magnitude.csv", "task_1_2_paillier_tiny_fl.csv"),
    metrics="MEASURED: enc/agg/dec seconds, ciphertext bytes, max abs error vs plaintext, accuracy.",
    metrics_code="show_csv('task_1_2_fl_utility.csv')",
    stats_text="Per-seed results with 95% CIs where repeated (see 1.5 for the multi-seed comparison).", stats_code="show_csv('task_1_2_he_microbenchmarks.csv', 20)",
    figs=["task_1_2_he_scaling.png"], interp="Encrypted aggregation matches plaintext accuracy; the cost is paid in time and bandwidth (see printed table).",
    findings_code="d = pd.read_csv(RES / 'task_1_2_fl_utility.csv'); display(d.round(5))",
    limits=["Key-holder (decryptor) assumed honest and non-colluding with the server.", "Paillier evaluated on a tiny model only (cost)."],
    conclusions="HE aggregation is functionally correct; overhead is substantial (quantified in the table).", refs=["cheon2017", "fv2012", "paillier1999", "benaloh_tenseal"])
add(id="FL-SEC-001.3", story=S, title="Secure aggregation with dropout recovery",
    objective="Implement Bonawitz-style secure aggregation (X25519, HKDF, ChaCha20-Poly1305, Shamir, double masking) with dropout recovery and security checks.",
    background="Pairwise masks cancel in the sum; Shamir shares of seeds allow recovery when clients drop [bonawitz2017]; sparse neighbour graphs reduce cost [bell2020].",
    rqs=["Is the aggregate exact under dropout?", "What does the server view look like, and when does the protocol abort?"],
    hyps=["Aggregate error < 1e-6 with ≤ (1−threshold) dropout.", "Server view is statistically unrelated to plaintext."],
    math="y_i = x_i + PRG(b_i) + Σ_{j>i} PRG(s_ij) − Σ_{j<i} PRG(s_ij) (mod 2^64); Σ y_i − Σ b-masks = Σ x_i; Shamir over GF(2^521−1) with threshold t.",
    config_code=CONFIG_NONE, modules="`crypto/secagg.py`", live_code=live_sa,
    runs=csvs("task_1_3_cost_vs_clients.csv", "task_1_3_dropout.csv"), metrics="MEASURED: aggregate error, round time, bytes; security checks are protocol-level assertions, not a security proof.",
    metrics_code="show_csv('task_1_3_dropout.csv')", stats_text="Deterministic correctness checks; costs reported per client count.", stats_code="J = json.load(open(RES / 'task_1_3_security_checks.json')); print(json.dumps(J, indent=1))",
    figs=["task_1_3_secagg.png"], interp="Correctness under dropout, abort below threshold and tamper detection are demonstrated; a consistency/signature round (needed against a malicious server) is NOT implemented.",
    findings_code="J = json.load(open(RES / 'task_1_3_security_checks.json')); print({k: J[k] for k in ['server_view_corr_with_plain','shamir_t_shares_recover','shamir_t_minus_1_shares_recover','abort_below_threshold','tampered_share_detected','flower_end_to_end_run']})",
    limits=["Semi-honest server only.", "Flower's secaggplus primitives were inspected but not run end-to-end (NOT_RUN)."],
    conclusions="Secure aggregation with dropout recovery works in simulation and is exact to quantisation precision.", refs=["bonawitz2017", "bell2020", "shamir1979"])
add(id="FL-SEC-001.4", story=S, title="SMPC aggregation with authenticated shares (SPDZ-style)",
    objective="Implement additive secret sharing with information-theoretic MACs, Beaver triples and MAC-checked opening; detect share tampering.",
    background="SPDZ-style protocols authenticate shares with a global MAC key [damgard2012]; Beaver triples enable multiplication [beaver1991]. The offline phase here is a trusted dealer (NOT a real preprocessing protocol).",
    rqs=["Is the aggregate exact?", "Is share tampering detected?", "How do costs scale with parties and clients?"],
    hyps=["Aggregate error < 1e-5.", "Any single-share modification triggers MAC failure."],
    math="x = Σ_i x_i mod p, p=2^61−1; MAC m_j = α_j x; check Σ_i (m_j)_i − α_j x = 0 for each of K=2 keys; Beaver: z = c + εb + δa + εδ with ε=x−a, δ=y−b.",
    config_code=CONFIG_NONE, modules="`crypto/smpc.py`", live_code=live_smpc,
    runs=csvs("task_1_4_parties.csv", "task_1_4_scaling_clients.csv"), metrics="MEASURED: bytes (offline / party-to-party / online per client), time, error.",
    metrics_code="show_csv('task_1_4_parties.csv')", stats_text="Deterministic checks plus scaling tables.", stats_code="show_csv('task_1_4_scaling_clients.csv')",
    figs=["task_1_4_smpc.png"], interp="MAC checking detects tampering (abort); correctness holds to fixed-point precision.", findings_code="J = json.load(open(RES / 'task_1_4_smpc.json')); print({k: J[k] for k in J if k != 'provenance'} if isinstance(J, dict) else J)",
    limits=["MP-SPDZ binaries were unavailable → from-scratch implementation; not a production MPC.", "Trusted-dealer offline phase.", "Norm-bound check reveals norms to the parties."],
    conclusions="Authenticated SMPC aggregation works in simulation; cost is high relative to SecAgg.", refs=["damgard2012", "beaver1991", "shamir1979"])
add(id="FL-SEC-001.5", story=S, title="Comparative analysis of HE vs SecAgg vs SMPC",
    objective="Compare plaintext, CKKS, BFV, SecAgg (with/without dropout) and SMPC on accuracy, time, communication and aggregation error across seeds.",
    background="See tasks 1.1–1.4.", rqs=["Does any mechanism change accuracy?", "What are the training-time and communication overheads?"],
    hyps=["Accuracy loss < 1 pp for all mechanisms.", "Overheads differ by orders of magnitude between mechanisms."],
    math="overhead% = 100·(t_secure − t_plain)/t_plain; accuracy loss (pp) = 100·(acc_plain − acc_secure).",
    config_code=CONFIG_NONE, modules="`fl.py`, `crypto/*`, `monitoring.py`", live_code=live_sa,
    runs=csvs("task_1_5_comparison_runs.csv"), metrics="MEASURED means and 95% CIs over seeds; overhead formulas above.",
    metrics_code="show_csv('task_1_5_comparison_summary.csv')", stats_text="Bootstrap/t CIs over seeds (unit of replication = seed).", stats_code="d = pd.read_csv(RES / 'task_1_5_comparison_summary.csv'); display(d[['config','acc','acc_ci','train_overhead_pct','comm_overhead_pct','acc_loss_pp']])",
    figs=["task_1_5_comparison.png"], interp="Accuracy is unchanged by all mechanisms; cost differs strongly (printed).", findings_code="d = pd.read_csv(RES / 'task_1_5_comparison_summary.csv'); print(d[['config','train_overhead_pct','comm_overhead_pct','acc_loss_pp']].round(2).to_string(index=False))",
    limits=["Small seed count; timing depends on the 2-core machine."], conclusions="Mechanism choice is a cost/trust trade-off, not an accuracy trade-off, in this simulation.", refs=["bonawitz2017", "cheon2017", "damgard2012", "kairouz2021"])

# ------------------------------------------------------------------ FL-PERF-002
P = "fl_perf_002"
live_perf = '''from federated_learning.training.config import FLConfig
from federated_learning.servers.orchestrator import run_fl
from federated_learning.monitoring.resources import ResourceMonitor
parts = partition_iid(data.y_train, 5, SEED)
with ResourceMonitor(0.02) as m:
    r = run_fl(FLConfig(rounds=3, local_epochs=1, seed=SEED), data, parts)
s = m.summary(); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
assert s["cpu_time_s"] > 0 and s["wall_s"] > 0'''
def perf(i, title, n, runs, fig, obj, interp, extra_limits=None):
    add(id=f"FL-PERF-002.{i}", story=P, title=title, objective=obj,
        background="Performance of privacy-preserving FL is dominated by cryptographic computation and ciphertext/mask expansion [kairouz2021, li2020challenges]. All clients here are virtual and share one machine, so wall time includes serial execution of clients.",
        rqs=["What are the training-time, communication, memory and CPU overheads of each secure aggregation mechanism?", "Are the story KPIs (<30% training overhead, <40% communication overhead, <1 pp accuracy loss) met?"],
        hyps=["Secure aggregation overhead exceeds the 30%/40% KPIs for at least some mechanisms in simulation."],
        math="overhead% = 100(x_secure − x_plain)/x_plain; throughput = samples processed / client compute seconds; scaling exponent b from log-log fit time ∝ n^b.",
        config_code=CONFIG_FL, modules="`monitoring.py`, `fl.py`, `crypto/*`", live_code=live_perf, runs=runs,
        metrics="Defined in `monitoring.py` docstring and docs/fl_perf_002/metrics_definitions.md: cpu_time_s (cumulative), cpu util mean/peak, RSS mean/peak, logical comm bytes, GPU = unavailable (never estimated).",
        metrics_code=f"show_csv({runs[0]!r})", stats_text="Repeated seeds per configuration; CIs from t-distribution/bootstrap; no significance claims beyond CIs.",
        stats_code=f"show_csv({runs[-1]!r}, 15)", figs=fig, interp=interp,
        findings_code="print('see stored tables above; KPI verdicts are in task_2_5_kpi_check.csv (task 2.5)')" if i != 5 else "d = pd.read_csv(RES / 'task_2_5_kpi_check.csv'); print(d.round(1).to_string(index=False))",
        limits=["Virtual clients: time does not reflect parallel hospitals or network latency."] + (extra_limits or []),
        conclusions="See results_and_discussion.md for the KPI verdicts; they are reported as measured, including NOT MET.", refs=["kairouz2021", "li2020challenges", "bonawitz2017", "cheon2017"])
perf(1, "Resource and communication monitoring framework", 10, ["task_2_1_round_timeline.csv"], ["task_2_1_monitoring.png"], "Build and validate the monitor (CPU, memory, comm, MLflow) used by all benchmarks.", "The monitor tracks cumulative CPU time and RSS; validated against busy-loop and allocation tests in tests/.")
perf(2, "Benchmark with 10 virtual clients", 10, ["fl_perf_002.2_runs.csv"], ["task_2_2_benchmark_10.png"], "Measure secure-aggregation overhead at 10 clients.", "Overheads are reported per mechanism with CIs.")
perf(3, "Benchmark with 50 virtual clients", 50, ["fl_perf_002.3_runs.csv"], ["task_2_3_benchmark_50.png"], "Measure overhead at 50 clients.", "Overheads grow with client count for pairwise-masking designs.")
perf(4, "Benchmark with 100 and 500 virtual clients", 100, ["fl_perf_002.4a_runs.csv", "fl_perf_002.4b_runs.csv"], ["task_2_4_benchmark_100.png", "task_2_4_benchmark_500.png"], "Measure overhead at 100 and 500 virtual clients.", "At 500 clients complete-graph SecAgg was NOT_RUN (cost); a sparse-graph variant (k=14) was measured instead.", ["Complete-graph SecAgg at 500 clients: NOT_RUN."])
perf(5, "Scalability analysis and KPI check", 500, ["task_2_5_scalability_table.csv", "task_2_5_scaling_exponents.csv", "task_2_5_kpi_check.csv"], ["task_2_5_scalability_curves.png"], "Fit scaling exponents, check story KPIs, and discuss limits; cost figures use labelled illustrative price assumptions.", "KPIs <30% training overhead and <40% communication overhead are mostly NOT MET in this simulation (printed).", ["Price assumptions are illustrative, not quotes."])

# ------------------------------------------------------------------ FL-ATTACK-003
A = "fl_attack_003"
live_bd = '''from federated_learning import attacks as A
from federated_learning.training.config import FLConfig
from federated_learning.servers.orchestrator import run_fl
parts = partition_iid(data.y_train, 10, SEED); rng = np.random.default_rng(SEED); mal = {0, 1, 2}
cd = {k: A.poison_client_data(data.X_train[parts[k]], data.y_train[parts[k]], 0.6, 0, rng)[:2] for k in mal}
ev = lambda m, w: A.backdoor_metrics(m, w, data.X_test, data.y_test, 0)
atk = run_fl(FLConfig(rounds=8, local_epochs=1, seed=SEED), data, parts, client_data=cd, client_hook=A.boost_hook(mal, 3.0), eval_fn=ev)
cln = run_fl(FLConfig(rounds=8, local_epochs=1, seed=SEED), data, parts, eval_fn=ev)
print("ASR attacked:", round(atk.final["asr"], 3), "| ASR clean model:", round(cln.final["asr"], 3)); assert atk.final["asr"] > cln.final["asr"]'''
live_def = '''from federated_learning.aggregators.mean import WeightedMean
from federated_learning.aggregators.robust import CoordinateMedian, TrimmedMean, Krum
from federated_learning.aggregators.defenses import ClippedMean
r = np.random.default_rng(SEED); D = r.normal(1.0, .05, (10, 30)).astype(np.float32); D[:2] = -50.0; W = np.ones(10)
for a in (WeightedMean(), CoordinateMedian(), TrimmedMean(0.25), Krum(2, 1)):
    print(f"{type(a).__name__:18s} max|agg-1| = {np.abs(a.aggregate(D, W) - 1.0).max():.3f}")'''
live_mia = '''from federated_learning import attacks as A
rng = np.random.default_rng(SEED)
print("perfect separation:", A.mia_metrics(np.ones(100), np.zeros(100))["auc"], "| random scores AUC:", round(A.mia_metrics(rng.normal(size=3000), rng.normal(size=3000))["auc"], 3))'''
live_gl = '''from federated_learning import attacks as A
from federated_learning.training.models import build_model, get_flat
m = build_model(40, 32, "sigmoid"); w = get_flat(m); X, y = data.X_test[:2], data.y_test[:2]
g = A.model_grad(m, w, X, y); Xh = A.dlg_attack(m, w, g, 2, 40, labels=y, iters=300, seed=SEED)
print("DLG on B=2 plaintext gradient:", A.match_and_score(X, Xh))
m1 = build_model(40, 16, "sigmoid"); w1 = get_flat(m1); g1 = A.model_grad(m1, w1, data.X_test[:1], data.y_test[:1])
xa = A.analytic_first_layer_inversion(g1, 40, 16); print("analytic B=1 relative error:", float(np.linalg.norm(xa - data.X_test[0]) / np.linalg.norm(data.X_test[0])))'''
def att(i, title, obj, bg, rqs, hyps, math, live, runs, summ, figs, interp, find, limits, concl, refs, mods="`attacks.py`, `defenses.py`, `fl.py`"):
    add(id=f"FL-ATTACK-003.{i}", story=A, title=title, objective=obj, background=bg, rqs=rqs, hyps=hyps, math=math, config_code=CONFIG_FL, modules=mods, live_code=live,
        runs=runs, metrics="MEASURED over seeds: attack metric (ASR / AUC / success rate), clean accuracy, defense cost. Intensity levels low/medium/high are defined in docs/fl_attack_003/attack_methodology.md.",
        metrics_code=f"show_csv({summ!r})", stats_text="95% CIs over seeds; AUC/ASR CIs may extend beyond [0,1] because they are t-intervals on few seeds (reported as computed).",
        stats_code=f"show_csv({summ!r}, 30)", figs=figs, interp=interp, findings_code=find, limits=limits, conclusions=concl, refs=refs)
att(1, "Backdoor attack and defenses", "Measure backdoor attack success (trigger + poisoning + boosting) at three intensities and defence effect.", "Backdoors embed a trigger-conditioned behaviour while keeping clean accuracy [bagdasaryan2020, sun2019, wang2020tail].",
    ["How does ASR grow with attacker fraction/boost?", "What is the ASR of the trigger on a clean model?"], ["ASR rises with intensity; secure aggregation does not reduce it."],
    "ASR = P(f(x⊕t)=y_target | y≠y_target); poisoned local objective L_clean + L_poison; boosting Δ' = γΔ.", live_bd,
    ["task_3_1_runs.csv", "task_3_1_grid.csv", "task_3_1_trigger_selection.csv"], "task_3_1_summary.csv", ["task_3_1_backdoor.png"],
    "Backdoor ASR is high for all attack levels, while clean accuracy degrades with intensity (printed). Natural trigger ASR on a clean model is reported separately.",
    "d = pd.read_csv(RES / 'task_3_1_summary.csv'); print(d[['level','asr','clean_acc','kpi_asr_lt_5pct']].round(3).to_string(index=False))", ["Single trigger family; tabular features not images."], "Backdoor KPI (<5%) is NOT met without a defence.", ["bagdasaryan2020", "sun2019", "wang2020tail"])
att(2, "Byzantine attacks (random, sign-flip, scale, drift)", "Quantify accuracy damage of Byzantine updates and the effect of robust aggregators.", "Median, trimmed mean and Krum offer Byzantine robustness under assumptions on the malicious fraction [blanchard2017, yin2018, fang2020].",
    ["Which aggregator preserves accuracy under which behaviour and fraction?", "What is the clean-run cost of robustness?"], ["Mean aggregation degrades under sign-flip; robust rules do not."],
    "Krum score s_i = Σ_{j∈N_i} ||Δ_i−Δ_j||², n ≥ 2f+3; median per coordinate; trimmed mean removes β fraction each side.", live_def,
    ["task_3_2_runs.csv", "task_3_2_clean_cost.csv"], "task_3_2_summary.csv", ["task_3_2_byzantine.png"],
    "Plain mean is damaged by sign-flip (largest at 30%); robust rules hold. Note: 'scale' under mean can *raise* accuracy (a scaled benign update acts like a larger step) – reported as measured, not as robustness.",
    "d = pd.read_csv(RES / 'task_3_2_summary.csv'); print(d.pivot_table(index=['mal_frac','behavior'], columns='aggregator', values='acc').round(3).to_string())", ["Krum requires n ≥ 2f+3."], "Robust aggregation preserves accuracy in the tested settings; secure aggregation alone does not provide this.", ["blanchard2017", "yin2018", "fang2020"])
att(3, "Membership inference and privacy trade-off", "Evaluate shadow-model and loss-baseline membership inference with and without clipping/DP noise.", "MIA exploits generalisation gap [shokri2017, nasr2019, carlini2022]; DP-SGD/clipping reduce leakage at utility cost [abadi2016, mironov2017].",
    ["Does the attack exceed AUC 0.6 on an overfit target?", "What do clipping and noise change (AUC, TPR@1%FPR, utility, ε)?"], ["DP noise lowers AUC and utility monotonically."],
    "AUC of score s(x); advantage = max_τ(TPR−FPR); ε from RDP accountant (Opacus).", live_mia,
    ["task_3_3_runs.csv"], "task_3_3_summary.csv", ["task_3_3_membership_inference.png"], "AUC stays below 0.6 in this synthetic setting even undefended; reported with caveats about the regime.",
    "d = pd.read_csv(RES / 'task_3_3_summary.csv'); print(d[['config','eps','auc','utility_acc']].round(3).to_string(index=False))", ["Overfit regime on synthetic data; ε values are large and give little formal guarantee."], "KPI AUC<0.60 met in this regime; not a general claim.", ["shokri2017", "nasr2019", "carlini2022", "abadi2016", "mironov2017"])
att(4, "Gradient leakage (DLG/iDLG, analytic inversion)", "Reconstruct inputs from shared gradients and test mitigation via aggregation and noise.", "Gradients leak inputs [zhu2019, zhao2020, geiping2020]; for batch size 1 and a first dense layer, inversion is analytic.",
    ["When does reconstruction succeed (batch size, hidden width, noise)?", "Does aggregation over clients protect?"], ["Success is high for B=1,2 plaintext and falls with batch size/aggregation/noise."],
    "min_x ||∇_W L(f(x),y) − g||²; analytic: x = (∂L/∂W₁ row)/(∂L/∂b₁ entry). Success = relative L2 error ≤ 0.1 after Hungarian matching.", live_gl,
    ["task_3_4_runs.csv", "task_3_4_analytic.csv"], "task_3_4_runs.csv", ["task_3_4_gradient_leakage.png"], "Plaintext small-batch updates are reconstructed; aggregated/noised updates are not (printed).",
    "J = json.load(open(RES / 'task_3_4_gradient_leakage.json')); print([k for k in J if k != 'provenance'])", ["Sigmoid MLP, small models; deep CNN attacks not evaluated."], "Gradient-leakage KPI is NOT met for plaintext small-batch updates.", ["zhu2019", "zhao2020", "geiping2020"])
att(5, "Countermeasure evaluation (before/after)", "Compare defences on each attack before and after, including secure aggregation (which gives no poisoning robustness).", "See tasks 3.1–3.4 and [kairouz2021].",
    ["Which defences reduce ASR/accuracy damage and at what clean-accuracy cost?"], ["Secure aggregation and HE leave backdoor ASR unchanged; robust rules/norm bounds reduce it."],
    "Defence effect = metric_after − metric_before with CIs.", live_def, ["task_3_5_runs.csv"], "task_3_5_summary.csv", ["task_3_5_countermeasures.png"],
    "Printed table; SecAgg/HE rows equal 'none' for backdoor ASR as expected.", "d = pd.read_csv(RES / 'task_3_5_summary.csv'); print(d[d.metric=='asr'][['attack','defense','value','clean_acc']].round(3).to_string(index=False))", ["Adaptive attacks against defences not evaluated."], "Defences must be chosen per threat; confidentiality tools do not give integrity.", ["blanchard2017", "yin2018", "bonawitz2017", "kairouz2021"])
att(6, "Risk matrix and KPI verification", "Summarise attack/defence outcomes as a risk matrix and verify the story KPIs honestly.", "Risk = likelihood × impact is qualitative; here scores derive from measured outcomes.", ["Which KPIs are met?"], ["Not all KPIs are met."],
    "KPI check: measured vs threshold.", "print('loads stored KPI table')", ["task_3_6_risk_matrix.csv"], "task_3_6_kpi.csv", [], "Backdoor and gradient-leakage KPIs are NOT met undefended; MIA AUC KPI met in this regime.",
    "show_csv('task_3_6_kpi.csv')", ["Qualitative risk scoring."], "Honest verdicts are in the KPI table.", ["kairouz2021"])

# ------------------------------------------------------------------ FL-BENCH-004
B = "fl_bench_004"
def live_alg(a):
    return f'''from federated_learning.training.config import FLConfig
from federated_learning.servers.orchestrator import run_fl
parts = partition_dirichlet(data.y_train, 8, 0.3, SEED)
r = run_fl(FLConfig(algo="{a}", rounds=6, local_epochs=2, mu=0.05, seed=SEED), data, parts)
print("{a} final", {{k: round(v, 4) for k, v in r.final.items() if isinstance(v, float)}}, "comm MB", round(r.totals["comm_mb"], 3))
assert np.isfinite(r.w).all()'''
def ben(i, title, algo, obj, bg, math, runs, summ, figs, interp, find, limits=None):
    add(id=f"FL-BENCH-004.{i}", story=B, title=title, objective=obj, background=bg,
        rqs=["Does the implementation satisfy its validation identities?", "How does it behave under IID vs non-IID data?"], hyps=["Validation identities hold to numerical precision."],
        math=math, config_code=CONFIG_FL, modules="`fl.py`, `stats.py`", live_code=live_alg(algo or "fedavg"), runs=runs,
        metrics="accuracy, F1, rounds-to-target (acc≥0.85), logical communication MB, local drift ||w_k − w_t||.", metrics_code=f"show_csv({summ!r})",
        stats_text="Seeds are the unit of replication; paired tests across seeds (Wilcoxon signed-rank, paired t), Holm correction, Cohen's d_z; see task 4.6.", stats_code=f"show_csv({summ!r}, 30)", figs=figs, interp=interp, findings_code=find,
        limits=(limits or []) + ["Synthetic tabular data; small MLP."], conclusions="Implementation validated by identity tests (tests/ and task JSON).", refs=["mcmahan2017", "li2020fedprox", "karimireddy2020", "wang2020fednova", "li2021niid"])
ben(1, "FedAvg baseline", "fedavg", "Implement and validate FedAvg; sweep rounds/epochs/participation.", "FedAvg averages local models weighted by data size [mcmahan2017].", "w_{t+1} = Σ_k p_k w_k^{t+1}, p_k = n_k/Σ n_j; single client ⇒ local SGD.", ["task_4_1_runs.csv", "task_4_1_sweeps.csv"], "task_4_1_summary.csv", ["task_4_1_fedavg.png"], "See printed summary.", "J = json.load(open(RES / 'task_4_1_fedavg.json')); print([k for k in J if k != 'provenance'])")
ben(2, "FedProx", "fedprox", "Implement FedProx and measure drift reduction vs μ.", "A proximal term limits client drift under heterogeneity [li2020fedprox].", "min_w F_k(w) + (μ/2)||w − w_t||²; gradient + μ(w − w_t); μ=0 ⇒ FedAvg.", ["task_4_2_runs.csv", "task_4_2_local_drift.csv"], "task_4_2_summary.csv", ["task_4_2_fedprox.png"], "Drift decreases monotonically with μ (validated); accuracy benefit depends on setting.", "show_csv('task_4_2_local_drift.csv')")
ben(3, "SCAFFOLD", "scaffold", "Implement SCAFFOLD (option II control variates) and evaluate under partial participation.", "Control variates correct client drift [karimireddy2020]; communication doubles.", "y ← y − η(g − c_i + c); c_i⁺ = c_i − c + (x − y)/(Kη); invariant c = mean(c_i) under full participation.", ["task_4_3_runs.csv", "task_4_3_partial_participation.csv"], "task_4_3_summary.csv", ["task_4_3_scaffold.png"], "Communication is exactly 2× FedAvg; control-variate invariant verified.", "show_csv('task_4_3_partial_participation.csv')")
ben(4, "FedNova", "fednova", "Implement FedNova and verify objective-inconsistency correction on a closed-form quadratic toy.", "Heterogeneous local steps bias FedAvg's fixed point [wang2020fednova].", "w ← w + τ_eff Σ p_k Δ_k/τ_k; equal τ ⇒ FedAvg.", ["task_4_4_runs.csv", "task_4_4_quadratic_toy.csv"], "task_4_4_summary.csv", ["task_4_4_fednova.png"], "Toy problem: FedNova reaches the true weighted optimum; FedAvg is biased.", "show_csv('task_4_4_quadratic_toy.csv')")
ben(5, "Four-algorithm comparison grid", None, "Compare FedAvg/FedProx/SCAFFOLD/FedNova (+ proposed FedAvg+CKKS) across IID, Dirichlet α∈{0.3,0.1}, 50 clients, E=5, partial participation, and sign-flip attackers.", "See tasks 4.1–4.4.", "—", ["task_4_5_runs.csv"], "task_4_5_summary.csv", ["task_4_5_convergence_iid.png", "task_4_5_convergence_dir0.3.png", "task_4_5_summary_bars.png"], "See printed summary.", "d = pd.read_csv(RES / 'task_4_5_summary.csv'); print(d.columns.tolist()); display(d.head(12).round(4))")
ben(6, "Statistical analysis", None, "Friedman + pairwise Wilcoxon/paired-t with Holm correction and effect sizes; bootstrap CIs.", "Non-parametric multi-algorithm comparison [demsar2006, holm1979, wilcoxon1945].", "Wilcoxon minimum attainable p = 2/2ⁿ ⇒ n≥6 needed for p<0.05; 10 seeds used in main conditions; Cohen d_z = mean(d)/sd(d).", ["task_4_6_pairwise_tests.csv"], "task_4_6_pairwise_tests.csv", ["task_4_6_confidence_intervals.png"], "Significant differences (after Holm) are listed; many pairs are not significant.", "d = pd.read_csv(RES / 'task_4_6_pairwise_tests.csv'); print(d[d['significant_holm_0.05']==True][['condition','metric','a','b','mean_diff','holm_p_wilcoxon']].round(4).to_string(index=False))")
ben(7, "Leaderboard", None, "Multi-criteria leaderboard (utility, convergence, communication, robustness, cost).", "Mean-rank aggregation of per-criterion ranks [demsar2006].", "mean rank = average over criteria of rank (1 = best).", ["task_4_7_leaderboard_iid.csv", "task_4_7_leaderboard_dir0.3.csv"], "task_4_7_leaderboard_dir0.3.csv", [], "Rankings differ between IID and non-IID; the CKKS row is not ranked (robustness NOT_MEASURED).", "show_csv('task_4_7_leaderboard_iid.csv'); show_csv('task_4_7_leaderboard_dir0.3.csv')")

# ------------------------------------------------------------------ extension tasks (closing open items)
live_adapt = '''from federated_learning.attacks.adaptive import alie_z, alie_vector, minmax_vector, minmax_gamma, _max_pairwise
r = np.random.default_rng(SEED); B = r.normal(1.0, 0.3, (16, 50)).astype(np.float32)
print("ALIE z (n=20, f=4):", round(alie_z(20, 4), 4))
mu = B.mean(0); p = -mu / np.linalg.norm(mu); g = minmax_gamma(B, mu, p); thr = _max_pairwise(B)
print("min-max gamma:", round(g, 3), "| max distance to benign:", round(float(np.linalg.norm(mu + g * p - B, axis=1).max()), 3), "<= benign max pairwise:", round(thr, 3))
assert np.linalg.norm(mu + g * p - B, axis=1).max() <= thr * (1 + 1e-9)'''
add(id="FL-ATTACK-003.7", story=A, title="Adaptive (defence-aware, omniscient) attacks", objective="Test whether the defences that worked against static attacks (task 3.2/3.5) survive attackers that see the benign updates and adapt: ALIE, MIN-MAX, norm-matched backdoor, MIN-MAX backdoor.",
    background="ALIE shifts malicious updates by a small multiple of the benign standard deviation [baruch2019]; MIN-MAX/AGR-tailored attacks keep malicious updates inside the benign distance envelope [shejwalkar2021]; backdoors can be rescaled to pass norm bounds [bagdasaryan2020, sun2019, wang2020tail].",
    rqs=["Do median, trimmed mean, Krum/Multi-Krum, median-norm clipping and the anomaly filter keep accuracy under ALIE/MIN-MAX?", "Do norm- or distance-constrained backdoors defeat the defences that stopped the static backdoor?"],
    hyps=["At least one defence that performed well against static attacks degrades under an adaptive attack.", "A norm-matched backdoor passes norm-based defences."],
    math="ALIE: v = mu - z*sigma, z = Phi^{-1}((n-s)/n), s = floor(n/2+1) - f. MIN-MAX: v = mu + gamma p with gamma = max{g: max_i ||mu + g p - b_i|| <= max_{i,j} ||b_i - b_j||}, p = -mu/||mu||. Norm-matched backdoor: d' = d * median_i||b_i|| / ||d||. MIN-MAX backdoor: p = d_m - mu.",
    config_code=CONFIG_FL, modules="`attacks_adaptive.py`, `defenses.py`, `fl.py`, `experiments/ext.py`", live_code=live_adapt,
    runs=["task_3_7_untargeted_runs.csv", "task_3_7_backdoor_runs.csv"],
    metrics="test accuracy/F1/AUC (untargeted) and backdoor ASR + clean accuracy; mean and t-CI over 5 seeds; threat model is OMNISCIENT (upper bound on attacker knowledge).",
    metrics_code="show_csv('task_3_7_untargeted_summary.csv', 30)", stats_text="t-intervals over seeds; no multiple-comparison testing is claimed for this task (descriptive).", stats_code="show_csv('task_3_7_backdoor_summary.csv', 30)",
    figs=["task_3_7_adaptive.png"], interp="Printed tables show, per defence, accuracy/ASR with no attack, with the adaptive attack and (backdoor) with the static attack; the interpretation in results_and_discussion.md is limited to those numbers.",
    findings_code="d = pd.read_csv(RES / 'task_3_7_untargeted_summary.csv'); print(d[d.dataset=='syn'].pivot_table(index='defense', columns='attack', values='acc').round(3).to_string()); b = pd.read_csv(RES / 'task_3_7_backdoor_summary.csv'); print(b.pivot_table(index='defense', columns='attack', values='asr').round(3).to_string())",
    limits=["Omniscient attacker = upper bound; attacks are not optimised per defence beyond their constraints.", "Real-data settings are small (K=10, f=2) and untargeted only; the backdoor study is synthetic only."],
    conclusions="Static-attack robustness does not imply adaptive-attack robustness; see results_and_discussion.md for which defences held.", refs=["baruch2019", "shejwalkar2021", "bagdasaryan2020", "sun2019", "blanchard2017", "yin2018"])
live_sites = '''from federated_learning.preprocessing.multisite_heart import load_sites, load_loso, SITE_NAMES
try:
    sd = load_sites(SEED); print("train sizes per site:", sd.meta["site_train_sizes"], "| positive rate per site:", [round(x, 2) for x in sd.meta["site_positive_rate"]])
    lo = load_loso("switzerland"); print("LOSO Switzerland: train sites", lo.site_names, "| test rows", len(lo.y_test), "| test positive rate", round(float(lo.y_test.mean()), 2))
    assert len(sd.parts) == 4 and lo.X_test.shape[0] == 123
except Exception as e:  # needs network (GitHub) or data/raw/ on first use; stored results below do not
    print("live check skipped (data files not reachable):", type(e).__name__, str(e)[:120])'''
add(id="FL-BENCH-004.10", story=B, title="Natural multi-site benchmark: UCI heart disease (4 collection sites as clients)", objective="Replace synthetic client partitions by NATURAL site heterogeneity (Cleveland, Hungary, Switzerland, VA Long Beach), with within-site and leave-one-site-out protocols, nested tuning, pooled/local baselines and a secure-aggregation equivalence check.",
    background="Real federations differ by site in prevalence, measurement practice and missingness [kairouz2021, rieke2020]; benchmark rankings depend on how heterogeneity is created [hsu2019, li2021niid]. Data: UCI Heart Disease [uci_heart], four site files fetched from a commit-pinned mirror and hash-verified.",
    rqs=["Do the algorithm rankings from synthetic partitions hold when the clients are natural cohorts?", "How well does a model trained on three sites transfer to the fourth (external validation)?", "Does federated training beat local-only training and approach pooled training?"],
    hyps=["Differences between FL algorithms are within noise at n=920 with four clients.", "Transfer to Switzerland (positive rate 0.93, cholesterol missing) is clearly worse than to Cleveland in balanced accuracy."],
    math="Protocol A: per site 70/30 stratified; 4 clients = 4 sites; nested tuning on 25% validation per split; 10 splits. Protocol B: train on 3 sites, test on all rows of the 4th; tuned on 25% validation of the training sites; 10 initialisation seeds (data fixed). Balanced accuracy = (recall + specificity)/2.",
    config_code=CONFIG_NONE, modules="`real_sites.py`, `experiments/ext2.py`", live_code=live_sites,
    runs=["task_4_10_A_runs.csv", "task_4_10_A_site_metrics.csv", "task_4_10_A_tuned_params.csv", "task_4_10_A_baselines.csv", "task_4_10_B_runs.csv", "task_4_10_B_tuned_params.csv", "task_4_10_B_baselines.csv", "task_4_10_secagg_equivalence.csv"],
    metrics="test acc/F1/AUC/balanced accuracy (pooled and per site for A; held-out site for B); pooled and local-only baselines.", metrics_code="show_csv('task_4_10_A_summary.csv', 12); show_csv('task_4_10_A_baselines_summary.csv'); show_csv('task_4_10_B_macro.csv', 12)",
    stats_text="Protocol A: paired tuned-vs-default tests by split, Friedman + Holm across algorithms. Protocol B: seeds replicate only initialisation (data fixed) and only four sites exist, so results are descriptive; a Friedman test over the four site means is shown with that caveat.", stats_code="show_csv('task_4_10_A_tuned_vs_default_tests.csv'); show_csv('task_4_10_A_pairwise.csv'); show_csv('task_4_10_B_summary.csv', 40)",
    figs=["task_4_10_natural_sites.png"], interp="Switzerland has 8 negatives, so F1/accuracy there are inflated; AUC and balanced accuracy are the primary metrics for site-level statements.",
    findings_code="show_csv('task_4_10_A_site_summary.csv', 20); show_csv('task_4_10_secagg_equivalence.csv')",
    limits=["Four 1980s research cohorts, n=920; not a live hospital federation.", "Pooled imputation/standardisation is a centralised preprocessing step.", "Mirror licence not verified; files are not redistributed.", "Protocol B p-values (if any) reflect training noise only."],
    conclusions="See results_and_discussion.md.", refs=["uci_heart", "kairouz2021", "hsu2019", "li2021niid", "mcmahan2017"])
live_rs = '''from federated_learning.pipelines.experiments.ext2 import sample_configs, _algo_cfg
c = sample_configs(24, 7); assert c == sample_configs(24, 7)
print(len(c), "configs; lr range", round(min(x["lr"] for x in c), 4), "-", round(max(x["lr"] for x in c), 3), "| epochs used", sorted({x["local_epochs"] for x in c}))
print("fedavg config ignores mu/server_lr:", _algo_cfg(c[0], "fedavg"))
assert all(0.005 <= x["lr"] <= 0.8 and 1e-3 <= x["mu"] <= 1 for x in c)'''
add(id="FL-BENCH-004.11", story=B, title="Broader tuning: six-parameter random search vs lr/mu grid", objective="Test whether tuning more than learning rate and FedProx mu (batch size, local epochs, hidden width, SCAFFOLD server step) changes conclusions, on synthetic data and on the four natural sites.",
    background="Hyper-parameter sensitivity can reverse algorithm rankings in FL benchmarks [li2021niid, hsu2019]; random search is a standard budget-matched alternative to grids.",
    rqs=["Does a 24-configuration random search over six parameters beat the lr/mu grid on test F1?", "Does it change which algorithm is best?"], hyps=["Gains over the grid are small relative to seed variability."],
    math="Same 24 draws for every algorithm; selection by mean validation F1 (tie: lower validation loss) on 25% of TRAIN; three settings compared on the untouched test set: default, grid (lr[,mu]), random (6 parameters). Paired Wilcoxon / t by seed (synthetic, 10) or split (heart, 5).",
    config_code=CONFIG_NONE, modules="`experiments/ext2.py`", live_code=live_rs,
    runs=["task_4_11_runs.csv", "task_4_11_selected_configs.csv"], metrics="test acc/F1/AUC for default, grid and random settings.", metrics_code="show_csv('task_4_11_summary.csv', 30)",
    stats_text="Paired tests for random-grid, random-default, grid-default; uncorrected p-values shown; the heart setting has only 5 splits (Wilcoxon cannot reach p<0.05 below 6 pairs).", stats_code="show_csv('task_4_11_tests.csv', 30)",
    figs=["task_4_11_random_search.png"], interp="Budgets are matched in number of configurations, not in compute (local epochs and hidden width change cost).",
    findings_code="show_csv('task_4_11_selected_configs.csv', 40)",
    limits=["24 configurations is a small search budget.", "Selection on small validation sets can overfit (heart: about 160 validation rows).", "Compute is not matched across configurations."],
    conclusions="See results_and_discussion.md.", refs=["li2021niid", "hsu2019", "li2020fedprox", "karimireddy2020", "wang2020fednova"])
live_agr = '''from federated_learning.attacks.adaptive import agr_tailored_choose, minsum_vector, minmax_vector, alie_vector
from federated_learning.aggregators.mean import WeightedMean
from federated_learning.aggregators.robust import CoordinateMedian, Krum
rng = np.random.default_rng(SEED); n, f = 16, 4
D = rng.normal(1.0, 0.3, (n, 50)).astype(np.float32); B = D[f:]; w = np.ones(n); mal = list(range(f)); u = B.mean(0) / np.linalg.norm(B.mean(0))
for inner in (WeightedMean(), CoordinateMedian(), Krum(3, 1)):
    M, info = agr_tailored_choose(inner, D, w, mal, B)
    dev = lambda v: float(inner.aggregate(np.vstack([np.tile(v, (f, 1)) if v.ndim == 1 else v, B]), w) @ u)
    fixed = {"alie": alie_vector(B, n, f), "minmax": minmax_vector(B), "minsum": minsum_vector(B)}
    print(inner.name, "chosen:", info["choice"], "| deviation", round(dev(M), 3), "vs fixed:", {k: round(dev(v), 3) for k, v in fixed.items()})
    assert dev(M) <= min(dev(v) for v in fixed.values()) + 1e-6'''
add(id="FL-ATTACK-003.8", story=A, title="Stronger adaptive attacks: rule-aware (AGR-tailored), MIN-SUM, directed deviation, AGR-backdoor", objective="Test the same seven aggregation rules against attackers that additionally know the server rule and simulate it, and against MIN-SUM and Fang-style directed deviation.",
    background="Defence-aware attackers can defeat rank- and distance-based rules [shejwalkar2021, fang2020, baruch2019]; the AGR-tailored attacker here searches a fixed candidate set under the simulated rule.",
    rqs=["Is there any tested rule whose accuracy stays near the no-attack level under the strongest attack?", "Does AGR-tailored beat the best fixed attack (paired by seed)?", "Do AGR-aware backdoors raise ASR above the distance-constrained backdoor of task 3.7?"],
    hyps=["Every rule has at least one attack that lowers its accuracy by more than 1 pp.", "AGR-tailored is at least as damaging as the best fixed attack on most rules."],
    math="AGR-tailored: v* = argmin_c <agg(D with malicious rows = c), mu/||mu||> over candidates c = mu + g*||mu||*d (3 directions x 10 gammas) and ALIE/MIN-MAX/MIN-SUM/FANG. MIN-SUM: gamma = max{g: sum_i ||mu + g p - b_i||^2 <= max_i sum_j ||b_i - b_j||^2}.",
    config_code=CONFIG_FL, modules="`attacks_adaptive.py`, `experiments/ext2.py`", live_code=live_agr,
    runs=["task_3_8_untargeted_runs.csv", "task_3_8_backdoor_runs.csv"], metrics="test accuracy/F1/AUC and backdoor ASR/clean accuracy; mean and t-CI over 5 seeds; K=20 (f=2,4,8) synthetic and K=10 (f=2) WDBC.", metrics_code="show_csv('task_3_8_worst_case.csv', 40)",
    stats_text="Paired test (by seed) of AGR-tailored against the strongest fixed attack per rule; uncorrected p-values; 5 seeds limit power (Wilcoxon cannot reach p<0.05 below 6 pairs; the t-test is shown).", stats_code="show_csv('task_3_8_agr_vs_best_fixed.csv', 40)",
    figs=["task_3_8_stronger_adaptive.png"], interp="The attacker is an upper bound (omniscient, rule-aware, unconstrained gamma in the tailored candidates); a better optimiser could be stronger still.",
    findings_code="d = pd.read_csv(RES / 'task_3_8_untargeted_summary.csv'); print(d[d.dataset=='syn'].pivot_table(index='defense', columns='attack', values='acc').round(3).to_string()); b = pd.read_csv(RES / 'task_3_8_backdoor_summary.csv'); print(b.pivot_table(index='defense', columns='attack', values='asr').round(3).to_string())",
    limits=["Fixed candidate set; no gradient-based or loss-aware attacker.", "Synthetic and one real table only.", "Full participation required (row index = client id)."],
    conclusions="See results_and_discussion.md.", refs=["shejwalkar2021", "fang2020", "baruch2019", "blanchard2017", "yin2018"])
live_tune = '''from federated_learning.preprocessing.splits import split_validation
from federated_learning.pipelines.experiments.ext import tune
val = split_validation(data, 0.25, SEED)
best, tab = tune(val, "fedavg", 10, "0.3", 3, 1, [100])
print("best (3-round smoke test, NOT the reported tuning):", best, "| configs evaluated:", len(tab)); assert len(tab) == 7'''
add(id="FL-BENCH-004.8", story=B, title="Validation-based tuning of the four FL baselines", objective="Remove the 'untuned baseline' objection: select lr (all) and mu (FedProx) by validation F1 on a split of the TRAIN set, then compare tuned vs default on the untouched test set.",
    background="Baseline comparisons are only meaningful if every method gets comparable tuning effort [li2020fedprox, karimireddy2020, wang2020fednova, li2021niid]. Validation data is carved from train; test data is never used for selection.",
    rqs=["Does tuning change the ranking of FedAvg/FedProx/SCAFFOLD/FedNova?", "Is the selected hyper-parameter interior to the grid or at its boundary?"], hyps=["Tuned learning rates exceed the default 0.05 because the round budget is fixed.", "Rankings shrink toward ties after tuning."],
    math="theta* = argmax_theta mean_s F1_val(theta, seed s); grid: lr in {0.01,0.03,0.05,0.1,0.2,0.4,0.8}, mu in {0.001,0.01,0.05,0.1,0.5,1} (FedProx). Paired effect = F1_tuned - F1_default per seed.",
    config_code=CONFIG_FL, modules="`experiments/ext.py`, `real_data.split_validation`, `stats.py`", live_code=live_tune,
    runs=["task_4_8_tuning_grid.csv", "task_4_8_runs.csv"], metrics="validation F1 (selection); test acc/F1/AUC over 10 seeds (evaluation).", metrics_code="show_csv('task_4_8_summary.csv')",
    stats_text="Paired Wilcoxon / t (tuned vs default, by seed), Friedman + Holm-corrected pairwise tests across the tuned algorithms.", stats_code="show_csv('task_4_8_tuned_vs_default_tests.csv'); show_csv('task_4_8_tuned_pairwise.csv')",
    figs=["task_4_8_tuning.png"], interp="Boundary-optimum rows are flagged in the printed table (selected lr equals the largest grid value).",
    findings_code="g = pd.read_csv(RES / 'task_4_8_tuning_grid.csv'); best = g.loc[g.groupby('algo')['val_f1'].idxmax()]; best['at_lr_grid_edge'] = best['lr'] == g['lr'].max(); display(best.round(4))",
    limits=["Grid truncated at lr=0.8: where the optimum is at the edge, true optimum may be larger.", "Fixed round budget favours aggressive learning rates.", "Only lr/mu tuned (batch size, epochs, hidden width fixed)."],
    conclusions="See results_and_discussion.md.", refs=["li2020fedprox", "karimireddy2020", "wang2020fednova", "li2021niid", "demsar2006"])
live_real = '''from federated_learning.preprocessing.real_datasets import load_real
from federated_learning.training.config import FLConfig
from federated_learning.servers.orchestrator import run_fl
d = load_real("wdbc", SEED); print(d.meta)
parts = partition_dirichlet(d.y_train, 5, 0.5, SEED, min_size=3)
r = run_fl(FLConfig(rounds=10, local_epochs=2, lr=0.1, seed=SEED), d, parts)
print({k: round(r.final[k], 4) for k in ("acc", "f1", "auc")}); assert r.final["acc"] > 0.8'''
add(id="FL-BENCH-004.9", story=B, title="Benchmark on real public de-identified medical tables", objective="Rerun the four-algorithm benchmark, the secure-aggregation equivalence check and a Byzantine check on real public tabular medical data (WDBC, Pima diabetes, Cleveland heart disease) with nested tuning.",
    background="Small public clinical tables allow a first check that conclusions from synthetic data are not artefacts of the generator. They are NOT hospital federations: clients are synthetic partitions of a single table. Sources: UCI repository pages [uci_wdbc, uci_heart, pima_nidd, smith1988].",
    rqs=["Do the algorithm rankings from synthetic data hold on real tables?", "Do CKKS and secure aggregation reproduce plaintext accuracy on real data?", "Does coordinate median still protect against a sign-flip client?"], hyps=["Differences between algorithms are within noise on these tiny datasets.", "Secure aggregation is exact up to quantisation on real data as on synthetic data."],
    math="Nested protocol per outer split s: tune on 25% validation of TRAIN_s; evaluate on TEST_s; 10 splits per dataset; Wilcoxon on per-split F1; Friedman across algorithms.",
    config_code=CONFIG_NONE, modules="`real_data.py`, `experiments/ext.py`", live_code=live_real,
    runs=["task_4_9_real_runs.csv", "task_4_9_real_tuned_params.csv", "task_4_9_real_centralized.csv"], metrics="test acc/F1/AUC; pooled-data MLP as reference.", metrics_code="show_csv('task_4_9_real_summary.csv', 40)",
    stats_text="Paired tests tuned vs default, Friedman + Holm across tuned algorithms, per dataset.", stats_code="show_csv('task_4_9_real_tuned_vs_default.csv', 20); show_csv('task_4_9_real_ranking.csv', 30)",
    figs=["task_4_9_real_data.png"], interp="Test sets have 91-231 samples, so intervals are wide; only differences that survive Holm correction are called significant in results_and_discussion.md.",
    findings_code="show_csv('task_4_9_real_secagg_equivalence.csv', 12); b = pd.read_csv(RES / 'task_4_9_real_byzantine.csv'); print(b.pivot_table(index='dataset', columns='aggregator', values='f1').round(3).to_string())",
    limits=["Tiny public datasets; partitions into virtual clients are synthetic.", "Third-party GitHub mirrors for Pima/Heart (hash-pinned; licence of mirror not verified).", "No hospital heterogeneity, no real federation."],
    conclusions="See results_and_discussion.md.", refs=["uci_wdbc", "uci_heart", "pima_nidd", "smith1988", "karimireddy2020", "li2020fedprox", "wang2020fednova"])

# ------------------------------------------------------------------ FL-PUB-005
U = "fl_pub_005"
add(id="FL-PUB-005.1", story=U, title="Reproducibility audit", objective="Verify determinism, re-execution, provenance consistency and artifact integrity of all stored results.",
    background="Reproducibility practices in ML [pineau2020, henderson2018].", rqs=["Are results bit-reproducible with a fixed seed?", "Do stored summaries equal recomputation from raw runs?"], hyps=["All determinism and consistency checks pass; the git-commit check fails (no repository)."],
    math="Equality checks: max|w_a − w_b| = 0; SHA-256 of config JSON recomputed.", config_code=CONFIG_NONE, modules="`experiments/common.py`, `fl.py`",
    live_code='''from federated_learning.training.config import FLConfig
from federated_learning.servers.orchestrator import run_fl
parts = partition_iid(data.y_train, 5, SEED)
a = run_fl(FLConfig(rounds=2, seed=7), data, parts); b = run_fl(FLConfig(rounds=2, seed=7), data, parts); c = run_fl(FLConfig(rounds=2, seed=8), data, parts)
print("same seed identical:", bool(np.array_equal(a.w, b.w)), "| different seed differs:", bool(not np.array_equal(a.w, c.w))); assert np.array_equal(a.w, b.w)''',
    runs=["task_5_1_artifact_hashes.csv"], metrics="pass/fail per check with detail.", metrics_code="J = json.load(open(RES / 'task_5_1_reproducibility.json')); display(pd.DataFrame(J['checks']))",
    stats_text="Not applicable (deterministic checks).", stats_code="print('n/a')", figs=[], interp="Passed checks and the failed git-commit check are printed.",
    findings_code="J = json.load(open(RES / 'task_5_1_reproducibility.json')); ch = J['checks']; print(sum(c['passed'] for c in ch), 'of', len(ch), 'checks passed'); print([c['check'] for c in ch if not c['passed']])",
    limits=["Bit-reproducibility holds on this software stack/CPU only; other BLAS/torch builds may differ.", "No git commit id."], conclusions="Results are reproducible on this stack; failures are listed honestly.", refs=["pineau2020", "henderson2018"])
add(id="FL-PUB-005.2", story=U, title="Ablation study", objective="Ablate confidentiality (HE/SecAgg/SMPC), DP and robust aggregation and their feasible combinations; document incompatible combinations.",
    background="Defences address different threats and can conflict (e.g. hidden updates vs server-side robust statistics) [kairouz2021].", rqs=["What does each component cost and protect against?", "Which combinations are incompatible?"], hyps=["Confidentiality components do not change robustness; robust/clipping components do."],
    math="Effects relative to plain FL with CIs over seeds.", config_code=CONFIG_FL, modules="`fl.py`, `defenses.py`, `crypto/*`", live_code=live_def, runs=["task_5_2_runs.csv", "task_5_2_incompatible.csv"], metrics="accuracy, overhead, accuracy under sign-flip, backdoor ASR, gradient-leakage success (from task 3.4), MIA AUC (from 3.3).",
    metrics_code="show_csv('task_5_2_summary.csv')", stats_text="CIs over seeds.", stats_code="show_csv('task_5_2_incompatible.csv')", figs=["task_5_2_ablation.png"], interp="See printed table; the NOT_RUN combinations are explained in the incompatible-combinations table.",
    findings_code="d = pd.read_csv(RES / 'task_5_2_summary.csv'); print(d[['config','acc','train_overhead_pct','acc_under_signflip','backdoor_asr']].round(3).to_string(index=False))",
    limits=["Gradient-leakage and MIA columns are reused from tasks 3.3/3.4, not re-measured per row."], conclusions="Confidentiality and integrity need different components; no single layer covers all threats.", refs=["kairouz2021", "abadi2016", "bonawitz2017"])

def docnb(i, title, obj, bg, math, doc, extra):
    add(id=f"FL-PUB-005.{i}", story=U, title=title, objective=obj, background=bg, rqs=["Does the document state its assumptions, limits and sources honestly and consistently with the stored results?"],
        hyps=["Every quantitative statement is traceable to a stored result file; no compliance claim appears."], math=math, config_code=CONFIG_NONE, modules="`scripts/generate_publication.py`",
        live_code=f"""import re
P = ROOT / {doc!r}
txt = P.read_text() if P.exists() else ''
print(P.name, 'exists:', P.exists(), '| characters:', len(txt))
banned = [w for w in ['is compliant', 'GDPR-compliant', 'guarantees compliance', 'certified'] if w in txt]
print('compliance-claim phrases found:', banned); assert not banned""",
        runs=["task_5_2_incompatible.csv"], metrics="document checks: existence, absence of compliance claims, presence of limitations section.", metrics_code="show_csv('task_5_2_summary.csv')",
        stats_text="Not applicable.", stats_code="print('n/a')", figs=[], interp=extra, findings_code=f"""t = (ROOT / {doc!r}).read_text() if (ROOT / {doc!r}).exists() else ''
print({{'mentions_limitations': bool(re.search('(?i)limitation|threats to validity', t)), 'mentions_virtual_clients': 'virtual' in t.lower(), 'mentions_not_verified': 'NOT_VERIFIED' in t}})""",
        limits=["Documents are drafts; legal sources unverified."], conclusions="Draft produced; see file.", refs=["gdpr", "aiact", "nis2", "kairouz2021"])
docnb(3, "Healthcare threat model and security architecture", "Document adversaries, assets, STRIDE mapping and security architecture for healthcare FL.", "STRIDE-style threat modelling applied to FL; regulatory pointers (unverified).", "—", "docs/research/healthcare_threat_model.md", "Document check only.")
docnb(4, "Scientific report", "Produce a scientific report from stored results.", "—", "—", "reports/manuscript_support/scientific_report.md", "Document check only.")
docnb(5, "IEEE/ACM manuscript draft", "Produce an IEEE-style manuscript draft and verified-status bibliography.", "—", "—", "latex/manuscript_ieee.tex", "Document check only; IEEEtran.cls unavailable in the sandbox, compile-checked with the article class.")

def main():
    NB.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in T:
        d = NB / t["story"]; d.mkdir(parents=True, exist_ok=True)
        nbf.write(build(t), d / (t["id"].replace("-", "_").replace(".", "_").lower() + ".ipynb")); n += 1
    print(n, "notebooks written")
if __name__ == "__main__":
    main()
