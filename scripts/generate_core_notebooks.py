"""Generates the 14 numbered notebooks in notebooks/. Each is self-contained, runs at demonstration scale in about a minute,
writes publication-quality figures (300 dpi PNG + vector PDF) to reports/figures/ and a JSON summary to reports/notebooks/.
Run: python scripts/generate_core_notebooks.py && python scripts/execute_core_notebooks.py"""
from pathlib import Path

import nbformat as nbf

R = Path(__file__).resolve().parents[1]
NB = R / "notebooks"

SETUP = '''import os, sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
def _root():
    p = Path(os.environ.get("FL_ROOT") or Path.cwd()).resolve()
    for q in [p, *p.parents]:
        if (q / "pyproject.toml").exists() and (q / "src" / "federated_learning").exists():
            return q
    raise RuntimeError("repository root not found; set FL_ROOT")
ROOT = _root(); sys.path.insert(0, str(ROOT / "src")); os.environ["FL_ROOT"] = str(ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display
SEED = 0; np.random.seed(SEED)
NB_ID = "__NB_ID__"
FIG = ROOT / "reports" / "figures"; OUT = ROOT / "reports" / "notebooks"; FIG.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]   # Okabe-Ito, colour-blind safe
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.prop_cycle": matplotlib.cycler(color=PAL), "axes.grid": True, "grid.alpha": .25, "legend.frameon": False})
SUMMARY = {}
def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{NB_ID}_{name}.{ext}", bbox_inches="tight")
    plt.show()
def save_summary():
    (OUT / f"{NB_ID}.json").write_text(json.dumps(SUMMARY, indent=2, default=float)); print("summary ->", OUT / f"{NB_ID}.json")'''

NOTICE = ("> **Data statement.** SYN-IDS-v1 is completely synthetic despite its name (scikit-learn `make_classification`); it is not real "
          "intrusion data and not patient data. The public healthcare datasets are de-identified research tables; no real hospital data is used, "
          "all federated clients are simulated locally in one process, and raw source files are not redistributed.")

NBS = []
def add(i, name, title, purpose, cells):
    NBS.append((i, name, title, purpose, cells))

add(1, "dataset_overview", "Dataset overview", "Inventory of every dataset used in the study: size, features, positive rate, source, licence and integrity checksum.", [
("## 1. Registry\nEvery dataset is declared once in `federated_learning.preprocessing.registry` with source, licence status, citation and SHA-256 of the files.",
'''from federated_learning.preprocessing import SPECS
from federated_learning.preprocessing.real_datasets import load_real
from federated_learning.synthetic_data import make_dataset
rows = []
syn = make_dataset(n_samples=12000, seed=SEED)
rows.append(dict(dataset="SYN-IDS-v1 (synthetic)", samples=12000, features=40, positive_rate=round(float(np.r_[syn.y_train, syn.y_test].mean()), 3), license="n/a (generated)", setup="simulated IID / Dirichlet clients"))
for a in ("wdbc", "pima", "heart"):
    d = load_real(a); s = [x for x in SPECS.values() if x.alias == a][0]
    rows.append(dict(dataset=s.title, samples=d.meta["n_samples"], features=d.meta["n_features"], positive_rate=round(float(np.r_[d.y_train, d.y_test].mean()), 3), license=s.license, setup="simulated IID / Dirichlet clients"))
s4 = SPECS["uci_heart_disease_multisite"]
rows.append(dict(dataset=s4.title, samples=s4.n_samples, features=s4.n_features, positive_rate=None, license=s4.license, setup="4 natural site clients"))
tab = pd.DataFrame(rows); display(tab); SUMMARY["datasets"] = rows'''),
("## 2. Integrity\nSHA-256 values pinned in the registry (files are downloaded on first use and refused if the hash differs).",
'''chk = pd.DataFrame([dict(dataset=k, file=f.filename, sha256=(f.sha256 or "")[:16] + "...") for k, s in SPECS.items() for f in s.files]); display(chk)'''),
("## 3. Figure: size and positive rate",
'''fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
t = tab.dropna(subset=["positive_rate"])
ax[0].barh(t.dataset.str.slice(0, 26), t.samples, color=PAL[0]); ax[0].set_xlabel("samples"); ax[0].set_title("Size")
ax[1].barh(t.dataset.str.slice(0, 26), t.positive_rate, color=PAL[1]); ax[1].set_xlabel("positive-class rate"); ax[1].set_title("Class balance"); ax[1].set_yticklabels([])
save_fig(fig, "overview"); save_summary()'''),
])

add(2, "syn_ids_generation", "SYN-IDS-v1 generation, validation and quality", "Reproducible generation of the fully synthetic benchmark, with validation, quality and drift reports.", [
("## 1. Generate\n40 features, binary, 60/40 classes, 3 clusters per class.",
'''from federated_learning.synthetic_data import SynIDSConfig, generate_raw
from federated_learning.synthetic_data.pipelines import generate_and_validate
cfg = SynIDSConfig(n_samples=12000, n_features=40, seed=SEED)
rep = generate_and_validate(cfg)
print("validation passed:", rep["validation"]["passed"]); display(pd.DataFrame(rep["validation"]["checks"]))
SUMMARY.update(quality=rep["quality"], arrays_sha256=rep["arrays_sha256"], train_test_psi_max=rep["train_test_psi_max"])
print(json.dumps(rep["quality"], indent=1))'''),
("## 2. Determinism\nThe same seed must give byte-identical arrays.",
'''from federated_learning.utils.hashing import sha256_arrays
X, y = generate_raw(cfg); X2, y2 = generate_raw(cfg)
assert sha256_arrays(X, y) == sha256_arrays(X2, y2) == rep["arrays_sha256"]; print("deterministic: OK")'''),
("## 3. Figure: class structure and feature distributions",
'''from sklearn.decomposition import PCA
Z = PCA(2, random_state=0).fit_transform((X - X.mean(0)) / X.std(0))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3))
for c, col in ((0, PAL[0]), (1, PAL[1])): ax[0].scatter(*Z[y == c][:1500].T, s=3, alpha=.5, color=col, label=f"class {c}")
ax[0].set_title("PCA of SYN-IDS-v1 (1500 pts/class)"); ax[0].legend(markerscale=3)
ax[1].hist(X[:, 0][y == 0], 50, alpha=.6, color=PAL[0]); ax[1].hist(X[:, 0][y == 1], 50, alpha=.6, color=PAL[1]); ax[1].set_title("Feature 0 by class")
save_fig(fig, "structure"); save_summary()'''),
])

add(3, "real_dataset_preprocessing", "Real public dataset preprocessing", "Preprocessing of the public healthcare tables: impossible zeros as missing, duplicate removal, train-only imputation and scaling.", [
("## 1. Missing-value codes and duplicates",
'''from federated_learning.preprocessing.real_datasets import _raw
Xp, yp, _ = _raw("pima"); Xh, yh, _ = _raw("heart")
miss = pd.Series(np.isnan(Xp).mean(0), index=["pregnancies", "glucose", "blood_pressure", "skin", "insulin", "bmi", "pedigree", "age"])
display(miss.round(3).to_frame("missing fraction (zeros recoded)"))
print("Cleveland rows after duplicate removal:", len(yh), "(303 raw)"); SUMMARY.update(pima_missing=miss.to_dict(), cleveland_rows=int(len(yh)))'''),
("## 2. Train-only imputation and scaling (no leakage)",
'''from sklearn.model_selection import train_test_split
from federated_learning.feature_engineering import preprocess_split
a, b, c, d = train_test_split(Xp, yp, test_size=.3, stratify=yp, random_state=SEED)
A, B = preprocess_split(a, b)
print("train mean ~0:", float(abs(A.mean(0)).max()) < 1e-5, "| no NaN left:", bool(np.isfinite(A).all() and np.isfinite(B).all()))'''),
("## 3. Figure",
'''fig, ax = plt.subplots(figsize=(4.5, 2.6)); miss.plot.bar(ax=ax, color=PAL[1]); ax.set_ylabel("missing fraction"); ax.set_title("Pima: zeros treated as missing")
save_fig(fig, "pima_missing"); save_summary()'''),
])

add(4, "client_partitioning", "Client partitioning strategies", "How data is split across simulated clients: IID, Dirichlet, label shards, stratified, quantity skew; missing-data simulation.", [
("## 1. Partition SYN-IDS-v1 ten ways",
'''from federated_learning.synthetic_data import make_dataset
from federated_learning.data_partitioning import partition, label_skew_stat, simulate_missing
d = make_dataset(n_samples=6000, seed=SEED); y = d.y_train
cfgs = {"iid": {}, "stratified": {}, "dirichlet a=0.5": dict(alpha=.5), "dirichlet a=0.1": dict(alpha=.1), "label_shards": dict(shards_per_client=1), "quantity_skew": dict(sigma=1.5)}
P = {k: partition(k.split()[0], y, 10, SEED, **v) for k, v in cfgs.items()}
tab = pd.DataFrame([dict(strategy=k, min_size=min(map(len, p)), max_size=max(map(len, p)), label_skew=round(label_skew_stat(y, p), 3)) for k, p in P.items()]); display(tab); SUMMARY["partitions"] = tab.to_dict("records")'''),
("## 2. Figure: positive rate per client",
'''fig, ax = plt.subplots(2, 3, figsize=(7.4, 3.6), sharey=True)
for a, (k, p) in zip(ax.ravel(), P.items()):
    a.bar(range(10), [y[i].mean() for i in p], color=PAL[0]); a.axhline(y.mean(), color=PAL[1], lw=1); a.set_title(k, fontsize=8)
fig.supxlabel("client"); fig.supylabel("positive rate"); save_fig(fig, "partitions")'''),
("## 3. Missing-data simulation (MCAR / MAR / MNAR)",
'''X = d.X_train[:2000]
for m in ("mcar", "mar", "mnar"): print(m, "missing fraction:", round(float(np.isnan(simulate_missing(X, .2, m, SEED)).mean()), 3))
save_summary()'''),
])

add(5, "federated_training", "Federated training", "FedAvg, FedProx, SCAFFOLD and FedNova on a Dirichlet label-skew federation.", [
("## 1. Train four algorithms (virtual clients, one process)",
'''from federated_learning.synthetic_data import make_dataset
from federated_learning.data_partitioning import partition
from federated_learning.servers.orchestrator import run_fl
from federated_learning.training import FLConfig
d = make_dataset(n_samples=6000, seed=SEED); parts = partition("dirichlet", d.y_train, 10, SEED, alpha=.5)
res = {a: run_fl(FLConfig(algo=a, rounds=15, local_epochs=2, mu=.05 if a == "fedprox" else 0., seed=SEED), d, parts) for a in ("fedavg", "fedprox", "scaffold", "fednova")}
tab = pd.DataFrame([dict(algo=a, acc=r.final["acc"], f1=r.final["f1"], auc=r.final["auc"], up_mb=r.totals.get("up_mb")) for a, r in res.items()]).round(4); display(tab)
SUMMARY["final"] = tab.to_dict("records")'''),
("## 2. Figure: convergence",
'''fig, ax = plt.subplots(figsize=(4.6, 3))
for a, r in res.items(): ax.plot([h["acc"] for h in r.history], label=a)
ax.set_xlabel("round"); ax.set_ylabel("test accuracy"); ax.legend(); save_fig(fig, "convergence"); save_summary()'''),
])

add(6, "secure_aggregation", "Secure aggregation", "Bonawitz-style masking: exactness against the plain mean, communication cost and dropout handling (simulation, no network).", [
("## 1. Exactness",
'''from federated_learning.privacy.secure_aggregation import SecureAggregationProtocol
rng = np.random.default_rng(SEED); n, L = 10, 2000
V = {i: rng.normal(size=L).astype(np.float32) for i in range(n)}; W = {i: float(rng.integers(50, 200)) for i in range(n)}
proto = SecureAggregationProtocol(n, L); vec, wt = proto.run(V, W)
ref = sum(W[i] * V[i] for i in range(n)) / sum(W.values())
err = float(np.abs(vec - ref).max()); print("max |secure - plain| =", err); assert err < 1e-4; SUMMARY["max_abs_error"] = err'''),
("## 2. Cost versus number of clients and dropout tolerance",
'''import time
rows = []
for n in (5, 10, 20, 40):
    V = {i: rng.normal(size=L).astype(np.float32) for i in range(n)}; W = {i: 1. for i in range(n)}
    t = time.perf_counter(); SecureAggregationProtocol(n, L).run(V, W); rows.append(dict(clients=n, seconds=time.perf_counter() - t))
cost = pd.DataFrame(rows); display(cost)
n = 10; V = {i: rng.normal(size=L).astype(np.float32) for i in range(n)}; W = {i: 1. for i in range(n)}
v2, _ = SecureAggregationProtocol(n, L).run(V, W, drop_before_unmask={0, 1})
print("with 2 clients dropping after masking, result equals the mean of the 8 survivors:", bool(np.abs(v2 - np.mean([V[i] for i in range(2, 10)], 0)).max() < 1e-4))
SUMMARY["cost"] = rows'''),
("## 3. Figure",
'''fig, ax = plt.subplots(figsize=(4.2, 2.8)); ax.plot(cost.clients, cost.seconds, "o-"); ax.set_xlabel("clients"); ax.set_ylabel("seconds per aggregation"); ax.set_title("SecAgg cost (single process)")
save_fig(fig, "cost"); save_summary()'''),
])

add(7, "attack_simulation", "Attack simulation", "Byzantine (sign-flip) and backdoor attacks against undefended FedAvg.", [
("## 1. Sign-flip Byzantine clients",
'''from federated_learning.synthetic_data import make_dataset
from federated_learning.data_partitioning import partition
from federated_learning.servers.orchestrator import run_fl
from federated_learning.training import FLConfig
from federated_learning import attacks as A
d = make_dataset(n_samples=6000, seed=SEED); parts = partition("iid", d.y_train, 10, SEED); cfg = FLConfig(rounds=12, local_epochs=1, seed=SEED)
rows = []
for frac in (0, .1, .2, .3):
    mal = set(range(int(10 * frac)))
    r = run_fl(cfg, d, parts, client_hook=A.byzantine_hook("sign_flip", mal, 5.0) if mal else None); rows.append(dict(malicious_fraction=frac, acc=r.final["acc"]))
byz = pd.DataFrame(rows).round(4); display(byz); SUMMARY["byzantine"] = byz.to_dict("records")'''),
("## 2. Backdoor",
'''rng = np.random.default_rng(SEED); mal = {0, 1, 2}
cd = {k: A.poison_client_data(d.X_train[parts[k]], d.y_train[parts[k]], .6, 0, rng)[:2] for k in mal}
ev = lambda m, w: A.backdoor_metrics(m, w, d.X_test, d.y_test, 0)
atk = run_fl(cfg, d, parts, client_data=cd, client_hook=A.boost_hook(mal, 3.0), eval_fn=ev); cln = run_fl(cfg, d, parts, eval_fn=ev)
print("attack success rate: attacked =", round(atk.final["asr"], 3), "| clean =", round(cln.final["asr"], 3)); SUMMARY["backdoor"] = dict(attacked_asr=atk.final["asr"], clean_asr=cln.final["asr"], attacked_acc=atk.final["acc"])'''),
("## 3. Figure",
'''fig, ax = plt.subplots(1, 2, figsize=(7, 2.7))
ax[0].plot(byz.malicious_fraction, byz.acc, "o-"); ax[0].set_xlabel("malicious fraction"); ax[0].set_ylabel("accuracy"); ax[0].set_title("Sign-flip (x5)")
ax[1].bar(["clean", "backdoored"], [cln.final["asr"], atk.final["asr"]], color=[PAL[0], PAL[1]]); ax[1].set_ylabel("attack success rate"); ax[1].set_title("Backdoor")
save_fig(fig, "attacks"); save_summary()'''),
])

add(8, "defense_mechanisms", "Defence mechanisms", "Robust aggregation rules under 20% sign-flip clients, with their clean-data cost.", [
("## 1. Compare aggregators",
'''from federated_learning.synthetic_data import make_dataset
from federated_learning.data_partitioning import partition
from federated_learning.servers.orchestrator import run_fl
from federated_learning.training import FLConfig
from federated_learning import attacks as A
from federated_learning.aggregators import WeightedMean, CoordinateMedian, TrimmedMean, Krum, ClippedMean
d = make_dataset(n_samples=6000, seed=SEED); parts = partition("iid", d.y_train, 10, SEED); cfg = FLConfig(rounds=12, local_epochs=1, seed=SEED)
mal = {0, 1}; AG = {"mean": WeightedMean, "median": CoordinateMedian, "trimmed": lambda: TrimmedMean(.2), "krum": lambda: Krum(2, 1), "clipped": lambda: ClippedMean(1.0)}
rows = []
for k, f in AG.items():
    clean = run_fl(cfg, d, parts, aggregator=f()).final["acc"]
    att = run_fl(cfg, d, parts, aggregator=f(), client_hook=A.byzantine_hook("sign_flip", mal, 5.0)).final["acc"]
    rows.append(dict(aggregator=k, clean_acc=clean, attacked_acc=att))
tab = pd.DataFrame(rows).round(4); display(tab); SUMMARY["defences"] = tab.to_dict("records")'''),
("## 2. Figure",
'''fig, ax = plt.subplots(figsize=(5, 2.9)); x = np.arange(len(tab)); ax.bar(x - .2, tab.clean_acc, .4, label="no attack"); ax.bar(x + .2, tab.attacked_acc, .4, label="20% sign-flip")
ax.set_xticks(x, tab.aggregator); ax.set_ylabel("accuracy"); ax.legend(); save_fig(fig, "defences"); save_summary()'''),
])

add(9, "hyperparameter_tuning", "Hyper-parameter tuning", "Random search for FedProx on a held-out validation split carved from the training data (the test set is never used for selection).", [
("## 1. Random search with a validation split",
'''from federated_learning.synthetic_data import make_dataset
from federated_learning.preprocessing import split_validation
from federated_learning.data_partitioning import partition
from federated_learning.servers.orchestrator import run_fl
from federated_learning.training import FLConfig
d = make_dataset(n_samples=6000, seed=SEED); tr = split_validation(d, .25, SEED)   # tr.X_test is the VALIDATION split carved from train
parts = partition("dirichlet", tr.y_train, 10, SEED, alpha=.5); rng = np.random.default_rng(SEED); rows = []
for t in range(8):
    p = dict(lr=float(10 ** rng.uniform(-2.5, -0.7)), mu=float(10 ** rng.uniform(-3, 0)), local_epochs=int(rng.integers(1, 4)))
    r = run_fl(FLConfig(algo="fedprox", rounds=10, seed=SEED, **p), tr, parts); rows.append(dict(**p, val_acc=r.final["acc"]))
tab = pd.DataFrame(rows).sort_values("val_acc", ascending=False).round(4); display(tab); best = tab.iloc[0]
SUMMARY["search"] = tab.to_dict("records")'''),
("## 2. Refit the best configuration on the full training split and evaluate once on test",
'''parts_full = partition("dirichlet", d.y_train, 10, SEED, alpha=.5)
fin = run_fl(FLConfig(algo="fedprox", rounds=10, seed=SEED, lr=float(best.lr), mu=float(best.mu), local_epochs=int(best.local_epochs)), d, parts_full)
dflt = run_fl(FLConfig(algo="fedprox", rounds=10, seed=SEED), d, parts_full)
print("tuned test acc", round(fin.final["acc"], 4), "| default", round(dflt.final["acc"], 4)); SUMMARY["test"] = dict(tuned=fin.final["acc"], default=dflt.final["acc"])
fig, ax = plt.subplots(figsize=(4.4, 2.8)); ax.scatter(np.log10(tab.lr), tab.val_acc, c=np.log10(tab.mu), cmap="viridis"); ax.set_xlabel("log10 learning rate"); ax.set_ylabel("validation accuracy"); ax.set_title("Random search (colour = log10 mu)")
save_fig(fig, "search"); save_summary()'''),
])

add(10, "evaluation", "Evaluation", "Federated versus centralised model: metrics with bootstrap confidence intervals, ROC curve and permutation importance.", [
("## 1. Train and evaluate",
'''from federated_learning.synthetic_data import make_dataset
from federated_learning.data_partitioning import partition
from federated_learning.servers.orchestrator import run_fl
from federated_learning.training import FLConfig, centralized_baseline, build_model
from federated_learning.evaluation.metrics import evaluate
from federated_learning.training.models import set_flat
import torch
d = make_dataset(n_samples=6000, seed=SEED); parts = partition("dirichlet", d.y_train, 10, SEED, alpha=1.0)
fl = run_fl(FLConfig(rounds=20, seed=SEED), d, parts); cen = centralized_baseline(d, epochs=20)
print("federated:", {k: round(v, 3) for k, v in fl.final.items() if k in ("acc", "f1", "auc")}); print("centralised:", {k: round(v, 3) for k, v in cen.items() if k in ("acc", "f1", "auc")})'''),
("## 2. Bootstrap 95% CI of the federated accuracy and ROC",
'''from sklearn.metrics import roc_curve
m = fl.extras["model"]; set_flat(m, fl.w); m.eval()
with torch.no_grad(): p = torch.softmax(m(torch.from_numpy(d.X_test)), 1)[:, 1].numpy()
pred = (p >= .5).astype(int); rng = np.random.default_rng(SEED)
bs = [(pred[i] == d.y_test[i]).mean() for i in (rng.integers(0, len(p), len(p)) for _ in range(1000))]
ci = np.percentile(bs, [2.5, 97.5]); print("accuracy 95% CI:", ci.round(4)); SUMMARY["acc_ci"] = ci.tolist()
fpr, tpr, _ = roc_curve(d.y_test, p)'''),
("## 3. Permutation importance of the global model and figure",
'''from federated_learning.explainability import permutation_importance
imp = permutation_importance(build_model(40, 32, "relu"), fl.w, d.X_test, d.y_test, n_repeats=3, seed=SEED); top = np.argsort(imp)[::-1][:10]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9)); ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", color="grey"); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate"); ax[0].set_title("ROC (federated model)")
ax[1].barh([f"f{j}" for j in top][::-1], imp[top][::-1]); ax[1].set_xlabel("accuracy drop"); ax[1].set_title("Permutation importance")
save_fig(fig, "evaluation"); SUMMARY["top_features"] = [int(j) for j in top]; save_summary()'''),
])

add(11, "cross_site_validation", "Cross-site validation", "UCI heart-disease sites as natural clients: within-site holdout (protocol A) and leave-one-site-out external validation (protocol B).", [
("## 1. Protocol A and B",
'''from federated_learning.preprocessing import load_sites, load_loso, SITE_NAMES, site_validation
from federated_learning.servers.orchestrator import run_fl
from federated_learning.training import FLConfig
cfg = FLConfig(rounds=25, local_epochs=2, hidden=16, seed=SEED)
A_ = load_sites(SEED); ra = run_fl(cfg, A_, A_.parts); print("Protocol A (within-site holdout) acc:", round(ra.final["acc"], 3), "| sites:", A_.site_names)
rows = []
for s in SITE_NAMES:
    L = load_loso(s); r = run_fl(cfg, L, L.parts); rows.append(dict(held_out_site=s, n_test=len(L.y_test), acc=r.final["acc"], auc=r.final["auc"], test_positive_rate=float(L.y_test.mean())))
tab = pd.DataFrame(rows).round(3); display(tab); SUMMARY["loso"] = tab.to_dict("records")'''),
("## 2. Figure",
'''fig, ax = plt.subplots(figsize=(5, 2.9)); ax.bar(tab.held_out_site, tab.acc, color=PAL[0]); ax.set_ylabel("accuracy on held-out site"); ax.set_title("Leave-one-site-out (small, 1980s cohorts)"); plt.setp(ax.get_xticklabels(), rotation=15)
save_fig(fig, "loso"); save_summary()'''),
])

add(12, "ablation_studies", "Ablation studies", "Effect of each protective layer (secure aggregation, clipping, robust median) on accuracy, with and without a sign-flip adversary.", [
("## 1. Ablate",
'''from federated_learning.synthetic_data import make_dataset
from federated_learning.data_partitioning import partition
from federated_learning.servers.orchestrator import run_fl
from federated_learning.training import FLConfig
from federated_learning import attacks as A
from federated_learning.aggregators import WeightedMean, CoordinateMedian, ClippedMean
from federated_learning.privacy.secure_aggregation import SecAggAggregator
d = make_dataset(n_samples=6000, seed=SEED); parts = partition("iid", d.y_train, 10, SEED); cfg = FLConfig(rounds=10, local_epochs=1, seed=SEED); hook = A.byzantine_hook("sign_flip", {0, 1}, 5.0)
cfgs = {"plain mean": WeightedMean, "+ secure aggregation": SecAggAggregator, "+ clipping": lambda: ClippedMean(1.0), "+ coordinate median": CoordinateMedian}
rows = []
for k, f in cfgs.items():
    rows.append(dict(configuration=k, clean_acc=run_fl(cfg, d, parts, aggregator=f()).final["acc"], sign_flip_acc=run_fl(cfg, d, parts, aggregator=f(), client_hook=hook).final["acc"]))
tab = pd.DataFrame(rows).round(4); display(tab); SUMMARY["ablation"] = tab.to_dict("records")
print("Secure aggregation hides individual updates but cannot detect a poisoned one: confidentiality and integrity need different components.")'''),
("## 2. Figure",
'''fig, ax = plt.subplots(figsize=(5.6, 2.9)); x = np.arange(len(tab)); ax.bar(x - .2, tab.clean_acc, .4, label="clean"); ax.bar(x + .2, tab.sign_flip_acc, .4, label="20% sign-flip")
ax.set_xticks(x, tab.configuration, rotation=15); ax.set_ylabel("accuracy"); ax.legend(); save_fig(fig, "ablation"); save_summary()'''),
])

add(13, "reproducibility_check", "Reproducibility check", "Same seed gives bit-identical results; data checksums verify; software environment is recorded.", [
("## 1. Bit-identical reruns",
'''from federated_learning.synthetic_data import make_dataset
from federated_learning.data_partitioning import partition
from federated_learning.servers.orchestrator import run_fl
from federated_learning.training import FLConfig
from federated_learning.utils.hashing import sha256_arrays
d = make_dataset(n_samples=3000, seed=SEED); parts = partition("dirichlet", d.y_train, 8, SEED, alpha=.5)
a = run_fl(FLConfig(rounds=5, seed=3), d, parts); b = run_fl(FLConfig(rounds=5, seed=3), d, parts); c = run_fl(FLConfig(rounds=5, seed=4), d, parts)
chk = dict(same_seed_identical=bool(np.array_equal(a.w, b.w)), different_seed_differs=bool(not np.array_equal(a.w, c.w)), sha256_same_seed=sha256_arrays(a.w) == sha256_arrays(b.w))
print(chk); assert all(chk.values()); SUMMARY["checks"] = chk
dist = {"same seed (3 vs 3)": float(np.linalg.norm(a.w - b.w)), "different seed (3 vs 4)": float(np.linalg.norm(a.w - c.w))}
fig, ax = plt.subplots(figsize=(4.2, 2.6)); ax.bar(list(dist), list(dist.values()), color=[PAL[2], PAL[1]]); ax.set_ylabel("L2 distance of final weights"); ax.set_title("Reruns are bit-identical")
save_fig(fig, "rerun_distance")'''),
("## 2. Dataset checksums and environment",
'''from federated_learning.preprocessing import SPECS
from federated_learning.preprocessing.download import raw_path
from federated_learning.utils.hashing import sha256_file
rows = []
for k, s in SPECS.items():
    for f in s.files:
        p = raw_path(k, f.filename)
        if f.url and p.exists(): rows.append(dict(dataset=k, file=f.filename, ok=sha256_file(p) == f.sha256))
display(pd.DataFrame(rows)); assert all(r["ok"] for r in rows)
import importlib.metadata as md, platform
env = dict(python=platform.python_version(), platform=platform.platform(), **{p: md.version(p) for p in ("numpy", "scipy", "scikit-learn", "torch", "pandas", "matplotlib")}); print(json.dumps(env, indent=1)); SUMMARY["environment"] = env; save_summary()'''),
])

add(14, "publication_figures", "Publication figures from stored results", "Figures for the manuscript built from the stored full-scale experiment results in experiments/results/ (no re-running).", [
("## 1. Load stored results",
'''R_ = ROOT / "experiments" / "results"
perf = pd.read_csv(R_ / "fl_perf_002" / "task_2_5_kpi_check.csv"); byz = pd.read_csv(R_ / "fl_attack_003" / "task_3_2_summary.csv")
display(perf.head()); display(byz.head())'''),
("## 2. Figure: secure-aggregation overhead and robust aggregation under attack",
'''fig, ax = plt.subplots(1, 2, figsize=(7.4, 3))
for cfg_, g in perf.groupby("config"): ax[0].plot(g.n_clients, g.training_overhead_pct, "o-", label=cfg_)
ax[0].axhline(30, ls="--", color="grey"); ax[0].set_xscale("log"); ax[0].set_xlabel("clients"); ax[0].set_ylabel("training overhead (%)"); ax[0].set_title("Overhead vs 30% budget"); ax[0].legend(fontsize=7)
b = byz[(byz.behavior == "sign_flip") & (byz.mal_frac == byz.mal_frac.max())]
ax[1].bar(b.aggregator, b.acc, color=PAL[0]); ax[1].set_ylabel("accuracy"); ax[1].set_title(f"Sign-flip, {byz.mal_frac.max():.0%} malicious"); plt.setp(ax[1].get_xticklabels(), rotation=20)
save_fig(fig, "overhead_and_robustness"); SUMMARY["source_files"] = ["fl_perf_002/task_2_5_kpi_check.csv", "fl_attack_003/task_3_2_summary.csv"]; save_summary()'''),
])


def build(i, name, title, purpose, cells):
    nb_id = f"{i:02d}_{name}"
    c = [nbf.v4.new_markdown_cell(f"# {i:02d} — {title}\n\n**Purpose.** {purpose}\n\n**Author:** Guy M. Kaptue T. · **Execution:** independent, seeded (`SEED = 0`), demonstration scale (CPU, about a minute). Outputs: `reports/figures/{nb_id}_*.png|pdf` and `reports/notebooks/{nb_id}.json`.\n\n{NOTICE}"),
         nbf.v4.new_code_cell(SETUP.replace('__NB_ID__', nb_id))]
    for md, code in cells:
        c += [nbf.v4.new_markdown_cell(md), nbf.v4.new_code_cell(code)]
    c.append(nbf.v4.new_markdown_cell("## Limitations\nDemonstration scale (few rounds, one seed unless stated): numbers here illustrate the workflow and are not the study's reported results, which come from the full experiment pipeline in `experiments/results/` with multiple seeds and confidence intervals."))
    nb = nbf.v4.new_notebook(cells=c, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    return nb_id, nb


def main():
    NB.mkdir(exist_ok=True)
    for spec in NBS:
        nb_id, nb = build(*spec); nbf.write(nb, NB / f"{nb_id}.ipynb")
    print(len(NBS), "notebooks written")


if __name__ == "__main__":
    main()
