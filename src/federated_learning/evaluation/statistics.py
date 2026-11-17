"""Statistics for repeated independent FL runs (the unit of replication is the *seed*, never the round)."""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def bootstrap_ci(x, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if x.size < 2:
        raise ValueError("need >= 2 runs for a CI")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))

def t_ci(x, alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    m, se = x.mean(), stats.sem(x)
    h = se * stats.t.ppf(1 - alpha / 2, x.size - 1)
    return float(m), float(m - h), float(m + h)

def cohens_dz(a, b) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("inf") if d.mean() != 0 else 0.0

def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > y) for x in a for y in b); lt = sum((x < y) for x in a for y in b)
    return float((gt - lt) / (a.size * b.size))

def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()

def paired_tests(a, b) -> dict:
    """Paired comparison of two methods over the same seeds (same splits/inits => pairing justified)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.size < 3:
        raise ValueError("need paired samples with n >= 3")
    out = {"n": int(a.size), "mean_diff": float((a - b).mean()), "dz": cohens_dz(a, b)}
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue) if np.any(a != b) else 1.0
    except ValueError:
        out["wilcoxon_p"] = 1.0
    out["ttest_p"] = float(stats.ttest_rel(a, b).pvalue) if np.any(a != b) else 1.0
    return out

def pairwise_table(samples: dict[str, np.ndarray], key: str = "wilcoxon_p") -> list[dict]:
    rows = []
    for x, y in itertools.combinations(samples, 2):
        r = paired_tests(samples[x], samples[y]); r.update(a=x, b=y); rows.append(r)
    adj = holm([r[key] for r in rows])
    for r, p in zip(rows, adj):
        r["holm_p"] = p
    return rows

def friedman(samples: dict[str, np.ndarray]) -> dict:
    arrs = [np.asarray(v, float) for v in samples.values()]
    if len(arrs) < 3:
        return {"statistic": None, "p": None}
    s = stats.friedmanchisquare(*arrs)
    return {"statistic": float(s.statistic), "p": float(s.pvalue)}

def min_wilcoxon_p(n: int) -> float:
    """Smallest attainable two-sided exact Wilcoxon signed-rank p-value with n pairs (= 2/2^n)."""
    return 2.0 / (2 ** n)
