"""Helpers shared by generate_manuscript.py: LaTeX table/figure builders and result accessors (all numbers come from stored result files)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]; O = R / "experiments/results"; PUB = R / "latex"; FIG = PUB / "figures"; PUB.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
def rd(s, f): return pd.read_csv(O / s / f)
def js(s, f): return json.load(open(O / s / f))
def esc(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("$", r"\$")): s = s.replace(a, b)
    return s
def fmt(v, nd, raw=False):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v): return "--"
        return f"{v:.{nd}f}"
    return str(v) if raw else esc(v)
def table(df, caption, label, nd=3, cols=None, rename=None, star=False, align=None, raw=False, fit=False):
    """raw=True: string cells and headers are trusted LaTeX (math allowed); default escapes them."""
    df = df.copy()
    if cols: df = df[cols]
    if rename: df = df.rename(columns=rename)
    al = align or ("l" + "r" * (len(df.columns) - 1))
    body = " \\\\\n".join(" & ".join(fmt(v, nd, raw) for v in r) for r in df.values)
    env = "table*" if star else "table"
    rs0, rs1 = (("\\resizebox{\\columnwidth}{!}{", "}") if fit and not star else ("", ""))
    return (f"\\begin{{{env}}}[t]\\centering\\caption{{{caption}}}\\label{{{label}}}\\scriptsize\n{rs0}\\begin{{tabular}}{{{al}}}\\toprule\n" +
            " & ".join((str(c) if raw else esc(c)) for c in df.columns) + " \\\\\\midrule\n" + body + f" \\\\\\bottomrule\\end{{tabular}}{rs1}\\end{{{env}}}")
def figure(src, caption, label, star=False):
    """src = 'STORY/figures/name.png'; copied into publication/figures."""
    s, _, n = src.partition("/figures/")
    shutil.copy(O / s / "figures" / n, FIG / n)
    env = "figure*" if star else "figure"
    w = "0.95\\textwidth" if star else "\\columnwidth"
    return f"\\begin{{{env}}}[t]\\centering\\includegraphics[width={w}]{{figures/{n}}}\\caption{{{caption}}}\\label{{{label}}}\\end{{{env}}}"
def ci(s):  # "[a, b]" -> (a,b)
    a, b = str(s).strip("[]").split(","); return float(a), float(b)
