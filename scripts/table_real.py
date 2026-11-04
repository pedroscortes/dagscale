# ruff: noqa: N803
"""Table 6 (raw-units real data) from the methods.py and light_tunnel.py results, written to argv[1]."""

from __future__ import annotations

import csv
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dagscale.check import check
from sachs import COND1, load


def rows(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as fh:
        return [r for r in csv.DictReader(fh) if not r.get("error")]


R = "results"
# log-likelihood variant: rows from the reruns (methods_logll, light_tunnel_rerun)
B = rows(f"{R}/methods/methods.csv.gz") + rows(f"{R}/methods/methods_recent.csv.gz")
B = [r for r in B if r["method"] != "dagma_logll"] + rows(f"{R}/methods/methods_logll.csv")
C = rows(f"{R}/light_tunnel/light_tunnel.csv") + rows(f"{R}/light_tunnel/light_tunnel_recent.csv")
CF = rows(f"{R}/light_tunnel/light_tunnel_rerun.csv")
C = [r for r in C if r["method"] != "dagma_logll"] + [r for r in CF if r["method"] == "dagma_logll"]
sachs = {r["method"]: r for r in B if r["part"] == "sachs" and r["graph"] == "cond1_853" and r["variant"] == "raw"}
cham = {r["method"]: r for r in C if r["variant"] == "raw"}
NO_LOGDET = ("notears", "golem_ev", "golem_nv", "topo", "sortnregress", "r2sortnregress")


def cell(r):
    if not r:
        return ("--", "--")
    return (f"{float(r['proj_f1']):.2f}", r"\checkmark" if acyclic(r) else r"$\times$")


def acyclic(r):
    if r["method"] in NO_LOGDET:
        return r["cyclic"] != "True"
    h = r["h_logdet"]
    return h not in ("nan", "") and float(h) <= 0.01


def floor_leading(X, lambda1=0.02, mu=1e-4, s=0.6):
    """Conditional-variance floor of dagscale.check: mu^2 s^2 / 4 sum_j (Var(X_j | X_-j) - lambda1)_+^2."""
    return check(X, lambda1=lambda1, mu=mu, s=s).floor


X = load(COND1)
fl_s = floor_leading(X)
fl_c = next(float(r["floor_leading"]) for r in CF if r["variant"] == "raw")  # light tunnel, from the rerun CSV (1/(n-1) covariance, no lambda1; differs from check() by under 0.02%)
order = [
    ("dagma", "DAGMA (least squares)"),
    ("dagma_nnorm", "DAGMA, noise units"),
    ("colide_ev", "CoLiDE-EV"),
    ("colide_nv", "CoLiDE-NV"),
    ("dagma_logll", "DAGMA, log-likelihood"),
    ("analytic2", "Analytic constraint$^{\\dagger}$"),
    ("notears", "NOTEARS"),
    ("topo", "TOPO"),
    ("golem_ev", "GOLEM-EV"),
    ("golem_nv", "GOLEM-NV"),
]
lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\small\setlength{\tabcolsep}{4pt}",
    r"""\caption{Raw-units real data. Projected F1 against the consensus graph (Sachs, 17 edges, $n=853$) and the ground-truth graph (Causal Chambers light tunnel, 39 edges, $n=10{,}000$); a cross marks a non-acyclic output in the sense of \cref{sec:exp-setup}. The conditional-variance floor is the leading-order value of \cref{sec:exp-setup}, computed from the data alone. Values above $0.01$ flag a non-acyclic DAGMA output.}""",
    r"\label{tab:realdata}",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r" & \multicolumn{2}{c}{Sachs (raw)} & \multicolumn{2}{c}{Causal Chambers (raw)} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
    r"Method & F1 & acyclic & F1 & acyclic \\",
    r"\midrule",
]
for m, name in order:
    s, c = sachs.get(m), cham.get(m)
    a, b = cell(s), cell(c)
    lines.append(f"{name} & {a[0]} & {a[1]} & {b[0]} & {b[1]} \\\\")
lines += [
    r"\midrule",
    f"Conditional-variance floor & \\multicolumn{{2}}{{c}}{{{fl_s:.1f}}} & \\multicolumn{{2}}{{c}}{{{fl_c:.0f}}} \\\\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\par\smallskip\footnotesize$^{\dagger}$Our re-implementation of \citet{zhang2025analytic}; the official repository "
    r"is empty.",
    r"\end{table}",
]
out = sys.argv[1] if len(sys.argv) > 1 else "table_realdata.tex"
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
