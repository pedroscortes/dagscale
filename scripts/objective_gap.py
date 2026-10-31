# ruff: noqa: E402, N802, N803, N806
"""Objective F(W) = 0.5 tr((I - W)^T S (I - W)) + 0.02 |W|_1 of the W saved by simulate.py, the sortnregress DAG and the
true-parents OLS DAG (Section 5.6).
"""

from __future__ import annotations

import csv
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from simulate import GRAPHS, make_data, project_dag
from sorting import regress_along

BASE = "results/simulate"
LAM = 0.02


def one(job):
    graph, d, c, seed = job
    X, A = make_data(graph, d, 3000, seed, c)
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / Xc.shape[0]
    ev = np.linalg.eigvalsh(S)
    kap = ev[-1] / ev[0] if ev[0] > 0 else np.inf

    def F(Wd):
        D = np.eye(d) - Wd
        return 0.5 * np.trace(D.T @ S @ D) + LAM * np.abs(Wd).sum()

    W_snr = regress_along(Xc, np.argsort(Xc.var(0), kind="stable"))
    W_pa = np.zeros((d, d))
    for i in range(d):
        pa = np.nonzero(A[i])[0]
        if len(pa):
            W_pa[pa, i] = np.linalg.lstsq(Xc[:, pa], Xc[:, i], rcond=None)[0]
    row = dict(graph=graph, d=d, wscale=c, seed=seed, kappa=kap, F_snr=F(W_snr), F_truth=F(W_pa))
    for arm in ("ref", "pc_abs10"):
        Wg = np.load(os.path.join(BASE, "W", f"{graph}_d{d}_w{c:g}_s{seed}_{arm}.npz"))["W"].astype(np.float64)
        row[f"F_{arm}"] = F(Wg.T)
        P = project_dag(Wg)  # OLS refit on the projected support
        Wr = np.zeros((d, d))
        for i in range(d):
            pa = np.nonzero(P[i])[0]
            if len(pa):
                Wr[pa, i] = np.linalg.lstsq(Xc[:, pa], Xc[:, i], rcond=None)[0]
        row[f"F_{arm}_projrefit"] = F(Wr)
    return row


def main():
    jobs = [(g, d, c, s) for g in GRAPHS for d in (30, 50, 100) for c in (1.0, 2.0, 4.0) for s in range(30, 50)]
    out = "results/objective_gap/objective_gap.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 16) as ex:
        rows = list(ex.map(one, sorted(jobs, key=lambda j: -j[1])))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows))


if __name__ == "__main__":
    main()
