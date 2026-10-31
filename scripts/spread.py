# ruff: noqa: E402, N803, N806
"""Global trace normalization (Section 5.6, Table S7).

Arms: ref, gnorm (X / sqrt(tr(S)/d)), gnorm_adapt (gnorm, then stages mu x0.1 at s 0.6 until h <= 1e-6, at most 10).
Part A: ER2/ER4/SF4, d 30, w 0.5/1/2, c 0.01/1/100, seeds 30-34. Part B: all graphs, d 30/50/100, w 2/4, seeds 30-39.
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dagma_fit import Logged, scores
from simulate import GRAPHS, make_data, project_dag

OUT = "results/spread/spread.csv"


def fit_arm(X, arm):
    m = Logged()
    if arm != "ref":
        X = X / np.sqrt(np.trace(np.cov(X, rowvar=False)) / X.shape[1])
    with contextlib.redirect_stderr(io.StringIO()):
        W = np.asarray(m.fit(X.copy(), lambda1=0.02, w_threshold=0.0))
    stages = 5
    if arm == "gnorm_adapt":
        mu, s = 1e-5, 0.6
        while stages < 10 and m._h(W, s)[0] > 1e-6:
            W2, ok = m.minimize(W.copy(), mu, 60000, s, lr=3e-4)
            if not ok:
                break
            W, mu, stages = W2, mu * 0.1, stages + 1
    return W, float(m._h(W, 0.6)[0]), stages


def one(job):
    part, g, d, w, s, c, arm = job
    try:
        X, A = make_data(g, d, 3000, s, w)
        X = X * c
        S = np.cov(X, rowvar=False)
        ev = np.linalg.eigvalsh(S)
        var = np.diag(S)
        sd = np.sqrt(var)
        t0 = time.time()
        W, h, stages = fit_arm(X, arm)
        Wg = W.T
        return dict(
            part=part,
            graph=g,
            d=d,
            wscale=w,
            seed=s,
            c=c,
            arm=arm,
            kappa=ev[-1] / ev[0] if ev[0] > 0 else np.inf,
            spread=np.log10(var.max() / var.min()),
            h=h,
            stages=stages,
            wall=round(time.time() - t0, 1),
            proj_f1=scores(project_dag(Wg), A)["f1"],
            auprc_std=scores(Wg * sd[None, :] / sd[:, None], A)["auprc"],
        )
    except Exception as e:
        return dict(part=part, graph=g, d=d, wscale=w, seed=s, c=c, arm=arm, error=repr(e)[:150])


def main():
    arms = ("ref", "gnorm", "gnorm_adapt")
    jobs = [
        ("A", g, 30, w, s, c, a)
        for g in ("ER2", "ER4", "SF4")
        for w in (0.5, 1.0, 2.0)
        for s in range(30, 35)
        for c in (0.01, 1.0, 100.0)
        for a in arms
    ]
    jobs += [
        ("B", g, d, w, s, 1.0, a)
        for g in GRAPHS
        for d in (30, 50, 100)
        for w in (2.0, 4.0)
        for s in range(30, 40)
        for a in arms
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with ProcessPoolExecutor(18) as ex:
        rows = list(ex.map(one, sorted(jobs, key=lambda j: -j[2])))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print("DONE", len(rows))


if __name__ == "__main__":
    main()
