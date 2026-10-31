# ruff: noqa: E402, N806  (matrix names follow the dagma package; env vars must be set before numpy import)
"""DAGMA with 8 or 12 central-path stages and NOTEARS (d <= 50) on the data of simulate.py, weight scales 2 and 4,
seeds 30-39 (Sections 5.6 and S6, Tables 5 and S6). Rows already in --out are kept and not rerun.
"""

from __future__ import annotations

import argparse
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
from dagma_fit import Logged, diagnostics, parse_arm, scores
from simulate import FIELDS, GRAPHS, PROJ_KEYS, make_data, project_dag

ARMS = ("ref_T8", "ref_T12", "notears")


def run_one(job):
    graph, arm, d, seed, c, wdir = job
    try:
        X, A = make_data(graph, d, 3000, seed, c)
        kap, kapR, vs = diagnostics(X, A)
        Xc = X - X.mean(0)
        S = Xc.T @ Xc / Xc.shape[0]
        t0 = time.time()
        if arm == "notears":
            from notears.linear import notears_linear

            W = notears_linear(Xc.copy(), lambda1=0.02, loss_type="l2", w_threshold=0.0)
            from scipy.linalg import expm

            h = float(np.trace(expm(W * W)) - d)  # NOTEARS' own acyclicity measure (0 iff DAG)
        else:
            mk, fk = parse_arm(arm)
            m = Logged(**mk)
            with contextlib.redirect_stderr(io.StringIO()):
                W = m.fit(X.copy(), lambda1=0.02, w_threshold=0.0, **fk)
            h = float(m._h(W)[0])
        wall = time.time() - t0
        W = np.asarray(W)
        D = np.eye(d) - W
        F = 0.5 * np.trace(D.T @ S @ D) + 0.02 * np.abs(W).sum()
        W_geo = W.T
        np.savez_compressed(os.path.join(wdir, f"{graph}_d{d}_w{c:g}_s{seed}_{arm}.npz"), W=W_geo.astype(np.float32))
        row = dict(
            graph=graph,
            arm=arm,
            d=d,
            n=3000,
            seed=seed,
            wscale=c,
            n_true=int(A.sum()),
            kappa_cov=kap,
            kappa_corr=kapR,
            varsort=round(vs, 4),
            h=h,
            wall=round(wall, 1),
            objective=F,
        )
        row.update(scores(W_geo, A))
        row.update({f"proj_{k}": v for k, v in scores(project_dag(W_geo), A).items() if k in PROJ_KEYS})
        return row
    except Exception as e:
        return dict(
            graph=graph,
            arm=arm,
            d=d,
            n=3000,
            seed=seed,
            wscale=c,
            error=f"{type(e).__name__}: {e}"[:200],
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--out", default="results/schedule/schedule.csv")
    a = ap.parse_args()
    wdir = os.path.join(os.path.dirname(a.out), "W")
    os.makedirs(wdir, exist_ok=True)
    jobs = [
        (g, arm, d, s, c, wdir)
        for g in GRAPHS
        for d in (30, 50, 100)
        for c in (2.0, 4.0)
        for s in range(30, 40)
        for arm in ARMS
        if not (arm == "notears" and d > 50)
    ]
    done, old = set(), []
    if os.path.exists(a.out):
        with open(a.out) as fh:
            for r in csv.DictReader(fh):
                if not r.get("error"):
                    old.append(r)
                    done.add((r["graph"], r["arm"], int(r["d"]), int(r["seed"]), float(r["wscale"])))
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3], float(j[4])) not in done]
    jobs.sort(key=lambda j: -j[2] * (3 if j[1] == "ref_T12" else 2 if j[1] == "ref_T8" else 1))
    print(f"{len(jobs)} jobs", flush=True)
    fields = [*FIELDS[:12], "objective", *FIELDS[12:]]
    k = 0
    with open(a.out, "w", newline="") as fh, ProcessPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in old:
            w.writerow(r)
        fh.flush()
        for row in ex.map(run_one, jobs):
            w.writerow(row)
            fh.flush()
            k += 1
            if k % 50 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
