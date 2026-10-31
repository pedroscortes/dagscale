# ruff: noqa: E402, N803, N806  (matrix names follow the dagma package; env vars must be set before numpy import)
"""DAGMA and pc_abs10 on ER2/ER4/SF2/SF4 data, d 30/50/100, weight scale 1/2/4, n 3000 (Sections 5.6, S6, Tables 5, S6).

Each W is saved under <out dir>/W (read by objective_gap.py). Rows already in --out are kept and not rerun.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import contextlib
import io
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dagma_fit import Logged, diagnostics, parse_arm, scores

ARMS = ("ref", "pc_abs10")
GRAPHS = ("ER2", "ER4", "SF2", "SF4")


def make_data(graph, d, n, seed, wscale):
    import jax

    from dagscale.evaluation import adjacency_from_scm_weights
    from dagscale.scm import LinearGaussianSCM

    wr, nr = (0.5 * wscale, 2.0 * wscale), (0.5, 1.5)
    k = int(graph[2:])
    if graph.startswith("ER"):
        scm = LinearGaussianSCM.random(
            d=d, edge_probability=2 * k / (d - 1), seed=seed, noise_range=nr, weight_range=wr
        )
    else:
        scm = LinearGaussianSCM.random_scale_free(d=d, m=k, seed=seed, noise_range=nr, weight_range=wr)
    X = np.asarray(scm.sample(jax.random.PRNGKey(seed), n=n), dtype=np.float64)
    A = adjacency_from_scm_weights(scm.weights)
    if graph.startswith("SF"):  # scale-free nodes come in topological order; relabel
        pi = np.random.default_rng(seed + 11).permutation(d)
        X, A = X[:, pi], A[np.ix_(pi, pi)]
    return X, A


def project_dag(W_geo, t=0.3):
    """Greedy maximal DAG: add thresholded edges in decreasing |w|, skip those that close a cycle."""
    d = W_geo.shape[0]
    ch, pa = np.nonzero(np.abs(W_geo) > t)  # [target, source]
    order = np.argsort(-np.abs(W_geo[ch, pa]), kind="stable")
    reach = np.eye(d, dtype=bool)  # reach[a, b]: path a -> b
    keep = np.zeros_like(W_geo)
    for e in order:
        i, j = ch[e], pa[e]  # edge j -> i
        if reach[i, j]:
            continue
        keep[i, j] = W_geo[i, j]
        reach |= np.outer(reach[:, j], reach[i, :])
    return keep


def run_one(job):
    graph, arm, d, n, seed, wscale, wdir = job
    try:
        X, A = make_data(graph, d, n, seed, wscale)
        kap, kapR, vs = diagnostics(X, A)
        model_kw, fit_kw = parse_arm(arm)
        m = Logged(**model_kw)
        t0 = time.time()
        with contextlib.redirect_stderr(io.StringIO()):
            W = m.fit(X.copy(), lambda1=0.02, w_threshold=0.0, **fit_kw)
        wall = time.time() - t0
        h, _ = m._h(W)
        W_geo = np.asarray(W).T
        np.savez_compressed(
            os.path.join(wdir, f"{graph}_d{d}_w{wscale:g}_s{seed}_{arm}.npz"),
            W=W_geo.astype(np.float32),
        )
        row = dict(
            graph=graph,
            arm=arm,
            d=d,
            n=n,
            seed=seed,
            wscale=wscale,
            n_true=int(A.sum()),
            kappa_cov=kap,
            kappa_corr=kapR,
            varsort=round(vs, 4),
            h=float(h),
            wall=round(wall, 1),
        )
        row.update(scores(W_geo, A))
        row.update({f"proj_{k}": v for k, v in scores(project_dag(W_geo), A).items() if k in PROJ_KEYS})
        return row
    except Exception as e:
        return dict(
            graph=graph,
            arm=arm,
            d=d,
            n=n,
            seed=seed,
            wscale=wscale,
            error=f"{type(e).__name__}: {e}"[:200],
        )


PROJ_KEYS = ("shd_0.3", "n_learned", "precision", "recall", "f1")
FIELDS = (
    [
        "graph",
        "arm",
        "d",
        "n",
        "seed",
        "wscale",
        "n_true",
        "kappa_cov",
        "kappa_corr",
        "varsort",
        "h",
        "wall",
    ]
    + [
        "shd_0.1",
        "shd_0.3",
        "shd_0.5",
        "n_learned",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "auprc",
    ]
    + ["prec_at_E"]
    + [f"proj_{k}" for k in PROJ_KEYS]
    + ["error"]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--out", default="results/simulate/simulate.csv")
    ap.add_argument("--seeds", default="30-49", help="inclusive range")
    ap.add_argument("--smoke", action="store_true", help="one tiny job per graph type, d=10")
    a = ap.parse_args()
    wdir = os.path.join(os.path.dirname(a.out), "W")
    os.makedirs(wdir, exist_ok=True)
    lo, hi = map(int, a.seeds.split("-"))
    if a.smoke:
        jobs = [(g, arm, 10, 500, 20, 1.0, wdir) for g in GRAPHS for arm in ("ref", "pc_abs10")]
    else:
        jobs = [
            (g, arm, d, 3000, s, c, wdir)
            for g in GRAPHS
            for d in (30, 50, 100)
            for c in (1.0, 2.0, 4.0)
            for s in range(lo, hi + 1)
            for arm in ARMS
        ]
    done, old = set(), []
    if os.path.exists(a.out):
        with open(a.out) as fh:
            for r in csv.DictReader(fh):
                if not r.get("error"):
                    old.append(r)
                    done.add((r["graph"], r["arm"], int(r["d"]), int(r["seed"]), float(r["wscale"])))
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[4], float(j[5])) not in done]
    jobs.sort(key=lambda j: -j[2])
    print(f"{len(jobs)} jobs", flush=True)
    k = 0
    with open(a.out, "w", newline="") as fh, ProcessPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
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
