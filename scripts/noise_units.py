# ruff: noqa: E402, N803, N806
"""Noise-unit normalization and CoLiDE-EV under a global multiplier (Section 5.5, Table S5).

ER2/ER4/SF2/SF4, d 30/50, weight scale 1/2, seeds 30-39, c in {0.01, 1, 100}; ref at c = 1 only.
"""

import contextlib
import csv
import io
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dagma_fit import Logged, scores
from simulate import GRAPHS, make_data, project_dag


def nnorm(X):
    S = np.cov(X, rowvar=False)
    return X / np.sqrt(np.median(1.0 / np.diag(np.linalg.inv(S))))


def one(job):
    g, d, w, s, c, arm = job
    try:
        X, A = make_data(g, d, 3000, s, w)
        X = X * c
        var = np.var(X, axis=0)
        h = float("nan")
        if arm == "colide_ev":
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "third_party"))
            from colide_model import colide_ev

            with contextlib.redirect_stderr(io.StringIO()):
                W, _ = colide_ev().fit(
                    X.copy(),
                    lambda1=0.05,
                    T=4,
                    s=[1.0, 0.9, 0.8, 0.7],
                    warm_iter=2e4,
                    max_iter=7e4,
                    lr=0.0003,
                )
            M = 0.7 * np.eye(d) - W * W
            sg, ld = np.linalg.slogdet(M)
            h = float(-ld + d * np.log(0.7)) if sg > 0 else float("inf")
        else:
            Xin = nnorm(X) if arm == "nnorm" else X
            m = Logged()
            with contextlib.redirect_stderr(io.StringIO()):
                W = np.asarray(m.fit(Xin.copy(), lambda1=0.02, w_threshold=0.0))
            h = float(m._h(W)[0])
        Wg = np.asarray(W).T
        B = (np.abs(Wg) > 0.3).astype(np.int8)
        return dict(
            graph=g,
            d=d,
            w=w,
            seed=s,
            c=c,
            arm=arm,
            spread=float(np.log10(var.max() / var.min())),
            h=h,
            proj_f1=scores(project_dag(Wg), A)["f1"],
            graph_bits="".join(map(str, B.ravel())),
        )
    except Exception as e:
        return dict(graph=g, d=d, w=w, seed=s, c=c, arm=arm, error=repr(e)[:150])


if __name__ == "__main__":
    jobs = []
    for g in GRAPHS:
        for d in (30, 50):
            for w in (1.0, 2.0):
                for s in range(30, 40):
                    jobs.append((g, d, w, s, 1.0, "ref"))
                    for c in (0.01, 1.0, 100.0):
                        jobs += [(g, d, w, s, c, "nnorm"), (g, d, w, s, c, "colide_ev")]
    with ProcessPoolExecutor(18) as ex:
        out = list(ex.map(one, sorted(jobs, key=lambda j: -j[1])))
    keys = list(dict.fromkeys(k for r in out for k in r))
    os.makedirs("results/noise_units", exist_ok=True)
    with open("results/noise_units/noise_units.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(out)
    print("DONE", len(out), "errors", sum(1 for r in out if r.get("error")))
