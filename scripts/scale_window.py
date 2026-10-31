# ruff: noqa: E402, N803, N806
"""Scale window (Section 5.2, Fig. 2, Table S3): DAGMA and pc_abs10 on c X, c in {0.01, ..., 1000}.

ER2/ER4/SF4, d 30/50, weight scale 0.5/1/2, n 3000, seeds SEED_LO..SEED_HI-1.
"""

from __future__ import annotations

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
from simulate import make_data, project_dag
from sorting import regress_along

OUT = os.environ.get("OUT", "results/scale_window/scale_window.csv")


def one(job):
    g, d, w, s, c, arm = job
    try:
        X, A = make_data(g, d, 3000, s, w)
        X = X * c
        Xc = X - X.mean(0)
        S = Xc.T @ Xc / Xc.shape[0]
        ev = np.linalg.eigvalsh(S)
        sd = np.sqrt(np.diag(S))
        m = Logged() if arm == "ref" else Logged(direction="full", damping=10.0)
        with contextlib.redirect_stderr(io.StringIO()):
            W = np.asarray(m.fit(X.copy(), lambda1=0.02, w_threshold=0.0))

        def obj(M):
            D = np.eye(d) - M
            return 0.5 * np.trace(D.T @ S @ D) + 0.02 * np.abs(M).sum()

        Wsnr = regress_along(Xc, np.argsort(Xc.var(0), kind="stable"))
        Wg = W.T
        return dict(
            graph=g,
            d=d,
            wscale=w,
            seed=s,
            c=c,
            arm=arm,
            kappa=ev[-1] / ev[0] if ev[0] > 0 else np.inf,
            lmax=ev[-1],
            h=float(m._h(W)[0]),
            proj_f1=scores(project_dag(Wg), A)["f1"],
            auprc_std=scores(Wg * sd[None, :] / sd[:, None], A)["auprc"],
            F=obj(W),
            witness=obj(Wsnr) < obj(W),
        )
    except Exception as e:
        return dict(graph=g, d=d, wscale=w, seed=s, c=c, arm=arm, error=repr(e)[:150])


def main():
    jobs = [
        (g, d, w, s, c, arm)
        for g in ("ER2", "ER4", "SF4")
        for d in (30, 50)
        for w in (0.5, 1.0, 2.0)
        for s in range(int(os.environ.get("SEED_LO", 30)), int(os.environ.get("SEED_HI", 40)))
        for c in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
        for arm in ("ref", "pc_abs10")
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 18))) as ex:
        rows = list(ex.map(one, sorted(jobs, key=lambda j: -j[1])))
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(OUT, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print("DONE", len(rows))


if __name__ == "__main__":
    main()
