# ruff: noqa: E402, N803, N806
"""Leading-order conditional-variance floor on the data sets of predict.py, seeds 30-79 (Table 3); argv[1] = output."""

import csv
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, os.path.dirname(__file__))
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from simulate import make_data


def floor_leading(X, mu=1e-4, s=0.6):
    cv = 1.0 / np.diag(np.linalg.inv(np.cov(X, rowvar=False)))
    return mu**2 * s**2 / 4 * float(np.sum(cv**2))


def one(job):
    g, d, w, s = job
    X, _ = make_data(g, d, 3000, s, w)
    f1 = floor_leading(X)
    return [dict(graph=g, d=d, w=w, seed=s, c=c, floor=f1 * c**4) for c in (10.0, 15.0, 20.0, 30.0, 50.0)]


if __name__ == "__main__":
    jobs = [(g, d, w, s) for g in ("ER2", "ER4", "SF4") for d in (30, 50) for w in (0.5, 1.0) for s in range(30, 80)]
    with ProcessPoolExecutor(6) as ex:
        out = [r for rs in ex.map(one, jobs) for r in rs]
    with open(sys.argv[1], "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(len(out))
