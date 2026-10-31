# ruff: noqa: E402, N806
"""Scale window with diag(W) fixed at zero through DAGMA's exclude_edges (Table S8).

ER2/ER4/SF4, d 30/50, weight scale 0.5/1, n 3000, seeds 30-39, c in {1, ..., 1000}.
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
from methods import has_cycle
from simulate import make_data, project_dag

CS = (1.0, 10.0, 20.0, 30.0, 50.0, 100.0, 300.0, 1000.0)


def one(job):
    g, d, w, s = job
    X0, A = make_data(g, d, 3000, s, w)
    out = []
    for c in CS:
        X = X0 * c
        m = Logged()
        with contextlib.redirect_stderr(io.StringIO()):
            W = np.asarray(
                m.fit(
                    X.copy(),
                    lambda1=0.02,
                    w_threshold=0.0,
                    exclude_edges=tuple((j, j) for j in range(d)),
                )
            )
        M = np.eye(d) - W * W
        sign, ld = np.linalg.slogdet(M)
        in_dom = bool(sign > 0 and np.all(np.linalg.inv(M) >= -1e-12))
        h = float(-ld) if in_dom else float("nan")
        B = (np.abs(W.T) > 0.3).astype(int)
        np.fill_diagonal(B, 0)
        out.append(
            dict(
                graph=g,
                d=d,
                w=w,
                seed=s,
                c=c,
                in_domain=in_dom,
                h=h,
                non_acyclic=(not in_dom) or h > 0.01,
                cyclic=has_cycle(B),
                max_diag=float(np.max(np.abs(np.diag(W)))),
                proj_f1=scores(project_dag(W.T), A)["f1"],
            )
        )
    return out


if __name__ == "__main__":
    jobs = [(g, d, w, s) for g in ("ER2", "ER4", "SF4") for d in (30, 50) for w in (0.5, 1.0) for s in range(30, 40)]
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 8))) as ex:
        rows = [r for rs in ex.map(one, sorted(jobs, key=lambda j: -j[1])) for r in rs]
    path = os.environ.get("OUT", "results/masked_diagonal/masked_diagonal.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for c in CS:
        r = [x for x in rows if x["c"] == c]
        print(
            f"c={c:g}: non-acyclic {np.mean([x['non_acyclic'] for x in r]):.2f}  "
            f"thresholded cycle {np.mean([x['cyclic'] for x in r]):.2f}  projF1 {np.mean([x['proj_f1'] for x in r]):.2f}"
        )
