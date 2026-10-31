# ruff: noqa: E402, N806
"""Leak-law prediction of non-acyclic DAGMA outputs (Section 5.3, Table 3).

ER2/ER4/SF4, d 30/50, weight scale 0.5/1, seeds SEED_LO..SEED_HI-1, c in {10, 15, 20, 30, 50}.
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
from dagma_fit import Logged
from leaklaw import law_h
from simulate import make_data


def one(job):
    g, d, w, s, c = job
    X, A = make_data(g, d, 3000, s, w)
    X = X * c
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / Xc.shape[0]
    hl, selfloop = law_h(S, A)
    m = Logged()
    with contextlib.redirect_stderr(io.StringIO()):
        W = np.asarray(m.fit(X.copy(), lambda1=0.02, w_threshold=0.0))
    # dagma's _h ignores the slogdet sign: h is valid only if I - W o W is a nonsingular M-matrix
    sign = np.linalg.slogdet(np.eye(d) - W * W)[0]
    in_domain = bool(sign > 0 and np.all(np.linalg.inv(np.eye(d) - W * W) >= -1e-12))
    h_pkg = float(m._h(W)[0])
    return dict(
        in_domain=in_domain,
        graph=g,
        d=d,
        w=w,
        seed=s,
        c=c,
        h_law=hl,
        pred_infeasible=hl > 0.01,
        h_pkg=h_pkg,
        infeasible_raw=h_pkg > 0.01,
        infeasible=(h_pkg > 0.01) or not in_domain,
        pred_selfloop=selfloop >= 0.3,
        selfloop=bool(np.max(np.abs(np.diag(W))) >= 0.3),
        max_diag=float(np.max(np.abs(np.diag(W)))),
    )


if __name__ == "__main__":
    jobs = [
        (g, d, w, s, c)
        for g in ("ER2", "ER4", "SF4")
        for d in (30, 50)
        for w in (0.5, 1.0)
        for s in range(int(os.environ.get("SEED_LO", 30)), int(os.environ.get("SEED_HI", 40)))
        for c in (10.0, 15.0, 20.0, 30.0, 50.0)
    ]
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 18))) as ex:
        out = list(ex.map(one, sorted(jobs, key=lambda j: -j[1])))
    path = os.environ.get("OUT", "results/predict/predict.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    acc = np.mean([o["pred_infeasible"] == o["infeasible"] for o in out])
    sacc = np.mean([o["pred_selfloop"] == o["selfloop"] for o in out])
    print(f"fits {len(out)}; feasibility predicted correctly {acc:.3f}; self-loop predicted correctly {sacc:.3f}")
    for c in (10.0, 15.0, 20.0, 30.0, 50.0):
        x = [o for o in out if o["c"] == c]
        print(
            f"  c={c:g}: infeasible observed {np.mean([o['infeasible'] for o in x]):.2f} predicted "
            f"{np.mean([o['pred_infeasible'] for o in x]):.2f}; acc {np.mean([o['pred_infeasible'] == o['infeasible'] for o in x]):.2f}"
        )
    acc_raw = np.mean([o["pred_infeasible"] == o["infeasible_raw"] for o in out])
    print(
        f"out of domain: {np.mean([not o['in_domain'] for o in out]):.3f}; accuracy with the raw h only: {acc_raw:.3f}"
    )
