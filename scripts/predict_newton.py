# ruff: noqa: E402, N806
"""Leak-law prediction as in predict.py, with the Newton-CG path-follower and a free diagonal (Section 5.3, Table 3)."""

import csv
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from leaklaw import law_h
from newton_solver import run
from simulate import make_data

NC = dict(gram="qr", negcurv="step", stall_rtol=1e-13, ftol_early=1e-1, ftol_final=1e-2, max_newton=100)


def one(job):
    g, d, w, s, c = job
    try:
        X, A = make_data(g, d, 3000, s, w)
        X = X * c
        Xc = X - X.mean(0)
        S = Xc.T @ Xc / Xc.shape[0]
        hl, selfloop = law_h(S, A)
        W, _log = run(X.copy(), lambda1=0.02, free_diag=True, **NC)
        W = np.asarray(W)
        M = np.eye(d) - W * W
        sign, ld = np.linalg.slogdet(M)
        in_dom = bool(sign > 0 and np.all(np.linalg.inv(M) >= -1e-12))
        h = float(-ld) if in_dom else float("inf")
        return dict(
            graph=g,
            d=d,
            w=w,
            seed=s,
            c=c,
            h_law=hl,
            h_newton=h,
            in_domain=in_dom,
            pred_infeasible=hl > 0.01,
            infeasible=h > 0.01,
            ratio=h / hl if hl > 0 else float("nan"),
            pred_selfloop=selfloop >= 0.3,
            selfloop=bool(np.max(np.abs(np.diag(W))) >= 0.3),
        )
    except Exception as e:
        return dict(graph=g, d=d, w=w, seed=s, c=c, error=repr(e)[:150])


if __name__ == "__main__":
    lo, hi = int(os.environ.get("SEED_LO", 30)), int(os.environ.get("SEED_HI", 40))
    jobs = [
        (g, d, w, s, c)
        for g in ("ER2", "ER4", "SF4")
        for d in (30, 50)
        for w in (0.5, 1.0)
        for s in range(lo, hi)
        for c in (10.0, 15.0, 20.0, 30.0, 50.0)
    ]
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 18))) as ex:
        out = list(ex.map(one, sorted(jobs, key=lambda j: -j[1])))
    path = os.environ.get("OUT", "results/predict_newton/predict_newton.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(out)
    ok = [o for o in out if "error" not in o]
    acc = np.mean([o["pred_infeasible"] == o["infeasible"] for o in ok])
    fa = sum(o["pred_infeasible"] and not o["infeasible"] for o in ok)
    miss = sum(o["infeasible"] and not o["pred_infeasible"] for o in ok)
    print(f"fits {len(out)} errors {len(out) - len(ok)}; accuracy {acc:.3f}; false alarms {fa}; missed failures {miss}")
    print(f"self-loop predicted correctly {np.mean([o['pred_selfloop'] == o['selfloop'] for o in ok]):.3f}")
    for c in (10.0, 15.0, 20.0, 30.0, 50.0):
        x = [o for o in ok if o["c"] == c]
        r = [o["ratio"] for o in x if np.isfinite(o["ratio"])]
        print(
            f"  c={c:g}: acc {np.mean([o['pred_infeasible'] == o['infeasible'] for o in x]):.2f}; "
            f"median h_newton/h_law {np.median(r):.2f}; out of domain {np.mean([not o['in_domain'] for o in x]):.2f}"
        )
