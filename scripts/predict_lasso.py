# ruff: noqa: E402, N806, SIM115
"""Leak law at the order-restricted lasso fit (true order) for the fits of predict.py and predict_newton.py."""

import csv
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from leaklaw import law_h_precise
from simulate import make_data
from sorting import topo_order


def pred(key):
    g, d, w, s, c = key
    X, A = make_data(g, int(d), 3000, int(s), float(w))
    X = X * float(c)
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / Xc.shape[0]
    return key, law_h_precise(S, topo_order(A))


if __name__ == "__main__":
    files = {
        "package": [
            "results/predict/predict.csv",
            "results/predict/predict_s40_79.csv",
        ],
        "newton": ["results/predict_newton/predict_newton.csv"],
    }
    rows = {k: [r for f in v for r in csv.DictReader(open(f)) if not r.get("error")] for k, v in files.items()}
    keys = sorted({(r["graph"], r["d"], r["w"], r["seed"], r["c"]) for v in rows.values() for r in v})
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 18))) as ex:
        P = dict(ex.map(pred, keys))
    out = []
    for solver, R in rows.items():
        hk = "h_newton" if solver == "newton" else "h_pkg"
        ok = np.array(
            [(P[(r["graph"], r["d"], r["w"], r["seed"], r["c"])][0] > 0.01) == (r["infeasible"] == "True") for r in R]
        )
        old = np.mean([(r["pred_infeasible"] == "True") == (r["infeasible"] == "True") for r in R])
        fa = sum(
            (P[(r["graph"], r["d"], r["w"], r["seed"], r["c"])][0] > 0.01) and r["infeasible"] == "False" for r in R
        )
        ms = sum(
            (P[(r["graph"], r["d"], r["w"], r["seed"], r["c"])][0] <= 0.01) and r["infeasible"] == "True" for r in R
        )
        ratio = [float(r[hk]) / P[(r["graph"], r["d"], r["w"], r["seed"], r["c"])][0] for r in R if float(r["c"]) <= 15]
        print(
            f"{solver}: n={len(R)} accuracy first-order {old:.3f} -> precise {ok.mean():.3f}; false alarms {fa}, "
            f"missed {ms}; median h/precise-law at c<=15: {np.median(ratio):.2f}"
        )
        for r in R:
            k = (r["graph"], r["d"], r["w"], r["seed"], r["c"])
            out.append(
                dict(
                    solver=solver,
                    graph=k[0],
                    d=k[1],
                    w=k[2],
                    seed=k[3],
                    c=k[4],
                    h=r[hk],
                    h_law_precise=P[k][0],
                    infeasible=r["infeasible"],
                )
            )
    os.makedirs("results/predict_lasso", exist_ok=True)
    with open("results/predict_lasso/predict_lasso.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
