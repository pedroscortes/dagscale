# ruff: noqa: E402, N803, N806
"""Causal Chambers light tunnel (lt_interventions_standard_v1, uniform_reference) in raw, log and standardized units
(Section 5.7, Table 6). Methods: those of methods.py plus the sorting baselines.
"""

from __future__ import annotations

import csv
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from leaklaw import law_h, truth_free_floor
from methods import METHODS, fit, measure
from sorting import r2_order, regress_along

ROOT = os.environ.get("CC_ROOT", "data/causal_chambers")


def load():
    import causalchamber.datasets as ds
    from causalchamber.ground_truth import graph

    os.makedirs(ROOT, exist_ok=True)
    d = ds.Dataset(name="lt_interventions_standard_v1", root=ROOT, download=True)
    df = d.get_experiment(name="uniform_reference").as_pandas_dataframe()
    G = graph(chamber="lt", configuration="standard")
    cols = [c for c in G.columns if c in df.columns and df[c].std() > 0]
    X = df[cols].to_numpy(dtype=np.float64)
    Gm = G.loc[cols, cols].to_numpy()  # Gm[i, j] = 1 for i -> j
    A = Gm.T.astype(int)  # A[child, parent]
    return X, A, cols


def variant(X, name):
    if name == "log":
        Y = X.copy()
        nonneg = (X >= 0).all(0)
        Y[:, nonneg] = np.log1p(X[:, nonneg])
        return Y
    if name == "std":
        return (X - X.mean(0)) / X.std(0)
    return X


def job(args):
    vname, method = args
    try:
        X0, A, _ = load()
        X = variant(X0, vname)
        Xc = X - X.mean(0)
        S = Xc.T @ Xc / Xc.shape[0]
        t0 = time.time()
        if method == "sortnregress":
            W = regress_along(Xc, np.argsort(Xc.var(0), kind="stable"))
        elif method == "r2sortnregress":
            W = regress_along(Xc, r2_order(Xc))
        else:
            W = fit(method, X)
        r = measure(
            np.asarray(W),
            A,
            X,
            method,
            part="chambers",
            graph="lt_standard",
            d=A.shape[0],
            w="",
            seed="",
            variant=vname,
        )
        var = np.diag(S)
        r.update(
            wall=round(time.time() - t0, 1),
            floor=truth_free_floor(S),
            floor_leading=1e-8 * 0.09 * float(np.sum((1.0 / np.diag(np.linalg.inv(S))) ** 2)),
            h_law=law_h(S, A)[0],
            spread=float(np.log10(var.max() / var.min())),
            lmax=float(np.linalg.eigvalsh(S)[-1]),
        )
        return r
    except Exception as e:
        return dict(part="chambers", variant=vname, method=method, error=repr(e)[:300])


def main():
    X, A, cols = load()
    print(f"variables ({len(cols)}): {cols}")
    print(f"edges in ground truth (restricted): {int(A.sum())}; n = {X.shape[0]}")
    default = ",".join((*METHODS, "sortnregress", "r2sortnregress"))
    methods = os.environ.get("METHODS", default).split(",")
    jobs = [(v, m) for v in ("raw", "log", "std") for m in methods]
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 27))) as ex:
        rows = list(ex.map(job, jobs))
    out = os.environ.get("OUT", "results/light_tunnel/light_tunnel.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            {
                k: r.get(k)
                for k in (
                    "variant",
                    "method",
                    "error",
                    "floor",
                    "h_law",
                    "h_logdet",
                    "in_domain",
                    "cyclic",
                    "proj_f1",
                    "auprc_std",
                    "wall",
                )
            }
        )
    print("DONE", len(rows), "errors", sum(1 for r in rows if r.get("error")))


if __name__ == "__main__":
    main()
