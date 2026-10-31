# ruff: noqa: E402, N803, N806
"""Comparison of methods on synthetic and Sachs data (Section 5.4, Fig. 4, Tables 4, 6 and S4, Section S5).

METHODS: keys of fit(). VARIANTS: global multiplier (0.01, 1, 100), resc (column j times 10^U(-2, 2)) or std.
Synthetic data: ER2/ER4/SF4, d 30, weight scale 0.5/1/2, n 3000, seeds SEED_LO..SEED_HI-1.
"""

from __future__ import annotations

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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "third_party"))
from dagma_fit import Logged, scores
from simulate import make_data, project_dag

METHODS = ("dagma", "dagma_nnorm", "notears", "golem_ev", "golem_nv", "colide_ev", "colide_nv")
RECENT = (
    "dagma_logll_zero",
    "dagma_logll_nnorm",
    "topo",
    "dagma_logll",
    "analytic2",
)


def nnorm(X):
    S = np.cov(X, rowvar=False)
    return X / np.sqrt(np.median(1.0 / np.diag(np.linalg.inv(S))))


def fit(method, X):
    """W[source, target] of a method on X."""
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        if method in ("dagma", "dagma_nnorm"):
            Xin = nnorm(X) if method == "dagma_nnorm" else X
            return np.asarray(Logged().fit(Xin.copy(), lambda1=0.02, w_threshold=0.0))
        if method == "notears":
            from notears.linear import notears_linear

            return np.asarray(notears_linear(X - X.mean(0), lambda1=0.02, loss_type="l2", w_threshold=0.0))
        if method in ("golem_ev", "golem_nv"):
            from dagscale.golem import fit_golem

            return fit_golem(X, equal_variances=(method == "golem_ev"), seed=0, timeout=7200)
        if method in ("colide_ev", "colide_nv"):
            from colide_model import colide_ev, colide_nv

            M = colide_ev() if method == "colide_ev" else colide_nv()
            W, _ = M.fit(
                X.copy(),
                lambda1=0.05,
                T=4,
                s=[1.0, 0.9, 0.8, 0.7],
                warm_iter=2e4,
                max_iter=7e4,
                lr=0.0003,
            )
            return np.asarray(W)
        if method in ("sortnregress", "r2sortnregress"):
            from sorting import r2_order, regress_along

            Xc = X - X.mean(0)
            order = np.argsort(Xc.var(0), kind="stable") if method == "sortnregress" else r2_order(Xc)
            return np.asarray(regress_along(Xc, order))
        if method in RECENT:
            import recent_methods as rm

            if method == "topo":
                return np.asarray(rm.fit_topo(X, seed=0, w_threshold=0.0))
            if method == "dagma_logll":
                return np.asarray(rm.fit_dagma_logll(X, w_threshold=0.0))
            if method in ("dagma_logll_zero", "dagma_logll_nnorm"):
                return np.asarray(rm.fit_dagma_logll(X, w_threshold=0.0, warm=method.rsplit("_", 1)[1]))
            return np.asarray(rm.fit_analytic(X, order=2, w_threshold=0.0))
    raise ValueError(method)


def has_cycle(B):
    d = B.shape[0]
    indeg = B.sum(1).astype(int)
    ready = [i for i in range(d) if indeg[i] == 0]
    seen = 0
    while ready:
        j = ready.pop()
        seen += 1
        for i in np.nonzero(B[:, j])[0]:
            indeg[i] -= 1
            if indeg[i] == 0:
                ready.append(i)
    return seen < d


def measure(W, A, X, method, **meta):
    from scipy.linalg import expm

    d = W.shape[0]
    Wg = W.T
    sd = X.std(0)
    M = np.eye(d) - W * W
    sign, ld = np.linalg.slogdet(M)
    in_dom = bool(sign > 0 and np.all(np.linalg.inv(M) >= -1e-12))
    B = (np.abs(Wg) > 0.3).astype(int)
    np.fill_diagonal(B, 0)
    return dict(
        **meta,
        method=method,
        h_logdet=float(-ld) if in_dom else float("nan"),
        in_domain=in_dom,
        h_expm=float(np.trace(expm(np.minimum(W * W, 50.0))) - d),
        cyclic=has_cycle(B),
        proj_f1=scores(project_dag(Wg), A)["f1"],
        auprc_std=scores(Wg * sd[None, :] / sd[:, None], A)["auprc"],
        n_edges=int(B.sum()),
    )


def synth_job(job):
    g, w, s, variant, method = job
    try:
        X, A = make_data(g, 30, 3000, s, w)
        if variant == "resc":
            X = X * 10.0 ** np.random.default_rng(s + 101).uniform(-2, 2, X.shape[1])
        elif variant == "std":
            X = (X - X.mean(0)) / X.std(0)
        else:
            X = X * float(variant)
        t0 = time.time()
        W = fit(method, X)
        r = measure(W, A, X, method, part="synth", graph=g, d=30, w=w, seed=s, variant=variant)
        r["wall"] = round(time.time() - t0, 1)
        return r
    except Exception as e:
        return dict(
            part="synth",
            graph=g,
            d=30,
            w=w,
            seed=s,
            variant=variant,
            method=method,
            error=repr(e)[:200],
        )


def sachs_job(job):
    dname, variant, method = job
    from sachs import COND1, EDGES, POOLED, load

    try:
        X = load(COND1 if dname == "cond1_853" else POOLED)
        if variant == "log":
            X = np.log(np.maximum(X, 1e-3))
        elif variant == "std":
            X = (X - X.mean(0)) / X.std(0)
        A = np.zeros((11, 11), dtype=int)
        for i, j in EDGES:
            A[j, i] = 1
        t0 = time.time()
        W = fit(method, X)
        r = measure(W, A, X, method, part="sachs", graph=dname, d=11, w="", seed="", variant=variant)
        r["wall"] = round(time.time() - t0, 1)
        return r
    except Exception as e:
        return dict(part="sachs", graph=dname, d=11, variant=variant, method=method, error=repr(e)[:200])


def main():
    lo, hi = int(os.environ.get("SEED_LO", 30)), int(os.environ.get("SEED_HI", 40))
    workers = int(os.environ.get("WORKERS", 18))
    out = os.environ.get("OUT", "results/methods/methods.csv")
    methods = os.environ.get("METHODS", ",".join(METHODS)).split(",")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sjobs = [(d, v, m) for d in ("cond1_853", "pooled_7466") for v in ("raw", "log", "std") for m in methods]
    jobs = [
        (g, w, s, v, m)
        for g in ("ER2", "ER4", "SF4")
        for w in (0.5, 1.0, 2.0)
        for s in range(lo, hi)
        for v in os.environ.get("VARIANTS", "0.01,1,100,resc").split(",")
        for m in methods
    ]
    cost = {"notears": 5, "golem_nv": 4, "golem_ev": 2}
    jobs.sort(key=lambda j: -cost.get(j[4], 1))
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(sachs_job, sjobs):
            rows.append(r)
        for i, r in enumerate(ex.map(synth_job, jobs)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
                _write(out, rows)
    _write(out, rows)
    print("DONE", len(rows), "errors", sum(1 for r in rows if r.get("error")), flush=True)


def _write(out, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
