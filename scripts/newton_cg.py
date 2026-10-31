# ruff: noqa: E402, N803, N806, N802
"""Newton-CG path-follower against DAGMA and pc_abs10 (Sections 5.6 and S6).

Cells: ER4 weights x2 and SF4 x4, d 30/50, seeds 30-32, plus SF4 x4 d 100 seeds 30-31. Arms (ARMS): covariance or
square-root (qr) loss, negative-curvature steps (nc), warm starts from sortnregress (snr) or the true-parents OLS DAG.
"""

from __future__ import annotations

import argparse
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
import scipy.linalg as sla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newton_solver import run

NC = dict(gram="qr", negcurv="step", stall_rtol=1e-13, ftol_early=1e-1, ftol_final=1e-2, max_newton=100)
ARMS = {
    "nt_cov": dict(),
    "nt_qr": dict(gram="qr"),
    "nt_nc": NC,
    "nt_nc_snr": dict(NC, W0="snr"),
    "nt_nc_snrlast": dict(NC, W0="snr", mu_init=1e-4, T=1, s=[0.6], max_newton=200),
    "nt_nc_truthlast": dict(NC, W0="truth", mu_init=1e-4, T=1, s=[0.6], max_newton=200),
}


def fit_dagma(X, arm):
    from dagma_fit import Logged

    m = Logged() if arm == "ref" else Logged(direction="full", damping=10.0)
    t0 = time.time()
    with contextlib.redirect_stderr(io.StringIO()):
        W = m.fit(X.copy(), lambda1=0.02, w_threshold=0.0)
    return np.asarray(W), time.time() - t0


def one_job(job):
    graph, d, seed, wscale, arm = job
    from dagma_fit import scores
    from simulate import make_data, project_dag
    from sorting import regress_along

    try:
        X, A = make_data(graph, d, 3000, seed, wscale)
        Xc = X - X.mean(0)
        n = Xc.shape[0]
        R = sla.qr(Xc, mode="r", check_finite=False)[0][:d] / np.sqrt(n)
        ev = np.linalg.svd(R, compute_uv=False)
        kap = (ev[0] / ev[-1]) ** 2 if ev[-1] > 0 else np.inf

        def F(W):
            return 0.5 * np.sum((R @ (np.eye(d) - W)) ** 2) + 0.02 * np.abs(W).sum()

        W_pa = np.zeros((d, d))
        for i in range(d):
            pa = np.nonzero(A[i])[0]
            if len(pa):
                W_pa[pa, i] = np.linalg.lstsq(Xc[:, pa], Xc[:, i], rcond=None)[0]
        W_snr = regress_along(Xc, np.argsort(Xc.var(0), kind="stable"))
        Ft = F(W_pa)
        extra = {}
        t0, c0 = time.time(), time.process_time()
        if arm in ("ref", "pc_abs10"):
            W, wall = fit_dagma(X, arm)
        else:
            kw = dict(ARMS[arm])
            if kw.get("W0") == "snr":
                kw["W0"] = W_snr
            elif kw.get("W0") == "truth":
                kw["W0"] = W_pa
            W, log = run(X, **kw)
            wall = time.time() - t0
            st = log["stages"]
            extra = dict(
                newton=log["newton"],
                cg=log["cg"],
                stage_newton="/".join(str(s["newton"]) for s in st),
                status="/".join(s["status"] for s in st),
                nc_steps="/".join(str(s.get("nc_steps", 0)) for s in st),
                s_final="/".join(f"{s:g}" for s in log["s_final"]),
            )
        cpu = time.process_time() - c0
        sign, ld = np.linalg.slogdet(np.eye(d) - W * W)
        h = -ld if sign > 0 else np.inf
        # DAGMA's last-stage objective (mu 1e-4, s 0.6); h_s = 0 on a DAG, so Phi(truth) = mu F(truth)
        s6, ld6 = np.linalg.slogdet(0.6 * np.eye(d) - W * W)
        phi_last = 1e-4 * F(W) + (-ld6 + d * np.log(0.6) if s6 > 0 else np.inf)
        Wg = np.asarray(W).T
        sc = scores(Wg, A)
        pj = scores(project_dag(Wg), A)
        sc_snr = scores(W_snr.T, A)
        row = dict(
            graph=graph,
            d=d,
            seed=seed,
            wscale=wscale,
            log10_kappa=round(float(np.log10(kap)), 2),
            arm=arm,
            wall=round(wall, 2),
            cpu=round(cpu, 2),
            F=F(W),
            F_truth=Ft,
            rF=F(W) / Ft,
            rF_snr=F(W_snr) / Ft,
            rPhi=phi_last / (1e-4 * Ft),
            h=h,
            feasible=int(h <= 0.01),
            f1=sc["f1"],
            pf1=pj["f1"],
            pshd=pj["shd_0.3"],
            f1_snr=sc_snr["f1"],
            **extra,
        )
    except Exception as e:
        row = dict(
            graph=graph,
            d=d,
            seed=seed,
            wscale=wscale,
            arm=arm,
            error=f"{type(e).__name__}: {e}"[:200],
        )
    return row


FIELDS = [
    "graph",
    "d",
    "seed",
    "wscale",
    "log10_kappa",
    "arm",
    "wall",
    "cpu",
    "F",
    "F_truth",
    "rF",
    "rF_snr",
    "rPhi",
    "h",
    "feasible",
    "f1",
    "pf1",
    "pshd",
    "f1_snr",
    "newton",
    "cg",
    "stage_newton",
    "status",
    "nc_steps",
    "s_final",
    "error",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/newton_cg/newton_cg.csv")
    a = ap.parse_args()
    arms = ["ref", "pc_abs10", *ARMS]
    cells = [(g, d, s, c) for d in (30, 50) for g, c in (("ER4", 2.0), ("SF4", 4.0)) for s in (30, 31, 32)]
    cells += [("SF4", 100, s, 4.0) for s in (30, 31)]
    cost = {"ref": 6, "pc_abs10": 6}
    jobs = sorted(((*c, arm) for c in cells for arm in arms), key=lambda j: -j[1] * cost.get(j[4], 1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh, ProcessPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in ex.map(one_job, jobs):
            w.writerow(r)
            fh.flush()
            print(
                {
                    k: (f"{v:.4g}" if isinstance(v, float) else v)
                    for k, v in r.items()
                    if k
                    in (
                        "graph",
                        "d",
                        "seed",
                        "arm",
                        "wall",
                        "rF",
                        "rF_snr",
                        "rPhi",
                        "h",
                        "pf1",
                        "stage_newton",
                        "status",
                        "error",
                    )
                },
                flush=True,
            )
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
