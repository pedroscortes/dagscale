# ruff: noqa: E402, N803, N806
"""DAGMA (linear, least squares) with an optional preconditioned Adam direction, and shared scoring helpers.

Arms: ref (package DAGMA: T 5, mu 1 -> 1e-4, s [1, .9, .8, .7, .6], lambda1 0.02, Adam lr 3e-4), ref_T<k> (k stages),
pc_abs<x> (Adam fed (Cov + x I)^-1 G inside the same central path).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import scipy.linalg as sla
from dagma.linear import DagmaLinear


class Logged(DagmaLinear):
    """DagmaLinear; direction="full" feeds Adam the preconditioned gradient."""

    def __init__(self, direction: str = "plain", damping: float = 10.0):
        super().__init__(loss_type="l2")
        self.direction, self.damping = direction, damping

    def fit(self, X, *args, s=None, **kwargs):
        # DagmaLinear.fit widens its default s list in place on retries; pass a fresh list
        return super().fit(X, *args, s=list(s) if s is not None else [1.0, 0.9, 0.8, 0.7, 0.6], **kwargs)

    def minimize(self, W, mu, max_iter, s, lr, tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        if self.direction == "plain":
            return super().minimize(W, mu, max_iter, s, lr, tol, beta_1, beta_2, pbar)
        return self._minimize_dir(W, mu, max_iter, s, lr, tol, beta_1, beta_2)

    def _minimize_dir(self, W, mu, max_iter, s, lr, tol, beta_1, beta_2):
        """DagmaLinear.minimize with the Adam input P G, P = (Cov + damping I)^-1."""
        P = sla.inv(self.cov + self.damping * self.Id)
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        grad = np.zeros_like(W)
        for it in range(1, max_iter + 1):
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < 0):
                if it == 1 or s <= 0.9:
                    return W, False
                W += lr * grad
                lr *= 0.5
                if lr <= 1e-16:
                    return W, True
                W -= lr * grad
                M = sla.inv(s * self.Id - W * W) + 1e-16
            G_score = -mu * self.cov @ (self.Id - W)
            G_pen = mu * self.lambda1 * np.sign(W) + 2 * W * M.T
            grad = self._adam_update(P @ (G_score + G_pen), it, beta_1, beta_2)
            W -= lr * grad
            if it % self.checkpoint == 0 or it == max_iter:
                obj_new, _, _ = self._func(W, mu, s)
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    break
                obj_prev = obj_new
        return W, True


def parse_arm(arm: str):
    """(model kwargs, fit kwargs) for an arm name."""
    if arm == "ref":
        return {}, {}
    if arm.startswith("ref_T"):
        return {}, {"T": int(arm[5:])}
    if arm.startswith("pc_abs"):
        return {"direction": "full", "damping": float(arm[6:])}, {}
    raise ValueError(arm)


def diagnostics(X, A):
    """(kappa of Cov, kappa of Corr, varsortability)."""
    cov = np.cov(X, rowvar=False)
    ev = np.linalg.eigvalsh(cov)
    sd = np.sqrt(np.diag(cov))
    R = cov / np.outer(sd, sd)
    evr = np.linalg.eigvalsh(R)
    var = np.diag(cov)
    ch, pa = np.nonzero(A)  # A[i, j] = 1: j -> i
    vs = float(np.mean(var[ch] > var[pa])) if len(ch) else float("nan")
    k = lambda e: float(e[-1] / e[0]) if e[0] > 0 else float("inf")
    return k(ev), k(evr), vs


def scores(W_geo, A):
    """Thresholded edge metrics, AUPRC and precision at |E| of |W_geo|; W_geo is W[target, source]."""
    from dagscale.evaluation import (
        edge_confusion,
        structural_hamming_distance,
        threshold_weights,
    )

    out = {}
    for t in (0.1, 0.3, 0.5):
        L = threshold_weights(W_geo, threshold=t)
        out[f"shd_{t}"] = structural_hamming_distance(L, A)
        if t == 0.3:
            tp, fp, fn = edge_confusion(L, A)
            out.update(
                n_learned=int(L.sum()),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=round(tp / max(tp + fp, 1), 4),
                recall=round(tp / max(tp + fn, 1), 4),
                f1=round(2 * tp / max(2 * tp + fp + fn, 1), 4),
            )
    d = A.shape[0]
    mask = ~np.eye(d, dtype=bool)
    s, y = np.abs(W_geo)[mask], A[mask].astype(bool)
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    tp_cum = np.cumsum(ys)
    prec = tp_cum / np.arange(1, len(ys) + 1)
    out["auprc"] = round(float(np.sum(prec[ys]) / max(ys.sum(), 1)), 4)
    out["prec_at_E"] = round(float(tp_cum[int(y.sum()) - 1] / max(int(y.sum()), 1)), 4) if y.sum() else float("nan")
    return out
