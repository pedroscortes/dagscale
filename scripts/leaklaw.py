# ruff: noqa: N803, N806, E741
"""Leak law (Section 3.4): predicted h_1 of a converged DAGMA run (mu 1e-4, s 0.6) from the least-squares fit on the
true support (law_h) or the order-restricted lasso fit (law_h_precise), and the graph-free floor (truth_free_floor).
"""

from __future__ import annotations

import numpy as np


def ols_on_support(S, A):
    """Least-squares W0[source, target] on the support A[child, parent] and the residual variances."""
    d = S.shape[0]
    W = np.zeros((d, d))
    sig2 = np.zeros(d)
    for j in range(d):
        pa = np.nonzero(A[j])[0]
        if len(pa):
            b = np.linalg.solve(S[np.ix_(pa, pa)], S[pa, j])
            W[pa, j] = b
            sig2[j] = S[j, j] - S[j, pa] @ b
        else:
            sig2[j] = S[j, j]
    return W, sig2


def predicted_leak(W0, sig2, s=0.6, mu=1e-4):
    """First-order stage minimizer W0 + T; T is nonzero on self-loops and back-edges (c in {p} U desc(p))."""
    d = W0.shape[0]
    Ainv = np.linalg.inv(np.eye(d) - W0)
    Q = s * np.linalg.inv(s * np.eye(d) - W0 * W0)
    reach = np.linalg.inv(np.eye(d) - (W0 != 0).astype(float) * 1e-3) > 1e-12  # reach[p, c]: p = c or p ->...-> c
    P, C = np.nonzero(reach)
    T = np.zeros((d, d))
    T[C, P] = mu * s * Ainv[P, C] * sig2[P] / (2 * Q[P, C])
    return W0 + T, T


def h1(W):
    sign, ld = np.linalg.slogdet(np.eye(W.shape[0]) - W * W)
    return -ld if sign > 0 else np.inf


def law_h(S, A, s=0.6, mu=1e-4):
    W0, sig2 = ols_on_support(S, A)
    Wp, T = predicted_leak(W0, sig2, s, mu)
    return h1(Wp), float(np.max(np.diag(T)))


def truth_free_floor(S, s=0.6, mu=1e-4):
    """Self-loop part of the leak with sigma_j^2 replaced by Var(X_j | X_-j) <= sigma_j^2 (a lower bound)."""
    cv = 1.0 / np.diag(np.linalg.inv(S))
    w = mu * s * cv / 2
    return float(-np.sum(np.log1p(-np.minimum(w * w, 1 - 1e-12))))


def _lasso_cov(Spp, spj, lam, iters=2000, tol=1e-12):
    """argmin_w 0.5 w' Spp w - spj' w + lam |w|_1 by coordinate descent."""
    k = len(spj)
    w = np.zeros(k)
    diag = np.diag(Spp).copy()
    for _ in range(iters):
        delta = 0.0
        for i in range(k):
            r = spj[i] - Spp[i] @ w + diag[i] * w[i]
            new = np.sign(r) * max(abs(r) - lam, 0.0) / diag[i]
            delta = max(delta, abs(new - w[i]))
            w[i] = new
        if delta < tol * (1 + np.abs(w).max()):
            break
    return w


def law_h_precise(S, order, lambda1=0.02, s=0.6, mu=1e-4):
    """(h_1, largest self-loop) predicted at the lasso fit restricted to DAGs compatible with `order`."""
    d = S.shape[0]
    W0 = np.zeros((d, d))
    pos = np.empty(d, dtype=int)
    pos[np.asarray(order)] = np.arange(d)
    for j in range(d):
        P = np.array([i for i in range(d) if pos[i] < pos[j]], dtype=int)
        if len(P):
            W0[P, j] = _lasso_cov(S[np.ix_(P, P)], S[P, j], lambda1)
    I = np.eye(d)
    G = S @ (I - W0)  # G[c, p] = cov(X_c, residual_p)
    Q = s * np.linalg.inv(s * I - W0 * W0)
    Q1 = np.linalg.inv(I - W0 * W0)
    reach = np.linalg.inv(I - (W0 != 0).astype(float) * 1e-3) > 1e-12  # reach[p, c]
    P_, C_ = np.nonzero(reach)  # includes p == c (self-loops)
    g = G[C_, P_]
    T = s * np.sign(g) * np.maximum(np.abs(g) - lambda1, 0.0) / (2 * Q[P_, C_])
    h1 = mu**2 * float(np.sum(Q1[P_, C_] * T**2))
    selfloop = float(np.max(np.abs(T[P_ == C_]))) * mu if np.any(P_ == C_) else 0.0
    return h1, selfloop
