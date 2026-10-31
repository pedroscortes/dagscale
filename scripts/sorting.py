# ruff: noqa: N803, N806
"""Order-based baselines: sortnregress and R2-sortnregress (Reisach et al. 2021, 2023)."""

from __future__ import annotations

import numpy as np


def regress_along(X, order):
    """Adaptive-lasso (BIC) regression of each node on its predecessors in `order`; returns W[source, target]."""
    from sklearn.linear_model import LassoLarsIC, LinearRegression

    d = X.shape[1]
    W = np.zeros((d, d))
    for k in range(1, d):
        cov, tgt = order[:k], order[k]
        w = np.abs(LinearRegression().fit(X[:, cov], X[:, tgt]).coef_)
        ll = LassoLarsIC(criterion="bic").fit(X[:, cov] * w, X[:, tgt])
        W[cov, tgt] = ll.coef_ * w
    return W


def r2_order(X):
    from sklearn.linear_model import LinearRegression

    d = X.shape[1]
    r2 = np.empty(d)
    for j in range(d):
        others = np.delete(np.arange(d), j)
        r2[j] = LinearRegression().fit(X[:, others], X[:, j]).score(X[:, others], X[:, j])
    return np.argsort(r2, kind="stable")


def topo_order(A):
    """Topological order of A[i, j] = 1 for j -> i."""
    d = A.shape[0]
    indeg = A.sum(1).astype(int)
    order, ready = [], [i for i in range(d) if indeg[i] == 0]
    while ready:
        j = ready.pop()
        order.append(j)
        for i in np.nonzero(A[:, j])[0]:
            indeg[i] -= 1
            if indeg[i] == 0:
                ready.append(i)
    return np.array(order)
