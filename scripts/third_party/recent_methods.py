# ruff: noqa
"""TOPO and log-likelihood DAGMA via dagrad (commit 4ce4c5a, not redistributed), and our re-implementation of the
analytic DAG constraints of Zhang et al. (2025) on DAGMA's path-following optimiser. All return W[source, target].
"""

import numpy as np


def fit_topo(X: np.ndarray, seed: int = 0, w_threshold: float = 0.3) -> np.ndarray:
    """TOPO (Deng et al. 2023): least-squares score, random initial order, sklearn inner solver (dagrad defaults)."""
    import dagrad

    np.random.seed(seed)
    X = np.asarray(X, dtype=np.float64)
    W = dagrad.dagrad(
        X,
        method="topo",
        model="linear",
        loss_fn="l2",
        reg="none",
        optimizer="sklearn",
        compute_lib="numpy",
        device="cpu",
    )
    W = np.array(W, dtype=float)
    W[np.abs(W) < w_threshold] = 0.0
    return W


def fit_dagma_logll(
    X: np.ndarray,
    lam0: float = 0.4,
    delta0: float = 0.2,
    decay: float = 0.8,
    n_steps: int = 8,
    w_threshold: float = 0.3,
    l1_init: float = 0.02,
    warm: str = "ls",
) -> np.ndarray:
    """Log-likelihood DAGMA with quasi-MCP (Deng et al. 2024), (lam, delta) decayed until the NLL rises."""
    from dagrad.method.dagma import dagma_linear_numpy
    from dagrad.score.score import loss_fn as _lf
    from dagma.linear import DagmaLinear

    X = np.asarray(X, dtype=np.float64)
    if warm == "zero":
        W = np.zeros((X.shape[1], X.shape[1]))
    else:
        Xw = X
        if warm == "nnorm":
            Xw = X / np.sqrt(np.median(1.0 / np.diag(np.linalg.inv(np.cov(X, rowvar=False)))))
        W = DagmaLinear(loss_type="l2").fit(Xw.copy(), lambda1=l1_init, w_threshold=0.0, s=[1.0, 0.9, 0.8, 0.7, 0.6])
    nll = lambda W_: _lf.logll_loss(W_, X)[0]
    # decay loop written out: dagrad's tuning_method="decay" rejects an initialization
    best, best_nll = W.copy(), np.inf
    for k in range(n_steps):
        W = dagma_linear_numpy(
            X=X,
            loss_fn="logll",
            h_fn="h_logdet_sq",
            reg="mcp",
            optimizer="adam",
            lambda1=lam0 * decay**k,
            gamma=delta0 * decay**k,
            w_threshold=0.0,
            initialization=W,
        )
        cur = nll(W)
        if cur > best_nll:
            break
        best, best_nll = W.copy(), cur
    W = np.array(best, dtype=float)
    W[np.abs(W) < w_threshold] = 0.0
    return W


def fit_analytic(
    X: np.ndarray,
    order: int = 2,
    lambda1: float = 0.1,
    T: int = 5,
    mu_init: float = 1.0,
    mu_factor: float = 0.1,
    s=(1.0, 0.9, 0.8, 0.7, 0.6),
    warm_iter: int = 30000,
    max_iter: int = 60000,
    lr: float = 3e-4,
    w_threshold: float = 0.3,
    checkpoint: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Analytic constraint of order k >= 2 (Zhang et al. 2025) in DAGMA's path-following optimiser."""
    import scipy.linalg as sla

    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    n, d = X.shape
    I = np.eye(d)
    cov = X.T @ X / n
    s = list(s) + [s[-1]] * max(0, T - len(s))
    k = int(order)

    def h_and_grad(W, si):  # h = s^(k-1) tr N^(k-1) - d, N = (sI - W o W)^-1
        B = W * W
        N = sla.inv(si * I - B)
        Nk1 = np.linalg.matrix_power(N, k - 1)
        h = si ** (k - 1) * np.trace(Nk1) - d
        GB = (k - 1) * si ** (k - 1) * (Nk1 @ N).T
        return h, 2 * W * GB, N

    def obj(W, mu, si):
        R = I - W
        score = 0.5 * np.trace(R.T @ cov @ R) + lambda1 * np.abs(W).sum()
        return mu * score + h_and_grad(W, si)[0]

    def minimize(W, mu, iters, si, lr):
        m1 = v1 = 0.0
        b1, b2 = 0.99, 0.999
        prev = 1e16
        grad = None
        for it in range(1, iters + 1):
            _, Gh, N = h_and_grad(W, si)
            while np.any(N + 1e-16 < 0):
                if it == 1 or si <= 0.9:
                    return W, False
                W += lr * grad
                lr *= 0.5
                if lr <= 1e-16:
                    return W, True
                W -= lr * grad
                _, Gh, N = h_and_grad(W, si)
            G = -mu * cov @ (I - W) + mu * lambda1 * np.sign(W) + Gh
            m1 = b1 * m1 + (1 - b1) * G
            v1 = b2 * v1 + (1 - b2) * G**2
            grad = (m1 / (1 - b1**it)) / (np.sqrt(v1 / (1 - b2**it)) + 1e-8)
            W = W - lr * grad
            if it % checkpoint == 0 or it == iters:
                o = obj(W, mu, si)
                if abs((prev - o) / prev) <= tol:
                    break
                prev = o
        return W, True

    W = np.zeros((d, d))
    mu = mu_init
    for t in range(T):
        lr_t, ok = lr, False
        it = max_iter if t == T - 1 else warm_iter
        while not ok:
            Wt, ok = minimize(W.copy(), mu, it, s[t], lr_t)
            if not ok:
                lr_t *= 0.5
                s[t] += 0.1
        W = Wt
        mu *= mu_factor
    W[np.abs(W) < w_threshold] = 0.0
    return W
