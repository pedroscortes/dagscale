# ruff: noqa: N803, N806
"""Newton-CG path-follower for DAGMA's least-squares estimator (Sections 5.3, 5.6, 5.7 and S6).

Same central path as dagma.linear.DagmaLinear: stage k minimizes mu_k (0.5 tr((I-W)^T C (I-W)) + lambda1 |W|_1)
+ h_s(W), h_s(W) = -log det(sI - W o W) + d log s. Inner solver: truncated block-preconditioned CG with
Levenberg-Marquardt damping, Armijo backtracking inside the M-matrix domain, l1 smoothed as sqrt(w^2 + eps^2) - eps.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla


class Stage:
    """Objective of one (mu, s) stage; R (R^T R = C) evaluates the loss in square-root form."""

    def __init__(self, C, mu, s, lambda1, eps=1e-4, free_diag=False, R=None):
        self.C, self.mu, self.s, self.lam, self.eps, self.R = C, mu, s, lambda1, eps, R
        self.d = C.shape[0]
        self.Id = np.eye(self.d)
        self.mask = np.ones((self.d, self.d)) if free_diag else 1.0 - self.Id

    def barrier(self, W):
        """(h_s, B^-1), or (inf, None) when B = sI - W o W is not a nonsingular M-matrix."""
        B = self.s * self.Id - W * W
        try:
            lu, piv = sla.lu_factor(B, check_finite=False)
        except (ValueError, np.linalg.LinAlgError):
            return np.inf, None
        dU = np.diag(lu)
        nswap = int(np.sum(piv != np.arange(self.d)))
        sign = (-1) ** nswap * np.prod(np.sign(dU))
        if not np.all(np.isfinite(dU)) or sign <= 0 or np.any(dU == 0):
            return np.inf, None
        Binv = sla.lu_solve((lu, piv), self.Id, check_finite=False)
        # a Z-matrix is a nonsingular M-matrix iff its inverse is >= 0 (up to rounding)
        if not np.all(np.isfinite(Binv)) or Binv.min() < -1e-12 * max(1.0, np.abs(Binv).max()):
            return np.inf, None
        h = -np.sum(np.log(np.abs(dU))) + self.d * np.log(self.s)
        # h_s >= 0 on the domain; a negative value means the tests passed a point numerically outside it
        if not np.isfinite(h) or h < -1e-9 * self.d:
            return np.inf, None
        return h, Binv

    def loss(self, W):
        A = self.Id - W
        if self.R is not None:
            RA = self.R @ A
            return 0.5 * np.einsum("ab,ab->", RA, RA), -(self.R.T @ RA)
        CA = self.C @ A
        return 0.5 * np.einsum("ab,ab->", A, CA), -CA

    def l1_value(self, W, true=False):
        if not true:
            return self.lam * np.sum((np.sqrt(W * W + self.eps**2) - self.eps) * self.mask)
        return self.lam * np.sum(np.abs(W) * self.mask)

    def evaluate(self, W):
        """Phi, gradient, B^-1 and the pieces f, h."""
        h, Binv = self.barrier(W)
        if not np.isfinite(h):
            return np.inf, None, None, None
        f, Gf = self.loss(W)
        G = self.mu * Gf + 2.0 * W * Binv.T
        G = G + self.mu * self.lam * W / np.sqrt(W * W + self.eps**2)
        G *= self.mask
        return self.mu * (f + self.l1_value(W)) + h, G, Binv, dict(f=f, h=h)

    def hvp(self, W, Binv, V):
        CV = self.R.T @ (self.R @ V) if self.R is not None else self.C @ V
        HV = self.mu * CV + 2.0 * V * Binv.T + 4.0 * W * (Binv @ (W * V) @ Binv).T
        r = np.sqrt(W * W + self.eps**2)
        HV = HV + self.mu * self.lam * self.eps**2 / r**3 * V
        return HV * self.mask

    def hess_diag(self, W, Binv):
        Bt = Binv.T
        D = 2.0 * Bt + 4.0 * (W * Bt) ** 2
        r = np.sqrt(W * W + self.eps**2)
        D = D + self.mu * self.lam * self.eps**2 / r**3
        return D * self.mask


class BlockPrecond:
    """Per column b, the inverse of the exact column block of the Hessian on the free rows."""

    def __init__(self, C, mu, tau):
        self.C, self.mu, self.tau = C, mu, tau
        self.d = C.shape[0]

    def update(self, Dh, free, U):
        """Dh: diagonal of the non-loss Hessian; U[a, b] = W_ab Binv_ba gives the coupling 4 u u^T."""
        d = self.d
        Minv = np.zeros((d, d, d))
        base = self.mu * self.C
        for b in range(d):
            F = np.flatnonzero(free[:, b])
            if F.size == 0:
                continue
            u = U[F, b]
            M = base[np.ix_(F, F)] + np.diag(Dh[F, b] + self.tau) + 4.0 * (np.outer(u, u) - np.diag(u * u))
            try:
                L = sla.cholesky(M, lower=True, check_finite=False)
                Li = sla.solve_triangular(L, np.eye(F.size), lower=True, check_finite=False)
                Minv[b][np.ix_(F, F)] = Li.T @ Li
            except np.linalg.LinAlgError:
                Minv[b][F, F] = 1.0 / np.maximum(np.diag(M), 1e-300)
        self.Minv = Minv

    def __call__(self, R, free):
        return np.einsum("bij,jb->ib", self.Minv, R * free, optimize=True) * free


def lm_cg(hv, g, prec, free, nu, Dm, eta, max_cg):
    """Truncated PCG on (H + nu diag(Dm)) z = -g; exits on negative curvature."""
    z = np.zeros_like(g)
    r = g * free
    y = prec(r, free)
    ry = float(np.sum(r * y))
    if ry <= 0:
        return z, dict(cg=0, status="zero_grad")
    p = -y
    pp = ry  # ||p||_M^2 by the PCG recurrence
    tol = eta * np.sqrt(ry)
    for j in range(1, max_cg + 1):
        Hp0 = hv(p) * free
        Hp = Hp0 + nu * Dm * p * free
        pHp = float(np.sum(p * Hp))
        if pHp <= 0 or not np.isfinite(pHp):
            nc = dict(p=p, pHp=float(np.sum(p * Hp0)), pMp=pp)
            return (p if j == 1 else z), dict(cg=j, status="neg_curv", nc=nc)
        alpha = ry / pHp
        z = z + alpha * p
        r = r + alpha * Hp
        y = prec(r, free)
        ry_new = float(np.sum(r * y))
        if np.sqrt(max(ry_new, 0.0)) <= tol:
            return z, dict(cg=j, status="converged")
        beta = ry_new / ry
        p = -y + beta * p
        pp = ry_new + beta * beta * pp
        ry = ry_new
    return z, dict(cg=max_cg, status="max_cg")


def _search(st, W, phi, G, z, max_ls, c1=1e-4):
    """Armijo backtracking; trials outside the domain are halved like non-decreasing ones."""
    lin0 = float(np.sum(G * z))
    if lin0 >= 0:
        return False, 1.0, None
    alpha = 1.0
    for _ in range(max_ls):
        Wn = W + alpha * z
        ev = st.evaluate(Wn)
        if np.isfinite(ev[0]) and ev[0] <= phi + c1 * alpha * lin0:
            return True, alpha, (Wn, ev)
        alpha *= 0.5
    return False, alpha, None


def _nc_search(st, W, phi, G, nc, t0, max_ls, c1=1e-4, max_expand=12):
    """Step along a negative-curvature direction (second-order Armijo, expand by 2 from t0, else backtrack)."""
    pMp = nc["pMp"]
    if not (pMp > 0 and np.isfinite(pMp)):
        return False, 0.0, None
    v = nc["p"] / np.sqrt(pMp)
    kap = nc["pHp"] / pMp
    gv = float(np.sum(G * v))
    if gv > 0:
        v, gv = -v, -gv
    if kap >= 0:
        return False, 0.0, None

    def test(t):
        Wn = W + t * v
        ev = st.evaluate(Wn)
        return np.isfinite(ev[0]) and ev[0] <= phi + c1 * (t * gv + 0.5 * t * t * kap), Wn, ev

    t = t0
    ok, Wn, ev = test(t)
    if ok:
        best = (t, Wn, ev)
        for _ in range(max_expand):
            ok2, Wn2, ev2 = test(2 * t)
            if not ok2 or ev2[0] >= best[2][0]:
                break
            t = 2 * t
            best = (t, Wn2, ev2)
        t, Wn, ev = best
        return True, t, (Wn, ev)
    for _ in range(max_ls):
        t *= 0.5
        ok, Wn, ev = test(t)
        if ok:
            return True, t, (Wn, ev)
    return False, t, None


def solve_stage(
    st,
    W,
    tol,
    max_newton,
    negcurv="none",
    stall_rtol=0.0,
    delta=1e-8,
    kkt_tol=1e-10,
    max_cg=250,
    eta_max=0.5,
    max_ls=40,
):
    """Newton-CG on one stage; negcurv="step" adds negative-curvature steps, stall_rtol > 0 a stall stop."""
    d = st.d
    phi, G, Binv, parts = st.evaluate(W)
    if not np.isfinite(phi):
        return W, None
    tau0 = delta * st.mu * np.trace(st.C) / d
    P = BlockPrecond(st.C, st.mu, tau0)
    free = st.mask
    n_cg, status = 0, "max_newton"
    nu, it, nc_t, n_ncstep, n_stall = 0.0, 0, None, 0, 0
    for it in range(1, max_newton + 1):
        if float(np.max(np.abs(G))) <= kkt_tol:
            status, it = "kkt", it - 1
            break
        g0 = G
        Dh = st.hess_diag(W, Binv)
        P.update(Dh, free, W * Binv.T)
        hv = lambda V, W=W, Binv=Binv: st.hvp(W, Binv, V)
        accepted, alpha = False, 1.0
        while not accepted:
            ry0 = float(np.sum(G * P(G, free)))
            if ry0 <= 0:
                status = "zero_grad"
                break
            eta = min(eta_max, ry0**0.25)
            Dm = (st.mu * np.diag(st.C)[:, None] + Dh + tau0) * free
            z, info = lm_cg(hv, G, P, free, nu, Dm, eta, max_cg)
            n_cg += info["cg"]
            accepted, alpha, acc = _search(st, W, phi, G, z, max_ls)
            if negcurv == "step" and info["status"] == "neg_curv":
                nc = info["nc"]
                t_init = nc_t if nc_t is not None else max(np.sqrt(ry0), abs(nc["pHp"] / nc["pMp"]))
                ok_nc, t_nc, acc_nc = _nc_search(st, W, phi, G, nc, t_init, max_ls)
                if ok_nc:
                    nc_t = t_nc
                    if not accepted or acc_nc[1][0] < acc[1][0]:
                        accepted, alpha, acc = True, 1.0, acc_nc
                        n_ncstep += 1
                        info = dict(info, status="nc_step")
            if info["status"] != "nc_step":
                if info["status"] == "neg_curv" or not accepted:
                    nu = max(4.0 * nu, 1e-6)
                elif alpha < 1.0:
                    nu = max(nu, 1e-8) * (8.0 if alpha < 0.25 else 2.0)
                else:
                    nu = nu / 4.0 if nu > 1e-10 else 0.0
            if accepted:
                phi_old = phi
                W, (phi, G, Binv, parts) = acc
                if stall_rtol > 0:
                    n_stall = n_stall + 1 if phi_old - phi <= stall_rtol * abs(phi) else 0
            elif nu > 1e12:
                break
        if not accepted:
            status = "no_step" if status != "zero_grad" else status
            break
        dec2 = -float(np.sum(g0 * z))  # Newton decrement squared
        if info["status"] == "converged" and nu <= 1e-8 and 0.5 * dec2 < tol:
            status = "decrement"
            break
        if stall_rtol > 0 and n_stall >= 2:
            status = "stall"
            break
    info = dict(
        mu=st.mu,
        s=st.s,
        newton=it,
        cg=n_cg,
        nc_steps=n_ncstep,
        status=status,
        h=parts["h"],
        F=parts["f"] + st.l1_value(W, true=True),
        phi=phi,
    )
    return W, info


def run(
    X,
    lambda1=0.02,
    T=5,
    mu_init=1.0,
    s=None,
    W0=None,
    free_diag=False,
    gram="cov",
    max_newton=50,
    ftol_early=None,
    ftol_final=None,
    tol=1e-9,
    **stage_kw,
):
    """DAGMA central path with the Newton-CG inner solver; returns (W[source, target], log)."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    n, d = X.shape
    C = X.T @ X / n
    R = None
    if gram == "qr":
        R = sla.qr(X, mode="r", check_finite=False)[0][:d] / np.sqrt(n)
        C = R.T @ R
    s_path = list(s) if s is not None else [1.0, 0.9, 0.8, 0.7, 0.6]
    s_path = s_path + [s_path[-1]] * max(T - len(s_path), 0)
    mask = np.ones((d, d)) if free_diag else 1.0 - np.eye(d)
    W = np.zeros((d, d)) if W0 is None else np.asarray(W0, dtype=float) * mask
    mu = mu_init
    stages = []
    for k in range(T):
        sk = s_path[k]
        while True:  # DAGMA's retry rule: widen s until the warm start is inside the domain
            st = Stage(C, mu, sk, lambda1, free_diag=free_diag, R=R)
            if np.isfinite(st.barrier(W)[0]):
                break
            sk += 0.1
        ftol = ftol_final if k == T - 1 else ftol_early  # decrement tolerance in F units
        W, info = solve_stage(st, W, tol if ftol is None else ftol * mu, max_newton, **stage_kw)
        stages.append(info)
        mu *= 0.1
    log = dict(
        stages=stages,
        newton=sum(x["newton"] for x in stages),
        cg=sum(x["cg"] for x in stages),
        s_final=[x["s"] for x in stages],
    )
    return W, log
