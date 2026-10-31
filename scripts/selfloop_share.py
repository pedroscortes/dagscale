# ruff: noqa: E402, N806
"""Share of the leading-order leak h_1 carried by self-loops on the c = 1 data sets of predict.py (Section 3.4)."""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from leaklaw import ols_on_support, predicted_leak
from simulate import make_data

shares = []
for g in ("ER2", "ER4", "SF4"):
    for d in (30, 50):
        for w in (0.5, 1.0):
            for s in range(30, 40):
                X, A = make_data(g, d, 3000, s, w)
                Xc = X - X.mean(0)
                S = Xc.T @ Xc / Xc.shape[0]
                W0, sig2 = ols_on_support(S, A)
                _, T = predicted_leak(W0, sig2)
                Q1 = np.linalg.inv(np.eye(d) - W0 * W0)
                terms = Q1.T * T**2  # term (c, p): Q1[p, c] T[c, p]^2
                shares.append(np.trace(terms) / terms.sum())
shares = np.array(shares)
print(
    f"n {len(shares)}  median {np.median(shares):.3f}  p10 {np.quantile(shares, 0.1):.3f}  p90 {np.quantile(shares, 0.9):.3f}"
)
