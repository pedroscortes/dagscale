# ruff: noqa: E402, N803, N806
"""One data set (ER2, d = 20, w = 1, seed 30) fitted by DAGMA at x0.01, x1 and x100 (Figure 1)."""

import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from dagma_fit import Logged
from simulate import make_data

from dagscale.check import check

OUT = "results/intro_example"

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    X, A = make_data("ER2", 20, 3000, 30, 1.0)
    out = {"A": A, "c": np.array([0.01, 1.0, 100.0])}
    for i, c in enumerate(out["c"]):
        m = Logged()
        with contextlib.redirect_stderr(io.StringIO()):
            W = np.asarray(m.fit(X * c, lambda1=0.02, w_threshold=0.0))
        out[f"W{i}"], out[f"h{i}"], out[f"status{i}"] = W, float(m._h(W)[0]), check(X * c).status
        print(c, out[f"h{i}"], out[f"status{i}"], flush=True)
    np.savez(f"{OUT}/intro_example.npz", **out)
