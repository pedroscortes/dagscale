# ruff: noqa: E402
"""Floor, DAGMA, noise-unit DAGMA and CoLiDE-EV on the Sachs data (Section 5.7); output in results/noise_units_real/."""

import contextlib
import io
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "third_party"))
from colide_model import colide_ev
from dagma_fit import Logged, scores
from leaklaw import truth_free_floor
from noise_units import nnorm
from sachs import COND1, EDGES, POOLED, load
from simulate import project_dag


def main():
    d = 11
    A = np.zeros((d, d), dtype=int)
    for i, j in EDGES:
        A[j, i] = 1
    for dname, path in (("cond1_853", COND1), ("pooled_7466", POOLED)):
        X0 = load(path)
        for variant in ("raw", "log", "std"):
            X = np.log(np.maximum(X0, 1e-3)) if variant == "log" else X0.copy()
            if variant == "std":
                X = (X - X.mean(0)) / X.std(0)
            S = np.cov(X, rowvar=False)
            fl = truth_free_floor(S)
            for arm in ("ref", "nnorm", "colide_ev"):
                with contextlib.redirect_stderr(io.StringIO()):
                    if arm == "colide_ev":
                        W, _ = colide_ev().fit(
                            X.copy(),
                            lambda1=0.05,
                            T=4,
                            s=[1.0, 0.9, 0.8, 0.7],
                            warm_iter=2e4,
                            max_iter=7e4,
                            lr=0.0003,
                        )
                        W = np.asarray(W)
                        h = -np.linalg.slogdet(np.eye(d) - W * W)[1]
                    else:
                        m = Logged()
                        W = np.asarray(
                            m.fit(
                                (nnorm(X) if arm == "nnorm" else X).copy(),
                                lambda1=0.02,
                                w_threshold=0.0,
                            )
                        )
                        h = float(m._h(W)[0])
                pf = scores(project_dag(W.T), A)
                print(
                    f"{dname:12} {variant:4} {arm:10} floor {fl:9.3g}  h {h:9.3g}  projF1 {pf['f1']:.3f}  edges {pf['n_learned']}"
                )


if __name__ == "__main__":
    main()
