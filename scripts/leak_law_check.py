# ruff: noqa: E402, N806, SIM115
"""Leak law against DAGMA's final h on the scale-window fits with c in {1, 10} (Section 5.3, Fig. 3)."""

import csv
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from leaklaw import law_h, truth_free_floor
from simulate import make_data


def one(r):
    X, A = make_data(r["graph"], int(r["d"]), 3000, int(r["seed"]), float(r["wscale"]))
    X = X * float(r["c"])
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / Xc.shape[0]
    hl = law_h(S, A)[0]
    var = np.diag(S)
    return dict(
        graph=r["graph"],
        d=r["d"],
        w=r["wscale"],
        c=r["c"],
        seed=r["seed"],
        h_pkg=float(r["h"]),
        h_law=hl,
        floor=truth_free_floor(S),
        spread=float(np.log10(var.max() / var.min())),
        f1=float(r["proj_f1"]),
    )


if __name__ == "__main__":
    R = [
        r
        for r in csv.DictReader(open("results/scale_window/scale_window.csv"))
        if r["arm"] == "ref" and r["c"] in ("1.0", "10.0") and not r.get("error")
    ]
    with ProcessPoolExecutor(18) as ex:
        out = list(ex.map(one, R))
    os.makedirs("results/leak_law_check", exist_ok=True)
    with open("results/leak_law_check/leak_law_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    ratio = np.array([o["h_pkg"] / o["h_law"] for o in out])
    sp = np.array([o["spread"] for o in out])
    m = sp <= 6
    print(f"fits {len(out)}; spread<=6: {m.sum()}")
    print(
        f"median ratio (spread<=6): {np.median(ratio[m]):.2f}  IQR [{np.percentile(ratio[m], 25):.2f}, {np.percentile(ratio[m], 75):.2f}]"
    )
    print(
        f"floor holds (h_pkg >= 0.5 h_law), spread<=6: {np.mean(ratio[m] >= 0.5):.3f}; all: {np.mean(ratio >= 0.5):.3f}"
    )
    for lo, hi in ((0, 3), (3, 6), (6, 9), (9, 99)):
        k = (sp >= lo) & (sp < hi)
        if k.any():
            print(f"  spread {lo}-{hi}: n={k.sum()} median ratio {np.median(ratio[k]):.2f}")
    fl = np.array([o["floor"] for o in out])
    hp = np.array([o["h_pkg"] for o in out])
    print(f"truth-free floor <= h_pkg in {np.mean(fl <= hp * 1.05):.3f} of fits")
