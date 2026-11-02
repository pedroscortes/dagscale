"""Pre-fit scale check for least-squares DAGMA (and other fixed-penalty, fixed-schedule learners).

Defaults are those of the DAGMA reference example: lambda1 = 0.02, last-stage score weight mu = 1e-4,
last-stage s = 0.6, and an output counted as non-acyclic when its barrier value exceeds 0.01.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScaleCheck:
    status: str  # "ok", "too small", "too large" or "spread"
    floor: float  # leading-order lower bound on the last-stage barrier value
    floor_headroom: float  # factor c at which the floor of cX reaches h_tol
    empty_columns: list[int]  # columns empty at every stationary point, whatever the data say
    spread: float  # decades between the largest and smallest column variance
    factor: float  # multiply X by this to put it in noise units

    def __str__(self):
        return (
            f"{self.status}: floor {self.floor:.3g} (fires above x{self.floor_headroom:.3g}), "
            f"{len(self.empty_columns)} forced-empty columns, spread {self.spread:.1f} decades, "
            f"noise-unit factor {self.factor:.3g}"
        )


def conditional_variances(X) -> np.ndarray:
    """Var(X_j | X_-j) = 1 / Theta_jj from the sample precision matrix."""
    return 1.0 / np.diag(np.linalg.inv(np.cov(X, rowvar=False)))


def noise_units(X) -> np.ndarray:
    """X divided by the median conditional standard deviation; the DAGMA weights are unchanged."""
    return X / np.sqrt(np.median(conditional_variances(X)))


def check(X, lambda1=0.02, mu=1e-4, s=0.6, h_tol=0.01, max_spread=8.0) -> ScaleCheck:
    """Covariance-only check before fitting: is X below, inside or above the scale window?"""
    X = np.asarray(X, dtype=float)
    S = np.cov(X, rowvar=False)
    cv = conditional_variances(X)
    floor = mu**2 * s**2 / 4 * float(np.sum(cv**2))
    off = np.abs(S - np.diag(np.diag(S)))
    empty = [int(j) for j in np.flatnonzero(off.max(axis=0) <= lambda1)]
    var = np.diag(S)
    spread = float(np.log10(var.max() / var.min()))
    if floor > h_tol:
        status = "too large"
    elif empty:
        status = "too small"
    elif spread > max_spread:
        status = "spread"
    else:
        status = "ok"
    return ScaleCheck(
        status=status,
        floor=floor,
        floor_headroom=(h_tol / floor) ** 0.25,
        empty_columns=empty,
        spread=spread,
        factor=float(1.0 / np.sqrt(np.median(cv))),
    )
