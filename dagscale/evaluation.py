"""Edge metrics; adjacencies are adj[i, j] = 1 for an edge j -> i."""

from __future__ import annotations

import numpy as np


def threshold_weights(weights, threshold: float = 0.3) -> np.ndarray:
    """Binary adjacency with A[i, j] = 1 iff |W[i, j]| > threshold."""
    return (np.abs(np.asarray(weights)) > threshold).astype(int)


def structural_hamming_distance(learned_adjacency, true_adjacency) -> int:
    """Edge additions, deletions and reversals (a reversal counts once) between two adjacencies."""
    learned = np.asarray(learned_adjacency).astype(bool).astype(int)
    true = np.asarray(true_adjacency).astype(bool).astype(int)
    disagreements = int(np.abs(learned - true).sum())
    # a reversed edge contributes 2 to disagreements but is one operation
    flipped = int(np.sum((learned == 1) & (true == 0) & (learned.T == 0) & (true.T == 1)))
    return disagreements - flipped


def edge_confusion(learned_adjacency, true_adjacency) -> tuple[int, int, int]:
    """(tp, fp, fn): tp = correctly oriented edges; reversed edges count in neither fp nor fn."""
    learned = np.asarray(learned_adjacency).astype(bool)
    true = np.asarray(true_adjacency).astype(bool)
    tp = int((learned & true).sum())
    fp = int((learned & ~true & ~true.T).sum())
    fn = int((~learned & ~learned.T & true).sum())
    return tp, fp, fn


def adjacency_from_scm_weights(weights) -> np.ndarray:
    """Exact graph of an SCM weight matrix (non-zero entries)."""
    return (np.asarray(weights) != 0.0).astype(int)
