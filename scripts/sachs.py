"""Sachs et al. (2005) data and the 17-edge consensus DAG, shared by the real-data scripts."""

import csv
import os

import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sachs")
COND1 = os.path.join(_DATA, "sachs_cd3cd28_853.csv")
POOLED = os.path.join(_DATA, "sachs_pooled_7466.csv")
EDGES = [
    (0, 1),
    (1, 5),
    (2, 3),
    (2, 4),
    (4, 3),
    (5, 6),
    (7, 0),
    (7, 1),
    (7, 5),
    (7, 6),
    (7, 9),
    (7, 10),
    (8, 0),
    (8, 1),
    (8, 9),
    (8, 10),
    (8, 2),
]  # i -> j, nodes Raf Mek Plcg PIP2 PIP3 Erk Akt PKA PKC P38 JNK


def load(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    return np.array(rows[1:], dtype=np.float64)
