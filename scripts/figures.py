# ruff: noqa: N806, RUF001
"""Figures 2 to 4 from the result files, written as PDFs to argv[1] (default figures/)."""

from __future__ import annotations

import csv
import gzip
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "pdf.fonttype": 42,
        "font.family": "DejaVu Sans",
    }
)
R = "results"


def rows(*paths):
    out = []
    for p in paths:
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as fh:
            out += [r for r in csv.DictReader(fh) if not r.get("error")]
    return out


def fig_window():
    S = rows(
        f"{R}/scale_window/scale_window.csv",
        f"{R}/scale_window/scale_window_s40_99.csv.gz",
    )
    S = [r for r in S if r["arm"] == "ref"]
    cs = ["0.01", "0.1", "1.0", "10.0", "100.0", "1000.0"]
    x = np.log10([float(c) for c in cs])
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6), constrained_layout=True)
    for k, (w, lab, mk) in enumerate((("0.5", "$w = 0.5$", "o"), ("1.0", "$w = 1$", "s"), ("2.0", "$w = 2$", "^"))):
        f1 = [np.mean([float(r["proj_f1"]) for r in S if r["wscale"] == w and r["c"] == c]) for c in cs]
        na = [np.mean([float(r["h"]) > 0.01 for r in S if r["wscale"] == w and r["c"] == c]) for c in cs]
        ax[0].plot(x, f1, marker=mk, color=C[k], label=lab)
        ax[1].plot(x, na, marker=mk, color=C[k], label=lab)
    for a, yl in ((ax[0], "projected F1"), (ax[1], "share non-acyclic (h > 0.01)")):
        a.set_xticks(x, [f"×{c.rstrip('0').rstrip('.')}" for c in cs])
        a.set_xlabel("global multiplier c of the data")
        a.set_ylabel(yl)
        a.set_ylim(-0.03, 1.03)
    ax[0].legend(frameon=False, loc="lower center")
    fig.savefig(f"{OUT}/fig_scale_window.pdf")
    plt.close(fig)


def fig_leaklaw():
    G = rows(f"{R}/leak_law_check/leak_law_check.csv")
    hp = np.array([float(r["h_pkg"]) for r in G])
    hl = np.array([float(r["h_law"]) for r in G])
    sp = np.array([float(r["spread"]) for r in G])
    fig, ax = plt.subplots(figsize=(3.6, 2.9), constrained_layout=True)
    sc = ax.scatter(
        hl,
        hp,
        c=np.minimum(sp, 12),
        cmap="Blues",
        vmin=0,
        vmax=12,
        s=9,
        edgecolors="#ffffff",
        linewidths=0.3,
    )
    lim = [min(hl.min(), hp.min()) * 0.5, max(hl.max(), hp.max()) * 2]
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("predicted h (leak law, first order)")
    ax.set_ylabel("observed final h")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("column-variance spread (decades)")
    fig.savefig(f"{OUT}/fig_leak_law.pdf")
    plt.close(fig)


def fig_breadth():
    # log-likelihood variant: rows from methods_logll.csv replace those of methods_recent
    B = rows(f"{R}/methods/methods.csv.gz", f"{R}/methods/methods_recent.csv.gz")
    B = [r for r in B if r["method"] != "dagma_logll"] + rows(f"{R}/methods/methods_logll.csv")
    B = [r for r in B if r["part"] == "synth"]
    groups = [
        (
            "fixed barrier schedule",
            0,
            ["dagma", "colide_ev", "colide_nv", "dagma_logll", "analytic2"],
        ),
        ("noise-unit DAGMA", 1, ["dagma_nnorm"]),
        ("adaptive / no schedule", 2, ["notears", "topo"]),
        ("likelihood (soft constraint)", 3, ["golem_ev", "golem_nv"]),
    ]
    names = {
        "dagma": "DAGMA",
        "colide_ev": "CoLiDE-EV",
        "colide_nv": "CoLiDE-NV",
        "dagma_logll": "DAGMA, log-likelihood",
        "analytic2": "analytic constraint†",
        "dagma_nnorm": "DAGMA, noise units",
        "notears": "NOTEARS",
        "topo": "TOPO",
        "golem_ev": "GOLEM-EV",
        "golem_nv": "GOLEM-NV",
    }
    variants = [
        ("0.01", "×0.01"),
        ("1", "×1"),
        ("100", "×100"),
        ("resc", "per variable"),
    ]
    order = [m for _, _, ms in groups for m in ms]
    col = {m: C[g] for _, g, ms in groups for m in ms}
    mk = {m: "osDo"[g] for _, g, ms in groups for m in ms}
    fig, ax = plt.subplots(1, 4, figsize=(6.6, 3.4), sharey=True, constrained_layout=True)
    y = np.arange(len(order))[::-1]
    for a, (v, lab) in zip(ax, variants, strict=True):
        for yi, m in zip(y, order, strict=True):
            x = [r for r in B if r["method"] == m and r["variant"] == v]
            f = np.mean([float(r["proj_f1"]) for r in x])
            hl = [r["h_logdet"] for r in x]
            nonacyc = (
                np.mean([(h in ("nan", "")) or float(h) > 0.01 for h in hl])
                if m not in ("notears", "golem_ev", "golem_nv", "topo")
                else np.mean([r["cyclic"] == "True" for r in x])
            )
            a.plot([0, f], [yi, yi], color=GRID, lw=1.2, zorder=1)
            a.scatter(
                [f],
                [yi],
                color=col[m],
                marker=mk[m],
                s=26,
                zorder=3,
                facecolors=col[m] if nonacyc < 0.5 else "#ffffff",
                linewidths=1.2,
            )
        a.set_title(lab)
        a.set_xlim(0, 1)
        a.set_xticks([0, 0.5, 1])
        a.grid(axis="y", visible=False)
    ax[0].set_yticks(y, [names[m] for m in order])
    fig.supxlabel("projected F1 (hollow marker: most fits non-acyclic)", fontsize=9, color=INK)
    handles = [plt.Line2D([], [], color=C[g], marker="osDo"[g], ls="", label=t) for t, g, _ in groups]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(f"{OUT}/fig_breadth.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_window()
    fig_leaklaw()
    fig_breadth()
    print("wrote", sorted(os.listdir(OUT)))
