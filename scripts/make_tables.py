# ruff: noqa: N803, N806, SIM115
"""All tables and in-text numbers from results/ (no fits), written to results/tables/; Table 6 via table_real.py."""

import csv
import gzip
import os
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from sachs import COND1, load
from simulate import GRAPHS

R = "results"
OUT = f"{R}/tables"
NO_BARRIER = ("notears", "golem_ev", "golem_nv", "topo", "sortnregress", "r2sortnregress")
NUMBERS = []


def rows(*paths):
    out = []
    for p in paths:
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as fh:
            out += [r for r in csv.DictReader(fh) if not r.get("error")]
    return out


def bad(h):
    """Barrier criterion: outside the domain (nan, inf, empty) or h > 0.01."""
    return h in ("nan", "", "inf") or not np.isfinite(float(h)) or float(h) > 0.01


def non_acyclic(r):
    if r["method"] in NO_BARRIER:
        return r["cyclic"] == "True"
    return bad(r["h_logdet"])


def num(label, value, fmt=".2f"):
    s = format(value, fmt) if not isinstance(value, str) else value
    NUMBERS.append(f"{label}: {s}")
    print(f"  {label}: {s}")


def table(name, lines):
    print(f"\n== {name}")
    print("\n".join(lines))
    with open(f"{OUT}/{name}.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")


def mean(xs, key="proj_f1"):
    return float(np.mean([float(r[key]) for r in xs]))


def share(xs, f):
    return float(np.mean([f(r) for r in xs]))


def wlab(w):
    return f"{float(w):g}"


def key(r, w="w"):
    return (r["graph"], int(r["d"]), float(r[w]), int(r["seed"]))


def scale_window():
    S = rows(
        f"{R}/scale_window/scale_window.csv",
        f"{R}/scale_window/scale_window_s40_99.csv.gz",
    )
    S = [r for r in S if r["arm"] == "ref"]
    cs = ("0.01", "0.1", "1.0", "10.0", "100.0", "1000.0")
    cell = {}
    for r in S:
        cell.setdefault((r["graph"], r["d"], wlab(r["wscale"]), r["c"]), []).append(r)
    f1, na = [], []
    for g in ("ER2", "ER4", "SF4"):
        for d in ("30", "50"):
            for w in ("0.5", "1", "2"):
                x = [cell[(g, d, w, c)] for c in cs]
                f1.append(f"{g} & {d} & {w} & " + " & ".join(f"{mean(v):.2f}" for v in x) + r" \\")
                na.append(
                    f"{g} & {d} & {w} & " + " & ".join(f"{share(v, lambda r: bad(r['h'])):.2f}" for v in x) + r" \\"
                )
    table("tab_app_window", [*f1, r"\midrule", *na])
    print("numbers (scale window)")
    num("window fits (ref arm)", f"{len(S)}")
    num("window seeds", f"{len({r['seed'] for r in S})}")
    by = {(wlab(w), c): [r for r in S if r["wscale"] == w and r["c"] == c] for w in ("0.5", "1.0", "2.0") for c in cs}
    num("fits per point (Fig 2)", f"{len(by[('1', '1.0')])}")
    for w in ("0.5", "1", "2"):
        for c in cs:
            x = by[(w, c)]
            num(f"Fig2 w={w} c={c} F1", mean(x))
            num(f"Fig2 w={w} c={c} non-acyclic", share(x, lambda r: bad(r["h"])))
    num("Fig2 w=1 c=1 AUPRC", mean(by[("1", "1.0")], "auprc_std"))
    num("Fig2 w=1 c=100 AUPRC", mean(by[("1", "100.0")], "auprc_std"))
    c1 = [r for r in S if r["c"] == "1.0"]
    num("data sets at c=1 (kappa/lmax)", f"{len(c1)}")
    for w in ("0.5", "1.0", "2.0"):
        x = [r for r in c1 if r["wscale"] == w]
        num(
            f"median log10 kappa w={wlab(w)}",
            float(np.median(np.log10([float(r["kappa"]) for r in x]))),
            ".1f",
        )
        num(
            f"median log10 lmax w={wlab(w)}",
            float(np.median(np.log10([float(r["lmax"]) for r in x]))),
            ".1f",
        )
    rho = spearmanr([float(r["kappa"]) for r in c1], [float(r["lmax"]) for r in c1])[0]
    num("Spearman log kappa vs log lmax (c=1)", rho, ".3f")
    return cell


def leak_law():
    G = rows(f"{R}/leak_law_check/leak_law_check.csv")
    ratio = np.array([float(r["h_pkg"]) / float(r["h_law"]) for r in G])
    sp = np.array([float(r["spread"]) for r in G])
    hp = np.array([float(r["h_pkg"]) for r in G])
    fl = np.array([float(r["floor"]) for r in G])
    m = sp <= 6
    print("\nnumbers (leak law)")
    num("leak-law fits", f"{len(G)}")
    num("median ratio, spread<=6", float(np.median(ratio[m])))
    num("IQR low, spread<=6", float(np.percentile(ratio[m], 25)))
    num("IQR high, spread<=6", float(np.percentile(ratio[m], 75)))
    for lo, hi in ((0, 3), (3, 6), (6, 9), (9, 99)):
        k = (sp >= lo) & (sp < hi)
        num(f"median ratio spread {lo}-{hi}", float(np.median(ratio[k])))
    num("min ratio, all fits", float(ratio.min()))
    num("min ratio, spread<=6", float(ratio[m].min()))
    num("share ratio > 1, all fits", float(np.mean(ratio > 1)))
    num("share ratio > 1, spread<=6", float(np.mean(ratio[m] > 1)))
    num("share ratio >= 0.5, spread<=6", float(np.mean(ratio[m] >= 0.5)))
    num("floor below observed h (leak-law fits)", f"{int(np.sum(fl < hp))} of {len(G)}")


def prediction():
    D = rows(f"{R}/predict/predict.csv", f"{R}/predict/predict_s40_79.csv")
    F = {(*key(r), float(r["c"])): float(r["floor"]) for r in rows(f"{R}/floor_alarm/floor_alarm.csv")}
    obs = np.array([r["infeasible"] == "True" for r in D])
    c = np.array([float(r["c"]) for r in D])
    rules = {
        "law": np.array([r["pred_infeasible"] == "True" for r in D]),
        "floor": np.array([F[(*key(r), float(r["c"]))] > 0.01 for r in D]),
        "rule": c >= 20,
    }

    def stats(p):
        return np.mean(p == obs), int(np.sum(p & ~obs)), np.sum(p & obs) / np.sum(obs)

    lines = []
    for k, name in (
        ("law", "Leak law, first order (true support)"),
        ("floor", "Conditional-variance floor (no graph)"),
        ("rule", r"Non-acyclic if and only if $c\ge20$"),
    ):
        a, fa, cc = stats(rules[k])
        lines.append(f"{name} & ${100 * a:.1f}\\%$ & {fa} & ${100 * cc:.1f}\\%$ \\\\")
    table("tab_predict", lines)
    print("numbers (prediction)")
    num("prediction fits", f"{len(D)}")
    num("non-acyclic / acyclic", f"{int(obs.sum())} / {int((~obs).sum())}")
    num("all outputs in domain", f"{all(r['in_domain'] == 'True' for r in D)}")
    err = rules["law"] != obs
    num("c values with law errors", ", ".join(f"{v:g}" for v in sorted(set(c[err]))))
    num("c=15 predicted non-acyclic %", 100 * float(np.mean(rules["law"][c == 15])), ".1f")
    num("c=15 observed non-acyclic %", 100 * float(np.mean(obs[c == 15])), ".1f")
    num("floor fires (fits)", f"{int(rules['floor'].sum())}")
    num("floor fires on acyclic", f"{int(np.sum(rules['floor'] & ~obs))}")
    for cv in (15, 20, 30, 50):
        k = (c == cv) & obs
        num(f"floor catches at c={cv} %", 100 * float(np.mean(rules["floor"][k])), ".0f")
    hp = np.array([float(r["h_pkg"]) for r in D])
    fl = np.array([F[(*key(r), float(r["c"]))] for r in D])
    num("floor below observed h (fits)", f"{int(np.sum(fl < hp))} of {len(D)}")

    N = rows(f"{R}/predict_newton/predict_newton.csv")
    num("Newton refit fits", f"{len(N)}")
    num(
        "Newton prediction accuracy %",
        100 * share(N, lambda r: r["pred_infeasible"] == r["infeasible"]),
        ".1f",
    )
    D10 = rows(f"{R}/predict/predict.csv")
    num(
        "reference accuracy, same 600 %",
        100 * share(D10, lambda r: r["pred_infeasible"] == r["infeasible"]),
        ".1f",
    )
    for cv in ("10.0", "15.0", "50.0"):
        x = [float(r["ratio"]) for r in N if r["c"] == cv and np.isfinite(float(r["ratio"]))]
        num(f"median h_newton/h_law c={float(cv):g}", float(np.median(x)))
    P = rows(f"{R}/predict_lasso/predict_lasso.csv.gz")
    for s in ("package", "newton"):
        x = [r for r in P if r["solver"] == s]
        acc = share(x, lambda r: (float(r["h_law_precise"]) > 0.01) == (r["infeasible"] == "True"))
        num(f"order-restricted Lasso accuracy ({s}) %", 100 * acc, ".1f")
    return D10


def breadth():
    B = rows(f"{R}/methods/methods.csv.gz", f"{R}/methods/methods_recent.csv.gz")
    logll_time = [r for r in B if r["part"] == "synth" and r["method"] == "dagma_logll"]
    B = [r for r in B if r["method"] != "dagma_logll"] + rows(f"{R}/methods/methods_logll.csv")
    B = [r for r in B if r["part"] == "synth"]
    V = ("0.01", "1", "100", "resc")
    order = [
        ("dagma", "DAGMA"),
        ("colide_ev", "CoLiDE-EV"),
        ("colide_nv", "CoLiDE-NV"),
        ("dagma_logll", "DAGMA, log-likelihood"),
        ("analytic2", r"Analytic constraint$^{\dagger}$"),
        ("dagma_nnorm", "DAGMA, noise units"),
        ("notears", "NOTEARS"),
        ("topo", "TOPO"),
        ("golem_ev", "GOLEM-EV"),
        ("golem_nv", "GOLEM-NV"),
    ]
    cell = {(m, v): [r for r in B if r["method"] == m and r["variant"] == v] for m, _ in order for v in V}
    lines = []
    for m, name in order:
        t = logll_time if m == "dagma_logll" else [r for v in V for r in cell[(m, v)]]
        lines.append(
            f"{name} & "
            + " & ".join(f"{mean(cell[(m, v)]):.2f} ({share(cell[(m, v)], non_acyclic):.2f})" for v in V)
            + f" & {mean(cell[(m, '1')], 'auprc_std'):.2f} & {mean(t, 'wall'):.0f} \\\\"
        )
    table("tab_app_methods", lines)
    print("numbers (comparison of methods)")
    num("fits per entry", f"{sorted({len(x) for x in cell.values()})}")
    f001 = {m: mean(cell[(m, "0.01")]) for m in ("dagma", "colide_ev", "colide_nv", "analytic2", "notears")}
    num(
        "x0.01 least-squares fixed-penalty F1 range",
        f"{min(f001.values()):.2f} to {max(f001.values()):.2f}",
    )
    num(
        "x100 fixed-schedule non-acyclic min",
        min(
            share(cell[(m, "100")], non_acyclic)
            for m in ("dagma", "colide_ev", "colide_nv", "dagma_logll", "analytic2")
        ),
    )
    cyc = share(cell[("dagma", "100")], lambda r: r["cyclic"] == "True")
    num("DAGMA x100 thresholded-cycle share", cyc)
    num("DAGMA x100 thresholded no cycle %", 100 * (1 - cyc), ".0f")
    topo1 = cell[("topo", "1")]
    num(
        "TOPO x1 domain-check failures",
        f"{sum(r['in_domain'] == 'False' for r in topo1)} of {len(topo1)}",
    )
    f1x1 = [mean(cell[(m, "1")]) for m, _ in order]
    fre = [mean(cell[(m, "resc")]) for m, _ in order]
    num("x1 F1 range (all methods)", f"{min(f1x1):.2f} to {max(f1x1):.2f}")
    num("per-variable F1 range (all methods)", f"{min(fre):.2f} to {max(fre):.2f}")

    L = rows(f"{R}/methods/methods_logll_control.csv.gz")
    LF = [r for r in rows(f"{R}/methods/methods_logll.csv") if r["part"] == "synth"]
    lines = []
    for name, x in (
        ("Least-squares DAGMA (reference protocol)", LF),
        ("$W = 0$", [r for r in L if r["method"] == "dagma_logll_zero"]),
        (
            "Least-squares DAGMA on noise-unit data",
            [r for r in L if r["method"] == "dagma_logll_nnorm"],
        ),
    ):
        c = [[r for r in x if r["variant"] == v] for v in ("0.01", "1", "100")]
        lines.append(f"{name} & " + " & ".join(f"{mean(v):.2f} ({share(v, non_acyclic):.2f})" for v in c) + r" \\")
    table("tab_logll", lines)
    print("numbers (log-likelihood invariance, noise-unit start)")
    nn = {}
    for r in L:
        if r["part"] == "synth" and r["method"] == "dagma_logll_nnorm":
            nn.setdefault(key(r), {})[r["variant"]] = (r["proj_f1"], r["n_edges"])
    same = [len(set(v.values())) == 1 for v in nn.values()]
    num("identical (F1, edge count) across all three multipliers", float(np.mean(same)))
    pair = [v[a] == v["1"] for v in nn.values() for a in ("0.01", "100")]
    num("identical vs x1 (pairs)", float(np.mean(pair)))
    dF = [abs(float(v[a][0]) - float(v["1"][0])) for v in nn.values() for a in ("0.01", "100")]
    num("mean |dF1| vs x1", float(np.mean(dF)))

    S = [r for r in rows(f"{R}/methods/methods_sorters.csv") if r["part"] == "synth"]
    print("numbers (sorting baselines)")
    for m in ("sortnregress", "r2sortnregress"):
        for v in ("1", "resc", "std"):
            num(f"{m} {v} F1", mean([r for r in S if r["method"] == m and r["variant"] == v]))

    T = [r for r in rows(f"{R}/methods/methods_std.csv") if r["part"] == "synth"]
    T += [r for r in rows(f"{R}/methods/methods_std_golem.csv") if r["part"] == "synth"]
    print("numbers (standardised data)")
    sm = {m: [r for r in T if r["method"] == m and r["proj_f1"]] for m, _ in order}
    num("std fits per method", ", ".join(f"{m} {len(x)}" for m, x in sm.items()))
    for m, _ in order:
        num(f"std {m} F1", mean(sm[m]))
    nt = sm["notears"]
    cy = [r for r in nt if r["cyclic"] == "True"]
    num("std NOTEARS max h_expm", max(float(r["h_expm"]) for r in nt), ".1e")
    num("std NOTEARS thresholded-cycle %", 100 * len(cy) / len(nt), ".0f")
    num("std NOTEARS fits with a cycle", f"{len(cy)}")
    num(
        "std NOTEARS median log-det h (cyclic fits)",
        float(np.median([float(r["h_logdet"]) for r in cy])),
        ".1e",
    )
    for m, _ in order:
        num(
            f"std {m} thresholded-cycle %",
            100 * share(sm[m], lambda r: r["cyclic"] == "True"),
            ".0f",
        )
    return cell


def noise_units():
    G = rows(f"{R}/noise_units/noise_units.csv.gz")
    cell = {}
    for r in G:
        cell.setdefault((r["graph"], r["d"], wlab(r["w"]), r["arm"], float(r["c"])), []).append(r)
    lines, within, maxdev = [], [], 0.0
    for g in GRAPHS:
        for d in ("30", "50"):
            for w in ("1", "2"):
                ref = cell[(g, d, w, "ref", 1.0)]
                nn = [cell[(g, d, w, "nnorm", c)] for c in (0.01, 1.0, 100.0)]
                cl = [cell[(g, d, w, "colide_ev", c)] for c in (0.01, 1.0, 100.0)]
                sp = float(np.median([float(r["spread"]) for r in ref]))
                na = "/".join(f"{share(x, lambda r: bad(r['h'])):.2f}" for x in (ref, nn[1], cl[1]))
                lines.append(
                    f"{g}/{d}/{w} & {sp:.1f} & {mean(ref):.2f} & {mean(nn[1]):.2f} & "
                    + " & ".join(f"{mean(x):.2f}" for x in cl)
                    + f" & {na} \\\\"
                )
                if sp <= 8:
                    within.append(mean(nn[1]) >= mean(ref) - 0.05)
                maxdev = max(maxdev, abs(mean(nn[0]) - mean(nn[1])), abs(mean(nn[2]) - mean(nn[1])))
    table("tab_app_noiseunits", lines)
    print("numbers (noise units)")
    bits = {}
    for r in G:
        bits.setdefault((r["arm"], *key(r)), {})[float(r["c"])] = r
    nn = [v for k, v in bits.items() if k[0] == "nnorm"]
    cl = [v for k, v in bits.items() if k[0] == "colide_ev"]
    num("noise-unit data sets", f"{len(nn)}")
    num(
        "identical thresholded graphs across c %",
        100 * float(np.mean([len({x["graph_bits"] for x in v.values()}) == 1 for v in nn])),
        ".2f",
    )
    num(
        "CoLiDE-EV identical across c %",
        100 * float(np.mean([len({x["graph_bits"] for x in v.values()}) == 1 for v in cl])),
        ".1f",
    )
    diff = [
        sum(a != b for a, b in zip(v[c]["graph_bits"], v[1.0]["graph_bits"], strict=True))
        for v in nn
        for c in (0.01, 100.0)
    ]
    dif = [x for x in diff if x > 0]
    num("median differing entries (non-identical pairs)", f"{np.median(dif):g}")
    num("max differing entries", f"{max(dif)}")
    dF = [abs(float(v[c]["proj_f1"]) - float(v[1.0]["proj_f1"])) for v in nn for c in (0.01, 100.0)]
    num("mean |dF1| vs c=1", float(np.mean(dF)), ".4f")
    num("max |dF1| vs c=1", float(np.max(dF)), ".3f")
    num("cells within 0.05 or better (spread<=8)", f"{sum(within)} of {len(within)}")
    num("max |nnorm c=0.01/100 - c=1| per cell", maxdev)


def spread_and_solver():
    F = rows(f"{R}/spread/spread.csv")
    print("\nnumbers (trace normalisation, part A, c=1)")
    for w in ("0.5", "1.0", "2.0"):
        for a in ("ref", "gnorm"):
            x = [r for r in F if r["part"] == "A" and r["arm"] == a and r["wscale"] == w and r["c"] == "1.0"]
            num(f"part A {a} w={wlab(w)} F1", mean(x))
    gA = [r for r in F if r["part"] == "A" and r["arm"] == "gnorm"]
    ws = {r["wscale"] for r in gA}
    num(
        "part A gnorm mean F1 range across c (per w, max)",
        max(
            np.ptp([mean([r for r in gA if r["wscale"] == w and r["c"] == c]) for c in ("0.01", "1.0", "100.0")])
            for w in ws
        ),
    )
    num(
        "part A gnorm F1 range across c (per data set, max)",
        max(np.ptp([float(r["proj_f1"]) for r in gA if key(r, "wscale") == k]) for k in {key(r, "wscale") for r in gA}),
    )
    Bp = [r for r in F if r["part"] == "B"]
    lines = []
    for lo, hi, lab in (
        (0, 4, "0 to 4"),
        (4, 8, "4 to 8"),
        (8, 12, "8 to 12"),
        (12, 99, "12 or more"),
    ):
        ref = [r for r in Bp if r["arm"] == "ref" and lo <= float(r["spread"]) < hi]
        gn = [r for r in Bp if r["arm"] == "gnorm" and lo <= float(r["spread"]) < hi]
        lines.append(
            f"{lab} & {len(ref)} & {mean(ref):.2f} ({share(ref, lambda r: bad(r['h'])):.2f}) & "
            f"{mean(gn):.2f} ({share(gn, lambda r: bad(r['h'])):.2f}) \\\\"
        )
    table("tab_spread", lines)
    ref = [r for r in Bp if r["arm"] == "ref"]
    num("raw F1, spread < 8", mean([r for r in ref if float(r["spread"]) < 8]))
    num(
        "trace-normalised non-acyclic, all part B",
        share([r for r in Bp if r["arm"] == "gnorm"], lambda r: bad(r["h"])),
    )
    fin = [r for r in ref if r["kappa"] not in ("", "inf") and np.isfinite(float(r["kappa"]))]
    num("high-spread data sets with finite kappa", f"{len(fin)}")
    num(
        "Spearman log kappa vs spread",
        spearmanr([float(r["kappa"]) for r in fin], [float(r["spread"]) for r in fin])[0],
    )

    W = rows(f"{R}/objective_gap/objective_gap.csv")
    print("numbers (objective ratios, witness)")
    num("witness data sets", f"{len(W)}")
    snr = []
    num("witness data sets with kappa = inf", f"{sum(float(r['kappa']) == np.inf for r in W)}")
    bins = (
        (0, 1e8, "<1e8"),
        (1e8, 1e12, "1e8-1e12"),
        (1e12, 1e14, "1e12-1e14"),
        (1e14, 1e16, "1e14-1e16"),
    )
    for lo, hi, lab in (*bins, (1e16, np.inf, ">=1e16 finite"), (np.inf, np.inf, "inf")):
        x = [r for r in W if lo <= float(r["kappa"]) < hi or lo == hi == float(r["kappa"])]
        num(
            f"median F_ref/F_truth kappa {lab} (n={len(x)})",
            float(np.median([float(r["F_ref"]) / float(r["F_truth"]) for r in x])),
            ".3g",
        )
        snr.append(float(np.median([float(r["F_snr"]) / float(r["F_truth"]) for r in x])))
    num(
        "sortnregress F/F_truth, range of bin medians (finite kappa)",
        f"{min(snr[:5]):.2f} to {max(snr[:5]):.2f}",
    )

    V = [r for r in rows(f"{R}/simulate/simulate.csv") if r["wscale"] in ("2.0", "4.0") and int(r["seed"]) < 40]
    Sch = rows(f"{R}/schedule/schedule.csv")
    kap = {key(r, "wscale"): float(r["kappa_cov"]) for r in V if r["arm"] == "ref"}
    Fw = {key(r, "wscale"): r for r in W}
    obj = {"ref": "F_ref", "pc_abs10": "F_pc_abs10"}

    def ratio(r):
        f = float(Fw[key(r, "wscale")][obj[r["arm"]]]) if r["arm"] in obj else float(r["objective"])
        return f / float(Fw[key(r, "wscale")]["F_truth"])

    arms = [
        ("ref", "DAGMA, $T = 5$"),
        ("ref_T8", "DAGMA, $T = 8$"),
        ("ref_T12", "DAGMA, $T = 12$"),
        ("pc_abs10", "Preconditioned direction"),
        ("notears", "NOTEARS"),
    ]
    sched, rem, counts = [], {}, {}
    for lo, hi, lab in (
        (1e8, 1e12, "$[10^{8},10^{12})$"),
        (1e12, 1e14, "$[10^{12},10^{14})$"),
        (1e14, 1e16, "$[10^{14},10^{16})$"),
    ):
        for i, (a, name) in enumerate(arms):
            x = [r for r in V + Sch if r["arm"] == a and lo <= kap[key(r, "wscale")] < hi]
            vals = (
                mean(x),
                share(x, lambda r: not bad(r["h"])),
                float(np.median([ratio(r) for r in x])),
                mean(x, "wall"),
            )
            rem[(lo, a)] = vals
            counts[(lo, a)] = len(x)
            r = f"{vals[2]:.1f}"
            sched.append(f"{lab if i == 0 else ''} & {name} & {vals[0]:.2f} & {vals[1]:.2f} & {r} & {vals[3]:.0f} \\\\")
        sched.append(r"\midrule")
    table("tab_app_schedule", sched[:-1])
    lines = []
    for a, name in arms:
        v1, v2 = rem[(1e12, a)], rem[(1e14, a)]
        lines.append(f"{name} & {v1[0]:.2f} & {v1[1]:.2f} & {v1[3]:.0f} & {v2[0]:.2f} & {v2[1]:.2f} & {v2[3]:.0f} \\\\")
    table("tab_remedies", lines)
    print("numbers (remedies)")
    num(
        "data sets kappa [1e12,1e14) / [1e14,1e16)",
        f"{counts[(1e12, 'ref')]} / {counts[(1e14, 'ref')]}",
    )
    num(
        "NOTEARS data sets kappa [1e12,1e14) / [1e14,1e16)",
        f"{counts[(1e12, 'notears')]} / {counts[(1e14, 'notears')]}",
    )
    num(
        "T12/T5 time ratio by bin",
        ", ".join(f"{rem[(lo, 'ref_T12')][3] / rem[(lo, 'ref')][3]:.1f}" for lo in (1e8, 1e12, 1e14)),
    )

    N = rows(f"{R}/newton_cg/newton_cg.csv")
    arm = {a: {(r["graph"], r["d"], r["seed"]): r for r in N if r["arm"] == a} for a in {r["arm"] for r in N}}
    print("numbers (Newton-CG)")
    num("Newton data sets", f"{len(arm['nt_nc'])}")
    num(
        "log10 kappa range",
        f"{min(float(r['log10_kappa']) for r in N):.2f} to {max(float(r['log10_kappa']) for r in N):.2f}",
    )
    for a in ("ref", "nt_nc"):
        num(
            f"{a} median F/F_truth",
            float(np.median([float(r["rF"]) for r in arm[a].values()])),
            ".3g",
        )
        num(f"{a} mean projected F1", mean(arm[a].values(), "pf1"))
        num(f"{a} acyclic outputs", f"{sum(r['feasible'] == '1' for r in arm[a].values())}")
    for a in ("nt_nc_truthlast", "nt_nc_snr", "nt_nc_snrlast"):
        num(f"{a} mean projected F1", mean(arm[a].values(), "pf1"))
    for t in ("wall", "cpu"):
        rr = [float(arm["nt_nc"][k][t]) / float(arm["ref"][k][t]) for k in arm["ref"]]
        num(
            f"nt_nc/ref {t} ratio median (range)",
            f"{np.median(rr):.2f} ({min(rr):.2f} to {max(rr):.2f})",
        )


def maskdiag(window_cell, D10):
    M = rows(f"{R}/masked_diagonal/masked_diagonal.csv")
    cs = (1.0, 10.0, 20.0, 30.0, 50.0, 100.0, 300.0, 1000.0)
    by = {c: [r for r in M if float(r["c"]) == c] for c in cs}
    free = {c: [r for r in D10 if float(r["c"]) == c] for c in cs}
    lines = [
        "Non-acyclic & " + " & ".join(f"{share(by[c], lambda r: r['non_acyclic'] == 'True'):.2f}" for c in cs) + r" \\",
        "Non-acyclic, free diagonal & "
        + " & ".join(f"{share(free[c], lambda r: r['infeasible'] == 'True'):.2f}" if free[c] else "--" for c in cs)
        + r" \\",
        "Cycle & " + " & ".join(f"{share(by[c], lambda r: r['cyclic'] == 'True'):.2f}" for c in cs) + r" \\",
        "F1 & " + " & ".join(f"{mean(by[c]):.2f}" for c in cs) + r" \\",
    ]
    table("tab_app_maskdiag", lines)
    print("numbers (masked diagonal)")
    num("data sets per column", f"{sorted({len(x) for x in by.values()})}")
    f = [mean(by[c]) for c in cs if c >= 20]
    num("masked F1 range c>=20", f"{min(f):.2f} to {max(f):.2f}")
    num(
        "free-diagonal F1 at c=1000, w<=1, max cell",
        max(mean(v) for k, v in window_cell.items() if k[3] == "1000.0" and k[2] in ("0.5", "1")),
    )


def realdata():
    subprocess.run(
        [sys.executable, "scripts/table_real.py", f"{OUT}/table_realdata.tex"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    lines = [
        ln for ln in open(f"{OUT}/table_realdata.tex").read().splitlines() if ln.count("&") == 4 or "floor &" in ln
    ]
    print("\n== table_realdata (written by table_real.py)")
    print("\n".join(lines))
    B = rows(
        f"{R}/methods/methods.csv.gz",
        f"{R}/methods/methods_recent.csv.gz",
        f"{R}/methods/methods_std.csv",
    )
    sa = {(r["graph"], r["variant"], r["method"]): r for r in B if r["part"] == "sachs"}
    C = rows(f"{R}/light_tunnel/light_tunnel.csv", f"{R}/light_tunnel/light_tunnel_recent.csv")
    ch = {(r["variant"], r["method"]): r for r in C}
    CF = {(r["variant"], r["method"]): r for r in rows(f"{R}/light_tunnel/light_tunnel_rerun.csv")}
    print("numbers (real data)")
    X = load(COND1)
    v = X.var(axis=0)
    num("Sachs cells x variables", f"{X.shape[0]} x {X.shape[1]}")
    num("Sachs spread (decades)", float(np.log10(v.max() / v.min())), ".1f")
    num("Chambers spread (decades)", float(CF[("raw", "dagma")]["spread"]), ".1f")
    num("Sachs raw DAGMA h", float(sa[("cond1_853", "raw", "dagma")]["h_logdet"]))
    num("Chambers raw DAGMA h", float(CF[("raw", "dagma")]["h_logdet"]), ".2g")
    cv = 1.0 / np.diag(np.linalg.inv(np.cov(X, rowvar=False)))
    num("Sachs largest leading-order self-loop mu s cv/2", float(1e-4 * 0.6 * cv.max() / 2), ".2g")
    readout = open(f"{R}/noise_units_real/readout.txt").read().splitlines()
    for ln in readout:
        if ln.startswith("cond1_853") and " ref " in ln and ("log" in ln or "std" in ln):
            num(f"Sachs floor ({ln.split()[1]}, noise_units_real readout)", ln.split()[4])
    for var in ("log", "std"):
        num(f"Sachs {var} DAGMA h", float(sa[("cond1_853", var, "dagma")]["h_logdet"]), ".2g")
    num("Sachs std DAGMA F1", float(sa[("cond1_853", "std", "dagma")]["proj_f1"]))
    num(
        "Chambers std DAGMA F1 (acyclic)",
        f"{float(CF[('std', 'dagma')]['proj_f1']):.2f} ({not bad(CF[('std', 'dagma')]['h_logdet'])})",
    )
    num(
        "Chambers AUPRC raw DAGMA / noise units",
        f"{float(ch[('raw', 'dagma')]['auprc_std']):.2f} / {float(ch[('raw', 'dagma_nnorm')]['auprc_std']):.2f}",
    )
    num("pooled raw DAGMA non-acyclic", f"{bad(sa[('pooled_7466', 'raw', 'dagma')]['h_logdet'])}")
    num(
        "pooled raw F1 noise units / DAGMA",
        f"{float(sa[('pooled_7466', 'raw', 'dagma_nnorm')]['proj_f1']):.2f} / {float(sa[('pooled_7466', 'raw', 'dagma')]['proj_f1']):.2f}",
    )
    lt = [float(r["proj_f1"]) for (v, m), r in {**ch, **CF}.items() if v in ("log", "std")]
    num("Chambers log/std max F1 (all methods)", max(lt))
    num("Chambers raw sortnregress F1", float(ch[("raw", "sortnregress")]["proj_f1"]))


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    wc = scale_window()
    leak_law()
    D10 = prediction()
    breadth()
    noise_units()
    spread_and_solver()
    maskdiag(wc, D10)
    realdata()
    with open(f"{OUT}/numbers.txt", "w") as fh:
        fh.write("\n".join(NUMBERS) + "\n")
    print(f"\nwrote {OUT}/ ({len(NUMBERS)} numbers) in {time.time() - t0:.1f} s")
