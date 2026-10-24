"""Analyse unstable positive equilibria from the stability scan.

The script recreates the fixed-seed draws, compares stable and unstable
parameter sets, relates convergence failures to local stability, continues the
reference regime in both Hill exponents, and characterizes a representative
limit cycle.

Output: ``results/p1_hopf_analysis.json``.
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as M  # noqa: E402
from p1_stability_scan import SEED, N_LOCAL, N_GLOBAL, SCALED, draw, load_C1  # noqa: E402

DAY = 24.0


def lead(pr):
    E = M.equilibrium(pr)
    ev = np.linalg.eigvals(M.jacobian(E, pr))
    i = np.argmax(ev.real)
    return ev[i], E


def main():
    base = load_C1()
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(N_LOCAL):
        pr = draw(base, rng)
        try:
            if M.R0(pr) <= 1.0:
                continue
            ev, E = lead(pr)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        rows.append(dict(pr=pr, re=ev.real, im=ev.imag, Nstar=E[3]))
    unst = [r for r in rows if r["re"] >= 0]
    st = [r for r in rows if r["re"] < 0]
    out = dict(n_invasive=len(rows), n_unstable=len(unst))

    def q(lst, f):
        v = np.array([f(r) for r in lst])
        return [float(np.percentile(v, x)) for x in (10, 50, 90)]

    feats = {"m (entry Hill)": lambda r: r["pr"].nal, "n (release Hill)": lambda r: r["pr"].n,
             "p": lambda r: r["pr"].p, "N*": lambda r: r["Nstar"],
             "Kal/N*": lambda r: r["pr"].Kal / r["Nstar"], "Kbe/N*": lambda r: r["pr"].Kbe / r["Nstar"],
             "albar/k1bar": lambda r: r["pr"].albar / r["pr"].k1bar,
             "bebar/k1bar": lambda r: r["pr"].bebar / r["pr"].k1bar}
    out["percentiles_10_50_90"] = {k: dict(stable=q(st, f), unstable=q(unst, f)) for k, f in feats.items()}
    out["unstable_complex_fraction"] = float(np.mean([abs(r["im"]) > 1e-12 for r in unst])) if unst else None
    for k, v in out["percentiles_10_50_90"].items():
        print(f"{k:16s} stable {np.round(v['stable'], 3)}  unstable {np.round(v['unstable'], 3)}")

    # Compare global convergence failures with local stability
    scan = json.load(open(os.path.join(ROOT, "results", "p1_stability_scan.json"), encoding="utf-8"))
    fail_sets = sorted({f["sample"] for f in scan["fails"]})
    first = rows[:N_GLOBAL]
    out["global_fail_sets"] = fail_sets
    out["global_fail_sets_unstable"] = [k for k in fail_sets if first[k]["re"] >= 0]
    out["global_fail_sets_stable"] = [k for k in fail_sets if first[k]["re"] < 0]
    out["n_unstable_in_first_300"] = int(sum(r["re"] >= 0 for r in first))
    print("failure sets:", fail_sets, " unstable among them:", out["global_fail_sets_unstable"],
          " unstable in first 300:", out["n_unstable_in_first_300"])
    # Extend integrations for slowly converging stable parameter sets
    recheck = []
    for k in out["global_fail_sets_stable"]:
        pr = first[k]["pr"]
        ev, E = lead(pr)
        T = 400.0 / abs(ev.real)
        y0 = np.r_[0.01 * E[:3].sum() * np.ones(3) / 3, pr.Nhat]
        s = solve_ivp(M.rhs, (0, T), y0, args=(pr,), method="LSODA", rtol=1e-9, atol=1e-13)
        err = max(np.abs(s.y[:3, -1] - E[:3]).max() / E[:3].sum(), abs(s.y[3, -1] - E[3]))
        recheck.append(dict(sample=k, lead_re=float(ev.real), T_days=float(T / DAY), err=float(err)))
    out["recheck_stable_failures"] = recheck
    print("recheck stable failures:", recheck)

    # Continue the reference regime in the entry and release Hill exponents
    def maxre(**kw):
        return lead(base.copy(**kw))[0].real
    cont = {}
    for name, grid in (("nal", np.linspace(1, 12, 111)), ("n", np.linspace(1, 12, 111))):
        vals = np.array([maxre(**{name: g}) for g in grid])
        cross = None
        idx = np.where(np.diff(np.sign(vals)) > 0)[0]
        if len(idx):
            i = idx[0]
            cross = float(brentq(lambda g: maxre(**{name: g}), grid[i], grid[i + 1]))
        cont[name] = dict(first_hopf=cross, maxre_min=float(vals.min()), maxre_max=float(vals.max()))
    out["continuation_from_C1"] = cont
    print("continuation from C1:", cont)

    # Select the unstable example with the largest relative real part
    if unst:
        ex = max(unst, key=lambda r: r["re"] / abs(complex(r["re"], r["im"])))
        pr = ex["pr"]
        ev, E = lead(pr)
        T = 3000 * DAY
        s = solve_ivp(M.rhs, (0, T), E * (1 + 1e-3), args=(pr,), method="LSODA", rtol=1e-9, atol=1e-13,
                      dense_output=True)
        tt = np.linspace(0.8 * T, T, 20001)
        Y = s.sol(tt)
        P = Y[:3].sum(0)
        fred = (Y[0] + Y[2]) / P
        # Estimate the period from upward crossings of the mean
        z = P - P.mean()
        up = tt[1:][(z[:-1] < 0) & (z[1:] >= 0)]
        per = float(np.mean(np.diff(up)) / DAY) if len(up) > 2 else None
        out["example"] = dict(params={k: float(getattr(pr, k)) for k in SCALED + ["nal", "n", "p"]},
                              lead_eig=[float(ev.real), float(ev.imag)],
                              P_range_over_Pstar=[float(P.min() / E[:3].sum()), float(P.max() / E[:3].sum())],
                              fred_range=[float(fred.min()), float(fred.max())], period_days=per)
        print("example limit cycle:", out["example"]["P_range_over_Pstar"], out["example"]["fred_range"],
              "period (d):", per)
    with open(os.path.join(ROOT, "results", "p1_hopf_analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
