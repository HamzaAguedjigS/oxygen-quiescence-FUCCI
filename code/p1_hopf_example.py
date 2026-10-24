"""Characterize a representative Hopf bifurcation.

Starting from the selected unstable set in ``p1_hopf_analysis.json``, the entry
Hill exponent is continued to locate the critical value. Simulations above the
critical point estimate the amplitude scaling and oscillation period.

Outputs: ``results/p1_hopf_example.json`` and
``results/p1_hopf_example.npz``.
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
from p1_stability_scan import SEED, N_LOCAL, draw, load_C1  # noqa: E402

DAY = 24.0


def lead(pr):
    E = M.equilibrium(pr)
    ev = np.linalg.eigvals(M.jacobian(E, pr))
    i = np.argmax(ev.real)
    return ev[i], E


def amplitude(pr, E, T_days=4000, tail=0.25):
    T = T_days * DAY
    s = solve_ivp(M.rhs, (0, T), E * (1 + 1e-2), args=(pr,), method="LSODA", rtol=1e-10, atol=1e-13,
                  dense_output=True)
    tt = np.linspace((1 - tail) * T, T, 40001)
    Y = s.sol(tt)
    P = Y[:3].sum(0) / E[:3].sum()
    fred = (Y[0] + Y[2]) / Y[:3].sum(0)
    z = P - P.mean()
    up = tt[1:][(z[:-1] < 0) & (z[1:] >= 0)]
    per = float(np.mean(np.diff(up)) / DAY) if len(up) > 2 else None
    return float(P.max() - P.min()), per, tt, Y, fred


def main():
    base = load_C1()
    # Hill-exponent statistics among unstable parameter sets
    rng = np.random.default_rng(SEED)
    tot = dict(m_le2=[0, 0], n_le2=[0, 0], both_le2=[0, 0])
    for _ in range(N_LOCAL):
        pr = draw(base, rng)
        try:
            if M.R0(pr) <= 1.0:
                continue
            ev, _ = lead(pr)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        u = ev.real >= 0
        for key, cond in (("m_le2", pr.nal <= 2), ("n_le2", pr.n <= 2), ("both_le2", pr.nal <= 2 and pr.n <= 2)):
            if cond:
                tot[key][0] += 1
                tot[key][1] += int(u)
    print("invasive / unstable with m<=2, n<=2, both:", tot)

    ex = json.load(open(os.path.join(ROOT, "results", "p1_hopf_analysis.json"), encoding="utf-8"))["example"]
    pr = base.copy(**ex["params"])
    m0 = pr.nal
    f = lambda m: lead(pr.copy(nal=m))[0].real
    grid = np.linspace(1.0, m0, 200)
    vals = np.array([f(m) for m in grid])
    i = np.where(np.diff(np.sign(vals)) > 0)[0][0]
    mH = brentq(f, grid[i], grid[i + 1], xtol=1e-12)
    evH, EH = lead(pr.copy(nal=mH))
    print(f"Hopf at m_H = {mH:.6f}, eigenvalue {evH}, period at onset {2 * np.pi / abs(evH.imag) / DAY:.2f} d")
    rows = []
    for dm in (0.005, 0.01, 0.02, 0.04, 0.08):
        q = pr.copy(nal=mH + dm)
        _, E = lead(q)
        amp, per, *_ = amplitude(q, E)
        rows.append(dict(dm=dm, amp=amp, period_days=per))
        print(f"  m = m_H + {dm}: amplitude of P/P* = {amp:.4f}, period {per} d")
    below = []
    for dm in (0.01, 0.04):
        q = pr.copy(nal=mH - dm)
        _, E = lead(q)
        amp, per, *_ = amplitude(q, E)
        below.append(dict(dm=-dm, amp=amp))
    # Test the expected square-root amplitude scaling
    a = np.array([r["amp"] for r in rows])
    d = np.array([r["dm"] for r in rows])
    slope = float(np.polyfit(np.log(d), np.log(a), 1)[0]) if (a > 0).all() else None
    print("log-log slope of amplitude vs (m - m_H):", slope, " below Hopf:", below)
    # Store a representative limit-cycle trajectory
    _, E0 = lead(pr)
    amp, per, tt, Y, fred = amplitude(pr, E0, T_days=3000, tail=0.1)
    np.savez(os.path.join(ROOT, "results", "p1_hopf_example.npz"), grid=grid, maxre=vals, t=tt, Y=Y, fred=fred,
             Pstar=E0[:3].sum(), mH=mH)
    out = dict(stats_m_n_le2=tot, base_params=ex["params"], m_example=m0, m_Hopf=mH,
               eig_at_Hopf=[float(evH.real), float(evH.imag)], period_onset_days=float(2 * np.pi / abs(evH.imag) / DAY),
               above=rows, below=below, amp_loglog_slope=slope,
               example_cycle=dict(P_range=[float(Y[:3].sum(0).min() / E0[:3].sum()),
                                           float(Y[:3].sum(0).max() / E0[:3].sum())],
                                  fred_range=[float(fred.min()), float(fred.max())], period_days=per))
    with open(os.path.join(ROOT, "results", "p1_hopf_example.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
