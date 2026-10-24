"""Compute exposure times near the extinction boundary.

At fixed epsilon, the calculation resolves the finite critical-boundary time
associated with algebraic decay. For deeper reductions, it compares the
simulated exposure time with the leading-order estimate log(1/epsilon)/|Lambda|.

Output: ``results/p5_exposure_boundary.json``.
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as M  # noqa: E402

DAY = 24.0
DELTAS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, "large")   # "large" = 0.999 - u1c
EPS = (1e-2, 1e-6)
TMAX_DAYS = 5e6


def load_C1():
    c = json.load(open(os.path.join(ROOT, "results", "p0_candidates.json"), encoding="utf-8"))
    return M.Params(**c["candidates"]["C1"]["params"])


def t_eps(pr, y0, Pst, eps):
    """Return the first day with P <= eps P*, or inf if the cap is reached."""
    ev = lambda t, y, q: y[:3].sum() - eps * Pst  # noqa: E731
    ev.terminal = True
    s = solve_ivp(M.rhs, (0, TMAX_DAYS * DAY), y0, args=(pr,), method="LSODA", rtol=1e-10, atol=1e-20, events=ev)
    assert s.status >= 0 and s.y.min() > -1e-8
    return float(s.t_events[0][0] / DAY) if len(s.t_events[0]) else float("inf")


def main():
    pr = load_C1()
    E = M.equilibrium(pr)
    Pst = E[:3].sum()
    u1c = M.u1_star(pr)
    rows = []
    for d in DELTAS:
        label = d
        d = 0.999 - u1c if d == "large" else d
        q = pr.copy(u1=u1c + d)
        la = M.growth_exponent(q)
        row = dict(label=label, delta=d, Lambda=la)
        for e in EPS:
            row[f"T_{e:g}"] = t_eps(q, E, Pst, e)
        row["leading_1e-06"] = float(np.log(1e6) / abs(la) / DAY) if d > 0 else float("inf")
        row["ratio_1e-06"] = row["T_1e-06"] / row["leading_1e-06"] if d > 0 else None
        rows.append(row)
        print(row)
    # Elasticities for n_beta, which is not included in the 17-parameter screen
    def outputs(qq):
        Eq = M.equilibrium(qq)
        return np.array([M.growth_exponent(qq), Eq[:3].sum(), (Eq[0] + Eq[2]) / Eq[:3].sum(), M.u1_star(qq)])

    y0 = outputs(pr)
    el = (outputs(pr.copy(n=pr.n * 1.01)) - outputs(pr.copy(n=pr.n * 0.99))) / (0.02 * y0)
    print("n_beta elasticities (Lambda, P*, fred*, u1*):", el)
    out = dict(u1c=u1c, Pstar=Pst, start="E* (u1 = 0)", eps=EPS, tmax_days=TMAX_DAYS, rows=rows,
               nbeta_elasticities=dict(zip(("Lambda", "Pstar", "fred", "u1star"), el.tolist())))
    with open(os.path.join(ROOT, "results", "p5_exposure_boundary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
