"""Local structural-identifiability analysis for clamped reoxygenation.

The cellular system is linear at fixed oxygen N_r = Nhat and starts from the
purely quiescent state x0 = (0, 0, 1):
    X' = A X,   A = [[-(kap+al+dR), 2 p k2, be], [kap, -(k2+dG), 0], [al, 2(1-p) k2, -(be+dQ)]].
Outputs are the green FUCCI fraction f = G/P, total abundance P, and the
state-weighted readout y_AO = (wR R + wG G + wQ Q)/P.

Taylor coefficients at t = 0 are computed through ``ORDER`` using exact
rational arithmetic. Exact parameter derivatives and Jacobian ranks establish
generic local identifiability following the rank principle used by SIAN.

Output: ``results/p1_identifiability.json``.
"""
import json
import os
import sys
from fractions import Fraction
from math import factorial

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as M  # noqa: E402

ORDER = 13
NAMES = ["kappa", "alpha", "dR", "k2", "dG", "p", "beta", "dQ"]
W_AO = (Fraction(1), Fraction(23, 20), Fraction(7, 20))      # State weights (R, G, Q)


def A_and_dA(th):
    kap, al, dR, k2, dG, p, be, dQ = (th[n] for n in NAMES)
    A = [[-(kap + al + dR), 2 * p * k2, be], [kap, -(k2 + dG), 0], [al, 2 * (1 - p) * k2, -(be + dQ)]]
    Z = lambda: [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    d = {n: Z() for n in NAMES}
    d["kappa"][0][0], d["kappa"][1][0] = -1, 1
    d["alpha"][0][0], d["alpha"][2][0] = -1, 1
    d["dR"][0][0] = -1
    d["k2"][0][1], d["k2"][1][1], d["k2"][2][1] = 2 * p, -1, 2 * (1 - p)
    d["dG"][1][1] = -1
    d["p"][0][1], d["p"][2][1] = 2 * k2, -2 * k2
    d["beta"][0][2], d["beta"][2][2] = 1, -1
    d["dQ"][2][2] = -1
    return A, d


def mv(A, x):
    return [sum(A[i][j] * x[j] for j in range(3)) for i in range(3)]


def series(th):
    """Return coefficients and derivatives for G, P, and the weighted numerator."""
    A, dA = A_and_dA(th)
    m = [Fraction(0), Fraction(0), Fraction(1)]
    dm = {n: [Fraction(0)] * 3 for n in NAMES}
    out = {"G": [], "P": [], "AO": []}
    dout = {k: {n: [] for n in NAMES} for k in out}
    for j in range(ORDER + 1):
        fj = Fraction(1, factorial(j))
        rows = {"G": (0, 1, 0), "P": (1, 1, 1), "AO": W_AO}
        for key, w in rows.items():
            out[key].append(fj * sum(wi * mi for wi, mi in zip(w, m)))
            for n in NAMES:
                dout[key][n].append(fj * sum(wi * di for wi, di in zip(w, dm[n])))
        new_dm = {n: [a + b for a, b in zip(mv(dA[n], m), mv(A, dm[n]))] for n in NAMES}
        m, dm = mv(A, m), new_dm
    return out, dout


def divide(num, dnum, den, dden):
    """Return the series q = num/den and its derivatives (den[0] != 0)."""
    q, dq = [], {n: [] for n in NAMES}
    for j in range(len(num)):
        s = num[j] - sum(q[i] * den[j - i] for i in range(j))
        q.append(s / den[0])
        for n in NAMES:
            ds = dnum[n][j] - sum(dq[n][i] * den[j - i] + q[i] * dden[n][j - i] for i in range(j))
            dq[n].append((ds - q[j] * dden[n][0]) / den[0])
    return q, dq


def jac_rows(th, outputs, params):
    s, ds = series(th)
    rows = []
    f, df = divide(s["G"], ds["G"], s["P"], ds["P"])
    if "fraction" in outputs:
        rows += [[df[n][j] for n in params] for j in range(ORDER + 1)]
    if "abundance" in outputs:
        rows += [[ds["P"][n][j] for n in params] for j in range(ORDER + 1)]
    if "AO" in outputs:
        a, da = divide(s["AO"], ds["AO"], s["P"], ds["P"])
        rows += [[da[n][j] for n in params] for j in range(ORDER + 1)]
    return rows


def exact_rank(rows):
    return sp.Matrix([[sp.Rational(x.numerator, x.denominator) for x in r] for r in rows]).rank()


def to_frac(x):
    return Fraction(x).limit_denominator(10 ** 9)


def point_from_params(pr):
    N = pr.Nhat
    return {"kappa": to_frac(M.kappa(N, pr)), "alpha": to_frac(M.alpha(N, pr)), "dR": to_frac(M.muR(N, pr)),
            "k2": to_frac(pr.k2), "dG": to_frac(M.muG(N, pr)), "p": to_frac(pr.p),
            "beta": to_frac(M.beta(N, pr)), "dQ": to_frac(M.muQ(N, pr))}


def random_point(rng):
    lu = lambda a, b: Fraction(int(np.exp(rng.uniform(np.log(a), np.log(b))) * 10 ** 6), 10 ** 6)
    return {"kappa": lu(0.005, 0.5), "alpha": lu(1e-4, 0.1), "dR": lu(1e-4, 0.05), "k2": lu(0.05, 0.3),
            "dG": lu(1e-4, 0.05), "p": Fraction(int(rng.uniform(0.5, 0.99) * 1000), 1000),
            "beta": lu(0.003, 0.3), "dQ": lu(1e-4, 0.05)}


SUBSETS = {"(kappa, beta)": ["kappa", "beta"],
           "(kappa, beta, k2, p)": ["kappa", "beta", "k2", "p"],
           "(kappa, beta, k2, p, alpha, dQ)": ["kappa", "beta", "k2", "p", "alpha", "dQ"],
           "all eight rates": NAMES}
SCEN = {"green fraction": ["fraction"], "green fraction + total abundance": ["fraction", "abundance"],
        "green fraction + AO readout": ["fraction", "AO"],
        "green fraction + abundance + AO": ["fraction", "abundance", "AO"]}


def main():
    import json as _json
    c1 = _json.load(open(os.path.join(ROOT, "results", "p0_candidates.json"), encoding="utf-8"))
    pr = M.Params(**c1["candidates"]["C1"]["params"])
    pts = {"C1": point_from_params(pr)}
    rng = np.random.default_rng(7)
    for i in range(3):
        pts[f"random{i + 1}"] = random_point(rng)
    table = {}
    for sname, params in SUBSETS.items():
        for scen, outs in SCEN.items():
            ranks = {k: exact_rank(jac_rows(th, outs, params)) for k, th in pts.items()}
            table[f"{sname} | {scen}"] = dict(n_params=len(params), ranks=ranks)
            print(f"{sname:34s} | {scen:36s} | rank " + " ".join(f"{r}" for r in ranks.values())
                  + f" / {len(params)}", flush=True)
    # Non-identifiable log-parameter direction for the six-parameter fraction-only case
    params = SUBSETS["(kappa, beta, k2, p, alpha, dQ)"]
    th = pts["C1"]
    Jm = np.array([[float(x) for x in r] for r in jac_rows(th, ["fraction"], params)])
    Jl = Jm * np.array([float(th[n]) for n in params])
    Jl = Jl[np.abs(Jl).max(axis=1) > 0]
    Jl /= np.abs(Jl).max(axis=1, keepdims=True)
    _, s, vt = np.linalg.svd(Jl)
    null = vt[-1] / np.abs(vt[-1]).max()
    print("null direction (log-params", params, "):", np.round(null, 4), " sv ratio", s[-1] / s[0])
    res = dict(order=ORDER, points={k: {n: str(v) for n, v in th.items()} for k, th in pts.items()},
               table=table, null_direction_C1_fraction_6params=dict(params=params, vector=null.tolist(),
                                                                   sv_ratio=float(s[-1] / s[0])))
    with open(os.path.join(ROOT, "results", "p1_identifiability.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
