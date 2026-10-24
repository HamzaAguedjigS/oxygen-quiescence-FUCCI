"""Compute threshold, reservoir, exposure, and sensitivity results.

The calculations cover the reproduction-number geometry, equilibrium branch,
two-control boundary, formation of the quiescent reservoir, finite exposure
protocols, post-withdrawal regrowth, and local elasticities at the reference
regime.

Outputs: ``results/p2_dynamics.npz`` and ``results/p2_dynamics.json``.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp as _solve_ivp
from scipy.optimize import brentq


def solve_ivp(*args, **kwargs):
    """Run ``solve_ivp`` and enforce solver success and state non-negativity."""
    s = _solve_ivp(*args, **kwargs)
    assert s.success, "solver failure"
    assert s.y.min() > -1e-8, "negative state"
    return s

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as M  # noqa: E402

DAY = 24.0
SEED = 2026
RTOL, ATOL = 1e-10, 1e-13


def load_C1():
    c = json.load(open(os.path.join(ROOT, "results", "p0_candidates.json"), encoding="utf-8"))
    return M.Params(**c["candidates"]["C1"]["params"])


def perron(N, pr):
    w, V = np.linalg.eig(M.A(N, pr))
    v = np.abs(V[:, np.argmax(w.real)].real)
    return v / v.sum()


def integrate(pr, y0, T, n=2001):
    s = solve_ivp(M.rhs, (0, T), y0, args=(pr,), method="LSODA", rtol=RTOL, atol=ATOL, dense_output=True)
    assert s.success
    t = np.linspace(0, T, n)
    Y = s.sol(t)
    assert Y.min() > -1e-8
    return t, np.clip(Y, 0, None)


def frontier_formula(pr, u2):
    """Return u1*(u2), or NaN when no threshold exists."""
    q = pr.copy(u1=0.0, u2=u2)
    N = q.Nhat
    piS, piM, piret, _ = M.R0_factors(q)
    k1h = q.k1bar * N / (q.KN + N)
    c = M.beta(N, q) + M.muQ(N, q)
    m = M.muR(N, q) + M.alpha(N, q) * M.muQ(N, q) / c
    if 2 * piM * piret <= 1:
        return np.nan
    return 1 - m / (k1h * (2 * piM * piret - 1))


def main():
    t0 = time.time()
    pr = load_C1()
    out, arr = {}, {}
    E = M.equilibrium(pr)
    Pst = E[:3].sum()
    u1c = M.u1_star(pr)
    out.update(Pstar=Pst, Nstar=E[3], u1c=u1c, Lambda=M.growth_exponent(pr))

    # (A) Threshold geometry
    u1 = np.linspace(0, 0.999, 400)
    fac = np.array([M.R0_factors(pr.copy(u1=u)) for u in u1])
    arr.update(A_u1=u1, A_piS=fac[:, 0], A_piM=fac[:, 1], A_piret=fac[:, 2], A_R0=fac[:, 3])
    Pb, stab = [], []
    for u in u1:
        q = pr.copy(u1=u)
        Eq = M.equilibrium(q)
        if Eq is None:
            Pb.append(0.0)
            stab.append(np.nan)
        else:
            Pb.append(Eq[:3].sum())
            stab.append(np.linalg.eigvals(M.jacobian(Eq, q)).real.max())
    arr.update(A_Pstar=np.array(Pb), A_Estar_maxRe=np.array(stab),
               A_Lambda=np.array([M.growth_exponent(pr.copy(u1=u)) for u in u1]))
    out["Estar_stable_on_branch"] = bool(np.nanmax(arr["A_Estar_maxRe"]) < 0)
    # Lambda(u1, u2) map
    g1, g2 = np.linspace(0, 0.999, 201), np.linspace(0, 1, 201)
    L = np.array([[M.growth_exponent(pr.copy(u1=a, u2=b)) for a in g1] for b in g2])
    arr.update(A_g1=g1, A_g2=g2, A_Lmap=L)
    ff = np.array([frontier_formula(pr, b) for b in g2])
    fn = []
    for b in g2:
        f = lambda a: M.growth_exponent(pr.copy(u1=a, u2=b))
        fn.append(brentq(f, 0, 0.999999, xtol=1e-14) if f(0) > 0 > f(0.999999) else (0.0 if f(0) <= 0 else np.nan))
    fn = np.array(fn)
    arr.update(A_front_formula=ff, A_front_numeric=fn)
    ok = np.isfinite(ff) & np.isfinite(fn) & (ff > 0)
    out["frontier_max_abs_dev"] = float(np.abs(ff[ok] - fn[ok]).max())
    f0 = lambda b: M.growth_exponent(pr.copy(u1=0.0, u2=b))
    out["u2c_at_u1_0"] = float(brentq(f0, 0, 1)) if f0(0) > 0 > f0(1) else None

    # (B) Quiescent-reservoir formation
    vN = perron(pr.Nhat, pr)
    t, Y = integrate(pr, np.r_[0.01 * Pst * vN, pr.Nhat], 80 * DAY)
    arr.update(B_t=t, B_Y=Y)
    P = Y[:3].sum(0)
    out["T90_days"] = float(t[np.argmax(P >= 0.9 * Pst)] / DAY)
    out["N_half_days"] = float(t[np.argmax(Y[3] <= 0.5 * (pr.Nhat + E[3]))] / DAY)
    out["Pmax_over_Pstar"] = float(P.max() / Pst)
    rng = np.random.default_rng(SEED)
    traj, P0s = [], []
    for _ in range(30):
        P0 = Pst * float(np.exp(rng.uniform(np.log(0.05), np.log(3))))
        y0 = np.r_[P0 * rng.dirichlet([1, 1, 1]), rng.uniform(0.05, pr.Nhat)]
        tt, YY = integrate(pr, y0, 80 * DAY, n=801)
        traj.append(YY[:3].sum(0) / Pst)
        P0s.append(P0 / Pst)
    arr.update(B_tm=tt, B_multi=np.array(traj))
    out["multistart_P0_range"] = [float(min(P0s)), float(max(P0s))]
    out["multistart_final_maxdev"] = float(np.abs(np.array(traj)[:, -1] - 1).max())

    # (C) Finite exposure
    yE = E.copy()
    U = np.linspace(0.80, 0.999, 120)
    T = np.linspace(0, 120 * DAY, 601)
    RES = np.zeros((len(T), len(U)))
    for j, u in enumerate(U):
        s = solve_ivp(M.rhs, (0, T[-1]), yE, args=(pr.copy(u1=u),), method="LSODA", rtol=1e-9, atol=1e-14,
                      dense_output=True)
        RES[:, j] = s.sol(T)[:3].sum(0) / Pst
    arr.update(C_U=U, C_T=T, C_RES=RES)
    d = np.logspace(-3, np.log10(0.999 - u1c), 24)
    lam = np.array([M.growth_exponent(pr.copy(u1=u1c + x)) for x in d])
    Teps = {}
    for e in (1e-2, 1e-4, 1e-6):
        Teps[e] = []
    for x, la in zip(d, lam):
        tend = 18.0 / abs(la) + 200 * DAY
        s = solve_ivp(M.rhs, (0, tend), yE, args=(pr.copy(u1=u1c + x),), method="LSODA", rtol=1e-10,
                      atol=1e-18, dense_output=True)
        tt = np.linspace(0, tend, 40001)
        PP = s.sol(tt)[:3].sum(0) / Pst
        for e in Teps:
            Teps[e].append(tt[np.argmax(PP < e)] if (PP < e).any() else np.nan)
    arr.update(C_d=d, C_lam=lam, C_T2=np.array(Teps[1e-2]), C_T4=np.array(Teps[1e-4]), C_T6=np.array(Teps[1e-6]))
    # Exposure-withdrawal protocol
    u_on, T_on, T_off = 0.95, 28 * DAY, 120 * DAY
    s1 = solve_ivp(M.rhs, (0, T_on), yE, args=(pr.copy(u1=u_on),), method="LSODA", rtol=RTOL, atol=1e-14,
                   dense_output=True)
    yw = s1.y[:, -1]
    s2 = solve_ivp(M.rhs, (0, T_off), yw, args=(pr,), method="LSODA", rtol=RTOL, atol=1e-14, dense_output=True)
    t1 = np.linspace(0, T_on, 400)
    t2 = np.linspace(0, T_off, 1200)
    arr.update(C_proto_t=np.r_[-10 * DAY, t1, T_on + t2], C_proto_Y=np.c_[yE[:, None], s1.sol(t1), s2.sol(t2)])
    Pw = yw[:3].sum()
    P2 = s2.sol(t2)[:3].sum(0)
    out["protocol"] = dict(u1=u_on, days_on=T_on / DAY, residual_fraction=float(Pw / Pst),
                           comp_at_withdrawal=(yw[:3] / Pw).tolist(),
                           Q_share_RQ=float(yw[2] / (yw[0] + yw[2])),
                           fred_at_withdrawal=float((yw[0] + yw[2]) / Pw),
                           T50_days=float(t2[np.argmax(P2 >= 0.5 * Pst)] / DAY),
                           T90_days=float(t2[np.argmax(P2 >= 0.9 * Pst)] / DAY),
                           Lambda_on=float(M.growth_exponent(pr.copy(u1=u_on))))
    # Sweep withdrawal composition while holding P, G, and N fixed
    S = yw[0] + yw[2]
    qs = np.linspace(0, 1, 41)
    T50, T90 = [], []
    for qf in qs:
        y0 = np.array([(1 - qf) * S, yw[1], qf * S, yw[3]])
        s = solve_ivp(M.rhs, (0, 400 * DAY), y0, args=(pr,), method="LSODA", rtol=RTOL, atol=1e-14,
                      dense_output=True)
        tt = np.linspace(0, 400 * DAY, 8001)
        PP = s.sol(tt)[:3].sum(0)
        T50.append(tt[np.argmax(PP >= 0.5 * Pst)] / DAY)
        T90.append(tt[np.argmax(PP >= 0.9 * Pst)] / DAY)
    arr.update(C_qs=qs, C_T50=np.array(T50), C_T90=np.array(T90))
    out["Qshare_sweep"] = dict(T50_q0=float(T50[0]), T50_q1=float(T50[-1]), T90_q0=float(T90[0]),
                               T90_q1=float(T90[-1]))

    # (D) Local sensitivity
    groups = {  # Effective parameters; shared parameters are perturbed together
        "k1bar": ["k1bar"], "KN": ["KN"], "k2": ["k2"], "p": ["p"], "albar": ["albar"], "Kal": ["Kal"],
        "nal": ["nal"], "bebar": ["bebar"], "Kbe": ["Kbe"], "mu0": ["muR0", "muG0"],
        "mumax": ["muRmax", "muGmax"], "muQmax": ["muQmax"], "Kmu": ["KmuR", "KmuG", "KmuQ"],
        "lamR": ["lamR"], "lamG": ["lamG"], "lamQ": ["lamQ"], "delta": ["delta"]}

    def outputs(q):
        Eq = M.equilibrium(q)
        return np.array([M.growth_exponent(q), Eq[:3].sum(), (Eq[0] + Eq[2]) / Eq[:3].sum(), M.u1_star(q)])

    y0 = outputs(pr)
    Es = []
    for g, keys in groups.items():
        up = pr.copy(**{k: getattr(pr, k) * 1.01 for k in keys})
        dn = pr.copy(**{k: getattr(pr, k) * 0.99 for k in keys})
        Es.append((outputs(up) - outputs(dn)) / (0.02 * y0))
    arr.update(D_E=np.array(Es), D_names=np.array(list(groups)))
    out["sensitivity_outputs_ref"] = y0.tolist()

    np.savez(os.path.join(ROOT, "results", "p2_dynamics.npz"), **arr)
    with open(os.path.join(ROOT, "results", "p2_dynamics.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
