"""Recompute the critical-threshold and numerical quantities added in Phase 7.

The calculations cover the ``R0 = 1`` asymptotics, exact regrowth times,
profile-likelihood limits, eigenvalues at the positive equilibrium, and the
validity of the clamped-oxygen approximation. Results are written to
``results/p7_critical_and_numbers.json``.
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as M  # noqa: E402
import obs_model as O  # noqa: E402
import p2_observability as PO  # noqa: E402

DAY = 24.0


def perron(Am):
    w, V = np.linalg.eig(Am)
    return np.abs(V[:, np.argmax(w.real)].real)


def first_time(pr, y0, target, tmax, direction):
    ev = lambda t, y, q: y[:3].sum() - target  # noqa: E731
    ev.terminal, ev.direction = True, direction
    s = solve_ivp(M.rhs, (0, tmax), y0, args=(pr,), method="LSODA", rtol=1e-10, atol=1e-20, events=ev)
    assert s.status >= 0 and s.y.min() > -1e-8
    return float(s.t_events[0][0]) if len(s.t_events[0]) else float("inf")


def critical(pr):
    u1c = M.u1_star(pr)
    q = pr.copy(u1=u1c)
    N, E = pr.Nhat, M.equilibrium(pr)
    Pst = float(E[:3].sum())
    Ac = M.A(N, q)
    v, psi = perron(Ac), perron(Ac.T)
    h_ = 1e-6
    sp_ = float(psi @ ((M.A(N + h_, q) - M.A(N - h_, q)) / (2 * h_)) @ v / (psi @ v))
    h = -(pr.S0 * pr.delta) - np.array([pr.lamR, pr.lamG, pr.lamQ]) * N
    nu = float(h @ v / pr.gamma)
    Pc_h = float(v.sum() / (abs(nu) * sp_))
    ts = np.array([1e3, 1e4, 1e5, 1e6, 1e7])
    s = solve_ivp(M.rhs, (0, ts[-1]), E, args=(q,), method="LSODA", rtol=1e-11, atol=1e-16, dense_output=True)
    Y = s.sol(ts)
    V = psi @ s.y[:3]
    teps = {}
    for eps in (1e-2, 1e-3, 1e-4):
        t = first_time(q, E, eps * Pst, 5e6 * DAY, -1) / DAY
        teps[f"{eps:g}"] = dict(exact_days=t, estimate_days=Pc_h / DAY / (eps * Pst))
    return dict(u1c=u1c, Pstar=Pst, psi_over_psiR=(psi / psi[0]).tolist(), sprime=sp_, nu=nu, Pc_hours=Pc_h, Pc_days=Pc_h / DAY,
                t_hours=ts.tolist(), tP_days=(ts / DAY * Y[:3].sum(0)).tolist(), V_nonincreasing=bool(np.all(np.diff(V) <= 1e-9 * V[:-1])),
                T_eps=teps)


def regrowth(pr):
    E = M.equilibrium(pr)
    Pst = E[:3].sum()
    s1 = solve_ivp(M.rhs, (0, 28 * DAY), E, args=(pr.copy(u1=0.95),), method="LSODA", rtol=1e-10, atol=1e-14)
    yw = s1.y[:, -1]
    S = yw[0] + yw[2]
    out = {}
    for qf in (0.0, 1.0):
        y0 = np.array([(1 - qf) * S, yw[1], qf * S, yw[3]])
        out[f"q{qf:g}"] = dict(T50=first_time(pr, y0, 0.5 * Pst, 400 * DAY, 1) / DAY, T90=first_time(pr, y0, 0.9 * Pst, 400 * DAY, 1) / DAY)
    return out


def profiles():
    th = PO.load_theta()
    ref = O.outputs(th, PO.T_OBS)
    cases = {"F2": (["kappa"], ["f"]), "abund": (["kappa", "k2", "p"], ["f", "P"]), "readout": (["kappa", "k2", "p"], ["f", "y"])}
    res = {}
    for name, (nuis, outs) in cases.items():
        g = np.exp(np.linspace(np.log(0.25), np.log(4.0), 49))
        vals = np.array([PO.profile_point(th, ref, x * th["beta"], nuis, outs)[0] for x in g])

        def F(lg):
            return PO.profile_point(th, ref, np.exp(lg) * th["beta"], nuis, outs)[0] - PO.CHI95

        i0 = int(np.where(g >= 1 - 1e-12)[0][0])
        lo = hi = None
        for i in range(i0, 0, -1):
            if vals[i] <= PO.CHI95 < vals[i - 1]:
                lo = float(np.exp(brentq(F, np.log(g[i - 1]), np.log(g[i]), xtol=1e-3)))
                break
        for i in range(i0, len(g) - 1):
            if vals[i] <= PO.CHI95 < vals[i + 1]:
                hi = float(np.exp(brentq(F, np.log(g[i]), np.log(g[i + 1]), xtol=1e-3)))
                break
        res[name] = dict(lower=lo, upper=hi)
    return res


def clamp(pr):
    th = O.theta_from_params(pr, M)
    t = PO.T_OBS
    fc = O.outputs(th, t)["f"]
    sd = np.sqrt(fc * (1 - fc) / 300)
    Pst = M.equilibrium(pr)[:3].sum()
    rows = []
    for frac in (1e-3, 1e-2, 3e-2, 1e-1):
        y0 = np.r_[0, 0, frac * Pst, pr.Nhat]
        s = solve_ivp(M.rhs, (0, 21 * DAY), y0, args=(pr,), method="LSODA", rtol=1e-10, atol=1e-14, dense_output=True)
        Y = s.sol(t)
        fu = Y[1] / Y[:3].sum(0)
        r = least_squares(lambda z: (O.outputs(O._pack(th, ["kappa", "beta"], z), t)["f"] - fu) / sd, np.log([th["kappa"], th["beta"]]))
        rows.append(dict(P0_over_Pstar=frac, Nmin=float(Y[3].min()), bias_beta=float(np.exp(r.x[1]) / th["beta"] - 1)))
    return dict(growth_factor_21d=float(O.outputs(th, np.array([21 * DAY]))["P"][0]), rows=rows)


def main():
    c = json.load(open(os.path.join(ROOT, "results", "p0_candidates.json"), encoding="utf-8"))
    pr = M.Params(**c["candidates"]["C1"]["params"])
    th = O.theta_from_params(pr, M)
    ev = np.linalg.eigvals(M.jacobian(M.equilibrium(pr), pr))
    lead = ev[np.argmax(ev.imag)]
    out = dict(critical=critical(pr), regrowth=regrowth(pr), profiles=profiles(), clamp=clamp(pr), alpha_hat=th["alpha"],
               g1_to_s_hours=1 / th["kappa"], g1_residence_hours=1 / (th["kappa"] + th["alpha"] + th["dR"]),
               eig_re=float(lead.real), eig_im=float(lead.imag), eig_period_days=float(2 * np.pi / lead.imag / DAY))
    with open(os.path.join(ROOT, "results", "p7_critical_and_numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
