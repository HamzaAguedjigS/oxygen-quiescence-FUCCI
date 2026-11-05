"""Analyse the Hopf example used in the Phase 7 revision.

The script computes the first Lyapunov coefficient using Kuznetsov's formula,
exact symbolic derivatives and a unit-norm eigenvector. It also evaluates
transversality, the eigenvalue-pair merger, and predicted versus simulated
cycle amplitudes. Results are written to ``results/p7_hopf_lyapunov.json``.
"""
import json
import os
import sys

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as M  # noqa: E402


def main():
    hx = json.load(open(os.path.join(ROOT, "results", "p1_hopf_example.json"), encoding="utf-8"))
    base = M.Params(**hx["base_params"])
    q = base
    R, G, Q, N = sp.symbols("R G Q N", positive=True)
    na = sp.Symbol("na", positive=True)
    ka = (1 - q.u1) * q.k1bar * N / (q.KN + N)
    al = q.albar * q.Kal ** na / (q.Kal ** na + N ** na)
    be = q.bebar * N ** q.n / (q.Kbe ** q.n + N ** q.n)
    mR = q.muR0 + q.muRmax * q.KmuR / (q.KmuR + N)
    mG = q.muG0 + q.muGmax * q.KmuG / (q.KmuG + N) + q.u2 * q.etaG
    mQ = q.muQ0 + q.muQmax * q.KmuQ / (q.KmuQ + N)
    f = sp.Matrix([2 * q.p * q.k2 * G - ka * R - al * R + be * Q - mR * R,
                   ka * R - q.k2 * G - mG * G,
                   al * R + 2 * (1 - q.p) * q.k2 * G - be * Q - mQ * Q,
                   q.S0 / (1 + q.delta * (R + G + Q)) - q.gamma * N - (q.lamR * R + q.lamG * G + q.lamQ * Q) * N])
    xs = [R, G, Q, N]
    fn = sp.lambdify([R, G, Q, N, na], f, "numpy")
    E0 = M.equilibrium(base.copy(nal=hx["m_Hopf"]))
    err = float(np.abs(np.array(fn(*E0, hx["m_Hopf"]), float).ravel() - M.rhs(0, E0, base.copy(nal=hx["m_Hopf"]))).max())
    assert err < 1e-12
    Jfn = sp.lambdify([R, G, Q, N, na], f.jacobian(xs), "numpy")
    Hfn = sp.lambdify([R, G, Q, N, na], [[[sp.diff(f[i], xs[j], xs[k]) for k in range(4)] for j in range(4)] for i in range(4)], "numpy")
    Tfn = sp.lambdify([R, G, Q, N, na],
                      [[[[sp.diff(f[i], xs[j], xs[k], xs[l]) for l in range(4)] for k in range(4)] for j in range(4)] for i in range(4)], "numpy")

    def eigs(n):
        return np.linalg.eigvals(np.array(Jfn(*M.equilibrium(base.copy(nal=n)), n), float))

    def re_lead(n):
        return float(np.sort(eigs(n).real)[-1])

    nH = brentq(re_lead, hx["m_Hopf"] - 0.05, hx["m_Hopf"] + 0.05, xtol=1e-11)
    EH = M.equilibrium(base.copy(nal=nH))
    A = np.array(Jfn(*EH, nH), float)
    w, Vr = np.linalg.eig(A)
    i = int(np.argmax(w.imag))
    om = float(w[i].imag)
    qv = Vr[:, i] / np.linalg.norm(Vr[:, i])
    wl, Vl = np.linalg.eig(A.T)
    pv = Vl[:, int(np.argmin(np.abs(wl + 1j * om)))]
    pv = pv / np.conj(np.conj(pv) @ qv)
    Hn, Tn = np.array(Hfn(*EH, nH), float), np.array(Tfn(*EH, nH), float)
    B = lambda a, b: np.einsum("ijk,j,k->i", Hn, a, b)  # noqa: E731
    C = lambda a, b, d: np.einsum("ijkl,j,k,l->i", Tn, a, b, d)  # noqa: E731
    qb = np.conj(qv)
    g21 = C(qv, qv, qb) - 2 * B(qv, np.linalg.solve(A, B(qv, qb))) + B(qb, np.linalg.solve(2j * om * np.eye(4) - A, B(qv, qv)))
    rec1 = float(np.real(np.conj(pv) @ g21) / 2)
    l1 = rec1 / om
    dn = 1e-4
    trans = (re_lead(nH + dn) - re_lead(nH - dn)) / (2 * dn)

    def imag_lead(n):
        ev = eigs(n)
        return float(np.abs(ev[np.argsort(ev.real)[-2:]].imag).max())

    grid = np.linspace(4.0, 4.6, 241)
    knee = float(next(x for x in grid if imag_lead(x) < 1e-12))
    amp = []
    qP = abs(qv[:3].sum())
    for dm in (0.01, 0.02):
        n = nH + dm
        pr = base.copy(nal=n)
        E = M.equilibrium(pr)
        T = 28800.0
        s = solve_ivp(M.rhs, (0, T), E * (1 + 0.05 * np.array([1, -1, 1, 0])), args=(pr,), method="LSODA", rtol=1e-9, atol=1e-11, dense_output=True)
        P = s.sol(np.linspace(0.9 * T, T, 6000))[:3].sum(0) / E[:3].sum()
        amp.append(dict(dn=dm, half_sim=float((P.max() - P.min()) / 2), half_pred=float(2 * np.sqrt(trans * dm / abs(rec1)) * qP / E[:3].sum())))
    out = dict(n_H=float(nH), omega_per_h=om, period_days=2 * np.pi / om / 24, l1=float(l1), re_c1=rec1, transversality=float(trans), knee_n=knee,
               amplitude=amp, params={k: v for k, v in hx["base_params"].items()})
    with open(os.path.join(ROOT, "results", "p7_hopf_lyapunov.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "params"}, indent=2))


if __name__ == "__main__":
    main()
