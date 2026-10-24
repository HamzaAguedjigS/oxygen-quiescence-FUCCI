"""Numerically evaluate the transcritical-bifurcation coefficients at u1 = u1c.

The state order is (R, G, Q, N) and E0 = (0, 0, 0, Nhat). The evaluated
quantities are:
  J(E0) = [[A(Nhat), 0], [h^T, -gamma]],  h = -(S0 delta + ell Nhat) (vecteur des dNdot/dX en E0)
  right null vector V = (v, nu), A v = 0, nu = h^T v / gamma < 0
  left null vector W = (w, 0), w^T A = 0
  (S1) W^T f_u1(E0) = 0 (E0 is an equilibrium for every u1)
  (S2) W^T D_x f_u1 V = -(k1hat * mhat / kappahat) w_R v_R < 0
  (S3) W^T D^2 f (V, V) = 2 nu w^T A'(Nhat) v = 2 nu s'(Nhat) (w^T v) < 0
along with the positive-equilibrium branch E*(u1) for u1 < u1c.

Output: ``results/p1_sotomayor.json``.
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import eig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as M  # noqa: E402


def load_C1():
    c = json.load(open(os.path.join(ROOT, "results", "p0_candidates.json"), encoding="utf-8"))
    return M.Params(**c["candidates"]["C1"]["params"])


def perron_pair(Am):
    w_, VL, VR = eig(Am, left=True, right=True)
    i = np.argmax(w_.real)
    v = np.abs(VR[:, i].real)
    w = np.abs(VL[:, i].real)
    return w_[i].real, v / v.sum(), w / (w @ (v / v.sum()))


def main():
    pr0 = load_C1()
    u1c = M.u1_star(pr0)
    pr = pr0.copy(u1=u1c)
    Nh = pr.Nhat
    A = M.A(Nh, pr)
    lam, v, w = perron_pair(A)
    ell = np.array([pr.lamR, pr.lamG, pr.lamQ])
    h = -(pr.S0 * pr.delta + ell * Nh)
    nu = h @ v / pr.gamma
    E0 = np.array([0, 0, 0, Nh])
    J0 = M.jacobian(E0, pr)
    V = np.r_[v, nu]
    W = np.r_[w, 0.0]
    # (S2) Finite-difference derivative of J with respect to u1, contracted with W and V
    du = 1e-6
    Jp = M.jacobian(E0, pr.copy(u1=u1c + du))
    Jm = M.jacobian(E0, pr.copy(u1=u1c - du))
    S2_num = W @ ((Jp - Jm) / (2 * du)) @ V
    kh = M.kappa(Nh, pr)
    k1h = pr.k1bar * Nh / (pr.KN + Nh)
    mh = M.muR(Nh, pr) + M.alpha(Nh, pr) * M.muQ(Nh, pr) / (M.beta(Nh, pr) + M.muQ(Nh, pr))
    S2_formula = -(k1h * mh / kh) * w[0] * v[0]
    # (S3) Second directional derivative d^2/ds^2 W.f(E0 + sV) at s = 0
    s = 1e-4
    f = lambda x: M.rhs(0, x, pr)
    S3_num = W @ (f(E0 + s * V) - 2 * f(E0) + f(E0 - s * V)) / s ** 2
    dN = 1e-6
    Ap = (M.A(Nh + dN, pr) - M.A(Nh - dN, pr)) / (2 * dN)
    S3_formula = 2 * nu * (w @ Ap @ v)
    sprime = (M.growth_exponent(pr, Nh + dN) - M.growth_exponent(pr, Nh - dN)) / (2 * dN)
    # Positive-equilibrium branch near u1c
    branch = []
    for d in [-0.05, -0.02, -0.01, -0.005, -0.001, 0.001, 0.01]:
        q = pr0.copy(u1=u1c + d)
        E = M.equilibrium(q)
        if E is None:
            branch.append(dict(du=d, Pstar=None, E0_maxRe=float(np.linalg.eigvals(M.jacobian(E0, q)).real.max())))
        else:
            ev = np.linalg.eigvals(M.jacobian(E, q))
            branch.append(dict(du=d, Pstar=float(E[:3].sum()), Nstar=float(E[3]), Estar_maxRe=float(ev.real.max()),
                               E0_maxRe=float(np.linalg.eigvals(M.jacobian(E0, q)).real.max())))
    out = dict(u1c=u1c, Lambda_at_u1c=lam, nu=float(nu), w=w.tolist(), v=v.tolist(),
               kernel_check=dict(Av=float(np.abs(A @ v).max()), wA=float(np.abs(w @ A).max()),
                                 J0V=float(np.abs(J0 @ V).max()), WJ0=float(np.abs(W @ J0).max())),
               S2=dict(numeric=float(S2_num), formula=float(S2_formula)),
               S3=dict(numeric=float(S3_num), formula=float(S3_formula), sprime_Nhat=float(sprime),
                       formula_via_sprime=float(2 * nu * sprime * (w @ v))),
               branch=branch)
    with open(os.path.join(ROOT, "results", "p1_sotomayor.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
