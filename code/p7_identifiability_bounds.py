"""Compute the exact transfer-function rank bounds reported in Table 2.

For ``X(0) = e_Q``, the script verifies the transfer-function numerators,
the ranks of the six- and eight-coordinate maps, invariance under a common
loss-rate shift, and the unresolved direction of the six-parameter problem.
The output is written to ``results/p7_identifiability_bounds.json``.
"""
import json
import os
import sys

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as M  # noqa: E402
import obs_model as O  # noqa: E402


def main():
    c = json.load(open(os.path.join(ROOT, "results", "p0_candidates.json"), encoding="utf-8"))
    th = O.theta_from_params(M.Params(**c["candidates"]["C1"]["params"]), M)
    ka, al, dR, k2, dG, p, be, dQ, s = sp.symbols("kappa alpha dR k2 dG p beta dQ s")
    params = [ka, al, dR, k2, dG, p, be, dQ]
    w = [sp.Rational(1), sp.Rational(115, 100), sp.Rational(35, 100)]
    A = sp.Matrix([[-(ka + al + dR), 2 * p * k2, be], [ka, -(k2 + dG), 0], [al, 2 * (1 - p) * k2, -(be + dQ)]])
    SI = s * sp.eye(3) - A
    den = sp.Poly(sp.expand(SI.det()), s)
    num = SI.adjugate()[:, 2]
    numR, numG, numQ = [sp.Poly(sp.expand(n), s) for n in num]
    nP = sp.Poly(sp.expand(sum(num)), s).all_coeffs()
    ny = sp.Poly(sp.expand(w[0] * num[0] + w[1] * num[1] + w[2] * num[2]), s).all_coeffs()
    assert sp.simplify(numG.as_expr() - ka * be) == 0
    assert numR.degree() == 1 and sp.simplify(numR.all_coeffs()[0] - be) == 0
    assert numQ.degree() == 2 and numQ.all_coeffs()[0] == 1
    assert nP[0] == 1 and sp.simplify(ny[0] - w[2]) == 0
    coords_GP = list(den.all_coeffs()[1:]) + [ka * be] + list(nP[1:])
    coords_all = coords_GP + list(ny[1:])
    ref = {sym: sp.nsimplify(th[n], rational=True) for sym, n in zip(params, O.NAMES)}
    JGP = sp.Matrix(coords_GP).jacobian(params).subs(ref)
    Jall = sp.Matrix(coords_all).jacobian(params).subs(ref)
    shift = sp.Matrix([0, 0, 1, 0, 1, 0, 0, 1])
    rGP, rAll = JGP.rank(), Jall.rank()
    assert (JGP * shift).norm() != 0
    # Kernel of the fraction map: kernel of the six coordinates plus the common shift.
    Kmat = sp.Matrix.hstack(*(list(JGP.nullspace()) + [shift]))
    ns = Kmat.extract([2, 4], list(range(Kmat.shape[1]))).nullspace()     # dR = dG = 0
    assert len(ns) == 1
    v = np.array([float(x) for x in Kmat * ns[0]])
    vals = np.array([th[k] for k in O.NAMES])
    logv = (v / vals) / (v[O.NAMES.index("p")] / vals[O.NAMES.index("p")])
    # Direct numerical check of the exact invariance.
    t = np.arange(1, 22) * O.DAY
    d = 0.003
    th2 = dict(th, dR=th["dR"] + d, dG=th["dG"] + d, dQ=th["dQ"] + d)
    a, b = O.outputs(th, t), O.outputs(th2, t)
    out = dict(n_coords_GP=len(coords_GP), rank_coords_GP=int(rGP), n_coords_all=len(coords_all), rank_coords_all=int(rAll),
               bound_f=int(rGP) - 1, bound_fP=int(rGP), bound_fy=int(rAll) - 1, bound_fPy=int(rAll),
               numerator_G=str(numG.as_expr()), numerator_R=str(numR.as_expr()),
               invariance_max_df=float(np.abs(a["f"] - b["f"]).max()), invariance_max_dy=float(np.abs(a["y"] - b["y"]).max()),
               abundance_changes=float(np.abs(np.log(a["P"]) - np.log(b["P"])).max()),
               six_param_log_direction=dict(zip(O.NAMES, map(float, logv))),
               log_kappa_beta_change=float((v / vals)[O.NAMES.index("kappa")] + (v / vals)[O.NAMES.index("beta")]))
    with open(os.path.join(ROOT, "results", "p7_identifiability_bounds.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
