"""Scan local stability and global convergence of the positive equilibrium.

Local stability is evaluated for ``N_LOCAL`` fixed-seed parameter draws around
the reference regime. For each invasive draw, the Jacobian spectrum and
Routh--Hurwitz margins are recorded. Global convergence is then tested for
``N_GLOBAL`` invasive draws and ``N_IC`` random initial states per draw.

Output: ``results/p1_stability_scan.json``.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as M  # noqa: E402

SEED = 1808
N_LOCAL = 20000
N_GLOBAL = 300
N_IC = 12
SCALED = ["k1bar", "KN", "k2", "albar", "Kal", "bebar", "Kbe", "muR0", "muRmax", "KmuR", "muG0", "muGmax",
          "KmuG", "muQmax", "KmuQ", "lamR", "lamG", "lamQ", "delta"]


def load_C1():
    c = json.load(open(os.path.join(ROOT, "results", "p0_candidates.json"), encoding="utf-8"))
    return M.Params(**c["candidates"]["C1"]["params"])


def draw(base, rng):
    kw = {k: getattr(base, k) * float(np.exp(rng.uniform(np.log(0.1), np.log(10)))) for k in SCALED}
    kw.update(nal=float(rng.uniform(1, 8)), n=float(rng.uniform(1, 8)), p=float(rng.uniform(0, 1)))
    return base.copy(**kw)


def rh(J):
    a3, a2, a1, a0 = np.poly(J)[1:]
    return a3, a0, a3 * a2 - a1, a1 * (a3 * a2 - a1) - a3 ** 2 * a0


def main():
    t0 = time.time()
    base = load_C1()
    rng = np.random.default_rng(SEED)
    invasive, unstable, rh_fail = [], 0, 0
    min_margin = np.inf
    for _ in range(N_LOCAL):
        pr = draw(base, rng)
        try:
            if M.R0(pr) <= 1.0:
                continue
            E = M.equilibrium(pr)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        J = M.jacobian(E, pr)
        ev = np.linalg.eigvals(J)
        lead = ev.real.max()
        scale = np.abs(ev).max()
        min_margin = min(min_margin, -lead / scale)
        if lead >= 0:
            unstable += 1
        a3, a0, d2, d3 = rh(J)
        if not (a3 > 0 and a0 > 0 and d2 > 0 and d3 > 0):
            rh_fail += 1
        invasive.append((pr, E, lead))
    t1 = time.time()
    print(f"local: {len(invasive)} invasive / {N_LOCAL}, unstable {unstable}, RH failures {rh_fail}, "
          f"min relative margin {min_margin:.2e} ({t1 - t0:.0f}s)", flush=True)

    fails, n_runs, worst = [], 0, 0.0
    for k, (pr, E, lead) in enumerate(invasive[:N_GLOBAL]):
        Pst = E[:3].sum()
        T = min(60.0 / abs(lead), 2e5)
        for _ in range(N_IC):
            P0 = Pst * float(np.exp(rng.uniform(np.log(1e-3), np.log(10))))
            comp = rng.dirichlet([1, 1, 1])
            y0 = np.r_[P0 * comp, rng.uniform(0, pr.Nhat)]
            s = solve_ivp(M.rhs, (0, T), y0, args=(pr,), method="LSODA", rtol=1e-8, atol=1e-12)
            n_runs += 1
            err = max(np.abs(s.y[:3, -1] - E[:3]).max() / Pst, abs(s.y[3, -1] - E[3]))
            worst = max(worst, err)
            if not s.success or err > 1e-3:
                fails.append(dict(sample=k, err=float(err), success=bool(s.success), T=T))
        if (k + 1) % 50 == 0:
            print(f"  global: {k + 1}/{N_GLOBAL} parameter sets, {len(fails)} failures, worst err {worst:.1e} "
                  f"({time.time() - t1:.0f}s)", flush=True)
    out = dict(seed=SEED, n_local=N_LOCAL, n_invasive=len(invasive), n_unstable=unstable, n_rh_fail=rh_fail,
               min_relative_margin=float(min_margin), n_global_sets=min(N_GLOBAL, len(invasive)), n_ic=N_IC,
               n_runs=n_runs, n_fail=len(fails), worst_err=float(worst), fails=fails[:50],
               ranges="x0.1-x10 log-uniform on " + ", ".join(SCALED) + "; m,n in [1,8]; p in [0,1]")
    with open(os.path.join(ROOT, "results", "p1_stability_scan.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "fails"}, indent=2))


if __name__ == "__main__":
    main()
