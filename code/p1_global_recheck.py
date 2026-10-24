"""Evaluate global convergence using the 3,600 archived initial states.

The initial states and parameter draws match ``p1_stability_scan.py``. Each
trajectory is integrated for ``HORIZON`` relaxation times, where the relaxation
time is 1/|Re(lambda)|, without an absolute time cap.

Output: ``results/p1_global_recheck.json``.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as M  # noqa: E402
from p1_stability_scan import SEED, N_LOCAL, N_GLOBAL, N_IC, draw, load_C1  # noqa: E402

HORIZON = 200.0
TOL = 1e-3


def run(args):
    pr, E, lead, y0 = args
    T = HORIZON / abs(lead)
    s = solve_ivp(M.rhs, (0, T), y0, args=(pr,), method="LSODA", rtol=1e-9, atol=1e-13)
    Pst = E[:3].sum()
    err = max(np.abs(s.y[:3, -1] - E[:3]).max() / Pst, abs(s.y[3, -1] - E[3]))
    return bool(s.success), float(err), float(T)


def main():
    base = load_C1()
    rng = np.random.default_rng(SEED)
    invasive = []
    for _ in range(N_LOCAL):                       # Reproduce the scan's draw sequence
        pr = draw(base, rng)
        try:
            if M.R0(pr) <= 1.0:
                continue
            E = M.equilibrium(pr)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        lead = np.linalg.eigvals(M.jacobian(E, pr)).real.max()
        invasive.append((pr, E, lead))
    tasks = []
    for pr, E, lead in invasive[:N_GLOBAL]:        # Reproduce the scan's initial states
        Pst = E[:3].sum()
        for _ in range(N_IC):
            P0 = Pst * float(np.exp(rng.uniform(np.log(1e-3), np.log(10))))
            comp = rng.dirichlet([1, 1, 1])
            tasks.append((pr, E, lead, np.r_[P0 * comp, rng.uniform(0, pr.Nhat)]))
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        res = pool.map(run, tasks, chunksize=4)
    ok = [r[0] and r[1] <= TOL for r in res]
    out = dict(seed=SEED, horizon_relaxation_times=HORIZON, tol=TOL, n_runs=len(res), n_converged=int(sum(ok)),
               worst_err=float(max(r[1] for r in res)), n_unstable_in_sets=int(sum(l >= 0 for _, _, l in invasive[:N_GLOBAL])),
               max_T_hours=float(max(r[2] for r in res)))
    print(out)
    with open(os.path.join(ROOT, "results", "p1_global_recheck.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
