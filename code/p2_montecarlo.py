"""Monte Carlo evaluation of the reoxygenation observation design.

``reference`` mode uses 500 replicates per scenario with common random numbers
and paired bootstrap comparisons. ``ensemble`` mode repeats the conclusions
across 30 parameter sets, with 200 replicates per set.

Usage: ``python code/p2_montecarlo.py reference|ensemble``.
Outputs: ``results/p2_mc_reference.json``,
``results/p2_mc_reference.npz``, and ``results/p2_mc_ensemble.json``.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as M  # noqa: E402
import obs_model as O  # noqa: E402

SEED = 271828
NB = 2000
SCHED = {"W7": np.arange(1, 8) * O.DAY, "W14": np.arange(1, 15) * O.DAY, "W21": np.arange(1, 22) * O.DAY,
         "E21": np.geomspace(0.25, 21, 21) * O.DAY}
BASE_NOISE = dict(n=300, cv_P=0.1, sd_y=0.03)
F2 = ["kappa", "beta"]
F4 = ["kappa", "beta", "k2", "p"]
W_LO = np.array([1.0, 1.15, 0.20])
W_HI = np.array([1.0, 1.15, 0.50])

SCENARIOS = {
    # (A) Observation duration and timing; kappa and beta unknown, green fraction only
    "A_W7": dict(sched="W7", outs=["f"], free=F2),
    "A_W14": dict(sched="W14", outs=["f"], free=F2),
    "A_W21": dict(sched="W21", outs=["f"], free=F2),
    "A_E21": dict(sched="E21", outs=["f"], free=F2),
    # (B) Added readouts; kappa, beta, k2, and p unknown over 21 days
    "B_f": dict(sched="W21", outs=["f"], free=F4),
    "B_fP": dict(sched="W21", outs=["f", "P"], free=F4),
    "B_fy": dict(sched="W21", outs=["f", "y"], free=F4),
    "B_fPy": dict(sched="W21", outs=["f", "P", "y"], free=F4),
    # (D) Systematic observation errors; kappa and beta unknown over 21 days
    "D_BB3": dict(sched="W21", outs=["f"], free=F2, phi=3.0),
    "D_CE3": dict(sched="W21", outs=["f"], free=F2, class_err=0.03),
    "D_SC10": dict(sched="W21", outs=["f"], free=F2, x0_true=(0.1, 0.0, 0.9)),
    "D_fy": dict(sched="W21", outs=["f", "y"], free=F2),
    "D_fy_w020": dict(sched="W21", outs=["f", "y"], free=F2, w_fit=W_LO),
    "D_fy_w050": dict(sched="W21", outs=["f", "y"], free=F2, w_fit=W_HI),
}
PAIRS = [("A_W21", "A_W7"), ("A_W21", "A_W14"), ("A_E21", "A_W21"), ("B_fP", "B_f"), ("B_fy", "B_f"),
         ("B_fPy", "B_f"), ("B_f", "A_W21")]
ENS_SCEN = ["A_W7", "A_W21", "B_f", "B_fP", "B_fy"]


def run_one(args):
    name, th, r = args
    sc = SCENARIOS[name]
    t = SCHED[sc["sched"]]
    noise = dict(BASE_NOISE, phi=sc.get("phi", 1.0), class_err=sc.get("class_err", 0.0))
    rng = np.random.default_rng([SEED, r])
    data = O.simulate(th, t, rng, sc["outs"], noise, x0_true=sc.get("x0_true", (0.0, 0.0, 1.0)))
    est, _, bnd = O.fit(data, t, th, sc["free"], noise, w_fit=sc.get("w_fit", O.W_REF))
    return (name, r, est["beta"] / th["beta"] - 1, est["kappa"] / th["kappa"] - 1,
            est["kappa"] * est["beta"] / (th["kappa"] * th["beta"]) - 1, bnd)


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def boot_ci(e, rng, stat=rmse):
    idx = rng.integers(0, len(e), size=(NB, len(e)))
    vals = np.array([stat(e[i]) for i in idx])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired_ci(e1, e2, rng):
    idx = rng.integers(0, len(e1), size=(NB, len(e1)))
    d = np.array([rmse(e1[i]) - rmse(e2[i]) for i in idx])
    return float(rmse(e1) - rmse(e2)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def run_set(th, names, nrep, pool):
    tasks = [(n, th, r) for n in names for r in range(nrep)]
    res = pool.map(run_one, tasks, chunksize=8)
    E = {n: dict(beta=np.zeros(nrep), kappa=np.zeros(nrep), prod=np.zeros(nrep), bnd=np.zeros(nrep, bool))
         for n in names}
    for n, r, eb, ek, ep, bd in res:
        E[n]["beta"][r], E[n]["kappa"][r], E[n]["prod"][r], E[n]["bnd"][r] = eb, ek, ep, bd
    return E


def load_C1():
    c = json.load(open(os.path.join(ROOT, "results", "p0_candidates.json"), encoding="utf-8"))
    return M.Params(**c["candidates"]["C1"]["params"])


def reference(pool):
    th = O.theta_from_params(load_C1(), M)
    t0 = time.time()
    E = run_set(th, list(SCENARIOS), 500, pool)
    rng = np.random.default_rng(7)
    summ = {}
    for n, e in E.items():
        summ[n] = dict(rmse_beta=rmse(e["beta"]), ci_beta=boot_ci(e["beta"], rng),
                       median_bias_beta=float(np.median(e["beta"])),
                       ci_median_bias=boot_ci(e["beta"], rng, stat=lambda x: float(np.median(x))),
                       rmse_kappa=rmse(e["kappa"]), rmse_prod=rmse(e["prod"]), ci_prod=boot_ci(e["prod"], rng),
                       corr_log=float(np.corrcoef(np.log1p(e["beta"]), np.log1p(e["kappa"]))[0, 1]),
                       bound_hits=int(e["bnd"].sum()), q=[float(np.percentile(e["beta"], x)) for x in (2.5, 97.5)],
                       spec={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in SCENARIOS[n].items()})
        print(f"{n:11s} RMSE beta {summ[n]['rmse_beta']:.3f} {np.round(summ[n]['ci_beta'], 3)}  median bias "
              f"{summ[n]['median_bias_beta']:+.3f}  RMSE prod {summ[n]['rmse_prod']:.3f}  bounds {summ[n]['bound_hits']}",
              flush=True)
    pairs = {}
    for a, b in PAIRS:
        d, ci = paired_ci(E[a]["beta"], E[b]["beta"], rng)
        pairs[f"{a} - {b}"] = dict(diff=d, ci=ci)
        print(f"  {a} - {b}: {d:+.3f} {np.round(ci, 3)}")
    d, ci = paired_ci(E["A_W21"]["prod"], E["A_W21"]["beta"], rng)
    pairs["A_W21 product - beta"] = dict(diff=d, ci=ci)
    print(f"  A_W21 product - beta: {d:+.3f} {np.round(ci, 3)}  ({time.time() - t0:.0f}s)")
    out = dict(seed=SEED, n_rep=500, n_boot=NB, noise=BASE_NOISE, theta_true=th, summary=summ, pairs=pairs,
               schedules_days={k: (v / O.DAY).round(3).tolist() for k, v in SCHED.items()})
    with open(os.path.join(ROOT, "results", "p2_mc_reference.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    np.savez(os.path.join(ROOT, "results", "p2_mc_reference.npz"),
             **{f"{n}_{k}": v for n, e in E.items() for k, v in e.items()})


def ensemble(pool):
    base = load_C1()
    rng = np.random.default_rng(31415)
    rows = []
    t0 = time.time()
    for i in range(30):
        tauG1 = rng.uniform(61, 77)
        tauExit = rng.uniform(3.4, 8.6) * O.DAY
        tauS = rng.uniform(7, 14)
        th = O.theta_from_params(base, M)
        f = lambda: float(np.exp(rng.uniform(np.log(0.5), np.log(2))))
        th.update(kappa=1 / tauG1, beta=1 / tauExit, k2=1 / tauS, p=float(rng.uniform(0.9, 0.99)),
                  dR=th["dR"] * f(), dG=th["dG"] * f(), dQ=th["dQ"] * f(), alpha=th["alpha"] * f())
        E = run_set(th, ENS_SCEN, 200, pool)
        brng = np.random.default_rng(100 + i)
        row = dict(theta=th, rmse={n: rmse(E[n]["beta"]) for n in ENS_SCEN},
                   rmse_prod_W21=rmse(E["A_W21"]["prod"]))
        for key, (a, b, fld) in {"window": ("A_W21", "A_W7", "beta"), "abundance": ("B_fP", "B_f", "beta"),
                                 "readout": ("B_fy", "B_f", "beta")}.items():
            d, ci = paired_ci(E[a][fld], E[b][fld], brng)
            row[key] = dict(diff=d, ci=ci)
        d, ci = paired_ci(E["A_W21"]["prod"], E["A_W21"]["beta"], brng)
        row["product"] = dict(diff=d, ci=ci)
        rows.append(row)
        print(f"set {i + 1:2d}/30  G1 {tauG1:.0f} h exit {tauExit / O.DAY:.1f} d  "
              + "  ".join(f"{n} {row['rmse'][n]:.2f}" for n in ENS_SCEN) + f"  ({time.time() - t0:.0f}s)", flush=True)
    concl = {}
    for key in ("product", "window", "abundance", "readout"):
        diffs = np.array([r[key]["diff"] for r in rows])
        sig = np.array([r[key]["ci"][1] < 0 for r in rows])
        concl[key] = dict(holds=int((diffs < 0).sum()), significant=int(sig.sum()), n=len(rows),
                          median_diff=float(np.median(diffs)))
        print(key, concl[key])
    with open(os.path.join(ROOT, "results", "p2_mc_ensemble.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(seed=31415, n_sets=30, n_rep=200, scenarios=ENS_SCEN, conclusions=concl, rows=rows), fh,
                  indent=2)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "reference"
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        reference(pool) if mode == "reference" else ensemble(pool)
