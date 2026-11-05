"""Run the Phase 7 robustness checks for the four-parameter Monte Carlo fit.

The scenarios use the same seeds as ``p2_montecarlo.py`` and test wider
bounds, alternative values or restrictions for ``p``, additional unknown
rates, and misspecified readout weights. Results are written to
``results/p7_mc_robustness.json``.
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
NREP = 500
T = np.arange(1, 22) * O.DAY
NOISE = dict(n=300, cv_P=0.1, sd_y=0.03, phi=1.0, class_err=0.0)
F4 = ["kappa", "beta", "k2", "p"]
DAY = O.DAY
B0 = dict(O.BOUNDS)
S0 = dict(O.STARTS)
EXTRA_B = {"alpha": (1e-6, 0.1), "dR": (1e-4, 0.05), "dG": (1e-4, 0.05), "dQ": (1e-4, 0.05)}
EXTRA_S = {"alpha": (2e-4,), "dR": (0.002,), "dG": (0.002,), "dQ": (0.002,)}
WIDE = {"kappa": (1 / 2000, 1.0), "beta": (1 / (240 * DAY), 1 / (0.1 * DAY)), "k2": (0.005, 2.0), "p": (0.1, 0.9999)}
PGE = dict(B0, p=(0.9, 0.999))

SC = {}
for lab, outs in (("f", ["f"]), ("fP", ["f", "P"]), ("fy", ["f", "y"]), ("fPy", ["f", "P", "y"])):
    SC["base_" + lab] = dict(outs=outs, free=F4, bounds=B0)
    SC["wide_" + lab] = dict(outs=outs, free=F4, bounds=WIDE)
    SC["p085_" + lab] = dict(outs=outs, free=F4, bounds=B0, p=0.85)
    SC["pge09_" + lab] = dict(outs=outs, free=F4, bounds=PGE)
SC["six_fPy"] = dict(outs=["f", "P", "y"], free=F4 + ["alpha", "dQ"], bounds=dict(B0, **EXTRA_B))
SC["eight_fPy"] = dict(outs=["f", "P", "y"], free=F4 + ["alpha", "dR", "dG", "dQ"], bounds=dict(B0, **EXTRA_B))
for wq in (0.20, 0.30, 0.40, 0.50):
    SC[f"wq{wq:.2f}_fPy"] = dict(outs=["f", "P", "y"], free=F4, bounds=B0, wq=wq)


def theta(p=None):
    c = json.load(open(os.path.join(ROOT, "results", "p0_candidates.json"), encoding="utf-8"))
    th = O.theta_from_params(M.Params(**c["candidates"]["C1"]["params"]), M)
    if p is not None:
        th["p"] = p
    return th


def run_one(args):
    name, r = args
    sc = SC[name]
    th = theta(sc.get("p"))
    O.BOUNDS.clear()
    O.BOUNDS.update(sc["bounds"])
    O.STARTS.clear()
    O.STARTS.update(S0)
    O.STARTS.update(EXTRA_S)
    rng = np.random.default_rng([SEED, r])
    data = O.simulate(th, T, rng, sc["outs"], NOISE)
    kw = {"w_fit": np.array([1.0, 1.15, sc["wq"]])} if "wq" in sc else {}
    est, _, bnd = O.fit(data, T, th, sc["free"], NOISE, **kw)
    hit = {}
    for k in sc["free"]:
        lo, hi = sc["bounds"][k]
        if abs(np.log(est[k] / lo)) < 1e-5:
            hit[k] = "lo"
        elif abs(np.log(est[k] / hi)) < 1e-5:
            hit[k] = "hi"
    return name, est["beta"] / th["beta"] - 1, bool(bnd), hit


def main():
    t0 = time.time()
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        res = pool.map(run_one, [(n, r) for n in SC for r in range(NREP)], chunksize=6)
    out = {}
    for n in SC:
        rows = [x for x in res if x[0] == n]
        e = np.array([x[1] for x in rows])
        hits = {}
        for x in rows:
            for k, v in x[3].items():
                hits[f"{k}_{v}"] = hits.get(f"{k}_{v}", 0) + 1
        out[n] = dict(rmse=float(np.sqrt(np.mean(e ** 2))), median_bias=float(np.median(e)), median_abs=float(np.median(np.abs(e))),
                      within25=float(np.mean(np.abs(e) <= 0.25)), bound=int(sum(x[2] for x in rows)), hits=hits)
        print(f"{n:13s} RMSE {out[n]['rmse']:.3f} bias {out[n]['median_bias']:+.3f} within25 {out[n]['within25']:.2f} "
              f"bounds {out[n]['bound']:3d} {hits}", flush=True)
    with open(os.path.join(ROOT, "results", "p7_mc_robustness.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(seed=SEED, n_rep=NREP, scenarios=out), fh, indent=2)
    print("elapsed", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
