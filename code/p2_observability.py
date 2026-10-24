"""Evaluate what reoxygenation readouts reveal about hidden-state kinetics.

The ridge calculation profiles beta while refitting nuisance parameters to
noise-free green-fraction data. Expected profile likelihoods then compare the
green fraction alone with designs that add abundance or a state-weighted
readout.

Outputs: ``results/p2_observability.json`` and
``results/p2_observability.npz``.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as M  # noqa: E402
import obs_model as O  # noqa: E402

NOISE = dict(n=300, cv_P=0.1, sd_y=0.03)
T_OBS = np.arange(1, 22) * O.DAY
CHI95 = 3.841


def load_theta():
    c = json.load(open(os.path.join(ROOT, "results", "p0_candidates.json"), encoding="utf-8"))
    return O.theta_from_params(M.Params(**c["candidates"]["C1"]["params"]), M)


def weighted_residuals(th, ref, outs):
    o = O.outputs(th, T_OBS)
    r = []
    if "f" in outs:
        sd = np.sqrt(ref["f"] * (1 - ref["f"]) / NOISE["n"])
        r.append((o["f"] - ref["f"]) / sd)
    if "P" in outs:
        r.append((np.log(o["P"]) - np.log(ref["P"])) / np.sqrt(np.log(1 + NOISE["cv_P"] ** 2)))
    if "y" in outs:
        r.append((o["y"] - ref["y"]) / NOISE["sd_y"])
    return np.concatenate(r)


def profile_point(th_true, ref, beta, nuis, outs):
    th0 = dict(th_true, beta=beta)
    z0 = np.array([O._to_z(k, th_true[k]) for k in nuis])
    lo = np.array([O._to_z(k, O.BOUNDS[k][0]) for k in nuis])
    hi = np.array([O._to_z(k, O.BOUNDS[k][1]) for k in nuis])
    best = None
    for scale in (1.0, 0.5, 2.0):
        zs = z0.copy()
        if "kappa" in nuis:
            zs[nuis.index("kappa")] = np.log(th_true["kappa"] * th_true["beta"] / beta) if scale == 1.0 \
                else np.log(th_true["kappa"] * scale)
        zs = np.clip(zs, lo + 1e-9, hi - 1e-9)
        s = least_squares(lambda z: weighted_residuals(O._pack(th0, nuis, z), ref, outs), zs, bounds=(lo, hi),
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=5000)
        if best is None or s.cost < best.cost:
            best = s
    return 2 * best.cost, O._pack(th0, nuis, best.x)


def main():
    th = load_theta()
    ref = O.outputs(th, T_OBS)
    out, arr = {"theta_true": th}, {}
    # (1) Ridge calculation
    mult = np.array([0.5, 0.71, 1.0, 1.41, 2.0])
    tt = np.linspace(0, 21 * O.DAY, 400)
    ridge = []
    for c in mult:
        chi, thf = profile_point(th, ref, c * th["beta"], ["kappa", "k2", "p"], ["f"])
        o = O.outputs(thf, tt)
        oo = O.outputs(thf, T_OBS)
        ridge.append(dict(mult=float(c), dchi2=float(chi), max_abs_df=float(np.abs(oo["f"] - ref["f"]).max()),
                          kappa=thf["kappa"], k2=thf["k2"], p=thf["p"],
                          Q_day3=float(O.outputs(thf, np.array([3 * O.DAY]))["Q"][0]),
                          Q_day7=float(O.outputs(thf, np.array([7 * O.DAY]))["Q"][0])))
        arr[f"ridge_f_{c}"] = o["f"]
        arr[f"ridge_Q_{c}"] = o["Q"]
        arr[f"ridge_R_{c}"] = o["R"]
    arr["ridge_t"] = tt
    arr["obs_t"] = T_OBS
    arr["obs_f"] = ref["f"]
    out["ridge"] = ridge
    for r in ridge:
        print(f"beta x{r['mult']:.2f}: dchi2 = {r['dchi2']:.3f}, max|df| = {r['max_abs_df']:.4f}, "
              f"Q/P day 3 = {r['Q_day3']:.3f}, day 7 = {r['Q_day7']:.3f}, p = {r['p']:.3f}")
    # (2) Expected profile likelihoods
    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 49))
    cases = {"F2: kappa free, green": (["kappa"], ["f"]),
             "F4: kappa, k2, p free, green": (["kappa", "k2", "p"], ["f"]),
             "F4 + total abundance": (["kappa", "k2", "p"], ["f", "P"]),
             "F4 + weighted readout": (["kappa", "k2", "p"], ["f", "y"])}
    prof = {}
    for name, (nuis, outs) in cases.items():
        refo = O.outputs(th, T_OBS)
        vals = np.array([profile_point(th, refo, g * th["beta"], nuis, outs)[0] for g in grid])
        arr["prof_" + name] = vals
        inside = grid[vals <= CHI95]
        prof[name] = dict(lower=float(inside.min()), upper=float(inside.max()),
                          bounded_low=bool(vals[0] > CHI95), bounded_high=bool(vals[-1] > CHI95))
        print(f"{name:32s} 95% range of beta/beta_true: [{inside.min():.3f}, {inside.max():.3f}]  "
              f"bounded: {vals[0] > CHI95}/{vals[-1] > CHI95}")
    arr["prof_grid"] = grid
    out["profiles"] = prof
    out["noise"] = NOISE
    out["t_obs_days"] = (T_OBS / O.DAY).tolist()
    np.savez(os.path.join(ROOT, "results", "p2_observability.npz"), **arr)
    with open(os.path.join(ROOT, "results", "p2_observability.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
