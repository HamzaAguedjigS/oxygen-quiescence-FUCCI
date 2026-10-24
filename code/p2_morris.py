"""Morris global-sensitivity screening around the reference regime.

Seventeen effective parameters vary on a logarithmic scale, with shared
parameters perturbed together. One-at-a-time trajectories estimate elementary
effects for Lambda, P*, f_red*, and u1*. Non-invasive points are recorded and
excluded. Rank robustness is evaluated across six fixed seeds.

Output: ``results/p2_morris.json``.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as M  # noqa: E402

GROUPS = {"k1bar": ["k1bar"], "KN": ["KN"], "k2": ["k2"], "p": ["p"], "albar": ["albar"], "Kal": ["Kal"],
          "nal": ["nal"], "bebar": ["bebar"], "Kbe": ["Kbe"], "mu0": ["muR0", "muG0"],
          "mumax": ["muRmax", "muGmax"], "muQmax": ["muQmax"], "Kmu": ["KmuR", "KmuG", "KmuQ"],
          "lamR": ["lamR"], "lamG": ["lamG"], "lamQ": ["lamQ"], "delta": ["delta"]}
NAMES = list(GROUPS)
LEVELS, JUMP = 6, 3


def load_C1():
    c = json.load(open(os.path.join(ROOT, "results", "p0_candidates.json"), encoding="utf-8"))
    return M.Params(**c["candidates"]["C1"]["params"])


def make(base, u):
    """Map u in [0,1]^k to log-uniform parameters; map p linearly."""
    kw = {}
    for ui, g in zip(u, NAMES):
        for k in GROUPS[g]:
            if k == "p":
                kw[k] = 0.85 + ui * (0.995 - 0.85)
            else:
                kw[k] = getattr(base, k) * float(np.exp(np.log(0.5) + ui * np.log(4.0)))
    return base.copy(**kw)


def outputs(pr):
    if M.R0(pr) <= 1.0:
        return None
    E = M.equilibrium(pr)
    u1 = M.u1_star(pr)
    if u1 is None:
        return None
    return np.array([M.growth_exponent(pr), E[:3].sum(), (E[0] + E[2]) / E[:3].sum(), u1])


def morris(base, seed, R=100):
    rng = np.random.default_rng(seed)
    k = len(NAMES)
    grid = np.arange(LEVELS) / (LEVELS - 1)
    step = JUMP / (LEVELS - 1)
    EE = [[] for _ in range(k)]
    dropped = 0
    for _ in range(R):
        x = rng.choice(grid[: LEVELS - JUMP], size=k)
        x = np.where(rng.random(k) < 0.5, x, x + step)            # Random grid starting point
        x = np.clip(x, 0, 1)
        y = outputs(make(base, x))
        for i in rng.permutation(k):
            x2 = x.copy()
            x2[i] = x[i] + step if x[i] + step <= 1 + 1e-12 else x[i] - step
            y2 = outputs(make(base, x2))
            if y is not None and y2 is not None:
                if NAMES[i] == "p":
                    dlog = np.log((0.85 + x2[i] * 0.145) / (0.85 + x[i] * 0.145))
                else:
                    dlog = (x2[i] - x[i]) * np.log(4.0)
                EE[i].append((y2 - y) / (0.5 * np.abs(y2 + y)) / dlog)
            else:
                dropped += 1
            x, y = x2, y2
    mu_star = np.array([np.mean(np.abs(e), axis=0) for e in EE])
    sigma = np.array([np.std(e, axis=0, ddof=1) for e in EE])
    n = np.array([len(e) for e in EE])
    return mu_star, sigma, n, dropped


def main():
    base = load_C1()
    mu, sg, n, dropped = morris(base, seed=0)
    ranks = {}
    for s in range(1, 6):
        m2, *_ = morris(base, seed=s)
        ranks[s] = [float(spearmanr(mu[:, j], m2[:, j])[0]) for j in range(4)]
    outs = ["Lambda", "Pstar", "fred", "u1star"]
    top = {o: [NAMES[i] for i in np.argsort(-mu[:, j])[:4]] for j, o in enumerate(outs)}
    for o in outs:
        print(o, "top-4 mu*:", top[o])
    zero = {o: float(max(mu[NAMES.index(g), j] for g in ("lamR", "lamG", "lamQ", "delta")))
            for j, o in enumerate(outs)}
    print("max mu* of oxygen parameters per output:", zero)
    print("dropped effects (non-invasive or no threshold):", dropped)
    print("Spearman vs seed 0 (Lambda, P*, fred, u1*):", {s: np.round(v, 3).tolist() for s, v in ranks.items()})
    out = dict(names=NAMES, outputs=outs, mu_star=mu.tolist(), sigma=sg.tolist(), n_effects=n.tolist(),
               dropped=dropped, top4=top, oxygen_param_max_mu=zero, spearman_other_seeds=ranks,
               design=dict(R=100, levels=LEVELS, jump=JUMP, range="x0.5–x2 log-uniform; p in [0.85, 0.995]"))
    with open(os.path.join(ROOT, "results", "p2_morris.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
