"""Observation model for the clamped-reoxygenation experiment.

Oxygen is fixed at N_r = Nhat and the cellular subsystem is X' = A X in the
state order (R, G, Q). Effective rates are stored in ``theta``. Outputs are
the green fraction G/P, relative abundance P(t)/P(0), and the state-weighted
readout (w.X)/P. The observation model supports binomial or beta-binomial
fraction noise, log-normal abundance noise, additive Gaussian readout noise,
and symmetric classification error. Parameters are estimated by bounded,
weighted least squares from four deterministic starting points.
"""
import numpy as np
from scipy.optimize import least_squares

DAY = 24.0
NAMES = ["kappa", "alpha", "dR", "k2", "dG", "p", "beta", "dQ"]
W_REF = np.array([1.0, 1.15, 0.35])          # poids (R, G, Q) du lecteur

# Broad absolute bounds (h^-1), independent of the generating values
BOUNDS = {"kappa": (1 / 500, 1 / 5), "beta": (1 / (60 * DAY), 1 / (0.5 * DAY)), "k2": (0.02, 0.5),
          "p": (0.5, 0.999)}
STARTS = {"kappa": (1 / 100, 1 / 40), "beta": (1 / (10 * DAY), 1 / (3 * DAY)), "k2": (0.1,), "p": (0.9,)}


def theta_from_params(pr, M, N=None):
    N = pr.Nhat if N is None else N
    return dict(kappa=float(M.kappa(N, pr)), alpha=float(M.alpha(N, pr)), dR=float(M.muR(N, pr)), k2=float(pr.k2),
                dG=float(M.muG(N, pr)), p=float(pr.p), beta=float(M.beta(N, pr)), dQ=float(M.muQ(N, pr)))


def matrix(th):
    return np.array([[-(th["kappa"] + th["alpha"] + th["dR"]), 2 * th["p"] * th["k2"], th["beta"]],
                     [th["kappa"], -(th["k2"] + th["dG"]), 0.0],
                     [th["alpha"], 2 * (1 - th["p"]) * th["k2"], -(th["beta"] + th["dQ"])]])


def states(th, t, x0=(0.0, 0.0, 1.0)):
    """X(t) pour t en heures (tableau 3 x len(t))."""
    w, V = np.linalg.eig(matrix(th))
    c = np.linalg.solve(V, np.asarray(x0, float))
    return (V @ (c[:, None] * np.exp(np.outer(w, t)))).real


def outputs(th, t, x0=(0.0, 0.0, 1.0), w=W_REF):
    X = states(th, t, x0)
    P = X.sum(0)
    return dict(f=X[1] / P, P=P / np.sum(x0), y=(w @ X) / P, Q=X[2] / P, R=X[0] / P)


# Data simulation
def simulate(th_true, t, rng, outs, noise, x0_true=(0.0, 0.0, 1.0), w_true=W_REF):
    o = outputs(th_true, t, x0_true, w_true)
    data = {}
    if "f" in outs:
        f = o["f"]
        e = noise.get("class_err", 0.0)
        f = e + (1 - 2 * e) * f
        n = noise["n"]
        phi = noise.get("phi", 1.0)
        if phi > 1.0:
            conc = (n - phi) / (phi - 1.0)
            pp = rng.beta(np.maximum(f * conc, 1e-9), np.maximum((1 - f) * conc, 1e-9))
            data["f"] = rng.binomial(n, pp) / n
        else:
            data["f"] = rng.binomial(n, f) / n
    if "P" in outs:
        cv = noise["cv_P"]
        s = np.sqrt(np.log(1 + cv ** 2))
        data["P"] = o["P"] * np.exp(rng.normal(-0.5 * s * s, s, size=len(t)))
    if "y" in outs:
        data["y"] = o["y"] + rng.normal(0, noise["sd_y"], size=len(t))
    return data


# Parameter estimation
def _pack(th_fixed, free, z):
    th = dict(th_fixed)
    for k, v in zip(free, z):
        th[k] = 1 / (1 + np.exp(-v)) if k == "p" else np.exp(v)
    return th


def _to_z(k, v):
    return np.log(v / (1 - v)) if k == "p" else np.log(v)


def fit(data, t, th_fixed, free, noise, x0_fit=(0.0, 0.0, 1.0), w_fit=W_REF):
    n = noise["n"]
    sP = np.sqrt(np.log(1 + noise.get("cv_P", 0.1) ** 2))
    sy = noise.get("sd_y", 0.03)
    sdf = np.sqrt(np.maximum(data["f"] * (1 - data["f"]), 1.0 / n) / n) if "f" in data else None

    def res(z):
        o = outputs(_pack(th_fixed, free, z), t, x0_fit, w_fit)
        r = []
        if "f" in data:
            r.append((o["f"] - data["f"]) / sdf)
        if "P" in data:
            r.append((np.log(np.maximum(o["P"], 1e-300)) - np.log(data["P"])) / sP)
        if "y" in data:
            r.append((o["y"] - data["y"]) / sy)
        return np.concatenate(r)

    lo = np.array([_to_z(k, BOUNDS[k][0]) for k in free])
    hi = np.array([_to_z(k, BOUNDS[k][1]) for k in free])
    best = None
    grids = np.meshgrid(*[STARTS[k] for k in free], indexing="ij")
    for start in zip(*[g.ravel() for g in grids]):
        z0 = np.array([_to_z(k, v) for k, v in zip(free, start)])
        try:
            s = least_squares(res, z0, bounds=(lo, hi), method="trf", x_scale=1.0, xtol=1e-10, ftol=1e-10,
                              gtol=1e-10, max_nfev=2000)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or s.cost < best.cost:
            best = s
    th = _pack(th_fixed, free, best.x)
    at_bound = bool(np.any(np.isclose(best.x, lo, atol=1e-6) | np.isclose(best.x, hi, atol=1e-6)))
    return th, 2 * best.cost, at_bound
