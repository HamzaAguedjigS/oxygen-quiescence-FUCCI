"""Construct candidate reference parameter regimes.

The G1, S/G2/M, quiescence-exit, and anoxic-entry time scales are anchored to
the experimental setting described in the companion study. Remaining
parameters are sampled with a fixed seed over broad ranges and ranked against
prespecified regime targets rather than fitted data. Selected values are
rounded and then re-evaluated.

Outputs: ``results/p0_candidates.json`` and
``results/p0_candidates_table.md``.
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as M  # noqa: E402

SEED = 20261022
N_DRAWS = 20000
DAY = 24.0

ANCHORS = {
    "C1": dict(label="companion study, central values", tauG1=65.0, tauS=10.0, tauexit=7.2 * DAY, tauentry=1.0),
    "C2": dict(label="companion study, fast bounds", tauG1=61.0, tauS=10.0, tauexit=3.4 * DAY, tauentry=1.0),
}
TARGET = dict(R0=1.7, Nstar=0.2, qstar=0.5, qnorm_max=0.12)
FIXED = dict(S0=1.0, gamma=1.0, delta=0.02, lamR=0.1, lamG=0.1, lamQ=0.05, n=2.0, muQ0=0.0, etaG=0.2)


def sig(x, d=2):
    return float(f"{x:.{d}g}")


def build(free, anc, rounded=False):
    """Construct a full parameter set from free variables and anchors."""
    r = (lambda x: sig(x, 2)) if rounded else (lambda x: x)
    f = {k: r(v) for k, v in free.items()}
    k1bar = (1.0 / anc["tauG1"]) * (f["KN"] + 1.0)
    bebar = (1.0 / anc["tauexit"]) * (f["Kbe"] ** FIXED["n"] + 1.0)
    if rounded:
        k1bar, bebar = sig(k1bar, 3), sig(bebar, 3)
    return M.Params(**FIXED, k1bar=k1bar, KN=f["KN"], k2=1.0 / anc["tauS"], p=f["p"],
                    albar=1.0 / anc["tauentry"], Kal=f["Kal"], nal=f["nal"], bebar=bebar, Kbe=f["Kbe"],
                    muR0=f["mu0"], muRmax=f["mumax"], KmuR=f["Kmu"],
                    muG0=f["mu0"], muGmax=f["mumax"], KmuG=f["Kmu"],
                    muQmax=f["muQmax"], KmuQ=f["Kmu"])


def perron(N, pr):
    w, V = np.linalg.eig(M.A(N, pr))
    v = np.abs(V[:, np.argmax(w.real)].real)
    return v / v.sum()


def evaluate(pr):
    if M.R0(pr) <= 1.0:
        return None
    E = M.equilibrium(pr)
    P = E[:3].sum()
    ev = np.linalg.eigvals(M.jacobian(E, pr))
    return dict(R0=M.R0(pr), Nstar=E[3], Pstar=P, qstar=E[2] / P, comp=E[:3] / P,
                qnorm=perron(pr.Nhat, pr)[2], stable=bool(ev.real.max() < 0), eig=ev)


def loss(ev):
    if ev is None or not ev["stable"]:
        return np.inf
    L = np.log(ev["R0"] / TARGET["R0"]) ** 2 + np.log(ev["Nstar"] / TARGET["Nstar"]) ** 2 \
        + np.log(ev["qstar"] / TARGET["qstar"]) ** 2
    return L + max(0.0, ev["qnorm"] - TARGET["qnorm_max"]) ** 2 / 0.02 ** 2


def draw(rng):
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    return dict(KN=lu(0.01, 0.3), Kal=lu(0.03, 0.3), nal=float(rng.uniform(2, 6)), Kbe=float(rng.uniform(0.1, 0.6)),
                p=float(rng.uniform(0.6, 1.0)), mu0=lu(3e-4, 4e-3), mumax=lu(3e-3, 8e-2), Kmu=lu(0.02, 0.3),
                muQmax=lu(1e-3, 5e-2))


def search(anc, rng):
    best, bestL = None, np.inf
    for _ in range(N_DRAWS):
        f = draw(rng)
        try:
            L = loss(evaluate(build(f, anc)))
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            continue
        if L < bestL:
            best, bestL = f, L
    return best, bestL


# Regime characterization
def characterise(pr):
    ev = evaluate(pr)
    E = M.equilibrium(pr)
    piS, piM, piret, r0 = M.R0_factors(pr)
    lam = M.growth_exponent(pr)
    vn = perron(pr.Nhat, pr)
    out = dict(R0=r0, piS=piS, piM=piM, piret=piret, Lambda=lam, doubling_days=np.log(2) / lam / DAY,
               Nstar=E[3], Pstar=E[:3].sum(), comp_star=(E[:3] / E[:3].sum()).tolist(),
               fred_star=float((E[0] + E[2]) / E[:3].sum()), comp_normoxic=vn.tolist(),
               fgreen_normoxic=float(vn[1]), max_re_eig=float(ev["eig"].real.max()),
               oscillatory=bool(np.abs(ev["eig"].imag).max() > 1e-9), u1_star=M.u1_star(pr),
               tauG1_h=1 / M.kappa(pr.Nhat, pr), tauS_h=1 / pr.k2, tauexit_days=1 / M.beta(pr.Nhat, pr) / DAY,
               tauentry_h=1 / M.alpha(0.0, pr), alpha_normoxic_per_h=float(M.alpha(pr.Nhat, pr)))
    # Growth from 1% of P* with normoxic composition and oxygen
    y0 = [*(0.01 * out["Pstar"] * vn), pr.Nhat]
    T = 200 * DAY
    s = solve_ivp(M.rhs, (0, T), y0, args=(pr,), method="LSODA", rtol=1e-9, atol=1e-12, dense_output=True)
    t = np.linspace(0, T, 4001)
    Y = s.sol(t)
    P = Y[:3].sum(0)
    out["T90_days"] = float(t[np.argmax(P >= 0.9 * out["Pstar"])] / DAY) if (P >= 0.9 * out["Pstar"]).any() else None
    # Clamped reoxygenation from a pure quiescent initial state
    Am = M.A(pr.Nhat, pr)
    tr = np.linspace(0, 30 * DAY, 3001)
    X = np.array([expm(Am * ti) @ np.array([0, 0, 1.0]) for ti in tr]).T
    F = X / X.sum(0)
    plateau = vn[1]
    i50 = np.argmax(F[1] >= 0.5 * plateau)
    out["reox_t50_green_days"] = float(tr[i50] / DAY)
    out["reox_Qfrac_at_t50"] = float(F[2, i50])
    return out, (t, Y), (tr, F)


def main():
    rng = np.random.default_rng(SEED)
    cands = {}
    for key, anc in ANCHORS.items():
        free, L = search(anc, rng)
        pr = build(free, anc, rounded=True)
        cands[key] = dict(label=anc["label"], params=pr, free={k: sig(v, 2) for k, v in free.items()},
                          loss=float(loss(evaluate(pr))), loss_unrounded=float(L), anchors=anc)
    res = {}
    for key, c in cands.items():
        ch, _, _ = characterise(c["params"])
        res[key] = dict(label=c["label"], params=c["params"].as_dict(), characteristics=ch,
                        free=c["free"], loss=c["loss"], anchors=c.get("anchors"))
    meta = dict(seed=SEED, n_draws=N_DRAWS, targets=TARGET, fixed=FIXED,
                note="Free parameters drawn log-uniformly (KN, Kal, mu0, mumax, Kmu, muQmax) or uniformly "
                     "(nal, Kbe, p); muR=muG share (mu0, mumax, Kmu); KmuQ=Kmu; values rounded to 2 s.f.")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "p0_candidates.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(meta=meta, candidates=res), fh, indent=2, default=float)

    # Human-readable summary table
    rows = [("Normoxic G1 duration (h)", "tauG1_h", "{:.1f}"), ("S/G2/M duration (h)", "tauS_h", "{:.1f}"),
            ("Normoxic quiescence-exit time (d)", "tauexit_days", "{:.2f}"),
            ("Anoxic entry time (h)", "tauentry_h", "{:.2f}"),
            ("Normoxic entry rate α(N̂) (h⁻¹)", "alpha_normoxic_per_h", "{:.4f}"),
            ("𝓡₀", "R0", "{:.3f}"), ("π_S / π_M / π_ret", None, None),
            ("Λ (h⁻¹)", "Lambda", "{:.4f}"), ("Doubling time (d)", "doubling_days", "{:.1f}"),
            ("N*", "Nstar", "{:.3f}"), ("P* (CDU)", "Pstar", "{:.1f}"), ("Q*/P*", None, None),
            ("f_red*", "fred_star", "{:.3f}"), ("Normoxic Q fraction (Perron)", None, None),
            ("Normoxic green fraction (Perron)", "fgreen_normoxic", "{:.3f}"),
            ("max Re(eigenvalue) at E*", "max_re_eig", "{:.2e}"), ("Oscillatory E*", "oscillatory", "{}"),
            ("u₁* (u₂ = 0)", "u1_star", "{:.4f}"), ("T₉₀ from 1% of P* (d)", "T90_days", "{:.1f}"),
            ("Reoxygenation: green t½ (d)", "reox_t50_green_days", "{:.2f}"),
            ("Reoxygenation: Q/P at t½", "reox_Qfrac_at_t50", "{:.2f}")]
    keys = list(res)
    lines = ["| Grandeur | " + " | ".join(f"{k} ({res[k]['label']})" for k in keys) + " |",
             "|---|" + "---|" * len(keys)]
    for name, k, fmt in rows:
        vals = []
        for c in keys:
            ch = res[c]["characteristics"]
            if name.startswith("π_S"):
                vals.append(f"{ch['piS']:.3f} / {ch['piM']:.3f} / {ch['piret']:.3f}")
            elif name == "Q*/P*":
                vals.append(f"{ch['comp_star'][2]:.3f}")
            elif name.startswith("Normoxic Q"):
                vals.append(f"{ch['comp_normoxic'][2]:.3f}")
            else:
                v = ch[k]
                vals.append("—" if v is None else fmt.format(v))
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    lines += ["", "Rounded parameters:", "",
              "| Parameter | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for p in ["k1bar", "KN", "k2", "p", "albar", "Kal", "nal", "bebar", "Kbe", "n", "muR0", "muRmax", "KmuR",
              "muG0", "muGmax", "KmuG", "muQ0", "muQmax", "KmuQ", "lamR", "lamG", "lamQ", "delta", "S0", "gamma"]:
        lines.append(f"| {p} | " + " | ".join(f"{res[c]['params'][p]:.4g}" for c in keys) + " |")
    with open(os.path.join(ROOT, "results", "p0_candidates_table.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("Saved results/p0_candidates.json and results/p0_candidates_table.md")


if __name__ == "__main__":
    main()
