"""Generate the additional LaTeX macros introduced in Phase 7.

The script reads ``results/p7_*.json`` and writes only the new commands to
``results/macros_p7.tex``; the main ``results/macros.tex`` file is unchanged.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")


def sci(x, d=2):
    s = f"{x:.{d - 1}e}"
    m, e = s.split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}"


def spct(x):
    return f"{100 * x:+.0f}"


def main():
    ident = json.load(open(os.path.join(RES, "p7_identifiability_bounds.json"), encoding="utf-8"))
    num = json.load(open(os.path.join(RES, "p7_critical_and_numbers.json"), encoding="utf-8"))
    hopf = json.load(open(os.path.join(RES, "p7_hopf_lyapunov.json"), encoding="utf-8"))
    mc = json.load(open(os.path.join(RES, "p7_mc_robustness.json"), encoding="utf-8"))["scenarios"]
    assert (ident["bound_f"], ident["bound_fP"], ident["bound_fy"], ident["bound_fPy"]) == (5, 6, 7, 8)
    m = {}
    # Fraction of estimates within +/-25% of the true value.
    for key, lab in (("f", "Bf"), ("fP", "BfP"), ("fy", "Bfy"), ("fPy", "BfPy")):
        m["PsWithin" + lab] = pct(mc["base_" + key]["within25"])
    m["PsWideBf"] = pct(mc["wide_f"]["rmse"])
    m["PsPlowBf"] = pct(mc["p085_f"]["rmse"])
    m["PsPgeWithinBf"] = pct(mc["pge09_f"]["within25"])
    m["PsPgeWithinBfP"] = pct(mc["pge09_fP"]["within25"])
    m["PsPgeBfP"] = pct(mc["pge09_fP"]["rmse"])
    m["PsEightBfPy"] = pct(mc["eight_fPy"]["rmse"])
    m["PsEightBound"] = str(mc["eight_fPy"]["bound"])
    m["PsWPBiasLo"] = spct(mc["wq0.20_fPy"]["median_bias"])
    m["PsWPBiasHi"] = spct(mc["wq0.50_fPy"]["median_bias"])
    m["PsWPBiasLoS"] = spct(mc["wq0.30_fPy"]["median_bias"])
    m["PsWPBiasHiS"] = spct(mc["wq0.40_fPy"]["median_bias"])
    rs = sorted((mc["wq0.30_fPy"]["rmse"], mc["wq0.40_fPy"]["rmse"]))
    m["PsWPRmseMinS"], m["PsWPRmseMaxS"] = pct(rs[0]), pct(rs[1])
    # Critical-threshold and independently recomputed quantities.
    c = num["critical"]
    m["PsPc"] = f"{c['Pc_days']:.0f}"
    m["PsTcritPred"] = f"{c['T_eps']['0.01']['estimate_days']:.0f}"
    m["PsSweepTfiftyZero"] = f"{num['regrowth']['q0']['T50']:.1f}"
    m["PsSweepTninetyOne"] = f"{num['regrowth']['q1']['T90']:.1f}"
    p = num["profiles"]
    m["PsProfFtwoLo"] = f"{p['F2']['lower']:.2f}"
    m["PsProfFtwoHi"] = f"{p['F2']['upper']:.2f}"
    m["PsProfAbHi"] = f"{p['abund']['upper']:.2f}"
    m["PsProfYLo"] = f"{p['readout']['lower']:.2f}"
    m["PsProfYHi"] = f"{p['readout']['upper']:.2f}"
    m["PsAlphaHat"] = sci(num["alpha_hat"], 2)
    m["PsEigRe"] = f"{abs(num['eig_re']):.4f}"
    m["PsEigIm"] = f"{num['eig_im']:.4f}"
    m["PsEigPeriod"] = f"{num['eig_period_days']:.0f}"
    rows = {r["P0_over_Pstar"]: r["bias_beta"] for r in num["clamp"]["rows"]}
    m["PsClampOne"] = pct(rows[0.01], 1)
    m["PsClampThree"] = pct(rows[0.03], 0)
    m["PsClampTen"] = pct(rows[0.1], 0)
    # Hopf-bifurcation quantities.
    m["PsLyap"] = sci(hopf["l1"], 2)
    m["PsHopfTrans"] = f"{hopf['transversality']:.3f}"
    m["PsHopfKnee"] = f"{hopf['knee_n']:.1f}"
    a = hopf["amplitude"][0]
    m["PsHopfAmpDn"] = f"{a['dn']:.2f}"
    m["PsHopfAmpPred"] = f"{a['half_pred']:.3f}"
    m["PsHopfAmpSim"] = f"{a['half_sim']:.3f}"
    h = hopf["params"]
    names = [("k1bar", r"\overline k_1"), ("KN", "K_N"), ("k2", "k_2"), ("p", "p"), ("albar", r"\overline\alpha"), ("Kal", r"K_\alpha"),
             ("bebar", r"\overline\beta"), ("Kbe", r"K_\beta"), ("n", r"n_\beta"), ("muR0", r"\mu_R^0"), ("muRmax", r"\mu_R^{\max}"),
             ("KmuR", r"K_{\mu,R}"), ("muG0", r"\mu_G^0"), ("muGmax", r"\mu_G^{\max}"), ("KmuG", r"K_{\mu,G}"), ("muQmax", r"\mu_Q^{\max}"),
             ("KmuQ", r"K_{\mu,Q}"), ("lamR", r"\ell_R"), ("lamG", r"\ell_G"), ("lamQ", r"\ell_Q"), ("delta", r"\delta")]
    cells = [f"${s}$ & ${h[k]:.4g}$" for k, s in names]
    while len(cells) % 3:
        cells.append("&")
    m["PsHopfTableRows"] = " ".join(" & ".join(cells[i:i + 3]) + r"\\" for i in range(0, len(cells), 3))
    lines = ["% Automatically produced by code/make_macros_p7.py.",
             "% Edit the source data or generating script rather than this file."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items()]
    with open(os.path.join(RES, "macros_p7.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
