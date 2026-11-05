"""Build the LaTeX macro file from archived machine-readable results.

Run this script after recomputing an analysis so that numerical values in the
manuscript remain synchronized with the archived outputs.
"""
import json
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")


def g(x, d=2):
    """Format a number for LaTeX with ``d`` significant digits."""
    s = f"{x:.{d}g}"
    if "e" in s:
        m, e = s.split("e")
        return rf"{m}\times10^{{{int(e)}}}"
    return s


def f(x, nd):
    return f"{x:.{nd}f}"


def f_up(x, nd):
    """Round upward for a reported upper bound."""
    q = 10 ** nd
    return f"{np.ceil(x * q - 1e-9) / q:.{nd}f}"


def f_down(x, nd):
    """Round downward for a reported lower bound."""
    q = 10 ** nd
    return f"{np.floor(x * q + 1e-9) / q:.{nd}f}"


def main():
    c = json.load(open(os.path.join(RES, "p0_candidates.json"), encoding="utf-8"))
    C1 = c["candidates"]["C1"]
    P, ch = C1["params"], C1["characteristics"]
    idf = json.load(open(os.path.join(RES, "p1_identifiability.json"), encoding="utf-8"))

    m = {}
    # Model parameters (Table B1).
    names = dict(kOneBar="k1bar", KN="KN", kTwo="k2", p="p", alphaBar="albar", Kalpha="Kal", malpha="nal",
                 betaBar="bebar", Kbeta="Kbe", nbeta="n", muZero="muR0", muMax="muRmax", Kmu="KmuR",
                 muQMax="muQmax", lR="lamR", lG="lamG", lQ="lamQ", delta="delta", Szero="S0", gam="gamma",
                 etaG="etaG")
    for mac, key in names.items():
        m["Par" + mac] = g(P[key], 3)
    # Realised time scales.
    m["ValTauGOne"] = f(ch["tauG1_h"], 0)
    m["ValTauS"] = f(ch["tauS_h"], 0)
    m["ValTauExit"] = f(ch["tauexit_days"], 1)
    m["ValTauEntry"] = f(ch["tauentry_h"], 0)
    # Reference-regime characteristics.
    m["ValRnot"] = f(ch["R0"], 2)
    m["ValPiS"] = f(ch["piS"], 3)
    m["ValPiM"] = f(ch["piM"], 3)
    m["ValPiRet"] = f(ch["piret"], 3)
    m["ValLambda"] = g(ch["Lambda"], 2)
    m["ValDoubling"] = f(ch["doubling_days"], 1)
    m["ValNstar"] = f(ch["Nstar"], 2)
    m["ValPstar"] = f(ch["Pstar"], 1)
    m["ValQfrac"] = f(ch["comp_star"][2], 2)
    m["ValQfracPct"] = f(100 * ch["comp_star"][2], 0)
    m["ValFred"] = f(ch["fred_star"], 2)
    m["ValQnormoxPct"] = f(100 * ch["comp_normoxic"][2], 0)
    m["ValGreenNormox"] = f(ch["fgreen_normoxic"], 2)
    m["ValUoneStar"] = f(ch["u1_star"], 3)
    m["ValTninety"] = f(ch["T90_days"], 0)
    m["ValReoxHalf"] = f(ch["reox_t50_green_days"], 1)
    m["ValReoxQ"] = f(ch["reox_Qfrac_at_t50"], 2)
    m["ValSeedPzero"] = str(c["meta"]["seed"])
    m["ValNdraws"] = f"{c['meta']['n_draws']:,}".replace(",", "\\,")
    # Structural-identifiability ranks.
    t = idf["table"]
    key = lambda s, o: t[f"{s} | {o}"]["ranks"]["C1"]
    m["RankGreenSix"] = str(key("(kappa, beta, k2, p, alpha, dQ)", "green fraction"))
    m["RankGreenEight"] = str(key("all eight rates", "green fraction"))
    m["RankAbundEight"] = str(key("all eight rates", "green fraction + total abundance"))
    m["RankAOEight"] = str(key("all eight rates", "green fraction + AO readout"))
    m["RankAllEight"] = str(key("all eight rates", "green fraction + abundance + AO"))
    m["IdOrder"] = str(idf["order"])

    # Reference-regime dynamics.
    p2f = os.path.join(RES, "p2_dynamics.json")
    if os.path.exists(p2f):
        import numpy as np
        d2 = json.load(open(p2f, encoding="utf-8"))
        arr = np.load(os.path.join(RES, "p2_dynamics.npz"))
        pro, sw = d2["protocol"], d2["Qshare_sweep"]
        m["ValUtwoStar"] = f(d2["u2c_at_u1_0"], 3)
        m["ValFrontierDevExp"] = str(int(np.ceil(np.log10(max(d2["frontier_max_abs_dev"], 1e-300)))))
        m["ValNhalfDays"] = f(d2["N_half_days"], 0)
        m["ValOvershootPct"] = f(100 * (d2["Pmax_over_Pstar"] - 1), 0)
        m["ValMultiMin"] = f(d2["multistart_P0_range"][0], 2)
        m["ValMultiMax"] = f(d2["multistart_P0_range"][1], 1)
        m["ValProtoUone"] = f(pro["u1"], 2)
        m["ValProtoDays"] = f(pro["days_on"], 0)
        m["ValProtoResidualPct"] = f(100 * pro["residual_fraction"], 0)
        m["ValProtoRsharePct"] = f(100 * pro["comp_at_withdrawal"][0], 0)
        m["ValProtoQsharePct"] = f(100 * pro["comp_at_withdrawal"][2], 0)
        m["ValProtoFred"] = f(pro["fred_at_withdrawal"], 2)
        m["ValProtoTfifty"] = f(pro["T50_days"], 1)
        m["ValProtoTninety"] = f(pro["T90_days"], 1)
        m["ValProtoHalfLife"] = f(np.log(2) / abs(pro["Lambda_on"]) / 24, 0)
        m["ValSweepTfiftyZero"] = f(sw["T50_q0"], 1)
        m["ValSweepTfiftyOne"] = f(sw["T50_q1"], 1)
        m["ValSweepTninetyZero"] = f(sw["T90_q0"], 1)
        m["ValSweepTninetyOne"] = f(sw["T90_q1"], 1)
        dd, T6, T2 = arr["C_d"], arr["C_T6"] / 24, arr["C_T2"] / 24
        m["ValTepsDsmall"] = g(dd[0], 1)
        m["ValTepsSixSmall"] = f"{T6[0]:,.0f}".replace(",", "\\,")
        m["ValTepsTwoSmall"] = f"{T2[0]:,.0f}".replace(",", "\\,")
        m["ValTepsDlarge"] = f(dd[-1], 3)
        m["ValTepsSixLarge"] = f"{T6[-1]:,.0f}".replace(",", "\\,")
        E, names = arr["D_E"], [str(n) for n in arr["D_names"]]
        m["ValSensFredMax"] = f_up(np.abs(E[:, 2]).max(), 3)
        m["ValSensLambdaP"] = f(E[names.index("p"), 0], 2)
        m["ValSensPstarM"] = f(E[names.index("nal"), 1], 2)

    # Stability scan and Hopf example.
    for fn in ("p1_stability_scan.json", "p1_hopf_example.json", "p1_hopf_analysis.json"):
        if not os.path.exists(os.path.join(RES, fn)):
            break
    else:
        sc = json.load(open(os.path.join(RES, "p1_stability_scan.json"), encoding="utf-8"))
        hx = json.load(open(os.path.join(RES, "p1_hopf_example.json"), encoding="utf-8"))
        m["ValScanDraws"] = f"{sc['n_local']:,}".replace(",", "\\,")
        m["ValScanInvasive"] = f"{sc['n_invasive']:,}".replace(",", "\\,")
        m["ValScanUnstable"] = str(sc["n_unstable"])
        m["ValScanUnstablePct"] = f(100 * sc["n_unstable"] / sc["n_invasive"], 1)
        m["ValGlobalSets"] = str(sc["n_global_sets"])
        m["ValGlobalRuns"] = f"{sc['n_runs']:,}".replace(",", "\\,")
        m["ValScanMleTwoInvasive"] = f"{hx['stats_m_n_le2']['m_le2'][0]:,}".replace(",", "\\,")
        m["ValScanMleTwoUnstable"] = str(hx["stats_m_n_le2"]["m_le2"][1])
        m["ValHopfM"] = f(hx["m_Hopf"], 3)
        m["ValHopfPeriod"] = f(hx["period_onset_days"], 1)
        m["ValHopfSlope"] = f(hx["amp_loglog_slope"], 2)
        m["ValCyclePeriod"] = f(hx["example_cycle"]["period_days"], 0)
        m["ValCyclePmin"] = f(hx["example_cycle"]["P_range"][0], 2)
        m["ValCyclePmax"] = f(hx["example_cycle"]["P_range"][1], 1)
        m["ValCycleFredMin"] = f(hx["example_cycle"]["fred_range"][0], 2)
        m["ValCycleFredMax"] = f(hx["example_cycle"]["fred_range"][1], 2)

    # Observation ridge and profile calculations.
    obf = os.path.join(RES, "p2_observability.json")
    if os.path.exists(obf):
        ob = json.load(open(obf, encoding="utf-8"))
        rid = ob["ridge"]
        m["ValRidgeMaxChi"] = f(max(r["dchi2"] for r in rid), 2)
        m["ValRidgeMaxDf"] = f(max(r["max_abs_df"] for r in rid), 3)
        m["ValRidgeQminDaySeven"] = f(min(r["Q_day7"] for r in rid), 2)
        m["ValRidgeQmaxDaySeven"] = f(max(r["Q_day7"] for r in rid), 2)
        pf = ob["profiles"]
        m["ValProfFtwoLo"] = f(pf["F2: kappa free, green"]["lower"], 2)
        m["ValProfFtwoHi"] = f(pf["F2: kappa free, green"]["upper"], 2)
        m["ValProfAbHi"] = f(pf["F4 + total abundance"]["upper"], 2)
        m["ValProfYLo"] = f(pf["F4 + weighted readout"]["lower"], 2)
        m["ValProfYHi"] = f(pf["F4 + weighted readout"]["upper"], 2)
        m["ValObsN"] = str(ob["noise"]["n"])
        m["ValObsCvPct"] = f(100 * ob["noise"]["cv_P"], 0)
        m["ValObsSdY"] = f(ob["noise"]["sd_y"], 2)

    # Reference-regime Monte Carlo experiment.
    mcf = os.path.join(RES, "p2_mc_reference.json")
    if os.path.exists(mcf):
        mc = json.load(open(mcf, encoding="utf-8"))
        S = mc["summary"]
        names = {"A_W7": "AWseven", "A_W14": "AWfourteen", "A_W21": "AWtwentyone", "A_E21": "AEtwentyone",
                 "B_f": "Bf", "B_fP": "BfP", "B_fy": "Bfy", "B_fPy": "BfPy", "D_BB3": "DBB", "D_CE3": "DCE",
                 "D_SC10": "DSC", "D_fy": "Dfy", "D_fy_w020": "Dwlo", "D_fy_w050": "Dwhi"}
        for k, short in names.items():
            m["Mc" + short] = f(100 * S[k]["rmse_beta"], 0)
            m["McLo" + short] = f(100 * S[k]["ci_beta"][0], 0)
            m["McHi" + short] = f(100 * S[k]["ci_beta"][1], 0)
            m["McBias" + short] = ("+" if S[k]["median_bias_beta"] > 0 else "-") + \
                f(abs(100 * S[k]["median_bias_beta"]), 0)
            m["McBound" + short] = str(S[k]["bound_hits"])
        m["McProdAWtwentyone"] = f(100 * S["A_W21"]["rmse_prod"], 0)
        m["McReps"] = str(mc["n_rep"])
        m["McBoot"] = f"{mc['n_boot']:,}".replace(",", "\\,")
        pe = mc["pairs"]["A_E21 - A_W21"]
        m["McDiffEarlyLo"] = f(100 * pe["ci"][0], 1)
        m["McDiffEarlyHi"] = f(100 * pe["ci"][1], 1)

    # Global convergence check without a time cap.
    grf = os.path.join(RES, "p1_global_recheck.json")
    if os.path.exists(grf):
        import numpy as np
        gr = json.load(open(grf, encoding="utf-8"))
        m["ValGlobalHorizon"] = f(gr["horizon_relaxation_times"], 0)
        m["ValGlobalConverged"] = f"{gr['n_converged']:,}".replace(",", "\\,")
        m["ValGlobalWorstExp"] = str(int(np.ceil(np.log10(gr["worst_err"]))))

    # Morris screening.
    mof = os.path.join(RES, "p2_morris.json")
    if os.path.exists(mof):
        import numpy as np
        mo = json.load(open(mof, encoding="utf-8"))
        mu = np.array(mo["mu_star"])
        m["MorrisFredMax"] = f_up(mu[:, 2].max(), 3)
        m["MorrisSpearmanMin"] = f_down(min(min(v) for v in mo["spearman_other_seeds"].values()), 2)
        m["MorrisDropped"] = str(mo["dropped"])
        m["MorrisR"] = str(mo["design"]["R"])

    # Parameter-ensemble Monte Carlo experiment.
    enf = os.path.join(RES, "p2_mc_ensemble.json")
    if os.path.exists(enf):
        en = json.load(open(enf, encoding="utf-8"))
        m["EnsSets"] = str(en["n_sets"])
        m["EnsReps"] = str(en["n_rep"])
        for key, short in (("product", "Prod"), ("window", "Window"), ("abundance", "Abund"), ("readout", "Read")):
            m["EnsHolds" + short] = str(en["conclusions"][key]["holds"])
            m["EnsSig" + short] = str(en["conclusions"][key]["significant"])

    # Exposure times near the extinction boundary.
    exf = os.path.join(RES, "p5_exposure_boundary.json")
    if os.path.exists(exf):
        exj = json.load(open(exf, encoding="utf-8"))
        ex = {r["label"]: r for r in exj["rows"]}
        m["ValTepsTwoCrit"] = f(ex[0.0]["T_0.01"], 0)
        m["ValTepsRatioSmall"] = f(ex[1e-3]["ratio_1e-06"], 2)
        m["ValTepsRatioLarge"] = f(ex["large"]["ratio_1e-06"], 2)
        m["ValSensNbetaMax"] = f_up(max(abs(v) for v in exj["nbeta_elasticities"].values()), 3)

    # Additional quantities introduced during the final manuscript revision.
    import sys
    sys.path.insert(0, HERE)
    import model as M
    ref = M.Params(**c["candidates"]["C1"]["params"])

    def q_normox(pr):
        w, V = np.linalg.eig(M.A(pr.Nhat, pr))
        v = np.abs(V[:, np.argmax(w.real)].real)
        return v[2] / v.sum()

    m["ValQnormoxHypPct"] = f(100 * q_normox(ref.copy(nal=1.0)), 0)
    hxf = os.path.join(RES, "p1_hopf_example.json")
    if os.path.exists(hxf):
        he = json.load(open(hxf, encoding="utf-8"))
        hd = ref.as_dict()
        hd.update(he["base_params"])
        hp = M.Params(**hd)
        Eh = M.equilibrium(hp)
        m["ValHopfMsample"] = f(he["m_example"], 2)
        m["ValHopfNstar"] = f(Eh[3], 3)
        m["ValHopfKal"] = f(hp.Kal, 4)
        m["ValHopfLamRatio"] = f(hp.lamQ / hp.lamR, 0)
        m["ValHopfExitDays"] = f(1.0 / M.beta(hp.Nhat, hp) / 24.0, 1)

    lines = ["% Automatically produced by code/make_macros.py.",
             "% Edit the source data or generating script rather than this file."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items()]
    with open(os.path.join(RES, "macros.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
