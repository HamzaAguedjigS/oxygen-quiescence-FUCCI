"""Generate manuscript figures from the archived numerical outputs.

  fig2_threshold   <- results/p2_dynamics.npz (A_*)
  fig3_reservoir   <- results/p2_dynamics.npz (B_*)
  fig4_control     <- results/p2_dynamics.npz (C_*)
  fig5_sensitivity <- results/p2_dynamics.npz (D_*)

Usage: python code/make_figures.py
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from style import C, LABEL, W2, W1, panel, save, ROOT

DAY = 24.0
R = np.load(os.path.join(ROOT, "results", "p2_dynamics.npz"), allow_pickle=True)
S = json.load(open(os.path.join(ROOT, "results", "p2_dynamics.json"), encoding="utf-8"))


def fig2():
    u1, u1c = R["A_u1"], S["u1c"]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.35), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    a = ax[0]
    a.plot(u1, R["A_piS"], color=C["R"], lw=1.1, label=r"$\pi_S$")
    a.plot(u1, R["A_R0"], color=C["P"], lw=1.8, label=r"$\mathcal{R}_0=2\pi_S\pi_M\pi_{\rm ret}$")
    a.plot(u1, R["A_piM"], color=C["G"], lw=1.25, ls="--", zorder=3,
           label=rf"$\pi_M={float(R['A_piM'][0]):.2f}$")
    a.plot(u1, R["A_piret"], color=C["Q"], lw=1.25, ls=":", zorder=4,
           label=rf"$\pi_{{\rm ret}}={float(R['A_piret'][0]):.2f}$")
    a.axhline(1, color=C["grey"], lw=0.7, ls="--", zorder=0)
    a.axvline(u1c, color=C["grey"], lw=0.7, ls=":")
    a.text(0.02, 1.03, r"$\mathcal{R}_0=1$", fontsize=6.8, color="0.4", va="bottom")
    a.set_xlabel(r"restriction-point blockade $u_1$")
    a.set_ylabel("value")
    a.set_ylim(0, 2.0)
    a.legend(loc="lower left", fontsize=6.8)
    panel(a, "a")
    a = ax[1]
    P0 = R["A_Pstar"][0]
    m = u1 < u1c
    a.plot(u1[m], R["A_Pstar"][m] / P0, color=C["P"], lw=1.8, label=r"$E^\ast$ (stable)")
    a.plot(u1[m], 0 * u1[m], color=C["P"], lw=1.1, ls=(0, (3, 2)), label=r"$E_0$ (unstable)")
    a.plot(u1[~m], 0 * u1[~m], color=C["P"], lw=1.8)
    a.plot([u1c], [0], "o", ms=4, mfc="white", mec=C["P"], mew=1)
    a.annotate("transcritical\n" + rf"$u_{{1,c}}={u1c:.3f}$", (u1c, 0), (0.22, 0.22), fontsize=7,
               arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
    a.set_xlabel(r"restriction-point blockade $u_1$")
    a.set_ylabel(r"$P^\ast(u_1)/P^\ast(0)$")
    a.set_ylim(-0.05, 1.05)
    a.legend(loc="upper right", fontsize=6.8)
    panel(a, "b")
    a = ax[2]
    g1, g2, L = R["A_g1"], R["A_g2"], R["A_Lmap"]
    vmax = np.abs(L).max()
    im = a.pcolormesh(g1, g2, L, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), shading="auto",
                      rasterized=True)
    ff, fn = R["A_front_formula"], R["A_front_numeric"]
    okf = np.isfinite(ff) & (ff >= 0)
    okn = np.isfinite(fn) & (fn > 0)
    a.plot(ff[okf], g2[okf], color="k", lw=1.6, label="closed-form boundary")
    a.plot(fn[okn][::6], g2[okn][::6], "o", ms=2.8, mfc="white", mec="k", mew=0.4, label=r"zero of $\Lambda$")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.text(0.05, 0.12, r"persistence" + "\n" + r"($\mathcal{R}_0>1$)", fontsize=7, color="k")
    a.text(0.35, 0.62, r"extinction ($\mathcal{R}_0<1$)", fontsize=7, color="k")
    cb = fig.colorbar(im, ax=a, pad=0.02, fraction=0.06)
    cb.set_label(r"$\Lambda$ (h$^{-1}$)")
    cb.ax.tick_params(labelsize=6.5)
    a.set_xlabel(r"restriction-point blockade $u_1$")
    a.set_ylabel(r"S/G2/M loss $u_2$")
    a.legend(loc="upper right", fontsize=6.3, frameon=True, framealpha=0.85, edgecolor="none")
    panel(a, "c")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig2_threshold")


def fig3():
    t, Y = R["B_t"] / DAY, R["B_Y"]
    Pst = S["Pstar"]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
    a = ax[0]
    a.plot(t, Y[3], color=C["N"])
    a.axhline(S["Nstar"], color=C["grey"], lw=0.7, ls="--")
    a.text(78, S["Nstar"] + 0.03, rf"$N^\ast={S['Nstar']:.2f}$", ha="right", fontsize=7, color="0.3")
    a.set_ylabel(r"oxygen $N$")
    a.set_ylim(0, 1.05)
    panel(a, "a")
    a = ax[1]
    for k, i in (("R", 0), ("G", 1), ("Q", 2)):
        a.plot(t, Y[i], color=C[k], label=LABEL[k])
    a.plot(t, Y[:3].sum(0), color=C["P"], lw=1.6, label=LABEL["P"])
    a.set_ylabel("living-cell density (arb. units)")
    a.set_ylim(0, 42)
    a.legend(loc="upper center", ncol=2, fontsize=6.0, columnspacing=0.8, handlelength=1.4)
    panel(a, "b")
    a = ax[2]
    for tr in R["B_multi"]:
        a.plot(R["B_tm"] / DAY, tr, color=C["blue"], lw=0.7, alpha=0.45)
    a.axhline(1, color=C["P"], lw=1.2, ls="--")
    a.set_ylabel(r"$P(t)/P^\ast$")
    a.set_yscale("log")
    a.set_yticks([0.1, 0.3, 1, 3], ["0.1", "0.3", "1", "3"])
    a.minorticks_off()
    panel(a, "c")
    for a in ax:
        a.set_xlabel("time (days)")
        a.set_xlim(0, 80)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig3_reservoir")


def fig4():
    u1c = S["u1c"]
    pro = S["protocol"]
    fig, ax = plt.subplots(2, 2, figsize=(W2, 4.9))
    a = ax[0, 0]
    U, T, RES = R["C_U"], R["C_T"] / DAY, R["C_RES"]
    Lg = np.log10(np.clip(RES, 1e-3, 1))
    cf = a.contourf(U, T, Lg, levels=np.arange(-3, 0.01, 0.5), cmap="Blues_r")
    a.contour(U, T, Lg, levels=[-2], colors="k", linewidths=1.1, linestyles="solid")
    a.contour(U, T, Lg, levels=[-1], colors="k", linewidths=0.8, linestyles="dashed")
    a.axvspan(U[0], u1c, color="0.88", lw=0, zorder=2)
    a.axvline(u1c, color=C["R"], lw=1.0, zorder=3)
    a.text(u1c - 0.003, 116, r"$u_{1,c}$", color=C["R"], ha="right", va="top", fontsize=7.5, zorder=4)
    a.text(0.838, 60, "no duration\nsuffices\n" + r"($\mathcal{R}_0>1$)", ha="center", fontsize=7, color="0.3",
           zorder=4)
    a.plot([pro["u1"]], [pro["days_on"]], "o", color=C["orange"], ms=4.5, mec="white", zorder=5)
    a.text(pro["u1"] - 0.004, pro["days_on"] - 2, "panel (c)", fontsize=6.8, color=C["orange"], zorder=5,
           ha="right", va="top")
    a.text(0.99, 92, "1%", fontsize=7, ha="right")
    a.text(0.99, 34, "10%", fontsize=7, ha="right", va="bottom")
    cb = fig.colorbar(cf, ax=a, pad=0.02)
    cb.set_label(r"$\log_{10}P(T)/P^\ast$")
    a.set_xlabel(r"blockade $u_1$ maintained on $[0,T]$")
    a.set_ylabel(r"exposure $T$ (days)")
    panel(a, "a")
    a = ax[0, 1]
    d, lam = R["C_d"], R["C_lam"]
    for key, ls, lab in (("C_T2", "-", r"$\varepsilon=10^{-2}$"), ("C_T4", "--", r"$\varepsilon=10^{-4}$"),
                         ("C_T6", ":", r"$\varepsilon=10^{-6}$")):
        a.loglog(d, R[key] / DAY, ls=ls, color=C["blue"], lw=1.3, label=lab)
    a.loglog(d, np.log(1e6) / np.abs(lam) / DAY, color=C["orange"], lw=1.0, label=r"$\log(10^6)/|\Lambda|$")
    a.set_xlabel(r"distance to threshold $u_1-u_{1,c}$")
    a.set_ylabel(r"exposure needed $T_\varepsilon$ (days)")
    a.legend(loc="upper right", fontsize=6.8)
    panel(a, "b")
    a = ax[1, 0]
    t, Y = R["C_proto_t"] / DAY, R["C_proto_Y"]
    a.axvspan(0, pro["days_on"], color=C["light"], lw=0)
    a.text(pro["days_on"] / 2, 25, rf"$u_1={pro['u1']}$", ha="center", fontsize=7, color="0.3")
    for k, i in (("R", 0), ("G", 1), ("Q", 2)):
        a.plot(t, Y[i], color=C[k], label=LABEL[k])
    a.plot(t, Y[:3].sum(0), color=C["P"], lw=1.6, label=LABEL["P"])
    a.set_xlabel("time (days)")
    a.set_ylabel("living-cell density (arb. units)")
    a.set_xlim(-10, 100)
    a.set_ylim(0, 42)
    a.legend(loc="upper center", ncol=2, fontsize=6.3, columnspacing=1.0)
    panel(a, "c")
    a = ax[1, 1]
    qs = R["C_qs"]
    a.plot(qs, R["C_T50"], color=C["P"], label=r"50% of $P^\ast$")
    a.plot(qs, R["C_T90"], color=C["P"], ls="--", label=r"90% of $P^\ast$")
    q0 = S["protocol"]["Q_share_RQ"]
    for key in ("C_T50", "C_T90"):
        a.plot([q0], [np.interp(q0, qs, R[key])], "o", color=C["orange"], ms=4, mec="white")
    a.set_xlabel(r"quiescent share $Q/(R+Q)$ at withdrawal")
    a.set_ylabel("regrowth time (days)")
    a.legend(loc="upper left", fontsize=6.8)
    panel(a, "d")
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    save(fig, "fig4_control")


def fig5():
    E, names = R["D_E"], [str(n) for n in R["D_names"]]
    lab = {"k1bar": r"$\overline{k}_1$", "KN": r"$K_N$", "k2": r"$k_2$", "p": r"$p$", "albar": r"$\overline{\alpha}$",
           "Kal": r"$K_\alpha$", "nal": r"$n_\alpha$", "bebar": r"$\overline{\beta}$", "Kbe": r"$K_\beta$",
           "mu0": r"$\mu^{0}$", "mumax": r"$\mu^{\max}$", "muQmax": r"$\mu_Q^{\max}$", "Kmu": r"$K_\mu$",
           "lamR": r"$\ell_R$", "lamG": r"$\ell_G$", "lamQ": r"$\ell_Q$", "delta": r"$\delta$"}
    groups = ["cycle"] * 4 + ["quiescence"] * 5 + ["loss"] * 4 + ["oxygen"] * 4
    zero = {"lamR", "lamG", "lamQ", "delta"}
    fig, a = plt.subplots(figsize=(W1 * 1.3, 4.6))
    vmax = np.abs(E).max()
    im = a.imshow(E, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for i, n in enumerate(names):
        for j in range(4):
            exact = n in zero and j != 1
            a.text(j, i, r"$0^\ast$" if exact else f"{E[i, j]:+.2f}", ha="center", va="center", fontsize=6.6,
                   color="0.45" if exact else ("white" if abs(E[i, j]) > 0.55 * vmax else "black"))
    a.set_xticks(range(4), [r"$\Lambda$", r"$P^\ast$", r"$f_{\rm red}^\ast$", r"$u_1^\ast$"])
    a.xaxis.tick_top()
    a.set_yticks(range(len(names)), [lab[n] for n in names])
    for s in a.spines.values():
        s.set_visible(False)
    a.tick_params(length=0)
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            if i < len(groups):
                a.axhline(i - 0.5, color="white", lw=2.5)
            a.text(3.72, (start + i - 1) / 2, groups[start], rotation=90, va="center", fontsize=7, color="0.35")
            start = i
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.13)
    cb.set_label("local elasticity")
    cb.ax.tick_params(labelsize=6.5)
    save(fig, "fig5_sensitivity")


def figS_hopf():
    H = np.load(os.path.join(ROOT, "results", "p1_hopf_example.npz"))
    J = json.load(open(os.path.join(ROOT, "results", "p1_hopf_example.json"), encoding="utf-8"))
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    a = ax[0]
    a.plot(H["grid"], 24 * H["maxre"], color=C["P"])
    a.axhline(0, color=C["grey"], lw=0.7, ls="--")
    a.axvline(float(H["mH"]), color=C["R"], lw=0.8, ls=":")
    a.text(float(H["mH"]) + 0.05, a.get_ylim()[1] * 0.8, rf"$n_{{\alpha,H}}={float(H['mH']):.3f}$", fontsize=7,
           color=C["R"])
    a.set_xlabel(r"entry Hill exponent $n_\alpha$")
    a.set_ylabel(r"leading Re$\,z$ at $E^\ast$ (day$^{-1}$)")
    panel(a, "a")
    a = ax[1]
    d = np.array([r["dm"] for r in J["above"]])
    amp = np.array([r["amp"] for r in J["above"]])
    a.loglog(d, amp, "o", color=C["blue"], ms=4)
    a.loglog(d, amp[0] * np.sqrt(d / d[0]), color=C["orange"], lw=1, label=r"slope $1/2$")
    a.set_xlabel(r"$n_\alpha-n_{\alpha,H}$")
    a.set_ylabel(r"cycle amplitude of $P/P^\ast$")
    a.legend(loc="upper left", fontsize=7)
    panel(a, "b")
    a = ax[2]
    t = (H["t"] - H["t"][0]) / DAY
    a.plot(t, H["Y"][:3].sum(0) / float(H["Pstar"]), color=C["P"], label=r"$P/P^\ast$")
    a.plot(t, H["fred"], color=C["R"], ls="--", label=r"$f_{\rm red}$")
    a.plot(t, H["Y"][2] / H["Y"][:3].sum(0), color=C["Q"], lw=1.0, label=r"$Q/P$ (hidden)")
    a.set_xlabel("time on the limit cycle (days)")
    a.set_xlim(0, min(150, t[-1]))
    a.legend(loc="upper right", fontsize=6.8, ncol=3)
    a.set_ylim(0, 2.4)
    panel(a, "c")
    fig.tight_layout(w_pad=1.0)
    save(fig, "figS_hopf")


def fig6():
    Ob = np.load(os.path.join(ROOT, "results", "p2_observability.npz"))
    J = json.load(open(os.path.join(ROOT, "results", "p2_observability.json"), encoding="utf-8"))
    mults = [r["mult"] for r in J["ridge"]]
    cols = plt.cm.viridis(np.linspace(0.05, 0.88, len(mults)))
    t = Ob["ridge_t"] / DAY
    fig, ax = plt.subplots(1, 4, figsize=(W2, 2.25), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    for c, col in zip(mults, cols):
        lab = rf"${c:g}\,\beta_r^{{\rm ref}}$"
        ax[0].plot(t, Ob[f"ridge_f_{c}"], color=col, lw=1.2, label=lab)
        ax[1].plot(t, Ob[f"ridge_Q_{c}"], color=col, lw=1.2)
        ax[2].plot(t, Ob[f"ridge_R_{c}"], color=col, lw=1.2)
    ax[0].plot(Ob["obs_t"] / DAY, Ob["obs_f"], "o", ms=2.6, mfc="white", mec="k", mew=0.6, zorder=5)
    ax[0].set_ylabel(r"$f_{\rm green}$ (observed)")
    ax[1].set_ylabel(r"$Q/P$ (hidden)")
    ax[2].set_ylabel(r"$R/P$ (hidden)")
    ridge_handles, ridge_labels = ax[0].get_legend_handles_labels()
    fig.legend(ridge_handles, ridge_labels, loc="upper center", bbox_to_anchor=(0.38, 0.99), ncol=5,
               fontsize=5.8, handlelength=1.2, columnspacing=0.8, frameon=False)
    for a, s in zip(ax[:3], "abc"):
        a.set_xlabel("days after reoxygenation")
        a.set_xlim(0, 21)
        panel(a, s, dx=-0.24)
    ax[0].set_ylim(0, 0.14)
    ax[1].set_ylim(0, 1.02)
    ax[2].set_ylim(0, 1.02)
    a = ax[3]
    g = Ob["prof_grid"]
    sty = {"F2: kappa free, green": (C["grey"], "-", r"$\kappa_r$ free"),
           "F4: kappa, k2, p free, green": (C["P"], "-", r"$\kappa_r,k_2,p$ free"),
           "F4 + total abundance": (C["blue"], "--", r"$\ldots$ + abundance"),
           "F4 + weighted readout": (C["orange"], "-.", r"$\ldots$ + readout $y_w$")}
    for k, (col, ls, lab) in sty.items():
        a.plot(g, Ob["prof_" + k], color=col, ls=ls, lw=1.3, label=lab)
    a.axhline(3.841, color="0.5", lw=0.7, ls=":")
    a.set_xscale("log")
    a.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    a.minorticks_off()
    a.set_ylim(0, 40)
    a.set_xlabel(r"$\beta_r/\beta_r^{\rm ref}$")
    a.set_ylabel(r"$\Delta(-2\log L)$ (expected)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=5.6, handlelength=1.5, frameon=True,
             framealpha=0.95, edgecolor="none", borderpad=0.3)
    panel(a, "d", dx=-0.24)
    fig.tight_layout(rect=[0, 0, 1, 0.88], w_pad=0.6)
    save(fig, "fig6_ridge")


MC_LABEL = {"A_W7": "7 days", "A_W14": "14 days", "A_W21": "21 days", "A_E21": "21 points, early-dense",
            "B_f": r"$f_{\rm green}$", "B_fP": r"$f_{\rm green}$ + abundance", "B_fy": r"$f_{\rm green}$ + readout",
            "B_fPy": r"$f_{\rm green}$ + abundance + readout",
            "D_BB3": r"overdispersion $\times3$ (unmodelled)", "D_CE3": "3% misclassification (unmodelled)",
            "D_SC10": "10% cycling cells in seed (unmodelled)", "D_fy": r"readout, $w_Q$ correct",
            "D_fy_w020": r"readout, $w_Q$ fitted 0.20 (true 0.35)", "D_fy_w050": r"readout, $w_Q$ fitted 0.50 (true 0.35)"}


def fig7():
    mc = json.load(open(os.path.join(ROOT, "results", "p2_mc_reference.json"), encoding="utf-8"))
    S = mc["summary"]
    ens_f = os.path.join(ROOT, "results", "p2_mc_ensemble.json")
    has_ens = os.path.exists(ens_f)
    groups = [("Observation window\n" + r"($\kappa_r,\beta_r$ unknown; $f_{\rm green}$)", ["A_W7", "A_W14", "A_W21", "A_E21"]),
              ("Added measurements\n" + r"($\kappa_r,\beta_r,k_2,p$ unknown; 21 days)", ["B_f", "B_fP", "B_fy", "B_fPy"]),
              ("Systematic errors\n" + r"($\kappa_r,\beta_r$ unknown; 21 days)",
               ["D_BB3", "D_CE3", "D_SC10", "D_fy", "D_fy_w020", "D_fy_w050"])]
    short = {"A_W7": "7 d", "A_W14": "14 d", "A_W21": "21 d", "A_E21": "21 d, early-dense",
             "B_f": "green only", "B_fP": "+ abundance", "B_fy": "+ readout", "B_fPy": "+ both",
             "D_BB3": r"overdispersion $\times3$", "D_CE3": "3% misclassified", "D_SC10": "10% cycling seed",
             "D_fy": r"readout, $w_Q$ exact", "D_fy_w020": r"$w_Q$ fitted 0.20", "D_fy_w050": r"$w_Q$ fitted 0.50"}
    heads = ["window ($\\kappa_r,\\beta_r$ unknown)", "measurements ($\\kappa_r,\\beta_r,k_2,p$ unknown)",
             "systematic errors ($\\kappa_r,\\beta_r$ unknown)"]
    fig = plt.figure(figsize=(W2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], width_ratios=[1.15, 1], hspace=0.55, wspace=0.55)
    a = fig.add_subplot(gs[0, 0])
    y, ticks, labs = 0.0, [], []
    gcol = [C["blue"], C["P"], C["orange"]]
    for gi, (_, names) in enumerate(groups):
        a.text(1.5, y + 0.35, heads[gi], fontsize=6.3, color="0.35", va="bottom", ha="left")
        for n in names:
            s = S[n]
            a.errorbar(100 * s["rmse_beta"], y, xerr=[[100 * (s["rmse_beta"] - s["ci_beta"][0])],
                                                     [100 * (s["ci_beta"][1] - s["rmse_beta"])]],
                       fmt="o", color=gcol[gi], ms=3.5, capsize=2, lw=1)
            ticks.append(y)
            labs.append(short[n])
            y -= 1
        y -= 0.9
    a.set_yticks(ticks, labs, fontsize=6.8)
    a.set_xscale("log")
    a.set_xlim(1.5, 150)
    a.set_xticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
    a.minorticks_off()
    a.set_ylim(y + 0.5, 1.2)
    a.set_xlabel(r"relative RMSE of $\hat\beta_r$ (%), 95% CI")
    a.spines["left"].set_visible(False)
    a.tick_params(axis="y", length=0)
    a.grid(axis="x", color="0.92", lw=0.6)
    a.set_axisbelow(True)
    panel(a, "a", dx=-0.42, dy=1.0)
    a = fig.add_subplot(gs[0, 1])
    names = ["A_W21", "D_BB3", "D_CE3", "D_SC10", "D_fy", "D_fy_w020", "D_fy_w050"]
    for i, n in enumerate(names):
        s = S[n]
        m = 100 * s["median_bias_beta"]
        lo, hi = 100 * np.array(s["ci_median_bias"])
        ok = n in ("A_W21", "D_fy")
        a.barh(-i, m, color=C["grey"] if ok else C["orange"], height=0.6)
        a.errorbar(m, -i, xerr=[[m - lo], [hi - m]], fmt="none", color="k", lw=0.8, capsize=2)
    a.axvline(0, color="k", lw=0.7)
    a.set_yticks(-np.arange(len(names)), ["counting noise only" if n == "A_W21" else short[n] for n in names],
                 fontsize=6.8)
    a.set_xlim(-50, 80)
    a.set_xlabel(r"median relative bias of $\hat\beta_r$ (%)")
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    panel(a, "b", dx=-0.62, dy=1.0)
    if has_ens:
        en = json.load(open(ens_f, encoding="utf-8"))
        a = fig.add_subplot(gs[1, :])
        cats = [("A_W7", "7 d"), ("A_W21", "21 d"), ("B_f", "green only"), ("B_fP", "+ abundance"),
                ("B_fy", "+ readout")]
        R = np.array([[100 * r["rmse"][k] for k, _ in cats] for r in en["rows"]])
        xs = np.arange(len(cats))
        for row in R:
            a.plot(xs[:2], row[:2], color="0.75", lw=0.5, zorder=1)
            a.plot(xs[2:], row[2:], color="0.75", lw=0.5, zorder=1)
        rng = np.random.default_rng(0)
        for j in range(len(cats)):
            a.scatter(xs[j] + rng.uniform(-0.08, 0.08, len(R)), R[:, j], s=9,
                      color=C["blue"] if j < 2 else C["P"], zorder=2)
        ref = [100 * S[k]["rmse_beta"] for k, _ in cats]
        a.scatter(xs, ref, marker="*", s=70, color=C["orange"], zorder=3, label="reference vector")
        a.axvline(1.5, color="0.85", lw=0.8)
        a.set_xticks(xs, [l for _, l in cats], fontsize=7)
        a.set_ylabel(r"RMSE of $\hat\beta_r$ (%)")
        a.set_ylim(0, 100)
        a.text(0.5, 96, r"$\kappa_r,\beta_r$ unknown", ha="center", va="top", fontsize=6.6, color="0.35")
        a.text(3.0, 96, r"$\kappa_r,\beta_r,k_2,p$ unknown (21 d)", ha="center", va="top", fontsize=6.6, color="0.35")
        a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=6.6)
        a.set_title(f"{en['n_sets']} parameter vectors from the companion time-scale ranges", fontsize=7.5,
                    color="0.25")
        panel(a, "c", dx=-0.12, dy=1.06)
    save(fig, "fig7_design")


def figS_morris():
    J = json.load(open(os.path.join(ROOT, "results", "p2_morris.json"), encoding="utf-8"))
    mu, sg, names = np.array(J["mu_star"]), np.array(J["sigma"]), J["names"]
    lab = {"k1bar": r"$\overline{k}_1$", "KN": r"$K_N$", "k2": r"$k_2$", "p": r"$p$", "albar": r"$\overline{\alpha}$",
           "Kal": r"$K_\alpha$", "nal": r"$n_\alpha$", "bebar": r"$\overline{\beta}$", "Kbe": r"$K_\beta$",
           "mu0": r"$\mu^{0}$", "mumax": r"$\mu^{\max}$", "muQmax": r"$\mu_Q^{\max}$", "Kmu": r"$K_\mu$",
           "lamR": r"$\ell_R$", "lamG": r"$\ell_G$", "lamQ": r"$\ell_Q$", "delta": r"$\delta$"}
    oxy = {"lamR", "lamG", "lamQ", "delta"}
    titles = [r"$\Lambda$", r"$P^\ast$", r"$f_{\rm red}^\ast$", r"$u_1^\ast$"]
    fig, ax = plt.subplots(1, 4, figsize=(W2, 2.2))
    for j, a in enumerate(ax):
        col = [C["N"] if n in oxy else C["P"] for n in names]
        a.scatter(mu[:, j], sg[:, j], s=12, c=col, zorder=3)
        for i in np.argsort(-mu[:, j])[:4]:
            offset = (3, -7) if j == 3 and names[i] == "Kmu" else (3, 2)
            valign = "top" if offset[1] < 0 else "baseline"
            a.annotate(lab[names[i]], (mu[i, j], sg[i, j]), xytext=offset, textcoords="offset points",
                       fontsize=7, va=valign)
        a.set_title(titles[j], fontsize=8.5)
        a.set_xlabel(r"$\mu^\ast$")
        if j == 0:
            a.set_ylabel(r"$\sigma$")
        a.set_xlim(left=0)
        a.set_ylim(bottom=0)
        panel(a, "abcd"[j], dx=-0.25)
    fig.tight_layout(w_pad=0.6)
    save(fig, "figS_morris")


if __name__ == "__main__":
    which = sys.argv[1:] or ["2", "3", "4", "5", "S", "6", "7", "M"]
    table = {"2": fig2, "3": fig3, "4": fig4, "5": fig5, "S": figS_hopf, "6": fig6, "7": fig7, "M": figS_morris}
    for k in which:
        table[k]()
