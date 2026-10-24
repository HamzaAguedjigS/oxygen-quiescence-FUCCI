"""Shared publication style for all generated figures."""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")

# State colours used consistently across figures
C = {"R": "#C0392B", "G": "#2E8B57", "Q": "#7B5EA7", "N": "#42949E", "P": "#272727",
     "grey": "#9A9A9A", "light": "#E6E6E6", "blue": "#0F4D92", "orange": "#D97A2B"}
LABEL = {"R": r"$R$ (cycling G1)", "G": r"$G$ (S/G2/M)", "Q": r"$Q$ (quiescent)", "N": r"$N$ (oxygen)",
         "P": r"$P$ (all living cells)"}

# Figure widths in inches: 90 mm single column and 190 mm double column
W1, W2 = 3.54, 7.48

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.7, "lines.linewidth": 1.4,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def panel(ax, s, dx=-0.17, dy=1.03):
    ax.text(dx, dy, f"({s})", transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved figures/" + name)
