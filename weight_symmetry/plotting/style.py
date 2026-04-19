"""
Shared matplotlib style for ICML Workshop 2026 figures.
Import apply_style() before any plt calls; use METHOD_STYLE for per-method kwargs.
"""
import os
import time
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Global rcParams
# ---------------------------------------------------------------------------
def apply_style():
    mpl.rcParams.update({
        "font.family":      "serif",
        "font.size":        9,
        "axes.titlesize":   9,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  8,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "pdf.fonttype":     42,   # embed fonts for camera-ready
        "ps.fonttype":      42,
    })


# ---------------------------------------------------------------------------
# Figure sizes (inches)
# ---------------------------------------------------------------------------
SINGLE   = (3.5, 2.8)
TWO_PANEL = (7.0, 2.8)
THREE_PANEL = (7.0, 2.5)
BAR      = (3.5, 2.5)


# ---------------------------------------------------------------------------
# Per-method style (Wong 2011 colorblind-safe palette)
# ---------------------------------------------------------------------------
METHOD_STYLE = {
    "MSE LAE":                      dict(color="#888888", ls="--",  lw=1.0),
    "MSE LAE + Ortho":              dict(color="#AAAAAA", ls="-.",  lw=1.0),
    "Oftadeh":                      dict(color="#E69F00", ls="-",   lw=1.5),
    "Standard MRL (MSE)":           dict(color="#56B4E9", ls="-",   lw=1.5, marker="o", ms=4, markevery=4),
    "Full-prefix MRL":              dict(color="#009E73", ls="-",   lw=1.5),
    "Full-prefix MRL + Ortho":      dict(color="#009E73", ls="-",   lw=1.5, marker="s", ms=4, markevery=4),
    "Full-prefix MRL + Both Ortho": dict(color="#005C44", ls="-",   lw=1.5, marker="^", ms=4, markevery=4),
    "Full-prefix MRL (CE)":         dict(color="#D55E00", ls="-",   lw=1.5),
    "Standard MRL (CE)":            dict(color="#CC79A7", ls="--",  lw=1.5),
    "PCA + linear probe":           dict(color="#0072B2", ls=":",   lw=1.0),
}


# ---------------------------------------------------------------------------
# Axes helper
# ---------------------------------------------------------------------------
def style_ax(ax):
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="gray")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
FIGURES_DIR = (
    "/home/argha/Mat_embedding_hyperbole/files/results/"
    "ICMLWorkshop_weightSymmetry2026/figures"
)


def save_fig(fig, base_name, fig_stamp, out_dir=None):
    """Save figure as pdf, png, svg with timestamp suffix."""
    out_dir = out_dir or FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{base_name}_{fig_stamp}"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight")
    print(f"Saved: {stem}.{{pdf,png,svg}}  →  {out_dir}")
    return stem
