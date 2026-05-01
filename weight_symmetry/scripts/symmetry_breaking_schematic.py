"""
Schematic teaching figures for the privileged-basis / symmetry-breaking story.

Four self-contained panels (no training, no data):
    1. Loss-landscape contours        (2D slice cartoon)
    2. Gradient field on the sphere   (attractors)
    3. Symmetry-group icons           (disk / pie / points)
    4. Alignment heatmaps |U^T A|     (block structure)

Usage:
    python weight_symmetry/scripts/symmetry_breaking_schematic.py
    python weight_symmetry/scripts/symmetry_breaking_schematic.py --only heatmap
    python weight_symmetry/scripts/symmetry_breaking_schematic.py --blocks 2,2,4

Conda environment: mrl_env
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Wedge, Circle, Rectangle

# ---------- project style (shared with all WS figures) ----------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)
from weight_symmetry.plotting.style import (  # noqa: E402
    FIGURES_DIR, apply_style, THREE_PANEL,
)

apply_style()

OUT_DIR = FIGURES_DIR
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name, stamp):
    """Save as pdf / png / svg matching the WS-paper save convention."""
    for ext in ("pdf", "png", "svg"):
        path = os.path.join(OUT_DIR, f"{name}{stamp}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  -> {path}")


# =====================================================================
# 1. Loss landscape (2D slice cartoon)
# =====================================================================
def make_landscape(stamp):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2

    L_unordered = (R2 - 1)**2
    L_smrl      = (R2 - 1)**2 + 0.15 * Y**2
    L_fpmrl     = (R2 - 1)**2 + 0.40 * Y**2

    titles = ["Unordered LAE", "Sparse MRL", r"FP-MRL / NU-$\ell_2$"]
    losses = [L_unordered, L_smrl, L_fpmrl]
    theta  = np.linspace(0, 2 * np.pi, 200)

    for ax, L, title in zip(axes, losses, titles):
        ax.contourf(X, Y, L, levels=20, cmap="Blues")
        ax.contour(X, Y, L, levels=10, colors="k", linewidths=0.4, alpha=0.5)

        if title == "Unordered LAE":
            ax.plot(np.cos(theta), np.sin(theta), "r--", lw=1.4)
        elif "FP-MRL" in title:
            for (mx, my) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ax.plot(mx, my, "r*", ms=12)
        else:
            ax.plot(np.cos(theta), np.sin(theta), "r--", lw=0.7, alpha=0.4)
            for (mx, my) in [(1, 0), (-1, 0)]:
                ax.plot(mx, my, "r*", ms=12)

        ax.set_title(title, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlabel(r"$a_1[0]$", fontsize=9)
        ax.set_ylabel(r"$a_1[1]$", fontsize=9)

    fig.text(0.5, 0.0, r"schematic 2D slice of $\mathcal{L}$",
             ha="center", style="italic", fontsize=9)
    plt.tight_layout()
    _save(fig, "schem_loss_landscape", stamp)
    plt.close(fig)


# =====================================================================
# 2. Gradient field on the sphere
# =====================================================================
def make_gradient_field(stamp):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    n = 16
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    px, py = np.cos(theta), np.sin(theta)

    titles = ["Unordered LAE", "Sparse MRL", r"FP-MRL / NU-$\ell_2$"]

    for ax, title in zip(axes, titles):
        ax.add_patch(Circle((0, 0), 1, fill=False, ls="--", color="gray", lw=0.7))

        if title == "Unordered LAE":
            ax.scatter(px, py, c="red", s=22, zorder=3)
        elif "FP-MRL" in title:
            attractors = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
            for ax_x, ax_y in attractors:
                ax.plot(ax_x, ax_y, "r*", ms=14, zorder=3)
            for x0, y0 in zip(px, py):
                d2 = np.sum((attractors - np.array([x0, y0]))**2, axis=1)
                tgt = attractors[np.argmin(d2)]
                dx, dy = (tgt - np.array([x0, y0])) * 0.18
                ax.arrow(x0, y0, dx, dy, head_width=0.04, head_length=0.05,
                         fc="black", ec="black", lw=0.5)
        else:  # Sparse MRL
            for (ax_x, ax_y) in [(1, 0), (-1, 0)]:
                ax.plot(ax_x, ax_y, "r*", ms=14, zorder=3)
            attractors = np.array([(1, 0), (-1, 0)])
            for x0, y0 in zip(px, py):
                d2 = np.sum((attractors - np.array([x0, y0]))**2, axis=1)
                tgt = attractors[np.argmin(d2)]
                dx, dy = (tgt - np.array([x0, y0])) * 0.10
                ax.arrow(x0, y0, dx, dy, head_width=0.03, head_length=0.04,
                         fc="black", ec="black", lw=0.4, alpha=0.7)

        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, pad=4)

    plt.tight_layout()
    _save(fig, "schem_gradient_field", stamp)
    plt.close(fig)


# =====================================================================
# 3. Symmetry-group icons
# =====================================================================
def make_symmetry_diagrams(stamp):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    titles = ["Unordered LAE", "Sparse MRL", "Full-prefix MRL"]
    subs = [
        r"$\mathcal{G} = SO(d)$",
        r"$\mathcal{G} = SO(m_1)\times SO(m_2{-}m_1)\times\cdots$",
        r"$\mathcal{G} = \{\pm 1\}^d$",
    ]

    # 1. continuous disk
    ax = axes[0]
    ax.add_patch(Circle((0, 0), 1, fill=True, color="lightblue",
                        ec="black", lw=1))
    ax.text(0, 0, r"$SO(d)$", ha="center", va="center", fontsize=14)

    # 2. pie of prefix blocks
    ax = axes[1]
    sizes = [2, 2, 4]
    colors = ["#e8b8d8", "#a8d8e8", "#d8e8a8"]
    start = 0
    for s, c in zip(sizes, colors):
        end = start + s / sum(sizes) * 360
        ax.add_patch(Wedge((0, 0), 1, start, end, facecolor=c,
                           edgecolor="black", lw=1))
        start = end

    # 3. finite set of points
    ax = axes[2]
    ax.add_patch(Circle((0, 0), 1, fill=False, ls="--", color="gray", lw=0.5))
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for a in angles:
        ax.plot(np.cos(a), np.sin(a), "ko", ms=10)

    for ax, title, sub in zip(axes, titles, subs):
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, pad=4)
        ax.set_xlabel(sub, fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    _save(fig, "schem_symmetry_groups", stamp)
    plt.close(fig)


# =====================================================================
# 4. Alignment heatmaps |U^T A|
# =====================================================================
def make_heatmaps(stamp, prefixes=(2, 4, 8), seed=0, cmap="Blues"):
    Ms = sorted(prefixes)              # disjoint S-MRL block sizes
    d_main = max(Ms)                   # 8 — Unordered & FP-MRL
    d_smrl = sum(Ms)                   # 14 — S-MRL with disjoint blocks
    rng = np.random.default_rng(seed)

    # Bumped from apply_style defaults (read too small in this full-row figure).
    LBL_SZ = 14    # titles, axis labels, italic caption
    M_SZ   = 14    # m=k annotations on each S-MRL block

    Q = np.linalg.qr(rng.standard_normal((d_main, d_main)))[0]
    M_unordered = np.abs(Q)

    # Build S-MRL blocks smallest first (low row indices) so that, with the
    # math-style y-axis (origin='lower'), m=2 sits at the bottom-left corner
    # at (0,0), m=4 in the middle, and m=8 at the top-right.
    #
    # Within each block of size m, intensity tiers by local rank r = max(i,j):
    #   r in [0,1]  → inner 2×2  → appears in m=2, m=4, m=8  (3 prefixes) DARKEST
    #   r in [2,3]  → ring up to 4 → appears in m=4, m=8     (2 prefixes) MEDIUM
    #   r in [4,7]  → ring up to 8 → appears in m=8          (1 prefix)   LIGHTEST
    def _smrl_block(m):
        # Block-specific intensities so the same ring index reads differently
        # in m=4 vs m=8. In m=4 the outer 2×2 ring is a touch lighter than
        # before; in m=8 the inner 4×4 reads darker so it pops against the
        # noisier outer 4–7 ring.
        block = np.empty((m, m))
        for i in range(m):
            for j in range(m):
                r = max(i, j)
                # Default noise range; outer ring of m=8 gets more spread.
                diag_n, off_n = 0.05, 0.08
                if m == 2:
                    diag_b, off_b = 0.90, 0.70
                elif m == 4:
                    if r < 2:
                        diag_b, off_b = 0.90, 0.70
                    else:                         # outer 2×2 ring — lighter
                        diag_b, off_b = 0.45, 0.28
                else:                              # m == 8
                    if r < 2:
                        diag_b, off_b = 0.90, 0.70
                    elif r < 4:                   # inner 4×4 ring
                        diag_b, off_b = 0.60, 0.40
                    else:                         # outer 4–7 ring — random
                        diag_b, off_b = 0.30, 0.18
                        diag_n, off_n = 0.25, 0.25
                if i == j:
                    v = diag_b + rng.uniform(-diag_n, diag_n)
                else:
                    v = off_b + rng.uniform(-off_n, off_n)
                block[i, j] = np.clip(v, 0.03, 1.0)
        return block

    M_smrl = np.zeros((d_smrl, d_smrl))
    off = 0
    block_centers = []                  # (i0, i1) row-range per block
    for m in Ms:                         # natural order [2, 4, 8]
        M_smrl[off:off + m, off:off + m] = _smrl_block(m)
        block_centers.append((off, off + m))
        off += m

    # Manual touch-ups in the m=8 block (local col 0):
    #   (x=1, y=4) — last row of inner 4×4 ring → make a bit DARKER
    #   (x=1, y=5) — first row of outer 4–7 ring → make a bit LIGHTER
    # so the tier boundary reads cleanly at this column.
    m8_off = sum(Ms[:-1])
    M_smrl[m8_off + 3, m8_off + 0] = 0.55   # local (row 3, col 0) — darker
    M_smrl[m8_off + 4, m8_off + 0] = 0.10   # local (row 4, col 0) — lighter

    # FP-MRL: diagonal recovery (1.0 → 0.7 fade), with faint random off-
    # diagonal noise in [0, 0.2] so the off-axis cells aren't pure white.
    fp_diag = np.linspace(1.0, 0.7, d_main)
    M_fpmrl = rng.uniform(0.02, 0.18, size=(d_main, d_main))
    np.fill_diagonal(M_fpmrl, fp_diag)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")
    M_smrl_disp  = np.ma.masked_where(M_smrl  == 0, M_smrl)
    M_fpmrl_disp = M_fpmrl   # off-diagonal already populated with faint noise

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0))
    fig.suptitle("Visualization of privileged-basis alignment across loss families",
                 fontsize=LBL_SZ + 2, y=0.98)
    titles = ["Unordered", "S-MRL", "FP-MRL"]
    # Consistent natural-language descriptions of the residual rotation freedom.
    caps = ["any rotation allowed",
            "block rotations only",
            "sign flips only"]
    matrices = [M_unordered, M_smrl_disp, M_fpmrl_disp]

    last_im = None
    for i, (ax, M, title, cap) in enumerate(
            zip(axes, matrices, titles, caps)):
        last_im = ax.imshow(M, cmap=cmap_obj, vmin=0, vmax=1, origin="lower")
        ax.set_title(f"{title}\n({cap})", pad=4, fontsize=LBL_SZ)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Learned embedding dimension", fontsize=LBL_SZ)
        ax.set_ylabel("Privileged basis direction", fontsize=LBL_SZ)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        if i == 1:
            # m=k annotations on each S-MRL diagonal block.
            # block_centers iterates BOTTOM→TOP under origin='lower':
            # k=0 is bottom-left (m=2), last is top-right (m=8).
            # m=2, m=4 → right label; m=8 (top) → left label.
            label_gap = 0.5
            for k, ((i0, i1), m) in enumerate(zip(block_centers, Ms)):
                row_c = (i0 + i1 - 1) / 2
                if k == len(Ms) - 1:
                    ax.text(i0 - 0.5 - label_gap, row_c,
                            f"$m\\!=\\!{m}$",
                            color="black", fontsize=M_SZ,
                            ha="right", va="center")
                else:
                    ax.text(i1 - 0.5 + label_gap, row_c,
                            f"$m\\!=\\!{m}$",
                            color="black", fontsize=M_SZ,
                            ha="left", va="center")

    plt.tight_layout(rect=[0, 0, 0.94, 0.94])
    cbar_ax = fig.add_axes([0.95, 0.28, 0.008, 0.50])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)
    _save(fig, "schem_alignment_heatmaps", stamp)
    plt.close(fig)


# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["landscape", "gradient", "groups",
                                       "heatmap", "all"], default="all")
    p.add_argument("--prefixes", default="2,4,8",
                   help="comma-separated nested prefix sizes m_1<...<m_K. "
                        "Ambient dim d = max(prefixes). "
                        "default '2,4,8' → d=8, M={2,4,8}")
    p.add_argument("--cmap", default="Blues",
                   help="matplotlib colormap for the heatmap panel "
                        "(try: Blues, viridis, magma, cividis, gray_r)")
    args = p.parse_args()

    stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    print(f"Output dir: {OUT_DIR}")
    print(f"Stamp: {stamp}")

    if args.only in ("landscape", "all"):
        print("[1/4] loss landscape")
        make_landscape(stamp)
    if args.only in ("gradient", "all"):
        print("[2/4] gradient field")
        make_gradient_field(stamp)
    if args.only in ("groups", "all"):
        print("[3/4] symmetry groups")
        make_symmetry_diagrams(stamp)
    if args.only in ("heatmap", "all"):
        prefixes = tuple(int(x) for x in args.prefixes.split(","))
        print(f"[4/4] alignment heatmaps  (prefixes={prefixes}, "
              f"d={max(prefixes)})")
        make_heatmaps(stamp, prefixes=prefixes, cmap=args.cmap)


if __name__ == "__main__":
    main()
