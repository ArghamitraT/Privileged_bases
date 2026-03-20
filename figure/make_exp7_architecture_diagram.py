"""
Script: figure/make_exp7_architecture_diagram.py
--------------------------------------------------
Generates a figure illustrating the 5 model architectures compared in
Experiment 7 (MRL vs FF vs L1 vs Standard vs PCA).

Each architecture is drawn as a schematic block diagram with labelled boxes
and arrows, so the key differences are immediately visible.

Inputs:
    None (standalone script, no CLI args)

Outputs:
    figure/exp7_architecture_diagram.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Make sure the script works when called from any working directory ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH     = os.path.join(SCRIPT_DIR, "exp7_architecture_diagram.png")
OUT_PATH_SVG = os.path.join(SCRIPT_DIR, "exp7_architecture_diagram.svg")

# ── Colour palette ─────────────────────────────────────────────────────
C_INPUT  = "#E8F4FD"   # light blue  – input node
C_ENC    = "#AED6F1"   # medium blue – encoder block
C_EMB    = "#5DADE2"   # darker blue – embedding vector
C_HEAD   = "#F5CBA7"   # orange      – classifier head
C_LOSS   = "#F9E79F"   # yellow      – loss function
C_EVAL   = "#A9DFBF"   # green       – evaluation / truncation step
C_PCA    = "#D7BDE2"   # purple      – PCA-specific blocks
C_L1     = "#FADBD8"   # pink-red    – L1 penalty annotation
C_BORDER = "#444444"   # dark grey   – box borders

FONT    = "DejaVu Sans"
FS_BIG  = 13   # panel subtitle font
FS_BOX  = 11   # standard box label
FS_SM   = 9.5  # small box label
FS_TINY = 8.5  # very small / dense panels


# ── Low-level drawing helpers ───────────────────────────────────────────

def draw_box(ax, cx, cy, w, h, label, color,
             fs=None, bold=False, alpha=1.0):
    """
    Draw a rounded rectangle centred at (cx, cy) with given label.

    Args:
        ax    : matplotlib Axes
        cx,cy : centre coordinates in axes-fraction space [0,1]
        w,h   : box width and height
        label : text string (may contain newlines)
        color : facecolor
        fs    : font size (defaults to FS_BOX)
        bold  : whether to use bold weight
        alpha : facecolor alpha
    """
    if fs is None:
        fs = FS_BOX
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.025",
        linewidth=1.3,
        edgecolor=C_BORDER,
        facecolor=color,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        cx, cy, label,
        ha="center", va="center",
        fontsize=fs, fontfamily=FONT,
        fontweight="bold" if bold else "normal",
        multialignment="center",
        zorder=4,
    )


def draw_arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.5, zorder=2):
    """
    Draw an arrow from (x1,y1) to (x2,y2) using annotate.

    Args:
        ax          : matplotlib Axes
        x1,y1       : tail coordinates
        x2,y2       : head coordinates
        color       : arrow colour
        lw          : line width
        zorder      : drawing order
    """
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=lw,
        ),
        zorder=zorder,
    )


def setup_panel(ax, title, color="#F8F9FA"):
    """
    Configure axes for a single architecture panel: remove ticks, add title.

    Args:
        ax    : matplotlib Axes
        title : panel title string
        color : background colour for the panel
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(color)
    ax.set_title(title, fontsize=FS_BIG, fontweight="bold",
                 fontfamily=FONT, pad=6)


def column_ys(n, gap, top):
    """
    Return y-coordinates for n boxes stacked downward.

    Args:
        n   (int)   : number of boxes
        gap (float) : vertical gap between consecutive centres
        top (float) : y-coordinate of the first (top) box centre

    Returns:
        list[float]: y positions in descending order
    """
    return [top - i * gap for i in range(n)]


# ══════════════════════════════════════════════════════════════════════
# Figure layout: 2 rows × 3 columns
#   Top row  : Standard | L1 | MRL
#   Bottom row: FF-k (spans 2 cols) | PCA
# ══════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(26, 14))
fig.patch.set_facecolor("white")

gs = fig.add_gridspec(
    2, 3,
    hspace=0.45, wspace=0.28,
    left=0.02, right=0.98,
    top=0.90, bottom=0.04,
)

ax_std = fig.add_subplot(gs[0, 0])
ax_l1  = fig.add_subplot(gs[0, 1])
ax_mrl = fig.add_subplot(gs[0, 2])
ax_ff  = fig.add_subplot(gs[1, 0:2])
ax_pca = fig.add_subplot(gs[1, 2])

fig.suptitle(
    "Experiment 7 — Architecture Comparison\n"
    "Standard  ·  L1 (Sparsity Ablation)  ·  MRL / Matryoshka  ·  FF-k (Dedicated)  ·  PCA",
    fontsize=16, fontweight="bold", fontfamily=FONT, y=0.97,
)

# Shared geometry for single-column panels
CX  = 0.50   # horizontal centre of Standard / L1 / MRL panels
BW  = 0.54   # box width
BH  = 0.09   # box height


# ══════════════════════════════════════════════════════════════════════
# Panel 1 — STANDARD
# ══════════════════════════════════════════════════════════════════════
setup_panel(ax_std, "① Standard")

ys = column_ys(6, gap=0.148, top=0.87)
boxes_std = [
    ("Input  x\n(784-dim)",              C_INPUT),
    ("MLP Encoder\n(Linear→BN→ReLU ×2)", C_ENC),
    ("Embedding  z\n(64-dim, L2-norm)",  C_EMB),
    ("Linear Head\n(64 → #classes)",     C_HEAD),
    ("Loss:  CE on full z",              C_LOSS),
    ("Eval:  zero-pad  z[:k] → 64-dim\nthen feed into shared head", C_EVAL),
]
for i, ((label, col), y) in enumerate(zip(boxes_std, ys)):
    w = BW + 0.08 if i == 5 else BW   # eval box a bit wider
    h = BH + 0.02 if i == 5 else BH
    draw_box(ax_std, CX, y, w, h, label, col, fs=FS_SM if i == 5 else FS_BOX)
    if i < len(ys) - 1:
        draw_arrow(ax_std, CX, y - (h / 2), CX, ys[i + 1] + (BH + 0.02 if i + 1 == 5 else BH) / 2)

# Key property note
ax_std.text(CX, 0.01,
            "No ordering imposed on z.\nPrefix accuracy degrades quickly.",
            ha="center", va="bottom", fontsize=FS_TINY, fontfamily=FONT,
            color="#666666", style="italic")


# ══════════════════════════════════════════════════════════════════════
# Panel 2 — L1 (Sparsity Ablation)
# ══════════════════════════════════════════════════════════════════════
setup_panel(ax_l1, "② L1  (Sparsity Ablation)")

ys = column_ys(6, gap=0.148, top=0.87)
boxes_l1 = [
    ("Input  x\n(784-dim)",              C_INPUT),
    ("MLP Encoder\n(Linear→BN→ReLU ×2)", C_ENC),
    ("Embedding  z\n(64-dim, L2-norm)",  C_EMB),
    ("Linear Head\n(64 → #classes)",     C_HEAD),
    ("Loss:  CE  +  λ ‖z‖₁\n(sparse, unordered dims)", C_LOSS),
    ("Eval:  zero-pad  z[:k] → 64-dim\n(active dims may NOT be at front!)", C_EVAL),
]
for i, ((label, col), y) in enumerate(zip(boxes_l1, ys)):
    w = BW + 0.08 if i in (4, 5) else BW
    h = BH + 0.02 if i in (4, 5) else BH
    draw_box(ax_l1, CX, y, w, h, label, col, fs=FS_SM)
    if i < len(ys) - 1:
        draw_arrow(ax_l1, CX, y - h / 2, CX, ys[i + 1] + (BH + 0.02 if i + 1 in (4, 5) else BH) / 2)

# Side box: L1 penalty annotation on the embedding row
EMB_Y = ys[2]
lbox = FancyBboxPatch(
    (0.78, EMB_Y - 0.05), 0.20, 0.10,
    boxstyle="round,pad=0.02", linewidth=1.2,
    edgecolor="#CC0000", facecolor=C_L1, zorder=3,
)
ax_l1.add_patch(lbox)
ax_l1.text(0.88, EMB_Y, "λ ‖z‖₁\npenalty",
           ha="center", va="center", fontsize=FS_TINY, fontfamily=FONT,
           color="#CC0000", fontweight="bold", zorder=4)
ax_l1.annotate(
    "", xy=(CX + BW / 2, EMB_Y), xytext=(0.78, EMB_Y),
    arrowprops=dict(arrowstyle="<-", color="#CC0000", lw=1.4), zorder=2,
)

ax_l1.text(CX, 0.01,
           "Sparsity encourages zeros — but NO ordering.\n"
           "Ablation: proves ordering (not just sparsity) matters.",
           ha="center", va="bottom", fontsize=FS_TINY, fontfamily=FONT,
           color="#666666", style="italic")


# ══════════════════════════════════════════════════════════════════════
# Panel 3 — MRL / Matryoshka
# ══════════════════════════════════════════════════════════════════════
setup_panel(ax_mrl, "③ MRL  (Matryoshka Representation Learning)")

# Encoder row
ENC_Y = 0.87
draw_box(ax_mrl, CX, ENC_Y, BW + 0.20, BH,
         "Input  x  →  MLP Encoder  →  Embedding  z  (64-dim, L2-norm)",
         C_ENC, fs=FS_SM)

# Prefix boxes
PREFIX_KS  = ["1", "2", "4", "8", "16", "32", "64"]
N_K        = len(PREFIX_KS)
PFX_Y      = 0.68
xs_pfx     = np.linspace(0.07, 0.93, N_K)
PW, PH     = 0.105, 0.09

# Fan arrows from encoder to each prefix box
for xk in xs_pfx:
    draw_arrow(ax_mrl, CX, ENC_Y - BH / 2, xk, PFX_Y + PH / 2,
               color="#555555", lw=0.9)

for i, (xk, k) in enumerate(zip(xs_pfx, PREFIX_KS)):
    shade = 0.30 + 0.10 * i
    col = plt.cm.Blues(shade)
    pfx_box = FancyBboxPatch(
        (xk - PW / 2, PFX_Y - PH / 2), PW, PH,
        boxstyle="round,pad=0.02", linewidth=1.0,
        edgecolor=C_BORDER, facecolor=col, zorder=3,
    )
    ax_mrl.add_patch(pfx_box)
    ax_mrl.text(xk, PFX_Y, f"z[:k]\nk={k}",
                ha="center", va="center", fontsize=FS_TINY, fontfamily=FONT,
                zorder=4)

# CE loss boxes below each prefix box
CE_Y = 0.50
for xk in xs_pfx:
    draw_arrow(ax_mrl, xk, PFX_Y - PH / 2, xk, CE_Y + 0.04, lw=0.9)
    ce_box = FancyBboxPatch(
        (xk - PW / 2, CE_Y - 0.04), PW, 0.08,
        boxstyle="round,pad=0.015", linewidth=1.0,
        edgecolor=C_BORDER, facecolor=C_LOSS, zorder=3,
    )
    ax_mrl.add_patch(ce_box)
    ax_mrl.text(xk, CE_Y, "CE",
                ha="center", va="center", fontsize=FS_TINY, fontfamily=FONT,
                fontweight="bold", zorder=4)

# Horizontal bar connecting all CE boxes, then sum arrow
BAR_Y = CE_Y - 0.04
ax_mrl.plot([xs_pfx[0], xs_pfx[-1]], [BAR_Y, BAR_Y],
            color="#333333", lw=1.5, zorder=2)
SUM_Y = 0.30
draw_arrow(ax_mrl, CX, BAR_Y, CX, SUM_Y + BH / 2)
draw_box(ax_mrl, CX, SUM_Y, BW + 0.20, BH,
         "Loss  =  Σ  CE( head( z[:k] ), y )  for  k ∈ {1,2,4,8,16,32,64}",
         C_LOSS, fs=FS_SM)

# Eval note
EVAL_Y = 0.14
draw_arrow(ax_mrl, CX, SUM_Y - BH / 2, CX, EVAL_Y + BH / 2 + 0.02)
draw_box(ax_mrl, CX, EVAL_Y, BW + 0.20, BH + 0.04,
         "Eval:  truncate  z  to first  k  dims\n"
         "← ORDERING enforced during training  →  graceful degradation",
         C_EVAL, fs=FS_SM)

ax_mrl.text(CX, 0.01,
            "Key property: early dims carry the most class information.",
            ha="center", va="bottom", fontsize=FS_TINY, fontfamily=FONT,
            color="#1A5276", style="italic")


# ══════════════════════════════════════════════════════════════════════
# Panel 4 — FF-k  (wide, spans 2 columns)
# ══════════════════════════════════════════════════════════════════════
setup_panel(ax_ff, "④ FF-k  (Fixed-Feature / Dedicated Model per k)")
ax_ff.set_xlim(0, 1)
ax_ff.set_ylim(0, 1)

# Show 5 representative k values side by side
KS_FF   = ["k=1", "k=2", "k=4", "k=8", "k=16"]
XS_FF   = np.linspace(0.10, 0.74, len(KS_FF))
WF, HF  = 0.11, 0.10

ROW_YS = [0.83, 0.66, 0.50, 0.34, 0.17]   # 5 rows per mini-model
ROW_TEMPLATES = [
    ("Input x\n(784-dim)",   C_INPUT),
    ("MLP Encoder\n(out={k})",  C_ENC),
    ("z  ({k}-dim)",          C_EMB),
    ("Head  ({k}→C)",         C_HEAD),
    ("CE Loss",               C_LOSS),
]

for xf, k_label in zip(XS_FF, KS_FF):
    k_val = k_label.split("=")[1]
    # Column header
    ax_ff.text(xf, 0.94, k_label, ha="center", va="center",
               fontsize=FS_BOX, fontweight="bold", fontfamily=FONT,
               color="#1A5276")
    for j, (ry, (tmpl, col)) in enumerate(zip(ROW_YS, ROW_TEMPLATES)):
        label = tmpl.replace("{k}", k_val)
        draw_box(ax_ff, xf, ry, WF, HF, label, col, fs=FS_TINY)
        if j < len(ROW_YS) - 1:
            draw_arrow(ax_ff, xf, ry - HF / 2, xf, ROW_YS[j + 1] + HF / 2, lw=1.1)

# "…" gap and note for higher k values
ax_ff.text(0.80, 0.55, "…\n(one model\nper k)",
           ha="center", va="center", fontsize=FS_SM, fontfamily=FONT,
           color="#777777")

# Right-side summary box
SUMX, SUMY = 0.91, 0.55
draw_box(ax_ff, SUMX, SUMY, 0.16, 0.40,
         "Each model\nis entirely\nindependent.\n\nNo shared\nweights.\n\neval_prefixes\n=[1,2,4,8,\n16,32,64]\n→ 7 models",
         "#FDFEFE", fs=FS_TINY, alpha=0.85)

# Bottom note
ax_ff.text(0.50, 0.02,
           "Each FF-k model is trained from scratch with embed_dim = k.  "
           "No parameter sharing.  Upper-bound oracle for each k independently.",
           ha="center", va="bottom", fontsize=FS_TINY, fontfamily=FONT,
           color="#666666", style="italic")


# ══════════════════════════════════════════════════════════════════════
# Panel 5 — PCA
# ══════════════════════════════════════════════════════════════════════
setup_panel(ax_pca, "⑤ PCA  (Analytical Baseline)")

PCA_STEPS = [
    ("Training data  X_train\n(n × 784)", C_INPUT),
    ("Centre  X  by mean\n(zero-mean each feature)", C_ENC),
    ("Eigen-decompose covariance\nΣ = XᵀX / n\n→ eigenvectors V,\nordered by variance ↓", C_PCA),
    ("Projection matrix\nW = V[:, :64]", C_PCA),
    ("Eval:  project  X_test @ W[:, :k]\n= first  k  principal components", C_EVAL),
]

PCA_YS = column_ys(len(PCA_STEPS), gap=0.175, top=0.87)
PCA_W  = BW + 0.14
for i, ((label, col), y) in enumerate(zip(PCA_STEPS, PCA_YS)):
    h = BH + 0.05 if i == 2 else BH + 0.02
    draw_box(ax_pca, CX, y, PCA_W, h, label, col, fs=FS_SM)
    if i < len(PCA_STEPS) - 1:
        next_h = BH + 0.05 if i + 1 == 2 else BH + 0.02
        draw_arrow(ax_pca, CX, y - h / 2, CX, PCA_YS[i + 1] + next_h / 2)

ax_pca.text(CX, 0.01,
            "No training.  Purely analytical.\n"
            "Dimensions ordered by variance explained.\n"
            "Theoretical upper bound for unsupervised ordering.",
            ha="center", va="bottom", fontsize=FS_TINY, fontfamily=FONT,
            color="#6C3483", style="italic")


# ── Legend strip at the very bottom ────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INPUT,  edgecolor=C_BORDER, label="Input data"),
    mpatches.Patch(facecolor=C_ENC,    edgecolor=C_BORDER, label="Encoder (shared MLP)"),
    mpatches.Patch(facecolor=C_EMB,    edgecolor=C_BORDER, label="Embedding vector z"),
    mpatches.Patch(facecolor=C_HEAD,   edgecolor=C_BORDER, label="Classifier head"),
    mpatches.Patch(facecolor=C_LOSS,   edgecolor=C_BORDER, label="Loss function"),
    mpatches.Patch(facecolor=C_EVAL,   edgecolor=C_BORDER, label="Evaluation / truncation"),
    mpatches.Patch(facecolor=C_PCA,    edgecolor=C_BORDER, label="PCA step"),
    mpatches.Patch(facecolor=C_L1,     edgecolor="#CC0000", label="L1 penalty"),
]
fig.legend(
    handles=legend_items,
    loc="lower center",
    ncol=8,
    fontsize=FS_BOX,
    frameon=True,
    bbox_to_anchor=(0.50, 0.00),
)

# ── Save ──────────────────────────────────────────────────────────────
plt.savefig(OUT_PATH,     dpi=150, bbox_inches="tight", facecolor="white")
plt.savefig(OUT_PATH_SVG,          bbox_inches="tight", facecolor="white")
print(f"[✓] Saved architecture diagram → {OUT_PATH}")
print(f"[✓] Saved architecture diagram → {OUT_PATH_SVG}")
plt.close()
