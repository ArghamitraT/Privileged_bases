"""
CE Seed Stability: compare effective discriminative subspace across seeds.

For each CE model and prefix k, computes:
    M_k^(s) = W_k^(s) @ B_{1:k}^(s)   ∈ R^{C × p}

Centers rows to remove the common-logit direction:
    M̃_k = (I - (1/C) 11ᵀ) M_k

Takes orthonormal basis of col(M̃_k^T) via SVD, then measures the principal
angles between the two seeds' discriminative subspaces.

Also computes projector distance  ||P_s1 - P_s2||_F  as a sign-independent metric.

Usage:
    Conda environment: mrl_env

    python weight_symmetry/scripts/plot_ce_seed_stability.py \\
        --run1 exprmnt_2026_04_19__19_52_28 \\
        --run2 exprmnt_2026_04_19__16_00_49
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import torch
from weight_symmetry.models.linear_ae import LinearAE
from weight_symmetry.models.linear_ae_heads import LinearAEWithHeads
from weight_symmetry.plotting.style import apply_style, save_fig

DEFAULT_RESULTS_ROOT = os.path.abspath(
    os.path.join(_CODE_ROOT, "..", "files", "results")
)

CE_TAGS = [
    ("normal_ce",    "Normal CE",            "#888888", "--"),
    ("std_mrl_ce",   "Standard MRL (CE)",    "#CC79A7", "--"),
    ("fp_mrl_ce",    "Full-prefix MRL (CE)", "#D55E00", "-"),
    ("prefix_l1_ce", "PrefixL1 (CE) (rev)",  "#E69F00", "-"),
]


# ==============================================================================
# Core geometry
# ==============================================================================

def effective_classifier(model, k: int) -> np.ndarray:
    """
    M_k = W_k B_{1:k}  ∈ R^{C × p}
    where B_{1:k} is the first k rows of the encoder, W_k is the k-th head weight.
    """
    B_k = model.get_encoder_matrix()[:k].detach().cpu().numpy().astype(np.float64)  # (k, p)
    W_k = model.heads[k - 1].weight.detach().cpu().numpy().astype(np.float64)       # (C, k)
    return W_k @ B_k   # (C, p)


def centered_rowspace_basis(M: np.ndarray) -> np.ndarray:
    """
    Center M's rows (remove common-logit direction), then return an orthonormal
    basis for the row space, i.e. right singular vectors of M̃.

    Returns Q ∈ R^{p × r}  (r = rank of M̃, at most C-1).
    """
    C = M.shape[0]
    centering = np.eye(C) - np.ones((C, C)) / C   # (C, C)
    M_tilde   = centering @ M                      # (C, p)
    _, s, Vh  = np.linalg.svd(M_tilde, full_matrices=False)   # Vh: (C, p)
    # Keep directions with non-negligible singular values
    tol  = s[0] * max(M_tilde.shape) * np.finfo(float).eps * 10
    rank = int((s > tol).sum())
    Q    = Vh[:rank].T                             # (p, rank)
    return Q


def principal_angles_deg(Q1: np.ndarray, Q2: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between span(Q1) and span(Q2)."""
    from scipy.linalg import subspace_angles
    return np.degrees(subspace_angles(Q1, Q2))


def projector_distance(Q1: np.ndarray, Q2: np.ndarray) -> float:
    """Frobenius distance between orthogonal projectors ||P1 - P2||_F."""
    P1 = Q1 @ Q1.T
    P2 = Q2 @ Q2.T
    return float(np.linalg.norm(P1 - P2, "fro"))


# ==============================================================================
# Load one run
# ==============================================================================

def load_run(run_dir: str):
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    seed      = cfg["seeds"][0]
    embed_dim = cfg["embed_dim"]
    n_classes = None   # will be inferred from weights

    models = {}
    for tag, label, _, _ in CE_TAGS:
        pt_path = os.path.join(run_dir, f"seed{seed}_{tag}_best.pt")
        if not os.path.exists(pt_path):
            print(f"  [warn] not found: {pt_path} — skipping {tag}")
            continue
        sd = torch.load(pt_path, weights_only=True, map_location="cpu")
        # Infer n_classes from head weight shape
        if n_classes is None:
            head_key = next(k for k in sd if "heads.0.weight" in k)
            n_classes = sd[head_key].shape[0]
        # Infer input_dim from encoder weight shape
        enc_key   = next(k for k in sd if "encoder.weight" in k)
        input_dim = sd[enc_key].shape[1]

        model = LinearAEWithHeads(input_dim, embed_dim, n_classes)
        model.load_state_dict(sd)
        model.eval()
        models[tag] = model

    return models, embed_dim, seed, cfg


# ==============================================================================
# Main comparison
# ==============================================================================

def compare_runs(run1_dir: str, run2_dir: str, out_dir: str):
    print(f"[stability] Loading run1: {run1_dir}")
    models1, embed_dim, seed1, cfg1 = load_run(run1_dir)
    print(f"[stability] Loading run2: {run2_dir}")
    models2, embed_dim2, seed2, cfg2 = load_run(run2_dir)

    assert embed_dim == embed_dim2, "embed_dim mismatch between runs"

    results = {}   # tag -> {"mean_angle": [...], "max_angle": [...], "proj_dist": [...]}
    xs = list(range(1, embed_dim + 1))

    for tag, label, color, ls in CE_TAGS:
        if tag not in models1 or tag not in models2:
            continue
        m1 = models1[tag]
        m2 = models2[tag]

        mean_angles = []
        max_angles  = []
        proj_dists  = []

        for k in xs:
            Q1 = centered_rowspace_basis(effective_classifier(m1, k))
            Q2 = centered_rowspace_basis(effective_classifier(m2, k))

            # Match rank to the smaller basis
            r = min(Q1.shape[1], Q2.shape[1])
            Q1, Q2 = Q1[:, :r], Q2[:, :r]

            angles = principal_angles_deg(Q1, Q2)
            mean_angles.append(float(angles.mean()))
            max_angles.append(float(angles.max()))
            proj_dists.append(projector_distance(Q1, Q2))

        results[tag] = dict(mean_angle=mean_angles, max_angle=max_angles,
                            proj_dist=proj_dists, label=label,
                            color=color, ls=ls)
        print(f"  [{tag}] mean angle@k=1: {mean_angles[0]:.1f}°  "
              f"@k={embed_dim}: {mean_angles[-1]:.1f}°  "
              f"proj_dist@k={embed_dim}: {proj_dists[-1]:.3f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    apply_style()
    fig_stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for tag, label, color, ls in CE_TAGS:
        if tag not in results:
            continue
        r = results[tag]
        axes[0].plot(xs, r["mean_angle"], label=label, color=color, ls=ls, lw=1.5)
        axes[1].plot(xs, r["max_angle"],  label=label, color=color, ls=ls, lw=1.5)
        axes[2].plot(xs, r["proj_dist"],  label=label, color=color, ls=ls, lw=1.5)

    for ax, title, ylabel in [
        (axes[0], "Mean principal angle (seed stability)",  "Mean principal angle (°)"),
        (axes[1], "Max principal angle (seed stability)",   "Max principal angle (°)"),
        (axes[2], "Projector distance  ||P_s1 - P_s2||_F", "||P_s1 - P_s2||_F"),
    ]:
        ax.set_xlabel("Prefix size k")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.suptitle(
        f"CE seed stability: seed {seed1} vs seed {seed2}\n"
        f"Comparing row space of W_k B_{{1:k}} (centered) across seeds",
        fontsize=9,
    )
    fig.tight_layout()

    stem = save_fig(fig, "ce_seed_stability_Mk_rowspace", fig_stamp)
    plt.close(fig)
    print(f"\n[stability] Saved figure: {stem}")

    # ── Text summary ─────────────────────────────────────────────────────────
    summary_path = os.path.join(
        os.path.dirname(os.path.dirname(run1_dir)) if "figures" in out_dir else out_dir,
        f"ce_seed_stability_{fig_stamp}.txt",
    )
    lines = [
        "CE Seed Stability — row space of W_k B_{1:k} (centered)",
        f"Seed 1: {seed1}  ({run1_dir})",
        f"Seed 2: {seed2}  ({run2_dir})",
        "",
        f"{'Model':<30}  {'mean_angle@1':>14}  {'mean_angle@d':>13}  {'proj_dist@d':>12}",
        "-" * 80,
    ]
    for tag, label, _, _ in CE_TAGS:
        if tag not in results:
            continue
        r = results[tag]
        lines.append(
            f"{label:<30}  {r['mean_angle'][0]:>14.1f}°  "
            f"{r['mean_angle'][-1]:>13.1f}°  "
            f"{r['proj_dist'][-1]:>12.3f}"
        )
    lines += [
        "",
        "mean_angle: mean of all principal angles between the two seeds' subspaces",
        "            (0° = identical subspace, 90° = orthogonal)",
        "proj_dist:  ||P_s1 - P_s2||_F  (0 = identical, sqrt(2r) = orthogonal)",
    ]
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[stability] Saved summary: {summary_path}")


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CE seed stability: compare W_k B_{1:k} row spaces across seeds")
    parser.add_argument("--run1", required=True,
                        help="First run folder (contains config.json + .pt files)")
    parser.add_argument("--run2", required=True,
                        help="Second run folder (contains config.json + .pt files)")
    parser.add_argument("--results-root", default=None)
    args = parser.parse_args()

    results_root = args.results_root or DEFAULT_RESULTS_ROOT
    run1_dir = os.path.join(results_root, args.run1) if not os.path.isabs(args.run1) else args.run1
    run2_dir = os.path.join(results_root, args.run2) if not os.path.isabs(args.run2) else args.run2

    for d in [run1_dir, run2_dir]:
        if not os.path.exists(os.path.join(d, "config.json")):
            sys.exit(f"ERROR: config.json not found in {d}")

    compare_runs(run1_dir, run2_dir, results_root)


if __name__ == "__main__":
    main()
