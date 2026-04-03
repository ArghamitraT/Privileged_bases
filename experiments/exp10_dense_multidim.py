"""
Script: experiments/exp10_dense_multidim.py
--------------------------------------------
Experiment 10 — Dense Prefix Sweep: MRL vs Standard vs L1 vs PrefixL1 vs PCA (No FF).

Exp7 without Fixed-Feature models, evaluated at every dimension k = 1..embed_dim
(dense sweep). Produces smooth, continuous accuracy curves showing exactly where
each model's performance plateaus.

Key differences vs Exp7:
  - No FF models — focus is on prefix ordering, not capacity matching
  - Dense eval: k = 1, 2, ..., embed_dim  (not just powers of 2)
  - Supports --embed-dim flag for 8, 16, 32, 64
  - Linear x-axis on plots (not log scale) to show the continuous curve
  - Supports --use-weights to regenerate plots from a prior run without retraining

Models trained:
  1. Standard  — plain CE loss on full embed_dim embedding
  2. L1        — CE + L1 regularization on embedding activations
  3. MRL (Mat) — CE summed at every prefix scale (Matryoshka loss)
  4. PrefixL1  — CE + front-loaded weighted L1 (dim j penalised by embed_dim-j)
  5. PCA       — analytical baseline (variance-ordered components)

PrefixL1 note:
  Dimensions are *reversed* before the prefix sweep (option A — "flip").
  The early dims of PrefixL1 carry the most sparsity pressure (lightest info);
  flipping puts the most informative dims first, so x-axis reads best-first.
  Legend label: "PrefixL1 (rev)".

Figure naming:
  All figures produced by this script have a timestamp suffix in their filename
  (e.g. linear_accuracy_curve_2026_04_01__22_04_54.png) so that re-running with
  --use-weights into a fresh run folder never overwrites existing images.

Evaluation metrics at each k = 1..embed_dim:
  a. Linear classification accuracy  (logistic regression probe on k-dim embedding)
  b. 1-NN accuracy  (1-nearest-neighbor; train set as database, test set as queries)

Inputs:
  --fast             : smoke test — digits dataset, 5 epochs, subsampled 1-NN
  --embed-dim N      : override embed_dim (8, 16, 32, 64); ignored if --use-weights set
  --use-weights PATH : load saved weights from PATH (folder name or full path),
                       skip training and regenerate all plots

Outputs (all in a new timestamped run folder):
  linear_accuracy_curve_{stamp}.png  : linear accuracy vs k — up to 5 model lines
  1nn_accuracy_curve_{stamp}.png     : 1-NN accuracy vs k — up to 5 model lines
  combined_comparison_{stamp}.png    : 2-panel (linear top, 1-NN bottom)
  training_curves_{stamp}.png        : loss vs epoch (skipped when loading weights)
  results_summary.txt                : full table k x model x linear_acc x 1nn_acc
  experiment_description.log
  runtime.txt
  code_snapshot/

Usage:
    python experiments/exp10_dense_multidim.py                            # full run (MNIST, embed_dim=8)
    python experiments/exp10_dense_multidim.py --fast                     # smoke test (digits, 5 epochs)
    python experiments/exp10_dense_multidim.py --embed-dim 16             # full run, embed_dim=16
    python experiments/exp10_dense_multidim.py --embed-dim 32             # full run, embed_dim=32
    python experiments/exp10_dense_multidim.py --embed-dim 8 --fast       # smoke test at dim=8
    python experiments/exp10_dense_multidim.py --use-weights exprmnt_XYZ  # regenerate from saved weights
    python tests/run_tests_exp10.py --fast                                # unit tests only
    python tests/run_tests_exp10.py                                       # unit tests + e2e smoke
"""

import os

# Cap BLAS thread count before numpy/scipy imports to prevent deadlocks on macOS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import sys
import time
import argparse
import dataclasses

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExpConfig
from utility import (
    create_run_dir, save_runtime, save_code_snapshot,
    save_config_json, load_config_json, get_path,
)
from data.loader import load_data

# Reuse training helpers and 1-NN evaluation from exp7 (no duplication)
from experiments.exp7_mrl_vs_ff import (
    train_single_model,
    get_embeddings_np,
    evaluate_1nn,
    evaluate_prefix_1nn,
    evaluate_pca_1nn,
    plot_training_curves,
)

# Needed for loading saved weights without retraining
from models.encoder import MLPEncoder
from models.heads import build_head


# ==============================================================================
# CONFIG — edit here to change the full run; use --fast for a quick smoke test
# ==============================================================================
DATASET       = "mnist"
EMBED_DIM     = 8
HIDDEN_DIM    = 256
HEAD_MODE     = "shared_head"
EVAL_PREFIXES = list(range(1, EMBED_DIM + 1))   # always derived from EMBED_DIM; --embed-dim overrides both
EPOCHS        = 10
PATIENCE      = 5
LR            = 1e-3
BATCH_SIZE    = 128
WEIGHT_DECAY  = 1e-4
SEED          = 42
L1_LAMBDA     = 0.05
# LP_ORDERS: list of Lp exponents to train PrefixLp models for.
# Each value p trains one model with model_type="prefix_l{p}", saved as pl{p}_*.pt.
# Set to [] to skip all PrefixLp models.  Examples: [1], [1, 3, 4], [1, 3, 4, 6].
LP_ORDERS     = [5, 6]
MAX_1NN_DB    = 10_000   # cap on 1-NN training-set size for speed on dense sweep
USE_WEIGHTS   = ""       # folder name (e.g. "exprmnt_2026_04_01__22_04_54") or full path;
                         # if set, skip training and load weights to regenerate plots
# ==============================================================================


# ==============================================================================
# Reproducibility
# ==============================================================================

def set_seeds(seed: int):
    """
    Set random seeds for numpy, torch, and python random.

    Args:
        seed (int): Master random seed.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[exp10] Random seeds set to {seed}")


# ==============================================================================
# Weight loader — load saved encoder/head pairs from a prior run directory
# ==============================================================================

def load_weights_from_dir(weights_dir: str, cfg, input_dim: int, n_classes: int):
    """
    Load encoder and head weights saved by train_single_model.

    Always loads Standard, L1, and MRL weights (required).
    Scans the directory for any pl{N}_encoder_best.pt files and loads all
    that are present — supporting arbitrary LP_ORDERS without hardcoding p.

    Args:
        weights_dir (str)      : Directory containing the saved .pt files.
        cfg         (ExpConfig): Config used to reconstruct model architecture.
        input_dim   (int)      : Input feature dimension (from data).
        n_classes   (int)      : Number of output classes (from data).

    Returns:
        tuple: (std_enc, std_hd, l1_enc, l1_hd, mat_enc, mat_hd, pl_models)
               pl_models is a dict {p (int): (encoder, head)} for all found
               PrefixLp weight files.  Empty dict if none are present.
    """
    import re as _re
    device = torch.device("cpu")

    def _load_pair(enc_tag, hd_tag):
        enc = MLPEncoder(
            input_dim=input_dim, hidden_dim=cfg.hidden_dim, embed_dim=cfg.embed_dim,
        )
        enc.load_state_dict(torch.load(os.path.join(weights_dir, enc_tag), map_location=device))
        enc.eval()
        print(f"  [load] Encoder loaded from {enc_tag}")

        hd = build_head(cfg, n_classes=n_classes)
        hd.load_state_dict(torch.load(os.path.join(weights_dir, hd_tag), map_location=device))
        hd.eval()
        print(f"  [load] Head    loaded from {hd_tag}")
        return enc, hd

    print(f"[exp10] Loading saved weights from {weights_dir}")

    std_enc, std_hd = _load_pair("standard_encoder_best.pt", "standard_head_best.pt")
    l1_enc,  l1_hd  = _load_pair("l1_encoder_best.pt",       "l1_head_best.pt")
    mat_enc, mat_hd = _load_pair("mat_encoder_best.pt",       "mat_head_best.pt")

    # Scan for pl{N}_encoder_best.pt files (any p value)
    pl_models = {}
    for fname in sorted(os.listdir(weights_dir)):
        m = _re.fullmatch(r"pl(\d+)_encoder_best\.pt", fname)
        if m:
            p = int(m.group(1))
            enc, hd = _load_pair(f"pl{p}_encoder_best.pt", f"pl{p}_head_best.pt")
            pl_models[p] = (enc, hd)
    if pl_models:
        print(f"  [load] PrefixLp models found: p = {sorted(pl_models)}")
    else:
        print("  [load] No PrefixLp weight files found — PrefixLp will be skipped")

    return std_enc, std_hd, l1_enc, l1_hd, mat_enc, mat_hd, pl_models


# ==============================================================================
# Experiment description log
# ==============================================================================

def save_experiment_description(cfg, run_dir, fast):
    """
    Write a human-readable log describing this experiment run.

    Args:
        cfg     (ExpConfig) : Experiment configuration.
        run_dir (str)       : Output directory for this run.
        fast    (bool)      : Whether fast/smoke mode is active.
    """
    log_path = os.path.join(run_dir, "experiment_description.log")
    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT 10 — Dense Prefix Sweep (MRL vs Standard vs L1 vs PCA)\n")
        f.write("=" * 70 + "\n\n")

        f.write("WHAT THIS EXPERIMENT DOES\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Trains three model families and evaluates each at EVERY prefix k\n"
            "from 1 to embed_dim (dense sweep, not just powers of 2):\n"
            "  Standard : plain CE loss on full embed_dim embedding\n"
            "  L1       : CE + L1 penalty on embedding activations\n"
            "  MRL      : CE summed at every prefix scale (Matryoshka loss)\n"
            "  PCA      : analytical baseline (variance-ordered components)\n\n"
            "No FF (Fixed-Feature) models — this experiment isolates the ordering\n"
            "property without capacity matching.\n\n"
            "Evaluation metrics:\n"
            "  Linear accuracy : logistic regression probe on k-dim embedding\n"
            "  1-NN accuracy   : 1-nearest-neighbor (train=database, test=query)\n\n"
        )

        f.write("WHY WE ARE RUNNING IT\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Exp7 used sparse checkpoints ([1,2,4,8,16,32,64]) and included FF\n"
            "models. Dense eval reveals the exact shape of the accuracy curve —\n"
            "does MRL plateau early? Does Standard drop linearly or suddenly?\n"
            "The L1 ablation confirms sparsity alone does not enforce ordering.\n\n"
        )

        f.write("EXPECTED OUTCOME\n")
        f.write("-" * 40 + "\n")
        f.write(
            "  MRL:      high accuracy even at small k; smooth monotone increase\n"
            "  Standard: drops sharply at small k; catches up at large k\n"
            "  L1:       similar to Standard at small k (no ordering benefit)\n"
            "  PCA:      intermediate — variance-ordered but not task-aware\n\n"
        )

        f.write(f"  Fast mode: {fast}\n\n")

        f.write("FULL CONFIG\n")
        f.write("-" * 40 + "\n")
        for field in dataclasses.fields(cfg):
            f.write(f"  {field.name:<20} = {getattr(cfg, field.name)}\n")
        f.write("\n")

    print(f"[exp10] Experiment description saved to {log_path}")


# ==============================================================================
# Linear accuracy evaluation (dense prefix sweep)
# ==============================================================================

def evaluate_prefix_linear(Z_train, Z_test, y_train, y_test,
                            eval_prefixes, model_tag, seed=42):
    """
    Logistic regression accuracy at each prefix k (dense sweep).

    Slices the first k columns of embeddings and fits a LR probe.
    Subsamples to 10k training samples for speed on MNIST.

    Args:
        Z_train       (np.ndarray): Train embeddings, shape (n_train, embed_dim).
        Z_test        (np.ndarray): Test embeddings,  shape (n_test,  embed_dim).
        y_train       (np.ndarray): Train labels.
        y_test        (np.ndarray): Test labels.
        eval_prefixes (list[int]) : Prefix sizes to evaluate.
        model_tag     (str)       : Label for print output.
        seed          (int)       : For LogisticRegression reproducibility.

    Returns:
        dict[int, float]: {k: accuracy}
    """
    print(f"\n[exp10] Linear accuracy sweep for '{model_tag}' ...")

    # Subsample train to 10k for speed (avoids very slow LR on 56k MNIST samples)
    rng = np.random.default_rng(seed)
    max_train = 10_000
    if len(Z_train) > max_train:
        idx     = rng.choice(len(Z_train), max_train, replace=False)
        Ztr_sub = Z_train[idx]
        ytr_sub = y_train[idx]
    else:
        Ztr_sub, ytr_sub = Z_train, y_train

    results = {}
    for k in eval_prefixes:
        lr = LogisticRegression(
            solver="saga", max_iter=1000, random_state=seed, n_jobs=1,
        )
        lr.fit(Ztr_sub[:, :k], ytr_sub)
        acc = float(lr.score(Z_test[:, :k], y_test))
        results[k] = acc
        if k % 8 == 0 or k <= 4:
            print(f"  k={k:>3}  linear={acc:.4f}")

    print(f"  [done] {model_tag}: k=1..{eval_prefixes[-1]}")
    return results


def evaluate_pca_linear(data, cfg, seed=42):
    """
    Logistic regression accuracy for PCA baseline at each prefix k.

    Fits PCA on training data, then evaluates LR on first k components.

    Args:
        data (DataSplit) : Train/test splits.
        cfg  (ExpConfig) : Uses embed_dim, eval_prefixes, seed.
        seed (int)       : For LogisticRegression.

    Returns:
        dict[int, float]: {k: accuracy}
    """
    from sklearn.decomposition import PCA as SklearnPCA

    print("\n[exp10] Linear accuracy for PCA baseline ...")

    X_train_np = np.array(data.X_train.tolist(), dtype=np.float32)
    X_test_np  = np.array(data.X_test.tolist(),  dtype=np.float32)
    y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
    y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)

    n_comp  = min(cfg.embed_dim, X_train_np.shape[0], X_train_np.shape[1])
    pca     = SklearnPCA(n_components=n_comp, random_state=cfg.seed)
    Z_train = pca.fit_transform(X_train_np)
    Z_test  = pca.transform(X_test_np)

    return evaluate_prefix_linear(
        Z_train, Z_test, y_train_np, y_test_np,
        cfg.eval_prefixes, "PCA", seed=seed,
    )


# ==============================================================================
# Plotting
# ==============================================================================

# Consistent colour/marker/label per model (no FF in exp10)
_BASE_MODEL_STYLES = {
    "Standard": ("steelblue",  "o-",  "Standard"),
    "L1":       ("orchid",     "D-",  "L1"),
    "MRL":      ("darkorange", "s-",  "MRL"),
    "PCA":      ("seagreen",   "x--", "PCA"),
}

# Fixed colours for each Lp exponent — add more if needed
_PL_COLORS  = {1: "crimson", 2: "hotpink", 3: "darkorchid", 4: "indigo",
               5: "navy",    6: "teal"}
_PL_MARKERS = {1: "^-", 2: "v-", 3: "P-", 4: "X-", 5: "h-", 6: "D-"}


def get_model_style(model_name: str, l1_lambda: float = None):
    """
    Return (color, marker_style, legend_label) for a model name.

    Handles fixed base models and dynamic "PrefixL{p} (rev)" names.

    Args:
        model_name (str)  : Model key used in results dicts.
        l1_lambda  (float): Used to annotate the L1 line label (optional).

    Returns:
        Tuple[str, str, str]: (color, style, label)
    """
    import re
    if model_name in _BASE_MODEL_STYLES:
        color, style, label = _BASE_MODEL_STYLES[model_name]
        if model_name == "L1" and l1_lambda is not None:
            label = f"L1 (lambda={l1_lambda})"
        return color, style, label
    # PrefixLp (rev) — e.g. "PrefixL3 (rev)"
    m = re.fullmatch(r"PrefixL(\d+) \(rev\)", model_name)
    if m:
        p     = int(m.group(1))
        color = _PL_COLORS.get(p, "darkred")
        style = _PL_MARKERS.get(p, "^-")
        return color, style, model_name
    # Fallback for unrecognised names
    return "gray", "x-", model_name


# Keep backward-compat name in case any inline code still references MODEL_STYLES
MODEL_STYLES = _BASE_MODEL_STYLES


def _single_accuracy_plot(results_dict, eval_prefixes, ylabel, title,
                           out_path, l1_lambda, fig_stamp: str = ""):
    """
    Plot accuracy vs prefix k (linear x-axis) and save.

    Args:
        results_dict  (dict)  : {model_name: {k: accuracy}}.
        eval_prefixes (list)  : x-axis values.
        ylabel        (str)   : y-axis label.
        title         (str)   : Plot title.
        out_path      (str)   : Base output path (fig_stamp is inserted before .png).
        l1_lambda     (float) : Annotates the L1 line label.
        fig_stamp     (str)   : Timestamp suffix inserted before .png extension.
    """
    # Insert fig_stamp before the .png extension
    if fig_stamp:
        base, ext = os.path.splitext(out_path)
        out_path = f"{base}{fig_stamp}{ext}"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    for model_name, acc_dict in results_dict.items():
        color, style, label = get_model_style(model_name, l1_lambda=l1_lambda)
        accs = [acc_dict.get(k, float("nan")) for k in eval_prefixes]
        ax.plot(eval_prefixes, accs, style, color=color,
                label=label, linewidth=2, markersize=4)

    # Linear x-axis — dense eval shows the full curve shape
    ax.set_xlabel("Prefix size k  (embedding dimensions used)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, eval_prefixes[-1])
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp10] Saved {os.path.basename(out_path)}")


def plot_all_curves(linear_results, nn1_results, eval_prefixes, run_dir,
                    l1_lambda, fig_stamp: str = ""):
    """
    Save three plots: linear accuracy, 1-NN accuracy, and combined 2-panel.

    Args:
        linear_results (dict)  : {model_name: {k: accuracy}} — linear probe.
        nn1_results    (dict)  : {model_name: {k: accuracy}} — 1-NN.
        eval_prefixes  (list)  : x-axis values (dense 1..embed_dim).
        run_dir        (str)   : Output directory.
        l1_lambda      (float) : Used in L1 legend label.
        fig_stamp      (str)   : Timestamp suffix inserted before .png extension.
    """
    _single_accuracy_plot(
        linear_results, eval_prefixes,
        ylabel="Linear Classification Accuracy",
        title="Linear Accuracy vs Prefix k  (Standard vs L1 vs MRL vs PCA — dense)",
        out_path=os.path.join(run_dir, "linear_accuracy_curve.png"),
        l1_lambda=l1_lambda,
        fig_stamp=fig_stamp,
    )

    _single_accuracy_plot(
        nn1_results, eval_prefixes,
        ylabel="1-NN Accuracy",
        title="1-NN Accuracy vs Prefix k  (Standard vs L1 vs MRL vs PCA — dense)",
        out_path=os.path.join(run_dir, "1nn_accuracy_curve.png"),
        l1_lambda=l1_lambda,
        fig_stamp=fig_stamp,
    )

    # Combined 2-panel
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for model_name in linear_results:
        color, style, label = get_model_style(model_name, l1_lambda=l1_lambda)
        lin_accs = [linear_results[model_name].get(k, float("nan")) for k in eval_prefixes]
        nn1_accs = [nn1_results[model_name].get(k,    float("nan")) for k in eval_prefixes]
        ax_top.plot(eval_prefixes, lin_accs, style, color=color,
                    label=label, linewidth=2, markersize=4)
        ax_bot.plot(eval_prefixes, nn1_accs, style, color=color,
                    label=label, linewidth=2, markersize=4)

    for ax in (ax_top, ax_bot):
        ax.set_ylim(0, 1.05)
        ax.set_xlim(1, eval_prefixes[-1])
        ax.legend(fontsize=10)

    ax_top.set_ylabel("Linear Accuracy", fontsize=12)
    ax_top.set_title("Linear Classification Accuracy vs Prefix k  (dense)", fontsize=12)
    ax_bot.set_xlabel("Prefix size k  (embedding dimensions used)", fontsize=12)
    ax_bot.set_ylabel("1-NN Accuracy", fontsize=12)
    ax_bot.set_title("1-NN Accuracy vs Prefix k  (dense)", fontsize=12)
    fig.suptitle("Standard vs L1 vs MRL vs PCA — Dense Prefix Sweep", fontsize=14, y=1.01)
    plt.tight_layout()
    combined_name = f"combined_comparison{fig_stamp}.png"
    plt.savefig(os.path.join(run_dir, combined_name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp10] Saved {combined_name}")


# ==============================================================================
# Results table
# ==============================================================================

def save_results_summary(linear_results, nn1_results, eval_prefixes, run_dir):
    """
    Write a plain-text table of linear accuracy and 1-NN accuracy for every (model, k).

    Args:
        linear_results (dict)  : {model_name: {k: accuracy}}.
        nn1_results    (dict)  : {model_name: {k: accuracy}}.
        eval_prefixes  (list)  : Prefix sizes (rows).
        run_dir        (str)   : Output directory.
    """
    model_names = list(linear_results.keys())
    path = os.path.join(run_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write("EXPERIMENT 10 — Dense Prefix Sweep Results\n")
        f.write("=" * 60 + "\n\n")
        header = f"{'k':>4}  {'Model':<12}  {'Linear Acc':>12}  {'1-NN Acc':>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for k in eval_prefixes:
            for model_name in model_names:
                lin = linear_results[model_name].get(k, float("nan"))
                nn1 = nn1_results[model_name].get(k,    float("nan"))
                f.write(f"{k:>4}  {model_name:<12}  {lin:>12.4f}  {nn1:>10.4f}\n")
            f.write("\n")

    print(f"[exp10] Results summary saved to {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """
    Orchestrate Experiment 10:
      - Parse args, configure, create run dir
      - Load data
      - Train Standard, L1, MRL models
      - Plot training curves (MANDATORY)
      - Evaluate linear accuracy and 1-NN accuracy at every k = 1..embed_dim
      - Plot and save all outputs
    """
    run_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Experiment 10 — Dense Prefix Sweep (no FF)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Smoke test: digits dataset, 5 epochs, subsampled 1-NN database.",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=None, metavar="N",
        help="Override embed_dim (8, 16, 32, 64); eval_prefixes = [1..N]. Ignored when --use-weights is set.",
    )
    parser.add_argument(
        "--use-weights", type=str, default=None, metavar="PATH",
        help=(
            "Load saved weights from PATH (folder name or full path) and regenerate "
            "all plots without retraining. Config is adopted from config.json in PATH. "
            "Example: --use-weights exprmnt_2026_04_01__22_04_54"
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 2: Resolve weights_dir and build config
    # ------------------------------------------------------------------

    # Priority: CLI flag > CONFIG constant > None (train from scratch)
    weights_dir = args.use_weights or USE_WEIGHTS or None

    if weights_dir:
        # Resolve relative names (e.g. "exprmnt_XYZ") to full paths under files/results/
        if not os.path.isabs(weights_dir):
            weights_dir = os.path.join(get_path("files/results"), weights_dir)
        print(f"[exp10] --use-weights: loading from {weights_dir}")

        # Adopt ALL settings from the saved run's config.json (so architecture matches)
        cfg_dict = load_config_json(weights_dir)
        if cfg_dict:
            cfg = ExpConfig(**cfg_dict)
            print(f"[exp10] Config adopted from config.json (embed_dim={cfg.embed_dim})")
        else:
            # Fallback: build config from local constants (user must ensure they match)
            print("[exp10] WARNING: config.json not found — using local CONFIG constants")
            cfg = ExpConfig(
                dataset=DATASET, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                head_mode=HEAD_MODE, eval_prefixes=EVAL_PREFIXES,
                lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
                weight_decay=WEIGHT_DECAY, seed=SEED, l1_lambda=L1_LAMBDA,
                experiment_name="exp10_dense_multidim",
            )
        max_1nn_db = 500 if args.fast else MAX_1NN_DB

    elif args.fast:
        cfg = ExpConfig(
            dataset="digits", embed_dim=16, hidden_dim=128,
            head_mode="shared_head", eval_prefixes=list(range(1, 17)),
            lr=LR, epochs=5, batch_size=BATCH_SIZE, patience=3,
            weight_decay=WEIGHT_DECAY, seed=SEED, l1_lambda=L1_LAMBDA,
            experiment_name="exp10_dense_multidim",
        )
        max_1nn_db = 500
    else:
        cfg = ExpConfig(
            dataset       = DATASET,
            embed_dim     = EMBED_DIM,
            hidden_dim    = HIDDEN_DIM,
            head_mode     = HEAD_MODE,
            eval_prefixes = EVAL_PREFIXES,
            lr            = LR,
            epochs        = EPOCHS,
            batch_size    = BATCH_SIZE,
            patience      = PATIENCE,
            weight_decay  = WEIGHT_DECAY,
            seed          = SEED,
            l1_lambda     = L1_LAMBDA,
            experiment_name = "exp10_dense_multidim",
        )
        max_1nn_db = MAX_1NN_DB

    # Apply --embed-dim override (only when NOT loading saved weights)
    if args.embed_dim is not None and not weights_dir:
        cfg.embed_dim     = args.embed_dim
        cfg.eval_prefixes = list(range(1, cfg.embed_dim + 1))
        print(f"[exp10] --embed-dim override: embed_dim={cfg.embed_dim}, "
              f"eval_prefixes=1..{cfg.embed_dim}")

    set_seeds(cfg.seed)

    # ------------------------------------------------------------------
    # Step 3: Setup output directory + save description
    # When --use-weights is set, write all outputs (plots, logs, summaries)
    # into the same folder as the loaded weights so results stay together.
    # ------------------------------------------------------------------
    if weights_dir:
        # Create a new timestamped subfolder *inside* the weights folder so that
        # repeated --use-weights runs don't collide (e.g. code_snapshot already exists).
        sub_stamp = time.strftime("exprmnt_%Y_%m_%d__%H_%M_%S")
        run_dir   = os.path.join(weights_dir, sub_stamp)
        os.makedirs(run_dir, exist_ok=True)
        print(f"[exp10] --use-weights: outputs will be saved to: {run_dir}\n")
    else:
        run_dir = create_run_dir()
        print(f"[exp10] Outputs will be saved to: {run_dir}\n")
    # Timestamp suffix for all figure filenames — prevents overwriting when re-running
    # (e.g. with --use-weights) in the same folder or comparing runs side-by-side.
    fig_stamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
    save_experiment_description(cfg, run_dir, args.fast)
    save_config_json(cfg, run_dir)   # machine-readable config for downstream experiments (e.g. exp8)

    # ------------------------------------------------------------------
    # Step 4: Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Loading data")
    print("=" * 60)
    data = load_data(cfg)

    # ------------------------------------------------------------------
    # Steps 5-7b: Train all models  OR  load saved weights
    # ------------------------------------------------------------------
    input_dim = data.X_train.shape[1]
    n_classes = int(len(set(data.y_train.numpy().tolist())))

    if weights_dir:
        # ---- Load path: skip training, read weights from disk ----
        print("=" * 60)
        print("STEPS 5-7b: Loading saved model weights (skipping training)")
        print("=" * 60)
        (std_encoder, std_head,
         l1_encoder,  l1_head,
         mat_encoder, mat_head,
         pl_models) = load_weights_from_dir(
            weights_dir, cfg, input_dim=input_dim, n_classes=n_classes,
        )
    else:
        # ---- Train path: fit base models + one PrefixLp per p in LP_ORDERS ----
        print("=" * 60)
        print("STEP 5: Training Standard model")
        print("=" * 60)
        std_encoder, std_head = train_single_model(
            cfg, data, run_dir, model_type="standard", model_tag="standard"
        )

        print("=" * 60)
        print(f"STEP 6: Training L1 model  (lambda={cfg.l1_lambda})")
        print("=" * 60)
        l1_encoder, l1_head = train_single_model(
            cfg, data, run_dir, model_type="l1", model_tag="l1"
        )

        print("=" * 60)
        print("STEP 7: Training MRL model")
        print("=" * 60)
        mat_encoder, mat_head = train_single_model(
            cfg, data, run_dir, model_type="matryoshka", model_tag="mat"
        )

        # Train one PrefixLp model per p in LP_ORDERS
        pl_models = {}   # {p: (encoder, head)}
        for p in LP_ORDERS:
            print("=" * 60)
            print(f"STEP 7+: Training PrefixL{p} model  (lambda={cfg.l1_lambda})")
            print("=" * 60)
            enc, hd = train_single_model(
                cfg, data, run_dir,
                model_type=f"prefix_l{p}", model_tag=f"pl{p}",
            )
            pl_models[p] = (enc, hd)

    # ------------------------------------------------------------------
    # Step 8: Training curves (MANDATORY)
    # plot_training_curves skips any tag whose .log file is absent,
    # so calling it is safe even when loading weights (no logs written).
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 8: Plotting training curves")
    print("=" * 60)
    pl_tags = [f"pl{p}" for p in sorted(pl_models)]
    plot_training_curves(run_dir, model_tags=["standard", "l1", "mat"] + pl_tags,
                         fig_stamp=fig_stamp)

    # ------------------------------------------------------------------
    # Step 9: Extract embeddings as numpy arrays
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 9: Extracting embeddings")
    print("=" * 60)
    Z_train_std = get_embeddings_np(std_encoder, data.X_train)
    Z_test_std  = get_embeddings_np(std_encoder, data.X_test)
    print(f"[exp10] Standard: train={Z_train_std.shape}, test={Z_test_std.shape}")

    Z_train_l1  = get_embeddings_np(l1_encoder, data.X_train)
    Z_test_l1   = get_embeddings_np(l1_encoder, data.X_test)
    print(f"[exp10] L1:       train={Z_train_l1.shape},  test={Z_test_l1.shape}")

    Z_train_mrl = get_embeddings_np(mat_encoder, data.X_train)
    Z_test_mrl  = get_embeddings_np(mat_encoder, data.X_test)
    print(f"[exp10] MRL:      train={Z_train_mrl.shape}, test={Z_test_mrl.shape}")

    # For each PrefixLp model: extract embeddings then flip dimensions.
    # Flip puts the most informative dim (last, lightest penalty) at position 0
    # so the prefix sweep reads best-first, matching MRL's convention.
    pl_embeddings = {}   # {p: (Z_train_flipped, Z_test_flipped)}
    for p, (enc, _hd) in sorted(pl_models.items()):
        Ztr = get_embeddings_np(enc, data.X_train)
        Zte = get_embeddings_np(enc, data.X_test)
        Ztr = np.ascontiguousarray(Ztr[:, ::-1])
        Zte = np.ascontiguousarray(Zte[:,  ::-1])
        pl_embeddings[p] = (Ztr, Zte)
        print(f"[exp10] PrefixL{p}: train={Ztr.shape}, reversed (most informative first)")

    y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
    y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)

    # ------------------------------------------------------------------
    # Step 10: Linear classification accuracy (dense prefix sweep)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 10: Evaluating linear accuracy  (k=1..{})".format(cfg.embed_dim))
    print("=" * 60)

    std_lin = evaluate_prefix_linear(
        Z_train_std, Z_test_std, y_train_np, y_test_np,
        cfg.eval_prefixes, "standard", seed=cfg.seed,
    )
    l1_lin = evaluate_prefix_linear(
        Z_train_l1, Z_test_l1, y_train_np, y_test_np,
        cfg.eval_prefixes, "l1", seed=cfg.seed,
    )
    mat_lin = evaluate_prefix_linear(
        Z_train_mrl, Z_test_mrl, y_train_np, y_test_np,
        cfg.eval_prefixes, "mat", seed=cfg.seed,
    )
    pca_lin = evaluate_pca_linear(data, cfg, seed=cfg.seed)

    linear_results = {
        "Standard": std_lin,
        "L1":       l1_lin,
        "MRL":      mat_lin,
    }
    for p, (Ztr, Zte) in sorted(pl_embeddings.items()):
        linear_results[f"PrefixL{p} (rev)"] = evaluate_prefix_linear(
            Ztr, Zte, y_train_np, y_test_np,
            cfg.eval_prefixes, f"pl{p}", seed=cfg.seed,
        )
    linear_results["PCA"] = pca_lin   # PCA always last

    # ------------------------------------------------------------------
    # Step 11: 1-NN accuracy (dense prefix sweep)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 11: Evaluating 1-NN accuracy  (k=1..{})".format(cfg.embed_dim))
    print("=" * 60)

    std_1nn = evaluate_prefix_1nn(
        std_encoder, data, cfg.eval_prefixes, "standard",
        max_db_samples=max_1nn_db, seed=cfg.seed,
    )
    l1_1nn = evaluate_prefix_1nn(
        l1_encoder, data, cfg.eval_prefixes, "l1",
        max_db_samples=max_1nn_db, seed=cfg.seed,
    )
    mat_1nn = evaluate_prefix_1nn(
        mat_encoder, data, cfg.eval_prefixes, "mat",
        max_db_samples=max_1nn_db, seed=cfg.seed,
    )
    pca_1nn = evaluate_pca_1nn(data, cfg, max_db_samples=max_1nn_db)

    nn1_results = {
        "Standard": std_1nn,
        "L1":       l1_1nn,
        "MRL":      mat_1nn,
    }
    # PrefixLp 1-NN: use pre-flipped embeddings directly to preserve the flip.
    for p, (Ztr, Zte) in sorted(pl_embeddings.items()):
        label = f"PrefixL{p} (rev)"
        print(f"\n[exp10] 1-NN sweep for '{label}' ...")
        pl_1nn = {}
        for k in cfg.eval_prefixes:
            k_eff = min(k, Ztr.shape[1])
            acc = evaluate_1nn(
                Ztr[:, :k_eff], Zte[:, :k_eff],
                y_train_np, y_test_np,
                max_db_samples=max_1nn_db, seed=cfg.seed,
            )
            pl_1nn[k] = acc
            print(f"  k={k:>3}  1-NN={acc:.4f}")
        nn1_results[label] = pl_1nn
    nn1_results["PCA"] = pca_1nn   # PCA always last

    # ------------------------------------------------------------------
    # Step 12: Plots + results table
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 12: Saving plots and results")
    print("=" * 60)

    plot_all_curves(linear_results, nn1_results, cfg.eval_prefixes, run_dir,
                    cfg.l1_lambda, fig_stamp=fig_stamp)
    save_results_summary(linear_results, nn1_results, cfg.eval_prefixes, run_dir)

    # Compact stdout table (sample every 4 dims to keep output readable)
    sample_ks   = [k for k in cfg.eval_prefixes if k % 4 == 0 or k == 1]
    model_names = list(linear_results.keys())   # only models that were actually evaluated
    print(f"\n{'k':>4}  {'Model':<12}  {'Linear':>8}  {'1-NN':>8}")
    print("-" * 40)
    for k in sample_ks:
        for model_name in model_names:
            lin = linear_results[model_name].get(k, float("nan"))
            nn1 = nn1_results[model_name].get(k,    float("nan"))
            print(f"{k:>4}  {model_name:<12}  {lin:>8.4f}  {nn1:>8.4f}")
        print()

    # ------------------------------------------------------------------
    # Step 13: Runtime + code snapshot
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 13: Saving runtime and code snapshot")
    print("=" * 60)
    save_runtime(run_dir, time.time() - run_start)
    save_code_snapshot(run_dir)

    print(f"\n[exp10] Experiment 10 complete.")
    print(f"[exp10] All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
