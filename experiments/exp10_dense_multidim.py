"""
Script: experiments/exp10_dense_multidim.py
--------------------------------------------
Experiment 10 — Dense Prefix Sweep: MRL vs Standard vs L1 vs PCA (No FF).

Exp7 without Fixed-Feature models, evaluated at every dimension k = 1..embed_dim
(dense sweep). Produces smooth, continuous accuracy curves showing exactly where
each model's performance plateaus.

Key differences vs Exp7:
  - No FF models — focus is on prefix ordering, not capacity matching
  - Dense eval: k = 1, 2, ..., embed_dim  (not just powers of 2)
  - Supports --embed-dim flag for 8, 16, 32, 64
  - Linear x-axis on plots (not log scale) to show the continuous curve

Models trained:
  1. Standard  — plain CE loss on full embed_dim embedding
  2. L1        — CE + L1 regularization on embedding activations
  3. MRL (Mat) — CE summed at every prefix scale (Matryoshka loss)
  4. PCA       — analytical baseline (variance-ordered components)

Evaluation metrics at each k = 1..embed_dim:
  a. Linear classification accuracy  (logistic regression probe on k-dim embedding)
  b. 1-NN accuracy  (1-nearest-neighbor; train set as database, test set as queries)

Inputs:
  --fast          : smoke test — digits dataset, 5 epochs, subsampled 1-NN
  --embed-dim N   : override embed_dim (8, 16, 32, 64); eval_prefixes = [1..N]

Outputs (all in a new timestamped run folder):
  linear_accuracy_curve.png  : linear accuracy vs k — 4 model lines
  1nn_accuracy_curve.png     : 1-NN accuracy vs k — 4 model lines
  combined_comparison.png    : 2-panel (linear top, 1-NN bottom)
  training_curves.png        : loss vs epoch for Standard, L1, MRL (MANDATORY)
  results_summary.txt        : full table k x model x linear_acc x 1nn_acc
  experiment_description.log
  runtime.txt
  code_snapshot/

Usage:
    python experiments/exp10_dense_multidim.py                       # full run (MNIST, embed_dim=64)
    python experiments/exp10_dense_multidim.py --fast                # smoke test (digits, 5 epochs)
    python experiments/exp10_dense_multidim.py --embed-dim 8         # full run, embed_dim=8
    python experiments/exp10_dense_multidim.py --embed-dim 16        # full run, embed_dim=16
    python experiments/exp10_dense_multidim.py --embed-dim 32        # full run, embed_dim=32
    python experiments/exp10_dense_multidim.py --embed-dim 8 --fast  # smoke test at dim=8
    python tests/run_tests_exp10.py --fast                           # unit tests only
    python tests/run_tests_exp10.py                                  # unit tests + e2e smoke
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
from utility import create_run_dir, save_runtime, save_code_snapshot
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
MODEL_STYLES = {
    "Standard": ("steelblue",  "o-",  "Standard"),
    "L1":       ("orchid",     "D-",  "L1"),
    "MRL":      ("darkorange", "s-",  "MRL"),
    "PCA":      ("seagreen",   "x--", "PCA"),
}


def _single_accuracy_plot(results_dict, eval_prefixes, ylabel, title,
                           out_path, l1_lambda):
    """
    Plot accuracy vs prefix k (linear x-axis) and save.

    Args:
        results_dict  (dict)  : {model_name: {k: accuracy}}.
        eval_prefixes (list)  : x-axis values.
        ylabel        (str)   : y-axis label.
        title         (str)   : Plot title.
        out_path      (str)   : Where to save the PNG.
        l1_lambda     (float) : Annotates the L1 line label.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    for model_name, acc_dict in results_dict.items():
        color, style, label = MODEL_STYLES.get(model_name, ("gray", "x-", model_name))
        if model_name == "L1":
            label = f"L1 (lambda={l1_lambda})"
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


def plot_all_curves(linear_results, nn1_results, eval_prefixes, run_dir, l1_lambda):
    """
    Save three plots: linear accuracy, 1-NN accuracy, and combined 2-panel.

    Args:
        linear_results (dict)  : {model_name: {k: accuracy}} — linear probe.
        nn1_results    (dict)  : {model_name: {k: accuracy}} — 1-NN.
        eval_prefixes  (list)  : x-axis values (dense 1..embed_dim).
        run_dir        (str)   : Output directory.
        l1_lambda      (float) : Used in L1 legend label.
    """
    _single_accuracy_plot(
        linear_results, eval_prefixes,
        ylabel="Linear Classification Accuracy",
        title="Linear Accuracy vs Prefix k  (Standard vs L1 vs MRL vs PCA — dense)",
        out_path=os.path.join(run_dir, "linear_accuracy_curve.png"),
        l1_lambda=l1_lambda,
    )

    _single_accuracy_plot(
        nn1_results, eval_prefixes,
        ylabel="1-NN Accuracy",
        title="1-NN Accuracy vs Prefix k  (Standard vs L1 vs MRL vs PCA — dense)",
        out_path=os.path.join(run_dir, "1nn_accuracy_curve.png"),
        l1_lambda=l1_lambda,
    )

    # Combined 2-panel
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for model_name in linear_results:
        color, style, label = MODEL_STYLES.get(model_name, ("gray", "x-", model_name))
        if model_name == "L1":
            label = f"L1 (lambda={l1_lambda})"
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
    plt.savefig(os.path.join(run_dir, "combined_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("[exp10] Saved combined_comparison.png")


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
        help="Override embed_dim (8, 16, 32, 64); eval_prefixes = [1..N].",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 2: Config
    # ------------------------------------------------------------------
    if args.fast:
        cfg = ExpConfig(
            dataset="digits",
            embed_dim=16,
            hidden_dim=128,
            head_mode="shared_head",
            eval_prefixes=list(range(1, 17)),   # overridden below if --embed-dim
            epochs=5,
            patience=3,
            seed=42,
            l1_lambda=0.05,
            experiment_name="exp10_dense_multidim",
        )
        max_1nn_db = 500
    else:
        cfg = ExpConfig(
            dataset="mnist",
            embed_dim=64,
            hidden_dim=256,
            head_mode="shared_head",
            eval_prefixes=list(range(1, 65)),   # overridden below if --embed-dim
            epochs=20,
            patience=5,
            seed=42,
            l1_lambda=0.05,
            experiment_name="exp10_dense_multidim",
        )
        max_1nn_db = 10_000   # cap 1-NN database at 10k for speed on dense sweep

    # Apply --embed-dim override: always yields a dense [1..N] prefix list
    if args.embed_dim is not None:
        cfg.embed_dim     = args.embed_dim
        cfg.eval_prefixes = list(range(1, cfg.embed_dim + 1))
        print(f"[exp10] --embed-dim override: embed_dim={cfg.embed_dim}, "
              f"eval_prefixes=1..{cfg.embed_dim}")

    set_seeds(cfg.seed)

    # ------------------------------------------------------------------
    # Step 3: Setup output directory + save description
    # ------------------------------------------------------------------
    run_dir = create_run_dir()
    print(f"[exp10] Outputs will be saved to: {run_dir}\n")
    save_experiment_description(cfg, run_dir, args.fast)

    # ------------------------------------------------------------------
    # Step 4: Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Loading data")
    print("=" * 60)
    data = load_data(cfg)

    # ------------------------------------------------------------------
    # Step 5: Train Standard model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5: Training Standard model")
    print("=" * 60)
    std_encoder, std_head = train_single_model(
        cfg, data, run_dir, model_type="standard", model_tag="standard"
    )

    # ------------------------------------------------------------------
    # Step 6: Train L1-regularized model
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"STEP 6: Training L1 model  (lambda={cfg.l1_lambda})")
    print("=" * 60)
    l1_encoder, l1_head = train_single_model(
        cfg, data, run_dir, model_type="l1", model_tag="l1"
    )

    # ------------------------------------------------------------------
    # Step 7: Train MRL model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 7: Training MRL model")
    print("=" * 60)
    mat_encoder, mat_head = train_single_model(
        cfg, data, run_dir, model_type="matryoshka", model_tag="mat"
    )

    # ------------------------------------------------------------------
    # Step 8: Training curves (MANDATORY)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 8: Plotting training curves")
    print("=" * 60)
    plot_training_curves(run_dir, model_tags=["standard", "l1", "mat"])

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
        "PCA":      pca_lin,
    }

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
        "PCA":      pca_1nn,
    }

    # ------------------------------------------------------------------
    # Step 12: Plots + results table
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 12: Saving plots and results")
    print("=" * 60)

    plot_all_curves(linear_results, nn1_results, cfg.eval_prefixes, run_dir, cfg.l1_lambda)
    save_results_summary(linear_results, nn1_results, cfg.eval_prefixes, run_dir)

    # Compact stdout table (sample every 4 dims to keep output readable)
    sample_ks = [k for k in cfg.eval_prefixes if k % 4 == 0 or k == 1]
    print(f"\n{'k':>4}  {'Model':<12}  {'Linear':>8}  {'1-NN':>8}")
    print("-" * 40)
    for k in sample_ks:
        for model_name in ["Standard", "L1", "MRL", "PCA"]:
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
