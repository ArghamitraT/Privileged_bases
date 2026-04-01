"""
Script: experiments/exp7_mrl_vs_ff.py
---------------------------------------
Experiment 7 — MRL vs Fixed-Feature (FF) vs L1-Regularized Models.

Replicates the core comparison from the Matryoshka Representation Learning (MRL)
paper, extended with an L1-regularization ablation:

Models trained:
  1. Standard  — plain CE loss on full embed_dim embedding
  2. L1        — CE loss + L1 regularization on embedding activations
  3. MRL (Mat) — CE summed at every prefix scale (Matryoshka loss)
  4. FF-k      — one dedicated model per k, trained with embed_dim=k
  5. PCA       — analytical baseline (variance-ordered components)

Evaluation metrics at each prefix k:
  a. Linear classification accuracy  (logistic regression probe on k-dim embedding)
  b. 1-NN accuracy  (1-nearest-neighbor; train set as database, test set as queries)

Key hypothesis:
  - MRL >= FF at every k  (replicating paper claim for linear accuracy)
  - L1 ~= Standard at small k  (sparsity without ordering doesn't help prefix eval)
  - MRL beats L1 at small k  -> proves ORDERING is essential, not just sparsity

Inputs:
  --fast       : smoke test — digits dataset, 5 epochs, subsampled 1-NN
  --use-exp1   : load MRL weights from canonical exp1 run instead of training

Outputs (all in a new timestamped run folder):
  linear_accuracy_curve.png   : linear accuracy vs k — 5 model lines
  1nn_accuracy_curve.png      : 1-NN accuracy vs k — 5 model lines
  combined_comparison.png     : 2-panel (linear top, 1-NN bottom)
  training_curves.png         : loss vs epoch for Standard, L1, MRL (MANDATORY)
  results_summary.txt         : full table k x model x linear_acc x 1nn_acc
  experiment_description.log
  runtime.txt
  code_snapshot/

Usage:
    python experiments/exp7_mrl_vs_ff.py --fast       # smoke test (digits, 5 epochs)
    python experiments/exp7_mrl_vs_ff.py              # full run (MNIST, 20 epochs)
    python experiments/exp7_mrl_vs_ff.py --use-exp1   # load MRL from exp1, train rest
    python tests/run_tests_exp7.py --fast             # unit tests only
    python tests/run_tests_exp7.py                    # unit tests + e2e smoke
"""

import os
import sys
import time
import argparse
import dataclasses

# Cap BLAS thread count before numpy/scipy imports to prevent segfaults on macOS
# (scipy/sklearn imports can conflict with OpenBLAS thread pool during MNIST load).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA as SklearnPCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ExpConfig
from utility import create_run_dir, save_runtime, save_code_snapshot, get_path
from data.loader import load_data
from models.encoder import MLPEncoder
from models.heads import build_head
from losses.mat_loss import build_loss
from training.trainer import train
from evaluation.prefix_eval import evaluate_prefix_sweep, evaluate_pca_baseline


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
    print(f"[exp7] Random seeds set to {seed}")


# ==============================================================================
# Experiment description log
# ==============================================================================

def save_experiment_description(cfg, run_dir, mrl_weights_dir, fast):
    """
    Write a human-readable log describing this experiment run.

    Args:
        cfg             (ExpConfig): Experiment configuration.
        run_dir         (str)      : Output directory for this run.
        mrl_weights_dir (str|None) : Path to loaded MRL weights, or None.
        fast            (bool)     : Whether fast/smoke mode is active.
    """
    log_path = os.path.join(run_dir, "experiment_description.log")
    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT 7 — MRL vs Fixed-Feature vs L1-Regularized Models\n")
        f.write("=" * 70 + "\n\n")

        f.write("WHAT THIS EXPERIMENT DOES\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Trains five model families and evaluates each at every prefix k:\n"
            "  Standard : plain CE loss on full embed_dim embedding\n"
            "  L1       : CE + L1 penalty on embedding activations\n"
            "  MRL      : CE summed at every prefix scale (Matryoshka loss)\n"
            "  FF-k     : one dedicated model per k, trained with embed_dim=k\n"
            "  PCA      : analytical baseline (variance-ordered components)\n\n"
            "Evaluation metrics:\n"
            "  Linear accuracy : logistic regression probe on k-dim embedding\n"
            "  1-NN accuracy   : 1-nearest-neighbor (train=database, test=query)\n\n"
        )

        f.write("WHY WE ARE RUNNING IT\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Replicating the MRL paper's core comparison (MRL vs FF) on MNIST.\n"
            "The L1 ablation isolates whether SPARSITY alone helps prefix eval,\n"
            "or whether the ORDERING property of MRL is what matters.\n"
            "If L1 ~= Standard at small k, ordering is essential, not sparsity.\n\n"
        )

        f.write("EXPECTED OUTCOME\n")
        f.write("-" * 40 + "\n")
        f.write(
            "  MRL >= FF at every k (linear accuracy — replicating paper)\n"
            "  MRL up to 2% better than FF at low k (1-NN accuracy)\n"
            "  L1 ~= Standard at small k (sparsity without ordering doesn't help)\n"
            "  PCA intermediate — variance-ordered but not task-aware\n\n"
        )

        f.write("MRL WEIGHTS SOURCE\n")
        f.write("-" * 40 + "\n")
        if mrl_weights_dir:
            f.write(f"  Loaded from: {mrl_weights_dir}\n\n")
        else:
            f.write("  Trained from scratch in this run.\n\n")
        f.write(f"  Fast mode: {fast}\n\n")

        f.write("FULL CONFIG\n")
        f.write("-" * 40 + "\n")
        for field in dataclasses.fields(cfg):
            f.write(f"  {field.name:<20} = {getattr(cfg, field.name)}\n")
        f.write("\n")

    print(f"[exp7] Experiment description saved to {log_path}")


# ==============================================================================
# Training helpers
# ==============================================================================

def train_single_model(cfg, data, run_dir, model_type, model_tag):
    """
    Build, train, and return a single encoder+head model with best weights loaded.

    Args:
        cfg        (ExpConfig) : Experiment configuration.
        data       (DataSplit) : Train/val/test splits.
        run_dir    (str)       : Where to save weights and logs.
        model_type (str)       : 'standard', 'matryoshka', or 'l1'.
        model_tag  (str)       : Filename prefix (e.g. 'standard', 'l1', 'mat').

    Returns:
        Tuple (encoder, head): both in eval mode with best weights loaded.
    """
    encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
    head    = build_head(cfg, data.n_classes)
    loss_fn = build_loss(cfg, model_type)
    opt     = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    train(encoder, head, loss_fn, opt, data, cfg, run_dir, model_tag=model_tag)

    # Reload best checkpoint saved by trainer
    enc_path  = os.path.join(run_dir, f"{model_tag}_encoder_best.pt")
    head_path = os.path.join(run_dir, f"{model_tag}_head_best.pt")
    encoder.load_state_dict(torch.load(enc_path,  map_location="cpu"))
    head.load_state_dict(   torch.load(head_path, map_location="cpu"))
    encoder.eval()
    head.eval()
    return encoder, head


def train_ff_models(cfg_base, eval_prefixes, data, run_dir):
    """
    Train one Fixed-Feature (FF) model per k in eval_prefixes.

    Each FF-k model has embed_dim=k, trained with standard CE loss.
    Same lr, epochs, patience, and seed as cfg_base for a fair comparison.

    Args:
        cfg_base      (ExpConfig) : Base config (dataset, lr, epochs, etc.).
        eval_prefixes (List[int]) : Prefix sizes; one FF model is trained per k.
        data          (DataSplit) : Train/val/test splits.
        run_dir       (str)       : Where to save weights and logs.

    Returns:
        dict: {k: (encoder, head)} — all in eval mode with best weights loaded.
    """
    ff_models = {}
    for k in eval_prefixes:
        print(f"\n[exp7] Training FF-{k} model (embed_dim={k}) ...")
        # Config for this FF model: embed_dim=k, eval_prefixes=[k] only
        cfg_ff = ExpConfig(
            dataset=cfg_base.dataset,
            embed_dim=k,
            hidden_dim=cfg_base.hidden_dim,
            head_mode="shared_head",
            eval_prefixes=[k],
            lr=cfg_base.lr,
            epochs=cfg_base.epochs,
            batch_size=cfg_base.batch_size,
            patience=cfg_base.patience,
            weight_decay=cfg_base.weight_decay,
            seed=cfg_base.seed,
            l1_lambda=cfg_base.l1_lambda,
            experiment_name=cfg_base.experiment_name,
        )
        tag     = f"ff_k{k}"
        encoder = MLPEncoder(data.input_dim, cfg_ff.hidden_dim, cfg_ff.embed_dim)
        head    = build_head(cfg_ff, data.n_classes)
        loss_fn = build_loss(cfg_ff, "standard")
        opt     = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()),
            lr=cfg_ff.lr, weight_decay=cfg_ff.weight_decay,
        )
        train(encoder, head, loss_fn, opt, data, cfg_ff, run_dir, model_tag=tag)

        enc_path  = os.path.join(run_dir, f"{tag}_encoder_best.pt")
        head_path = os.path.join(run_dir, f"{tag}_head_best.pt")
        encoder.load_state_dict(torch.load(enc_path,  map_location="cpu"))
        head.load_state_dict(   torch.load(head_path, map_location="cpu"))
        encoder.eval()
        head.eval()
        ff_models[k] = (encoder, head)

    print(f"\n[exp7] All {len(eval_prefixes)} FF models trained.\n")
    return ff_models


# ==============================================================================
# Embedding extraction
# ==============================================================================

def get_embeddings_np(encoder, X_tensor):
    """
    Run encoder forward pass and return a numpy array.

    Args:
        encoder  (nn.Module)    : Trained MLPEncoder in eval mode.
        X_tensor (torch.Tensor) : Input tensor shape (n, input_dim).

    Returns:
        np.ndarray: Shape (n, embed_dim), L2-normalised.
    """
    encoder.eval()
    with torch.no_grad():
        emb = encoder(X_tensor)
    return np.array(emb.cpu().tolist(), dtype=np.float32)


# ==============================================================================
# 1-NN evaluation helpers
# ==============================================================================

def evaluate_1nn(train_emb, test_emb, y_train, y_test,
                 max_db_samples=None, seed=42):
    """
    Compute 1-nearest-neighbor accuracy on precomputed embeddings.

    Optionally subsamples the training-set database for speed. Embeddings are
    L2-normalised so L2 distance is equivalent to cosine distance ranking.

    Args:
        train_emb      (np.ndarray): Shape (n_train, d) — database.
        test_emb       (np.ndarray): Shape (n_test,  d) — queries.
        y_train        (np.ndarray): Shape (n_train,) — database labels.
        y_test         (np.ndarray): Shape (n_test,)  — query labels.
        max_db_samples (int|None)  : Cap on database size (None = all samples).
        seed           (int)       : Random seed for subsampling.

    Returns:
        float: 1-NN accuracy in [0, 1].
    """
    # Subsample database if requested (saves memory and time on large datasets)
    if max_db_samples is not None and len(train_emb) > max_db_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_emb), max_db_samples, replace=False)
        train_emb = train_emb[idx]
        y_train   = y_train[idx]

    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean", algorithm="auto")
    knn.fit(train_emb, y_train)
    return float(knn.score(test_emb, y_test))


def evaluate_prefix_1nn(encoder, data, eval_prefixes, model_tag,
                        max_db_samples=None, seed=42):
    """
    1-NN prefix sweep for a neural model (Standard, L1, or MRL).

    Computes full embed_dim embeddings once, then slices first k dims at each k.

    Args:
        encoder        (nn.Module)  : Trained encoder.
        data           (DataSplit)  : Train/test splits.
        eval_prefixes  (List[int])  : Prefix sizes to sweep.
        model_tag      (str)        : Label for print output.
        max_db_samples (int|None)   : Database subsample limit.
        seed           (int)        : Random seed.

    Returns:
        Dict[int, float]: {k: 1nn_accuracy}.
    """
    print(f"\n[exp7] 1-NN sweep for '{model_tag}' ...")

    train_full = get_embeddings_np(encoder, data.X_train)
    test_full  = get_embeddings_np(encoder, data.X_test)
    y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
    y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)

    results = {}
    for k in eval_prefixes:
        k_eff = min(k, train_full.shape[1])
        acc = evaluate_1nn(
            train_full[:, :k_eff], test_full[:, :k_eff],
            y_train_np, y_test_np,
            max_db_samples=max_db_samples, seed=seed,
        )
        results[k] = acc
        print(f"  k={k:>3}  1-NN={acc:.4f}")

    return results


def evaluate_ff_linear(ff_models, data, eval_prefixes):
    """
    Linear (head) accuracy for each FF-k model on its full k-dim embedding.

    No prefix truncation needed — FF-k has embed_dim=k by design.

    Args:
        ff_models     (dict)      : {k: (encoder, head)}.
        data          (DataSplit) : Test split.
        eval_prefixes (List[int]) : Prefix sizes.

    Returns:
        Dict[int, float]: {k: accuracy}.
    """
    print("\n[exp7] Linear accuracy for FF models ...")
    results = {}
    for k in eval_prefixes:
        encoder, head = ff_models[k]
        encoder.eval()
        head.eval()
        with torch.no_grad():
            emb    = encoder(data.X_test)
            logits = head(emb)
            preds  = logits.argmax(dim=1)
            acc    = (preds == data.y_test).float().mean().item()
        results[k] = acc
        print(f"  k={k:>3}  FF linear={acc:.4f}")
    return results


def evaluate_ff_1nn(ff_models, data, eval_prefixes, max_db_samples=None, seed=42):
    """
    1-NN accuracy for each FF-k model on its full k-dim embedding.

    Args:
        ff_models      (dict)      : {k: (encoder, head)}.
        data           (DataSplit) : Train/test splits.
        eval_prefixes  (List[int]) : Prefix sizes.
        max_db_samples (int|None)  : Database subsample limit.
        seed           (int)       : Random seed.

    Returns:
        Dict[int, float]: {k: 1nn_accuracy}.
    """
    print("\n[exp7] 1-NN for FF models ...")
    y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
    y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)

    results = {}
    for k in eval_prefixes:
        encoder, _ = ff_models[k]
        train_emb  = get_embeddings_np(encoder, data.X_train)
        test_emb   = get_embeddings_np(encoder, data.X_test)
        acc = evaluate_1nn(
            train_emb, test_emb, y_train_np, y_test_np,
            max_db_samples=max_db_samples, seed=seed,
        )
        results[k] = acc
        print(f"  k={k:>3}  FF 1-NN={acc:.4f}")
    return results


def evaluate_pca_1nn(data, cfg, max_db_samples=None):
    """
    1-NN accuracy for PCA baseline at each prefix k.

    Fits PCA on training data, then slices first k components at each k.

    Args:
        data           (DataSplit) : Train/test splits.
        cfg            (ExpConfig) : Uses embed_dim, eval_prefixes, seed.
        max_db_samples (int|None)  : Database subsample limit.

    Returns:
        Dict[int, float]: {k: 1nn_accuracy}.
    """
    print("\n[exp7] 1-NN for PCA baseline ...")

    X_train_np = np.array(data.X_train.tolist(), dtype=np.float32)
    X_test_np  = np.array(data.X_test.tolist(),  dtype=np.float32)
    y_train_np = np.array(data.y_train.tolist(), dtype=np.int64)
    y_test_np  = np.array(data.y_test.tolist(),  dtype=np.int64)

    n_comp = min(cfg.embed_dim, X_train_np.shape[0], X_train_np.shape[1])
    pca    = SklearnPCA(n_components=n_comp, random_state=cfg.seed)
    Z_train = pca.fit_transform(X_train_np)
    Z_test  = pca.transform(X_test_np)

    results = {}
    for k in cfg.eval_prefixes:
        k_eff = min(k, n_comp)
        acc = evaluate_1nn(
            Z_train[:, :k_eff], Z_test[:, :k_eff],
            y_train_np, y_test_np,
            max_db_samples=max_db_samples, seed=cfg.seed,
        )
        results[k] = acc
        print(f"  k={k:>3}  PCA 1-NN={acc:.4f}")
    return results


# ==============================================================================
# Plotting
# ==============================================================================

# Consistent colour/marker/label per model across all plots
MODEL_STYLES = {
    "Standard": ("steelblue",  "o-",  "Standard"),
    "L1":       ("orchid",     "D-",  "L1"),
    "MRL":      ("darkorange", "s-",  "MRL"),
    "FF":       ("firebrick",  "^--", "FF (dedicated)"),
    "PCA":      ("seagreen",   "x--", "PCA"),
}


def _single_accuracy_plot(results_dict, eval_prefixes, ylabel, title,
                          out_path, l1_lambda):
    """
    Plot accuracy vs prefix k for all models on one axes and save.

    Args:
        results_dict  (dict)  : {model_name: {k: accuracy}}.
        eval_prefixes (list)  : x-axis values.
        ylabel        (str)   : y-axis label.
        title         (str)   : Plot title.
        out_path      (str)   : Where to save the PNG.
        l1_lambda     (float) : Annotates the L1 line label.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, acc_dict in results_dict.items():
        color, style, label = MODEL_STYLES.get(model_name, ("gray", "x-", model_name))
        if model_name == "L1":
            label = f"L1 (lambda={l1_lambda})"
        accs = [acc_dict.get(k, float("nan")) for k in eval_prefixes]
        ax.plot(eval_prefixes, accs, style, color=color,
                label=label, linewidth=2, markersize=7)

    ax.set_xscale("log", base=2)
    ax.set_xticks(eval_prefixes)
    ax.set_xticklabels([str(k) for k in eval_prefixes])
    ax.set_xlabel("Prefix size k  (embedding dimensions used)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp7] Saved {os.path.basename(out_path)}")


def plot_all_curves(linear_results, nn1_results, eval_prefixes, run_dir, l1_lambda):
    """
    Save three plots: linear accuracy, 1-NN accuracy, and combined 2-panel.

    Args:
        linear_results (dict)  : {model_name: {k: accuracy}} — linear probe.
        nn1_results    (dict)  : {model_name: {k: accuracy}} — 1-NN.
        eval_prefixes  (list)  : x-axis values.
        run_dir        (str)   : Output directory.
        l1_lambda      (float) : Used in L1 legend label.
    """
    _single_accuracy_plot(
        linear_results, eval_prefixes,
        ylabel="Linear Classification Accuracy",
        title="Linear Accuracy vs Prefix k  (MRL vs FF vs L1 vs Standard vs PCA)",
        out_path=os.path.join(run_dir, "linear_accuracy_curve.png"),
        l1_lambda=l1_lambda,
    )

    _single_accuracy_plot(
        nn1_results, eval_prefixes,
        ylabel="1-NN Accuracy",
        title="1-NN Accuracy vs Prefix k  (MRL vs FF vs L1 vs Standard vs PCA)",
        out_path=os.path.join(run_dir, "1nn_accuracy_curve.png"),
        l1_lambda=l1_lambda,
    )

    # Combined 2-panel: linear accuracy (top) + 1-NN accuracy (bottom)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    for model_name in linear_results:
        color, style, label = MODEL_STYLES.get(model_name, ("gray", "x-", model_name))
        if model_name == "L1":
            label = f"L1 (lambda={l1_lambda})"
        lin_accs = [linear_results[model_name].get(k, float("nan")) for k in eval_prefixes]
        nn1_accs = [nn1_results[model_name].get(k,    float("nan")) for k in eval_prefixes]
        ax_top.plot(eval_prefixes, lin_accs, style, color=color,
                    label=label, linewidth=2, markersize=7)
        ax_bot.plot(eval_prefixes, nn1_accs, style, color=color,
                    label=label, linewidth=2, markersize=7)

    for ax in (ax_top, ax_bot):
        ax.set_xscale("log", base=2)
        ax.set_xticks(eval_prefixes)
        ax.set_xticklabels([str(k) for k in eval_prefixes])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)

    ax_top.set_ylabel("Linear Accuracy", fontsize=12)
    ax_top.set_title("Linear Classification Accuracy vs Prefix k", fontsize=12)
    ax_bot.set_xlabel("Prefix size k  (embedding dimensions used)", fontsize=12)
    ax_bot.set_ylabel("1-NN Accuracy", fontsize=12)
    ax_bot.set_title("1-NN Accuracy vs Prefix k", fontsize=12)
    fig.suptitle("MRL vs FF vs L1 vs Standard vs PCA", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "combined_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("[exp7] Saved combined_comparison.png")


def plot_training_curves(run_dir, model_tags):
    """
    Parse training log files and plot train/val loss vs epoch.

    Reads {model_tag}_train.log files written by trainer.py. Skips any tag
    whose log file is missing or contains no parseable loss lines.

    Args:
        run_dir    (str)       : Output directory containing log files.
        model_tags (List[str]) : Tags to attempt to read (e.g. ['standard','l1','mat']).
    """
    histories = {}
    for tag in model_tags:
        log_path = os.path.join(run_dir, f"{tag}_train.log")
        if not os.path.isfile(log_path):
            continue
        train_losses, val_losses = [], []
        with open(log_path) as f:
            for line in f:
                # Trainer writes lines like:
                # "epoch=1  train_loss=2.3102  val_loss=2.2948  ..."
                if "train_loss=" in line and "val_loss=" in line:
                    try:
                        tl = float(line.split("train_loss=")[1].split()[0])
                        vl = float(line.split("val_loss=")[1].split()[0])
                        train_losses.append(tl)
                        val_losses.append(vl)
                    except (IndexError, ValueError):
                        pass
        if train_losses:
            histories[tag] = {"train_losses": train_losses, "val_losses": val_losses}

    if not histories:
        print("[exp7] No training log data found — skipping training_curves.png")
        return

    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (tag, hist) in zip(axes, histories.items()):
        epochs = range(1, len(hist["train_losses"]) + 1)
        ax.plot(epochs, hist["train_losses"], label="Train", linewidth=2)
        ax.plot(epochs, hist["val_losses"],   label="Val",   linewidth=2, linestyle="--")
        ax.set_title(tag, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    fig.suptitle("Training Curves  —  Standard / L1 / MRL", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("[exp7] Saved training_curves.png")


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
        f.write("EXPERIMENT 7 — MRL vs FF vs L1 Results\n")
        f.write("=" * 68 + "\n\n")
        header = f"{'k':>4}  {'Model':<12}  {'Linear Acc':>12}  {'1-NN Acc':>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for k in eval_prefixes:
            for model_name in model_names:
                lin = linear_results[model_name].get(k, float("nan"))
                nn1 = nn1_results[model_name].get(k,    float("nan"))
                f.write(f"{k:>4}  {model_name:<12}  {lin:>12.4f}  {nn1:>10.4f}\n")
            f.write("\n")

    print(f"[exp7] Results summary saved to {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """
    Orchestrate Experiment 7:
      - Parse args, configure, create run dir
      - Load data
      - Train Standard, L1, MRL models (or load MRL from --use-exp1)
      - Train FF-k models (one per prefix k)
      - Evaluate linear accuracy and 1-NN accuracy for all models
      - Plot and save all outputs
    """
    run_start = time.time()

    # ------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Experiment 7 — MRL vs FF vs L1")
    parser.add_argument(
        "--use-exp1", action="store_true",
        help="Load MRL weights from canonical exp1 run instead of training from scratch.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Smoke test: digits dataset, 5 epochs, subsampled 1-NN database.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    if args.fast:
        cfg = ExpConfig(
            dataset="digits",
            embed_dim=16,
            hidden_dim=128,
            head_mode="shared_head",
            eval_prefixes=[1, 2, 4, 8, 16],
            epochs=5,
            patience=3,
            seed=42,
            l1_lambda=0.05,
            experiment_name="exp7_mrl_vs_ff",
        )
        max_1nn_db = 500
    else:
        cfg = ExpConfig(
            dataset="mnist",
            embed_dim=64,
            hidden_dim=256,
            head_mode="shared_head",
            eval_prefixes=[1, 2, 4, 8, 16, 32, 64],
            epochs=20,
            patience=5,
            seed=42,
            l1_lambda=0.05,
            experiment_name="exp7_mrl_vs_ff",
        )
        max_1nn_db = None   # use full 56k training set for 1-NN

    set_seeds(cfg.seed)

    # ------------------------------------------------------------------
    # Resolve MRL weights dir for --use-exp1
    # ------------------------------------------------------------------
    mrl_weights_dir = None
    if args.use_exp1 and not args.fast:
        results_base    = get_path("files/results")
        exp1_folder     = "exprmnt_2026_03_08__16_36_30"
        mrl_weights_dir = os.path.join(results_base, exp1_folder)
        print(f"[exp7] --use-exp1: MRL weights from {mrl_weights_dir}")

    # ------------------------------------------------------------------
    # Setup output directory
    # ------------------------------------------------------------------
    run_dir = create_run_dir()
    print(f"[exp7] Outputs will be saved to: {run_dir}\n")
    save_experiment_description(cfg, run_dir, mrl_weights_dir, args.fast)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    data = load_data(cfg)

    # ------------------------------------------------------------------
    # Step 2: Train Standard model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: Training Standard model")
    print("=" * 60)
    std_encoder, std_head = train_single_model(
        cfg, data, run_dir, model_type="standard", model_tag="standard"
    )

    # ------------------------------------------------------------------
    # Step 3: Train L1-regularized model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Training L1 model  (lambda={})".format(cfg.l1_lambda))
    print("=" * 60)
    l1_encoder, l1_head = train_single_model(
        cfg, data, run_dir, model_type="l1", model_tag="l1"
    )

    # ------------------------------------------------------------------
    # Step 4: Train or load MRL model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: MRL model")
    print("=" * 60)

    if mrl_weights_dir:
        print(f"[exp7] Loading MRL weights from {mrl_weights_dir}")
        mat_encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
        mat_head    = build_head(cfg, data.n_classes)
        mat_encoder.load_state_dict(
            torch.load(os.path.join(mrl_weights_dir, "mat_encoder_best.pt"),
                       map_location="cpu")
        )
        mat_head.load_state_dict(
            torch.load(os.path.join(mrl_weights_dir, "mat_head_best.pt"),
                       map_location="cpu")
        )
        mat_encoder.eval()
        mat_head.eval()
        print("[exp7] MRL weights loaded.\n")
    else:
        mat_encoder, mat_head = train_single_model(
            cfg, data, run_dir, model_type="matryoshka", model_tag="mat"
        )

    # ------------------------------------------------------------------
    # Step 5: Train FF-k models (one per prefix)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5: Training FF models  ({} models)".format(len(cfg.eval_prefixes)))
    print("=" * 60)
    ff_models = train_ff_models(cfg, cfg.eval_prefixes, data, run_dir)

    # ------------------------------------------------------------------
    # Step 6: Training curves (Standard, L1, MRL)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 6: Plotting training curves")
    print("=" * 60)
    plot_training_curves(run_dir, model_tags=["standard", "l1", "mat"])

    # ------------------------------------------------------------------
    # Step 7: Linear classification accuracy
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 7: Evaluating linear accuracy")
    print("=" * 60)

    std_lin = evaluate_prefix_sweep(std_encoder, std_head, data, cfg, "standard")
    l1_lin  = evaluate_prefix_sweep(l1_encoder,  l1_head,  data, cfg, "l1")
    mat_lin = evaluate_prefix_sweep(mat_encoder, mat_head, data, cfg, "mat")
    ff_lin  = evaluate_ff_linear(ff_models, data, cfg.eval_prefixes)
    pca_lin = evaluate_pca_baseline(data, cfg)

    linear_results = {
        "Standard": std_lin,
        "L1":       l1_lin,
        "MRL":      mat_lin,
        "FF":       ff_lin,
        "PCA":      pca_lin,
    }

    # ------------------------------------------------------------------
    # Step 8: 1-NN accuracy
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 8: Evaluating 1-NN accuracy")
    print("=" * 60)

    std_1nn = evaluate_prefix_1nn(
        std_encoder, data, cfg.eval_prefixes, "standard",
        max_db_samples=max_1nn_db, seed=cfg.seed,
    )
    l1_1nn  = evaluate_prefix_1nn(
        l1_encoder,  data, cfg.eval_prefixes, "l1",
        max_db_samples=max_1nn_db, seed=cfg.seed,
    )
    mat_1nn = evaluate_prefix_1nn(
        mat_encoder, data, cfg.eval_prefixes, "mat",
        max_db_samples=max_1nn_db, seed=cfg.seed,
    )
    ff_1nn  = evaluate_ff_1nn(
        ff_models, data, cfg.eval_prefixes,
        max_db_samples=max_1nn_db, seed=cfg.seed,
    )
    pca_1nn = evaluate_pca_1nn(data, cfg, max_db_samples=max_1nn_db)

    nn1_results = {
        "Standard": std_1nn,
        "L1":       l1_1nn,
        "MRL":      mat_1nn,
        "FF":       ff_1nn,
        "PCA":      pca_1nn,
    }

    # ------------------------------------------------------------------
    # Step 9: Plots + results table
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 9: Saving plots and results")
    print("=" * 60)

    plot_all_curves(linear_results, nn1_results, cfg.eval_prefixes, run_dir, cfg.l1_lambda)
    save_results_summary(linear_results, nn1_results, cfg.eval_prefixes, run_dir)

    # Print compact stdout table
    print(f"\n{'k':>4}  {'Model':<12}  {'Linear':>8}  {'1-NN':>8}")
    print("-" * 40)
    for k in cfg.eval_prefixes:
        for model_name in ["Standard", "L1", "MRL", "FF", "PCA"]:
            lin = linear_results[model_name].get(k, float("nan"))
            nn1 = nn1_results[model_name].get(k,    float("nan"))
            print(f"{k:>4}  {model_name:<12}  {lin:>8.4f}  {nn1:>8.4f}")
        print()

    # ------------------------------------------------------------------
    # Step 10: Runtime + code snapshot
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 10: Saving runtime and code snapshot")
    print("=" * 60)
    save_runtime(run_dir, time.time() - run_start)
    save_code_snapshot(run_dir)

    print(f"\n[exp7] Experiment 7 complete.")
    print(f"[exp7] All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
