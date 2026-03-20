"""
Script: training/trainer.py
----------------------------
Generic training loop shared by both the standard and Matryoshka models.

Handles:
  - Mini-batch iteration over train set
  - Validation loss tracking after each epoch
  - Early stopping (stops when val loss does not improve for cfg.patience epochs)
  - Best-model checkpointing (saves the weights that achieved lowest val loss)
  - Per-epoch logging (loss, accuracy) printed to console and written to a log file

The caller passes in the encoder, head, loss function, and optimiser — the
trainer does not care whether it is training a standard or MRL model.

Inputs:
    encoder   (nn.Module)  : MLPEncoder instance
    head      (nn.Module)  : SharedClassifier or MultiHeadClassifier
    criterion (nn.Module)  : StandardLoss or MatryoshkaLoss
    optimiser (Optimizer)  : e.g. torch.optim.Adam
    data      (DataSplit)  : train/val tensors from data/loader.py
    cfg       (ExpConfig)  : full experiment config
    run_dir   (str)        : path to the timestamped output folder for this run
    model_tag (str)        : short label used in log lines, e.g. 'standard' or 'mat'

Outputs:
    dict with keys:
        'train_losses' : list of per-epoch training losses
        'val_losses'   : list of per-epoch validation losses
        'best_epoch'   : epoch index (0-based) where best val loss was achieved
"""

import os
import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# tqdm is optional — fall back to plain range() / StreamHandler if not installed
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


# ==============================================================================
# Internal helpers
# ==============================================================================

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy from logits and ground-truth labels.

    Args:
        logits (torch.Tensor): Shape (N, n_classes) — raw class scores.
        labels (torch.Tensor): Shape (N,) — ground-truth class indices.

    Returns:
        float: Accuracy in [0, 1].
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def _setup_logger(run_dir: str, model_tag: str) -> logging.Logger:
    """
    Create a logger that writes to both the console and a log file.

    The log file is saved at: {run_dir}/{model_tag}_train.log

    Args:
        run_dir   (str): Path to the run output directory.
        model_tag (str): Short label for this model (e.g. 'standard', 'mat').

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(model_tag)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    # Console handler — route through _tqdm.write() so the progress bar is not stomped.
    # Falls back to a plain StreamHandler if tqdm is not installed.
    if _tqdm is not None:
        class TqdmHandler(logging.StreamHandler):
            def emit(self, record):
                _tqdm.write(self.format(record))
        ch = TqdmHandler()
    else:
        ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_path = os.path.join(run_dir, f"{model_tag}_train.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# ==============================================================================
# Main training function
# ==============================================================================

def train(
    encoder,
    head,
    criterion,
    optimiser,
    data,
    cfg,
    run_dir: str,
    model_tag: str,
) -> Dict:
    """
    Train encoder + head for cfg.epochs epochs with early stopping.

    Each epoch:
      1. Iterate over mini-batches of training data, compute loss, backprop.
      2. Evaluate on the validation set (no gradient).
      3. Log train loss, val loss, val accuracy.
      4. If val loss improved: save model weights as best checkpoint.
      5. If val loss has not improved for cfg.patience epochs: stop early.

    Args:
        encoder   (nn.Module) : MLPEncoder — maps input to embedding.
        head      (nn.Module) : Classifier head (shared or multi).
        criterion (nn.Module) : StandardLoss or MatryoshkaLoss.
        optimiser (Optimizer) : PyTorch optimiser (e.g. Adam).
        data      (DataSplit) : namedtuple with train/val tensors + metadata.
        cfg       (ExpConfig) : Experiment config.
        run_dir   (str)       : Directory to save checkpoints and logs.
        model_tag (str)       : Short label used in filenames and log lines.

    Returns:
        dict: {
            'train_losses' (List[float]): training loss per epoch,
            'val_losses'   (List[float]): validation loss per epoch,
            'best_epoch'   (int)        : epoch with lowest val loss (0-based),
        }
    """

    logger = _setup_logger(run_dir, model_tag)
    logger.info(f"=== Starting training: {model_tag} ===")
    logger.info(f"  epochs={cfg.epochs}, lr={cfg.lr}, batch={cfg.batch_size}, "
                f"patience={cfg.patience}")

    # ------------------------------------------------------------------
    # Build DataLoaders
    # ------------------------------------------------------------------
    train_ds = TensorDataset(data.X_train, data.y_train)
    val_ds   = TensorDataset(data.X_val,   data.y_val)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Set random seeds for reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(cfg.seed)

    # ------------------------------------------------------------------
    # Tracking variables
    # ------------------------------------------------------------------
    train_losses: List[float] = []
    val_losses:   List[float] = []
    best_val_loss  = float("inf")
    best_epoch     = 0
    epochs_no_improve = 0   # counter for early stopping

    # Paths for saving the best checkpoint
    enc_ckpt  = os.path.join(run_dir, f"{model_tag}_encoder_best.pt")
    head_ckpt = os.path.join(run_dir, f"{model_tag}_head_best.pt")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    # Use tqdm epoch bar if available; otherwise fall back to plain range().
    # No nested position= args — macOS Terminal.app does not render them correctly.
    epoch_iter = (
        _tqdm(range(cfg.epochs), desc=f"[{model_tag}]", unit="epoch", leave=True)
        if _tqdm is not None
        else range(cfg.epochs)
    )

    for epoch in epoch_iter:

        # ---- Train phase ----
        encoder.train()
        head.train()
        epoch_train_loss = 0.0

        n_batches = len(train_loader)
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            optimiser.zero_grad()

            embedding  = encoder(X_batch)
            loss       = criterion(embedding, y_batch, head)
            loss.backward()
            optimiser.step()

            epoch_train_loss += loss.item() * len(X_batch)

            # Update epoch bar every 10 batches so the terminal doesn't look frozen
            # during long epochs (e.g. MNIST). No nested bar — single bar stays clean.
            if _tqdm is not None and (batch_idx + 1) % 10 == 0:
                epoch_iter.set_postfix(
                    batch=f"{batch_idx+1}/{n_batches}",
                    loss=f"{loss.item():.4f}",
                )

        # Average over all training samples
        epoch_train_loss /= len(data.X_train)
        train_losses.append(epoch_train_loss)

        # ---- Validation phase ----
        encoder.eval()
        head.eval()
        epoch_val_loss = 0.0
        val_logits_all = []
        val_labels_all = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                embedding = encoder(X_batch)

                # For val loss, always use the full embedding regardless of mode
                # (this is just to track overall training progress)
                if cfg.head_mode == "shared_head":
                    logits = head(embedding)
                else:
                    # Mode B: use the largest prefix (= full embed_dim)
                    largest_k = max(cfg.eval_prefixes)
                    logits = head(embedding, largest_k)

                loss = criterion(embedding, y_batch, head)
                epoch_val_loss += loss.item() * len(X_batch)

                val_logits_all.append(logits)
                val_labels_all.append(y_batch)

        epoch_val_loss /= len(data.X_val)
        val_losses.append(epoch_val_loss)

        # Compute val accuracy on full embedding
        all_logits = torch.cat(val_logits_all)
        all_labels = torch.cat(val_labels_all)
        val_acc    = _accuracy(all_logits, all_labels)

        # ---- Logging ----
        logger.info(
            f"Epoch {epoch+1:>3}/{cfg.epochs}  "
            f"train_loss={epoch_train_loss:.4f}  "
            f"val_loss={epoch_val_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )
        # Update tqdm bar with live metrics (only when tqdm is active)
        if _tqdm is not None:
            epoch_iter.set_postfix(
                train_loss=f"{epoch_train_loss:.4f}",
                val_loss=f"{epoch_val_loss:.4f}",
                val_acc=f"{val_acc:.4f}",
            )

        # ---- Best model checkpoint ----
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch    = epoch
            epochs_no_improve = 0

            torch.save(encoder.state_dict(), enc_ckpt)
            torch.save(head.state_dict(),    head_ckpt)
            logger.info(f"  -> New best val_loss={best_val_loss:.4f}  checkpoint saved.")

        else:
            epochs_no_improve += 1

        # ---- Early stopping ----
        if cfg.patience is not None and epochs_no_improve >= cfg.patience:
            logger.info(
                f"Early stopping at epoch {epoch+1} "
                f"(no improvement for {cfg.patience} epochs)."
            )
            break

    # ------------------------------------------------------------------
    # Load best weights back into the models before returning
    # ------------------------------------------------------------------
    encoder.load_state_dict(torch.load(enc_ckpt, weights_only=True))
    head.load_state_dict(torch.load(head_ckpt,    weights_only=True))
    logger.info(f"Best checkpoint loaded from epoch {best_epoch+1}.")
    logger.info(f"=== Training complete: {model_tag} ===\n")

    return {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "best_epoch":   best_epoch,
    }


# ==============================================================================
# Quick sanity check
# ==============================================================================

if __name__ == "__main__":
    import sys, tempfile
    sys.path.insert(0, "..")

    from config import ExpConfig
    from data.loader import load_data
    from models.encoder import MLPEncoder
    from models.heads import build_head
    from losses.mat_loss import build_loss

    # Use a small fast config for the test
    cfg = ExpConfig(
        dataset="digits",
        embed_dim=16,
        eval_prefixes=[1, 2, 4, 8, 16],
        epochs=5,
        patience=3,
        batch_size=32,
    )

    print("--- Loading data ---")
    data = load_data(cfg)

    # Use a temp dir so the test does not pollute results/
    with tempfile.TemporaryDirectory() as run_dir:
        print(f"\n--- Training standard model (run_dir={run_dir}) ---")
        encoder = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
        head    = build_head(cfg, data.n_classes)
        loss_fn = build_loss(cfg, "standard")
        opt     = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()),
            lr=cfg.lr,
        )

        history = train(encoder, head, loss_fn, opt, data, cfg, run_dir, "standard")

        assert len(history["train_losses"]) <= cfg.epochs
        assert len(history["val_losses"])   <= cfg.epochs
        assert 0 <= history["best_epoch"] < cfg.epochs
        print(f"\n  train_losses: {[f'{v:.3f}' for v in history['train_losses']]}")
        print(f"  val_losses:   {[f'{v:.3f}' for v in history['val_losses']]}")
        print(f"  best_epoch:   {history['best_epoch']}")
        print("  PASSED")

        print("\n--- Training matryoshka model ---")
        encoder_m = MLPEncoder(data.input_dim, cfg.hidden_dim, cfg.embed_dim)
        head_m    = build_head(cfg, data.n_classes)
        loss_m    = build_loss(cfg, "matryoshka")
        opt_m     = torch.optim.Adam(
            list(encoder_m.parameters()) + list(head_m.parameters()),
            lr=cfg.lr,
        )

        history_m = train(encoder_m, head_m, loss_m, opt_m, data, cfg, run_dir, "mat")
        print(f"\n  train_losses: {[f'{v:.3f}' for v in history_m['train_losses']]}")
        print("  PASSED")

    print("\nAll trainer checks passed.")
