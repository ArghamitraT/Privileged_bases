"""
Script: weight_symmetry/training/trainer.py
--------------------------------------------
Training loop for linear autoencoder experiments.
Adapted from code/training/trainer.py.

Handles:
  - Mini-batch iteration over train set (no labels — reconstruction only)
  - Validation loss tracking after each epoch
  - Early stopping on val loss
  - Best-model checkpointing
  - Optional QR orthogonalisation after each optimizer step

Interface:
    history = train_ae(model, loss_fn, optimiser, data, cfg, run_dir, model_tag,
                       orthogonalize=False)

Returns dict with train_losses, val_losses, best_epoch.

Usage:
    from weight_symmetry.training.trainer import train_ae
"""

import os
import logging
from typing import Dict, List

import torch
from torch.utils.data import TensorDataset, DataLoader

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


def _setup_logger(run_dir: str, model_tag: str) -> logging.Logger:
    logger = logging.getLogger(f"ws_{model_tag}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    if _tqdm is not None:
        class TqdmHandler(logging.StreamHandler):
            def emit(self, record):
                _tqdm.write(self.format(record))
        ch = TqdmHandler()
    else:
        ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(run_dir, f"{model_tag}_train.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def train_ae(
    model,
    loss_fn,
    optimiser,
    data,
    cfg: dict,
    run_dir: str,
    model_tag: str,
    orthogonalize: bool = False,
    orthogonalize_encoder: bool = False,
    supervised: bool = False,
    scheduler=None,
) -> Dict:
    """
    Train a LinearAE model.

    Args:
        model         : LinearAE or LinearAEWithHeads instance
        loss_fn       : callable(x, model) -> scalar loss           (supervised=False)
                        callable(x, model, y) -> scalar loss        (supervised=True)
        optimiser     : torch optimizer
        data          : DataSplit namedtuple
        cfg           : dict with keys: epochs, batch_size, patience, seed
        run_dir       : output directory for checkpoints and logs
        model_tag     : short string label for filenames/logs
        orthogonalize         : if True, call model.orthogonalize() after each step (decoder A^T A = I)
        orthogonalize_encoder : if True, call model.orthogonalize_encoder() after each step (encoder B B^T = I)
        supervised            : if True, pass labels to loss_fn (for CE losses)
        scheduler             : optional torch LR scheduler; stepped once per epoch after validation

    Returns:
        dict: train_losses, val_losses, best_epoch
    """
    logger = _setup_logger(run_dir, model_tag)
    device = next(model.parameters()).device
    logger.info(f"=== Training: {model_tag}  ortho={orthogonalize}  ortho_enc={orthogonalize_encoder}  device={device} ===")
    logger.info(f"  epochs={cfg['epochs']}  lr={cfg['lr']}  "
                f"batch={cfg['batch_size']}  patience={cfg['patience']}")

    torch.manual_seed(cfg["seed"])

    if supervised:
        train_loader = DataLoader(
            TensorDataset(data.X_train, data.y_train),
            batch_size=cfg["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(data.X_val, data.y_val),
            batch_size=cfg["batch_size"], shuffle=False
        )
    else:
        train_loader = DataLoader(
            TensorDataset(data.X_train),
            batch_size=cfg["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(data.X_val),
            batch_size=cfg["batch_size"], shuffle=False
        )

    train_losses: List[float] = []
    val_losses:   List[float] = []
    best_val_loss     = float("inf")
    best_epoch        = 0
    epochs_no_improve = 0

    ckpt_path = os.path.join(run_dir, f"{model_tag}_best.pt")

    epoch_iter = (
        _tqdm(range(cfg["epochs"]), desc=f"[{model_tag}]", unit="epoch", leave=True)
        if _tqdm is not None else range(cfg["epochs"])
    )

    for epoch in epoch_iter:
        # ---- Train ----
        model.train()
        epoch_train_loss = 0.0
        if supervised:
            for (x_batch, y_batch) in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimiser.zero_grad()
                loss = loss_fn(x_batch, model, y_batch)
                loss.backward()
                optimiser.step()
                if orthogonalize:
                    model.orthogonalize()
                if orthogonalize_encoder:
                    model.orthogonalize_encoder()
                epoch_train_loss += loss.item() * len(x_batch)
        else:
            for (x_batch,) in train_loader:
                x_batch = x_batch.to(device)
                optimiser.zero_grad()
                loss = loss_fn(x_batch, model)
                loss.backward()
                optimiser.step()
                if orthogonalize:
                    model.orthogonalize()
                if orthogonalize_encoder:
                    model.orthogonalize_encoder()
                epoch_train_loss += loss.item() * len(x_batch)
        epoch_train_loss /= len(data.X_train)
        train_losses.append(epoch_train_loss)

        # ---- Validation ----
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            if supervised:
                for (x_batch, y_batch) in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    loss = loss_fn(x_batch, model, y_batch)
                    epoch_val_loss += loss.item() * len(x_batch)
            else:
                for (x_batch,) in val_loader:
                    x_batch = x_batch.to(device)
                    loss = loss_fn(x_batch, model)
                    epoch_val_loss += loss.item() * len(x_batch)
        epoch_val_loss /= len(data.X_val)
        val_losses.append(epoch_val_loss)

        logger.info(
            f"Epoch {epoch+1:>4}/{cfg['epochs']}  "
            f"train={epoch_train_loss:.5f}  val={epoch_val_loss:.5f}"
        )
        if _tqdm is not None:
            epoch_iter.set_postfix(
                train=f"{epoch_train_loss:.4f}", val=f"{epoch_val_loss:.4f}"
            )

        # ---- Checkpoint + early stopping ----
        if epoch_val_loss < best_val_loss:
            best_val_loss     = epoch_val_loss
            best_epoch        = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        if scheduler is not None:
            scheduler.step()

        if cfg["patience"] and epochs_no_improve >= cfg["patience"]:
            logger.info(f"Early stopping at epoch {epoch+1}.")
            break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    logger.info(f"Best checkpoint loaded (epoch {best_epoch+1}, val={best_val_loss:.5f})")
    logger.info(f"=== Done: {model_tag} ===\n")

    return {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "best_epoch":   best_epoch,
    }


# ==============================================================================
# Sanity check
# ==============================================================================

if __name__ == "__main__":
    import sys, os, tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from weight_symmetry.models.linear_ae import LinearAE
    from weight_symmetry.losses.losses import MSELoss, FullPrefixMRLLoss
    from weight_symmetry.data.loader import load_data

    data = load_data("digits", seed=42)
    cfg  = dict(epochs=5, lr=1e-3, batch_size=32, patience=3, seed=42)

    with tempfile.TemporaryDirectory() as run_dir:
        for tag, loss_fn, ortho in [
            ("mse",           MSELoss(),           False),
            ("fullprefix",    FullPrefixMRLLoss(),  False),
            ("fullprefix_qr", FullPrefixMRLLoss(),  True),
        ]:
            model = LinearAE(input_dim=data.input_dim, embed_dim=8)
            opt   = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            h     = train_ae(model, loss_fn, opt, data, cfg, run_dir, tag, ortho)
            assert len(h["train_losses"]) <= cfg["epochs"]
            print(f"  {tag}: best_epoch={h['best_epoch']}  PASSED")

    print("\nAll trainer checks passed.")
