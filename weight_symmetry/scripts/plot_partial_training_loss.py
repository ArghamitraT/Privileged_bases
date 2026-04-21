"""
Plot training (and val) loss from partial run logs.

Usage:
    python weight_symmetry/scripts/plot_partial_training_loss.py \
        --run-dir exprmnt_2026_04_20__18_30_09
"""

import os, sys, re, argparse
import numpy as np
import matplotlib.pyplot as plt

_WS_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_ROOT = os.path.dirname(_WS_ROOT)
for _p in [_WS_ROOT, _CODE_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from weight_symmetry.utility import get_path

LABEL_MAP = {
    "mse_lae":             "Unordered LAE",
    "fullprefix_mrl_ortho":"FP MRL+ortho",
    "standard_mrl":        "Standard MRL",
}

def parse_log(path):
    epochs, train_loss, val_loss = [], [], []
    pat = re.compile(r"Epoch\s+(\d+)/\d+\s+train=([\d.]+)\s+val=([\d.]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
    return np.array(epochs), np.array(train_loss), np.array(val_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(get_path("files/results"), run_dir)

    logs = [f for f in os.listdir(run_dir) if f.endswith("_train.log")]
    if not logs:
        print("No *_train.log files found in", run_dir); return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training curves (partial run)", fontsize=13)

    for log_file in sorted(logs):
        # extract model key from filename: seed42_<key>_train.log
        key = re.sub(r"^seed\d+_", "", log_file).replace("_train.log", "")
        label = LABEL_MAP.get(key, key)
        epochs, train_loss, val_loss = parse_log(os.path.join(run_dir, log_file))
        if len(epochs) == 0:
            continue
        axes[0].plot(epochs, train_loss, label=label)
        axes[1].plot(epochs, val_loss,   label=label)

    for ax, title in zip(axes, ["Train loss", "Val loss"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(run_dir, "partial_training_curves.png")
    plt.savefig(out, dpi=150)
    print("Saved:", out)


if __name__ == "__main__":
    main()
