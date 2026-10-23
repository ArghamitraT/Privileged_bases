"""Plot existing training logs; no model loading or training is needed."""
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_meeting_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("/home/argha/Mat_embedding/files/results/exprmnt_2026_10_13__21_22_10")
OUTPUT = Path(__file__).resolve().parent
COLORS = ("#0072B2", "#D55E00", "#009E73")
CONDITIONS = (("mrl_vae", 1), ("mrl_beta_vae", 4))
KEYS = ("reconstruction_nll", "full_reconstruction_nll", "kl", "weighted_kl", "loss")


def main():
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    histories, sources, records = {}, [], []
    for condition, beta in CONDITIONS:
        for seed in range(3):
            source = Path(manifest["models"][condition][str(seed)]["model_directory"]) / "training.jsonl"
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            assert [row["epoch"] for row in rows] == list(range(1, 101)), source
            for row in rows:
                assert row["reconstruction_mrl"]
                assert all(math.isfinite(row[key]) for key in KEYS)
                assert len(row["prefix_reconstruction_nll"]) == 10
                assert math.isclose(row["weighted_kl"], beta * row["kl"], rel_tol=1e-6)
                assert math.isclose(row["reconstruction_nll"], sum(row["prefix_reconstruction_nll"]) / 10, rel_tol=1e-5)
                records.append({"condition": condition, "beta": beta, "seed": seed,
                                "epoch": row["epoch"], **{key: row[key] for key in KEYS},
                                **{f"prefix_{k}_reconstruction_nll": value for k, value in enumerate(row["prefix_reconstruction_nll"], 1)}})
            histories[condition, seed] = rows
            sources.append({"condition": condition, "beta": beta, "seed": seed, "training_log": str(source)})
    with (OUTPUT / "mrl_epoch_losses_all_seeds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUTPUT / "mrl_epoch_loss_sources.json").write_text(json.dumps({
        "sources": sources,
        "measurement": "Per-image training losses averaged over the batches in each epoch; model weights change during the epoch.",
        "reconstruction_nll": "Mean of ten prefix reconstruction losses; each sums Bernoulli NLL over all 4096 pixels.",
        "kl": "Full-posterior KL, before beta weighting; calculated once per image.",
        "units": "nats per image", "smoothing": "none"}, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey="row")
    for col, (condition, beta) in enumerate(CONDITIONS):
        axes[0, col].set_title(f"MRL {'VAE' if beta == 1 else 'β-VAE'} (β={beta})")
        for seed, color in enumerate(COLORS):
            rows = histories[condition, seed]
            for row_index, key in enumerate(("reconstruction_nll", "kl")):
                axes[row_index, col].plot([r["epoch"] for r in rows], [r[key] for r in rows],
                                          color=color, label=f"Seed {seed}", linewidth=1.7)
        for ax in axes[:, col]:
            ax.grid(alpha=.25)
            ax.legend(frameon=False)
            ax.set_xlim(1, 100)
        axes[1, col].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Reconstruction: average of 10 prefixes\n(nats per image)")
    axes[1, 0].set_ylabel("KL before β weighting\n(nats per image)")
    fig.suptitle("MRL training losses — all three seeds, epochs 1–100")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"mrl_epoch_reconstruction_and_kl_all_seeds.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharex=True, sharey=True)
    for ax, (condition, beta) in zip(axes, CONDITIONS):
        for seed, color in enumerate(COLORS):
            rows = histories[condition, seed]
            ax.plot([r["epoch"] for r in rows], [r["full_reconstruction_nll"] for r in rows],
                    color=color, label=f"Seed {seed}", linewidth=1.7)
        ax.set_title(f"MRL {'VAE' if beta == 1 else 'β-VAE'} (β={beta})")
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, 100)
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Reconstruction using all 10 coordinates\n(nats per image)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"mrl_epoch_full_reconstruction_all_seeds.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Verified and exported {len(records)} epoch records from six training logs.")
    for condition, beta in CONDITIONS:
        for seed in range(3):
            row = histories[condition, seed][-1]
            print(f"β={beta}, seed={seed}, epoch=100: reconstruction={row['reconstruction_nll']:.2f}, KL={row['kl']:.2f}")


if __name__ == "__main__":
    main()
