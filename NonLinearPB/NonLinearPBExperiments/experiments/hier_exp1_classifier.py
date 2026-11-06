"""Train the frozen reconstruction classifier once (plan section 5D.3) and run its test gate.

Conda environment: mrl_env_cuda12
Usage (from code/):
    python NonLinearPB/NonLinearPBExperiments/experiments/hier_exp1_classifier.py --output /path/to/exprmnt_x/classifier --device cuda:0
    python NonLinearPB/NonLinearPBExperiments/experiments/hier_exp1_classifier.py --device cuda:0 --output /tmp/clf --fast

Trained on original v2 training images with ground-truth labels (coarse 3,
mid node 9, leaf 27), selected on validation, then frozen. It lives in the
experiment folder (<run-dir>/classifier/) and a finished one is never
overwritten, so every model in the run is scored by the same network. Gate: test
accuracy >= GATE on all three heads; metrics.json records pass/fail, and the
evaluation refuses to score reconstructions with a classifier that failed.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hier_exp1.data import dataset_record, file_sha256, load_split
from hier_exp1.model import ReconClassifier

# ==============================================================================
# CONFIG
# ==============================================================================
EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
SEED = 0
GATE = 0.98
HEADS = tuple(ReconClassifier.HEADS)
# ==============================================================================


def tensors(split, device, limit=None):
    images, labels = load_split(split)
    if limit:                # strided, not a prefix: the splits are stored sorted by leaf
        step = max(1, len(images) // limit)
        images, labels = images[::step], {k: v[::step] for k, v in labels.items()}
    x = torch.as_tensor(images, device=device).unsqueeze(1)
    y = {h: torch.as_tensor(labels[h], device=device) for h in HEADS}
    return x, y


@torch.no_grad()
def accuracy(model, x, y, chunk=2048):
    model.eval()
    correct = {h: 0 for h in HEADS}
    for s in range(0, len(x), chunk):
        out = model(x[s:s + chunk].float() / 255.0)
        for h in HEADS:
            correct[h] += int((out[h].argmax(1) == y[h][s:s + chunk]).sum())
    return {h: correct[h] / len(x) for h in HEADS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, required=True, help="normally <run-dir>/classifier")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if (args.output / "metrics.json").exists():
        sys.exit(f"{args.output} already has a finished classifier; it is trained once and reused")
    # A classifier.pt without metrics.json is an interrupted run: retrain from scratch.
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    limit = 2048 if args.fast else None
    xtr, ytr = tensors("train", device, limit)
    xva, yva = tensors("val", device, limit and 512)
    xte, yte = tensors("test", device, limit and 512)

    model = ReconClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator(device=device).manual_seed(SEED)
    epochs = 2 if args.fast else EPOCHS
    best, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(xtr), device=device, generator=gen)
        for s in range(0, len(xtr), BATCH_SIZE):
            idx = order[s:s + BATCH_SIZE]
            out = model(xtr[idx].float() / 255.0)
            loss = sum(F.cross_entropy(out[h], ytr[h][idx]) for h in HEADS)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val = accuracy(model, xva, yva)
        score = min(val.values())                     # select on the weakest head
        history.append({"epoch": epoch, **{f"val_{h}": v for h, v in val.items()}})
        print(f"epoch {epoch}: val " + " ".join(f"{h} {v:.4f}" for h, v in val.items()), flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, args.output / "classifier.pt")

    model.load_state_dict(torch.load(args.output / "classifier.pt", map_location=device, weights_only=True)["state_dict"])
    test = accuracy(model, xte, yte)
    passed = all(v >= GATE for v in test.values())
    metrics = {"gate": GATE, "gate_passed": passed, "test_accuracy": test,
               "selected_epoch": torch.load(args.output / "classifier.pt", map_location="cpu", weights_only=True)["epoch"],
               "history": history, "architecture": "Conv(1,32,s2)-ReLU-Conv(32,64,s2)-ReLU-Conv(64,64,s2)-ReLU-Flatten-Linear x3",
               "heads": ReconClassifier.HEADS, "epochs": epochs, "lr": LR, "batch_size": BATCH_SIZE, "seed": SEED,
               "data": dataset_record(), "checkpoint_sha256": file_sha256(args.output / "classifier.pt"),
               "fast": args.fast, "seconds": round(time.time() - start, 1)}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"test " + " ".join(f"{h} {v:.4f}" for h, v in test.items()) + f" -> gate {'PASSED' if passed else 'FAILED'}", flush=True)


if __name__ == "__main__":
    main()
