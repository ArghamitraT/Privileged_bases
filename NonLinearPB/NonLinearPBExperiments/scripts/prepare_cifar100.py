"""Download CIFAR-100 once and freeze it as npz splits for the Experiment 1 trainer (--dataset cifar100).

Conda environment: mrl_env_cuda12
Source: the Hugging Face mirror uoft-cs/cifar100 (parquet, lossless PNG, same images and order as the official
cifar-100-python files; the Toronto server ran at ~3 KB/s on 2026-11-05). Download once into data/cifar100/raw/hf/:
    curl -L -o train.parquet https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet
    curl -L -o test.parquet  https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet
Needs pyarrow (not in mrl_env_cuda12; e.g. pip install --target DIR pyarrow, then PYTHONPATH=DIR).
Writes data/cifar100/v1/{train,val,test}.npz + config.json:
images uint8 [N, 3, 32, 32], coarse (20 superclasses), fine (100 classes). Train/val = a fixed permutation (seed 0) of the
official 50,000 training images, 45,000 / 5,000; test = the official 10,000. Refuses to overwrite an existing v1.

Usage: python scripts/prepare_cifar100.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hier_exp1.data import PROJECT_ROOT, file_sha256

RAW = PROJECT_ROOT / "data/cifar100/raw"
OUT = PROJECT_ROOT / "data/cifar100/v1"
N_VAL, SPLIT_SEED = 5000, 0


def main():
    if (OUT / "config.json").exists():
        sys.exit(f"{OUT} already exists (frozen); not overwriting")
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    base = RAW / "hf"

    def read(name):
        t = pq.read_table(base / f"{name}.parquet").to_pydict()
        x = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in t["img"]])  # [N, 32, 32, 3]
        return (x.transpose(0, 3, 1, 2).copy(), np.asarray(t["coarse_label"], np.int16), np.asarray(t["fine_label"], np.int16))

    tr_x, tr_c, tr_f = read("train")
    te_x, te_c, te_f = read("test")
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(tr_x))
    val_idx, train_idx = np.sort(perm[:N_VAL]), np.sort(perm[N_VAL:])
    OUT.mkdir(parents=True)
    splits = {"train": (tr_x[train_idx], tr_c[train_idx], tr_f[train_idx]),
              "val": (tr_x[val_idx], tr_c[val_idx], tr_f[val_idx]),
              "test": (te_x, te_c, te_f)}
    for name, (x, c, f) in splits.items():
        np.savez(OUT / f"{name}.npz", images=x, coarse=c, fine=f)
    feats = json.loads(pq.read_schema(base / "train.parquet").metadata[b"huggingface"])["info"]["features"]
    meta = {"coarse_label_names": feats["coarse_label"]["names"], "fine_label_names": feats["fine_label"]["names"]}
    cfg = {"version": "cifar100_v1", "frozen": True, "source": "huggingface.co/datasets/uoft-cs/cifar100 parquet (PNG), same as cifar-100-python",
           "source_sha256": {f"{n}.parquet": file_sha256(base / f"{n}.parquet") for n in ("train", "test")},
           "shape": "uint8 [N, 3, 32, 32]", "sizes": {k: int(len(v[0])) for k, v in splits.items()},
           "split": f"official train permuted with numpy default_rng({SPLIT_SEED}); first {N_VAL} -> val, rest -> train (each kept in original order)",
           "val_indices_sha256": hashlib.sha256(val_idx.astype(np.int64).tobytes()).hexdigest(),
           "coarse_names": meta["coarse_label_names"], "fine_names": meta["fine_label_names"],
           "sha256": {f"{n}.npz": file_sha256(OUT / f"{n}.npz") for n in splits}}
    (OUT / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    np.save(OUT / "val_indices.npy", val_idx)
    print(OUT, cfg["sizes"])


if __name__ == "__main__":
    main()
