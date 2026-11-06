# NonLinearPB data setup (new cluster)

How to get the NonLinearPB data onto a new machine (e.g. Empire AI): what to download and where
it goes.

## 1. Folder layout

All paths are relative to the **project root** = the folder that contains `code/`
(`/home/argha/Mat_embedding` on aeolus, `/mnt/home/at3836/Mat_embedding` on Empire AI). The code
computes this root itself (`NonLinearPBExperiments/vae_reproduction/runs.py`,
`PROJECT_ROOT = parents[4]`), so nothing needs editing as long as this layout holds:

```text
<project root>/
├── code/                          # this git repo
├── data/
│   ├── dsprites/
│   │   └── dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz   # 19,318,135 B — REQUIRED
│   └── scream/
│       └── scream.jpg                                       # 4,219,156 B — disentanglement_lib only
└── files/
    └── results/                   # all run outputs (OUTPUT_ROOT); created by the scripts
```

## 2. Download the datasets

```bash
cd <project root>/code
bash scripts/bash/download_dsprites.sh
```

The script downloads both files to `<project root>/data/` and checks their sha256 against the
aeolus copies. Manual equivalent:

| File | URL | sha256 |
|---|---|---|
| `data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz` | https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz | `857dc001287e8efa226e0025fac6b29e5745c621b380b0d3363e4e33f4e28fa2` |
| `data/scream/scream.jpg` | https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg | `a9bd171fe0a8da87f90a467e30542be195fb2e2a46d650f5287421869e57d390` |

Only dSprites is needed for the PyTorch experiments in `NonLinearPBExperiments/`. Scream is used
only by `scream_dsprites` in the TF `disentanglement_lib` reproduction. Do **not** run
`dlib_download_data`: it downloads every dataset into the current directory. For
`disentanglement_lib`, set `export DISENTANGLEMENT_LIB_DATA=<project root>/data`.

Any script can also take a different archive location with `--data-path`.

## 3. What the dSprites archive contains

`np.load(path)` gives:

| Key | Shape / dtype | Meaning |
|---|---|---|
| `imgs` | `(737280, 64, 64)` uint8, values 0/1 | binary images, one white shape on black |
| `latents_classes` | `(737280, 6)` int | factor index per image |
| `latents_values` | `(737280, 6)` float | factor values per image |
| `metadata` | dict (needs `allow_pickle=True, encoding="latin1"`) | names and sizes |

Factors, in archive column order: `color` (1, constant — dropped), `shape` (3: square, ellipse,
heart), `scale` (6), `orientation` (40, circular), `posX` (32), `posY` (32). The images are
stored in full factorial order, so image index = factors · `[245760, 40960, 1024, 32, 1]`
(shape, scale, rotation, x, y). `vae_reproduction/data.py` checks the shape, binary values and
this ordering on load, and fails if anything differs.

## 4. Other machine-specific paths

These still point to aeolus and must be changed (or overridden) on a new cluster:

- `NonLinearPBExperiments/scripts/*_tmux.sh`: `PYTHON=/home/argha/miniconda3/envs/mrl_env_cuda12/bin/python`
- `NonLinearPBExperiments/scripts/plot_dsprites_figure6.py`: `RESULTS = Path("/home/argha/Mat_embedding/files/results")`
- `vae_reproduction/runs.py: PRETRAINED_MANIFEST` points to the aeolus pretrained checkpoints;
  only the experiments that start from them (`dsprites_rotation_basis.py`,
  `dsprites_phase0_audit.py`) use it.

Environment: `mrl_env_cuda12`, spec in `weight_symmetry/env/mrl_env_cuda12.yml`.

## 5. Check

```bash
cd <project root>/code/NonLinearPB/NonLinearPBExperiments
python -c "from vae_reproduction.runs import DATA_PATH; from vae_reproduction.data import DSprites; print(DATA_PATH, len(DSprites(DATA_PATH)))"
# -> /mnt/home/at3836/Mat_embedding/data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz 737280
```
