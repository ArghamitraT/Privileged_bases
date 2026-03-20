# Matryoshka Embedding Interpretability

Exploring whether Matryoshka Representation Learning (MRL) increases
embedding interpretability by inducing a meaningful ordering of dimensions
(privileged bases).

---

## Setup

### 1. Create and activate the conda environment

```bash
conda env create -f env/mrl_env.yml
conda activate mrl_env
```

> If you have a GPU, remove the `cpuonly` line from `env/mrl_env.yml` before running.

### 2. Adding new packages

Do NOT modify the existing yml file. Instead:
1. Create `env/mrl_env_2.yml` (increment the number) with the new package added.
2. Recreate the environment:
   ```bash
   conda deactivate
   conda env remove -n mrl_env
   conda env create -f env/mrl_env_2.yml
   conda activate mrl_env_2
   ```

---

## Running Experiments

All experiments are run from the `code/` directory with the conda environment active.

### Verify everything works before running

```bash
cd code
python run_tests.py --fast    # quick smoke test (~10s), skips MNIST + training
python run_tests.py           # full suite including MNIST load and a short training run
```

Each module also has its own self-test you can run directly:

```bash
python utility.py
python config.py
python data/loader.py           # downloads MNIST on first run (~11MB)
python models/encoder.py
python models/heads.py
python losses/mat_loss.py
python training/trainer.py      # quick 5-epoch test on digits dataset
python evaluation/prefix_eval.py
```

### Run Experiment 1 — Prefix Performance Curve

**What it tests:** Can a Matryoshka-trained embedding remain useful even when
you throw away most of its dimensions?

**How it works:**

1. Three models are trained to classify MNIST digits using a 64-dim embedding:
   - **Standard** — normal cross-entropy loss on the full 64-dim embedding
   - **Matryoshka (Mat)** — cross-entropy loss summed at every prefix scale
     (k=1, 2, 4, 8, 16, 32, 64), so every prefix is jointly optimised
   - **PCA baseline** — no neural training; PCA is fit on the training data
     and components are naturally ordered by explained variance

2. At test time, only the first k dimensions are kept (the rest are zeroed out
   or sliced off). Accuracy is measured at each k.

3. The result is plotted as a curve: x = prefix size k (log scale),
   y = test accuracy.

**Expected result:** The Mat curve stays high at small k (the model has
learned to pack the most useful information into the first few dims). The
Standard curve drops off sharply because it was never trained to care about
prefix order. PCA sits somewhere in between as a linear baseline.

```bash
python experiments/exp1_prefix_curve.py
```

To change settings (dataset, embedding dim, prefix sizes, etc.), edit the
`ExpConfig(...)` block near the top of `main()` in that file.

---

## Output

All results are saved outside the code directory (not tracked by git):

```
Mat_embedding_hyperbole/files/results/exprmnt_{timestamp}/
    experiment_description.log  # what this experiment does, why, expected outcome + config
    standard_train.log          # per-epoch loss and accuracy (standard model)
    mat_train.log               # per-epoch loss and accuracy (Mat model)
    standard_encoder_best.pt    # best standard encoder weights
    standard_head_best.pt       # best standard head weights
    mat_encoder_best.pt         # best Matryoshka encoder weights
    mat_head_best.pt            # best Matryoshka head weights
    results_summary.txt         # accuracy table: k vs Standard / Mat / PCA
    prefix_curve.png            # the main result plot
```

Each run creates its own timestamped folder so previous results are never overwritten.

---

## Project Structure

```
code/
├── README.md                       # this file
├── CLAUDE.md                       # project context for Claude Code
├── config.py                       # ExpConfig dataclass — all settings here
├── utility.py                      # shared helpers (paths, timestamps, run dirs)
├── env/
│   └── mrl_env.yml                 # conda environment spec
├── data/
│   └── loader.py                   # dataset loading + train/val/test splits
├── models/
│   ├── encoder.py                  # MLP encoder (input -> embed_dim)
│   └── heads.py                    # classifier heads (shared or multi)
├── losses/
│   └── mat_loss.py                 # StandardLoss and MatryoshkaLoss
├── training/
│   └── trainer.py                  # training loop with early stopping
├── evaluation/
│   └── prefix_eval.py              # prefix sweep + PCA baseline evaluation
├── experiments/
│   └── exp1_prefix_curve.py        # Experiment 1 main script
└── figure/                         # figures from older experiments
```
