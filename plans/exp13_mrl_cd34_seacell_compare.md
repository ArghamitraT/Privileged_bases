# Plan: Exp13 — Supervised MRL on CD34 vs SEACells Baseline

## Goal
Train multiple loss variants (MRL, CE, FixedLp, ScalarLp, VecLp, PCA) on the CD34
gene expression dataset using known cell type labels (HSC, HMP, MEP, Ery, Mono,
cDC, pDC, CLP). Compare prefix-sweep clustering quality against the SEACells
metacell baseline from `seacells_cd34.py`.

This is the **supervised upper-bound** experiment in the MRL-SEACells roadmap.

## Related plan
- `plans/mrl_seacells.md` — full paradigm design (supervised → contrastive → AE)

---

## Loss Variants

All trained models share the same encoder + shared head architecture.
PCA is an analytical baseline (no training).

| Legend tag | Class | Notes |
|---|---|---|
| `pca` | `sklearn.decomposition.PCA` | No training; first-k dims = max variance |
| `ce` | `StandardLoss` | Plain cross-entropy on full embedding |
| `mrl` | `MatryoshkaLoss` | CE summed at each prefix scale |
| `fixed_lp` | `PrefixLpLoss(p=FIXED_LP_P)` | Ordered Lp; `FIXED_LP_P=1` ≡ old PrefixL1 |
| `learned_lp` | `LearnedPrefixLpLoss` | Scalar p **learned** jointly with encoder |
| `learned_lp_vec` | `VectorLearnedPrefixLpLoss` | Per-dim p vector **learned** jointly |

**FixedLp reversal rule:** for any `FIXED_LP_P`, flip dims before prefix sweep
(dim 0 is most penalised = least informative); legend label `"FixedLp (rev)"`.

**User selects which losses to run via `MODELS_TO_RUN` in the CONFIG block.**
Any subset of `["pca", "ce", "mrl", "fixed_lp", "learned_lp", "learned_lp_vec"]` is valid.

---

## Model Architecture

```
CD34 RNA (N_HVG genes, log-normalised)
    ↓
MLPEncoder:  N_HVG → HIDDEN_DIM → HIDDEN_DIM → EMBED_DIM  (BN + ReLU + Dropout)
    ↓
z  (batch, EMBED_DIM)  — L2-normalised
    ↓
SharedClassifier:  EMBED_DIM → N_CLASSES (8 cell types)
```

- Same `MLPEncoder` and `SharedClassifier` as all other experiments.
- `input_dim = N_HVG` (not in `ExpConfig` — passed directly to `MLPEncoder`).
- Each loss variant trains a **separate** encoder+head pair.
- Weights saved per model: `{model_tag}_encoder.pt`, `{model_tag}_head.pt`.

---

## CONFIG Block (full-run defaults)

```python
# ==============================================================================
# CONFIG — edit here to change the full run; use --fast for a quick smoke test
# ==============================================================================
DATA_PATH    = os.path.join(os.environ["HOME"], "Mat_embedding_hyperbole",
                            "data", "cd34_multiome",
                            "GSE200046_cd34_multiome_rna.h5ad")

# HVG selection
N_HVG          = 2000       # number of highly variable genes used as input
RECOMPUTE_HVG  = False      # True: recompute HVGs via scanpy; False: use h5ad precomputed

# Model
EMBED_DIM      = 30         # match PCA dim used by SEACells
HIDDEN_DIM     = 256
HEAD_MODE      = "shared_head"

# MRL training prefixes — sparse subset used in MatryoshkaLoss during training only
MRL_TRAIN_PREFIXES = [2, 4, 8, 16, 30]

# Evaluation prefixes — dense sweep k=1..EMBED_DIM applied to ALL models
# (set automatically in main() as list(range(1, EMBED_DIM + 1)); not a manual CONFIG value)

# Training
EPOCHS         = 15
PATIENCE       = 5
LR             = 1e-3
BATCH_SIZE     = 128
WEIGHT_DECAY   = 1e-4
SEED           = 42

# Loss-specific
L1_LAMBDA      = 0.05       # regularisation weight for FixedLp / ScalarLp / VecLp
FIXED_LP_P     = 1          # exponent for FixedLp (1 = old PrefixL1; 3 = soft ordering)

# Which models to train and evaluate (any subset of the 6 options)
MODELS_TO_RUN  = ["pca", "ce", "mrl", "fixed_lp", "learned_lp", "learned_lp_vec"]

# Evaluation
N_CLUSTERS     = 100        # k-means clusters for purity/compactness/separation
                            # set to match SEACells n_metacells for apples-to-apples
# ==============================================================================
```

`--fast` overrides (smoke test, not in CONFIG block):
- Subsample 500 cells, 5 epochs, `N_HVG=200`, `N_CLUSTERS=20`, `EMBED_DIM=8`,
  `EVAL_PREFIXES=[2, 4, 8]`

---

## Step-by-step Implementation

### Step 1 — Data loading (`load_cd34_data`)

```python
adata = sc.read_h5ad(DATA_PATH)
# [optional fast subsampling via sc.pp.subsample]

if RECOMPUTE_HVG:
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)
hvg_mask = adata.var["highly_variable"]
X_hvg = np.asarray(adata[:, hvg_mask].X.todense()
        if scipy.sparse.issparse(adata[:, hvg_mask].X)
        else adata[:, hvg_mask].X, dtype=np.float32)   # (n_cells, n_hvg)

y_str = adata.obs["celltype"].values
le    = LabelEncoder()
y_int = le.fit_transform(y_str).astype(np.int64)       # 0–7
label_names = list(le.classes_)                        # 8 cell type strings

X_pca = adata.obsm["X_pca"].astype(np.float32)         # (n_cells, 30), PCA baseline
X_umap = adata.obsm["X_umap"].astype(np.float32)       # for visualisation
seacell_labels = adata.obs["SEACell"].values            # paper's 195 metacell IDs
```

Returns `(X_hvg, y_int, X_pca, X_umap, seacell_labels, label_names, adata)`.

### Step 2 — Train/val/test split

```python
# Stratified split so all cell types appear in every split
X_tr, X_te, y_tr, y_te = train_test_split(X_hvg, y_int, test_size=0.2,
                                            stratify=y_int, random_state=SEED)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.1,
                                            stratify=y_tr, random_state=SEED)
```

Convert to `torch.Tensor` → `TensorDataset` → `DataLoader` (same pattern as all other exps).

### Step 3 — Build and train each model

For each `model_tag` in `MODELS_TO_RUN` (skip `"pca"` — no training needed):

```python
torch.manual_seed(SEED)
encoder = MLPEncoder(input_dim=X_hvg.shape[1], hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM)
head    = build_head(cfg, n_classes=len(label_names))   # SharedClassifier
loss_fn = build_loss(cfg, model_type_map[model_tag])    # uses build_loss() factory

# LearnedLp variants: include loss_fn.parameters() in optimizer
params = list(encoder.parameters()) + list(head.parameters())
if hasattr(loss_fn, "parameters"):
    params += list(loss_fn.parameters())
opt = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)

history = train(encoder, head, loss_fn, opt, data, cfg, run_dir, model_tag)
```

Save weights after each model:
```python
torch.save(encoder.state_dict(), os.path.join(run_dir, f"{model_tag}_encoder.pt"))
torch.save(head.state_dict(),    os.path.join(run_dir, f"{model_tag}_head.pt"))
```

### Step 4 — Extract embeddings

Run each trained encoder in eval mode on **all** cells (`X_hvg` full, not just test set —
we want cluster assignments for every cell, matching SEACells coverage):

```python
encoder.eval()
with torch.no_grad():
    Z = encoder(torch.tensor(X_hvg)).numpy()   # (n_cells, EMBED_DIM)
```

Also apply FixedLp reversal:
```python
if model_tag == "fixed_lp":
    Z = np.ascontiguousarray(Z[:, ::-1])
```

Save all embeddings (only models that were run):
```python
# embeddings dict built dynamically: {tag: Z_array}
np.savez(os.path.join(run_dir, "cd34_embeddings.npz"), **embeddings)
```

### Step 5 — Prefix sweep + k-means clustering

For each model and each `k ∈ EVAL_PREFIXES`:

```python
Z_k = Z[:, :k]                               # (n_cells, k)
km  = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED)
cluster_labels = km.fit_predict(Z_k)         # int array (n_cells,)
```

For PCA: `Z_k = X_pca[:, :k]` directly.

### Step 6 — Compute metrics

For each model × prefix `k`, compute three metrics over the `N_CLUSTERS` clusters:

**Cell type purity** (per cluster → dominant cell type fraction → median over clusters):
```python
def celltype_purity(cluster_labels, y_int):
    purities = []
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        counts = np.bincount(y_int[mask])
        purities.append(counts.max() / mask.sum())
    return np.median(purities)
```

**Compactness** (mean intra-cluster pairwise distance in MRL/PCA embedding space):
```python
def compactness(Z_k, cluster_labels):
    scores = []
    for c in np.unique(cluster_labels):
        pts = Z_k[cluster_labels == c]
        if len(pts) < 2: continue
        scores.append(pairwise_distances(pts).mean())
    return np.median(scores)
```

**Separation** (distance from each cluster centroid to its nearest neighbor centroid):
```python
def separation(Z_k, cluster_labels):
    centroids = np.array([Z_k[cluster_labels == c].mean(0)
                          for c in np.unique(cluster_labels)])
    D = pairwise_distances(centroids)
    np.fill_diagonal(D, np.inf)
    return np.median(D.min(axis=1))
```

**SEACells reference line** (purity only, using paper's assignments in `adata.obs['SEACell']`
and `X_pca` for compactness/separation so it's directly comparable to the SEACells paper):
```python
seacell_purity = celltype_purity(seacell_int_labels, y_int)
```

### Step 7 — Plots

**Three prefix curves** (`purity`, `compactness`, `separation`), each with:
- x-axis: prefix k (from EVAL_PREFIXES)
- One line per model: `PCA`, `CE`, `MRL`, `FixedLp (rev)`, `ScalarLp`, `VecLp`
- Horizontal dashed line: SEACells reference value (labelled `"SEACells (paper)"`)
- Saved as `prefix_{metric}_curve_{stamp}.png`

**UMAP visualisation** — 3-panel figure at best prefix k for MRL:
- Panel 1: cells coloured by ground-truth cell type
- Panel 2: cells coloured by MRL cluster assignment (k-means at best k)
- Panel 3: cells coloured by SEACells paper assignment
- Saved as `umap_comparison_{stamp}.png`

**Training curves** — one plot with loss curves for all trained models:
- `training_curves_{stamp}.png`

### Step 8 — Save outputs

Standard required outputs:
```
training_curves_{stamp}.png
prefix_purity_curve_{stamp}.png
prefix_compactness_curve_{stamp}.png
prefix_separation_curve_{stamp}.png
umap_comparison_{stamp}.png
cd34_embeddings.npz
results_summary.txt         ← table: model × prefix × purity × compactness × separation
experiment_description.log
runtime.txt
code_snapshot/
{model_tag}_encoder.pt  × (len(MODELS_TO_RUN) - 1)   # pca has no weights
{model_tag}_head.pt     × (len(MODELS_TO_RUN) - 1)
```

---

## `--use-weights` Flag

Same pattern as exp8/exp10:

```
--use-weights exprmnt_2026_04_01__22_04_54
```

Creates a new timestamped subfolder inside the weights folder. Loads all
`{model_tag}_encoder.pt` and `{model_tag}_head.pt` files, skips training,
re-runs Steps 4–8 (embedding extraction, clustering, plots).

Implementation:
```python
sub_stamp = time.strftime("exprmnt_%Y_%m_%d__%H_%M_%S")
run_dir   = os.path.join(weights_dir, sub_stamp)
os.makedirs(run_dir, exist_ok=True)
```

---

## CLI Flags

```
python experiments/exp13_mrl_cd34_supervised.py                     # full run
python experiments/exp13_mrl_cd34_supervised.py --fast              # smoke test
python experiments/exp13_mrl_cd34_supervised.py --use-weights PATH  # plots only
```

---

## Conda Environment

`mrl_env` — scanpy and anndata are already available in this env.
Do NOT use the `seacells` env for this script.

---

## Expected Outcome

- MRL purity curve rises quickly and plateaus near (or above) the SEACells reference line
- PCA shows slower rise (dims not ordered by classification relevance)
- CE may match MRL at full embedding but collapse at small prefixes
- FixedLp/ScalarLp/VecLp provide intermediate comparison points
- This would confirm MRL finds a biologically meaningful ordering — a biological
  validation of the privileged basis hypothesis

---

## Open Design Decisions (resolved)

| Question | Decision |
|---|---|
| `input_dim` in ExpConfig? | No — pass directly to `MLPEncoder`; keep as CONFIG constant `N_HVG` |
| HVG recompute or precomputed? | Both; `RECOMPUTE_HVG` flag in CONFIG |
| n_clusters | CONFIG constant `N_CLUSTERS = 100`; user sets it |
| Metric embedding space | MRL/PCA embedding space (not raw PCA space) |
| Multi-seed k-means | No (single seed, `n_init=10` handles init sensitivity) |
| Separate experiment folder | No — lives in `experiments/` like all other exps |
| Model weights | Saved per model tag; `--use-weights` to reload |

---

## Files to Create / Modify

| File | Action |
|---|---|
| `experiments/exp13_mrl_cd34_supervised.py` | **New** — main experiment script |
| `tests/run_tests_exp13.py` | **New** — unit + smoke tests |
| `CLAUDE.md` | **Update** — add exp13 row to table |
| `EXPERIMENTS.md` | **Update** — add exp13 section |
