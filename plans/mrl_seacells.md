# Exp 11: MRL Metacells vs SEACells

## Goal
Test whether MRL embeddings can recover hidden cell states from single-cell genomics data
**without** the manual Gaussian kernel step used by SEACells.

## Background: SEACells

SEACells (Persad et al., Nature Biotechnology 2023) identifies metacells — groups of highly
similar cells representing distinct cell states. Its pipeline has three steps:

1. **PCA** → low-dim representation of cells
2. **Adaptive Gaussian kernel** → cell-cell similarity matrix (density-aware, so rare states
   are not overwhelmed by dense clusters)
3. **Kernel archetypal analysis** → finds extreme boundary states, assigns cells to metacells

The Gaussian kernel is the key hand-crafted piece: it adapts to local density so that the
resulting similarity matrix projects cells into tight, well-separated neighborhoods. SEACells
evaluates metacells by **cell type purity**, **compactness** (low intra-metacell variance),
and **separation** (distance between neighboring metacells).

## Core Hypothesis

MRL embeddings may replace the Gaussian kernel because:

| SEACells mechanism | MRL analogue |
|---|---|
| Gaussian kernel creates hierarchical similarity | MRL prefixes create hierarchy by design |
| Archetypal analysis finds boundary states | Early MRL dims push cell types to extremes via training objective |
| Number of metacells = manual hyperparameter | Prefix length = natural resolution knob |
| Rare cells undersampled by density | MRL loss is prefix-weighted, not density-weighted |

If MRL embeddings have **privileged bases** (the project's central hypothesis), specific
dimensions already encode biological axes (erythroid vs myeloid, stem vs differentiated),
making the kernel redundant. This would mean MRL not only matches SEACells practically but
also provides *interpretable* axes that kernel-based metacells do not.

## Key Design Question: Supervision Level

SEACells is fully unsupervised. Three paths of increasing complexity:

| Mode | Supervision | Notes |
|---|---|---|
| Supervised MRL | Cell type labels | Strongest baseline; requires labels SEACells doesn't need |
| Contrastive MRL | Augmented cell pairs (count dropout) as positives | Closest in spirit to SEACells |
| Reconstruction MRL (AE) | Reconstruct gene expression at each prefix | Fully unsupervised, most direct comparison |

**Start with supervised MRL** to establish the upper bound, then ablate toward unsupervised.

---

## Training Paradigms (Detailed)

### 1. Supervised MRL

Train the encoder to predict known cell type labels from each prefix independently.

```
cell → encoder → [z₁ | z₂ | ... | z₆₄]
                    ↓     ↓           ↓
                 classify at k=2, 4, ..., 64
                    ↓     ↓           ↓
                 cross-entropy loss (weighted by prefix)
```

**Loss**:
```
L = Σₖ  wₖ · CrossEntropy(classifier(z[:k]), cell_type_label)
```
`wₖ` down-weights larger prefixes — first dims must work alone (MRL convention).

**Pros**: Strong signal, directly separates known cell types, slots into existing code unchanged.  
**Cons**: Requires labels SEACells doesn't need. Optimizes for *known* cell types — may miss
novel or transitional states that SEACells excels at finding.

**SEACells analogy**: Closest to using ground-truth annotations as supervision. Fair comparison
only when labels exist; sets the upper bound.

---

### 2. Contrastive MRL

No labels. Create positive pairs via biological noise augmentations of the same cell; train
so augmented views of the same cell cluster together at every prefix level.

```
cell x  →  augment₁  →  encoder  →  z_a
        →  augment₂  →  encoder  →  z_b

z_a[:k] and z_b[:k] should be close  (same cell, different augmentation)
z_a[:k] and z_c[:k] should be far    (different cell)
```

**Augmentations for scRNA-seq** (biologically motivated):
- **Count dropout**: randomly zero out 20–30% of gene counts (mimics technical dropout)
- **Gaussian noise**: add small noise to log-normalized expression
- **Gene subsetting**: randomly mask 10% of genes → forces encoder to learn robust features

**Loss** (SimCLR-style NT-Xent at each prefix):
```
L = Σₖ  wₖ · -log [ exp(sim(z_a[:k], z_b[:k]) / τ) / Σⱼ exp(sim(z_a[:k], z_j[:k]) / τ) ]
```
τ = temperature hyperparameter; sim = cosine similarity.

**Pros**: No labels needed — directly comparable to SEACells. Augmentations encode
domain knowledge about scRNA-seq noise. Learns structure from data geometry.  
**Cons**: Augmentation collapse risk; τ tuning sensitive; harder to implement than supervised.

**SEACells analogy**: Closest analog to the Gaussian kernel. Both ask "which cells are
similar?" without labels — kernel uses neighborhood structure, contrastive MRL uses
augmentation invariance. The **scientifically interesting comparison** lives here: if
contrastive MRL matches SEACells without a kernel, MRL is a principled alternative.

---

### 3. Reconstruction MRL (Autoencoder)

Encode the cell into a hierarchical embedding, then decode each prefix back to full gene
expression. The encoder must pack sufficient information at each prefix level to reconstruct
the input.

```
cell (2000 genes)  →  encoder  →  [z₁ | z₂ | ... | z₆₄]
                                        ↓     ↓           ↓
                                   decode  decode  ...  decode
                                        ↓     ↓           ↓
                                   x̂₂    x̂₄    ...  x̂₆₄  ≈ x
```

**Loss**:
```
L = Σₖ  wₖ · MSE(decoder(z[:k]), x)
```
Prefix `z[:2]` must reconstruct the whole cell — forces maximum information compression
into first dims.

**Pros**: Fully unsupervised. Most directly analogous to PCA (which is SEACells' step 1),
but non-linear. First dims should capture largest variance.  
**Cons**: MSE on gene expression is noisy (dropout, zero-inflation); may capture technical
variance rather than biological cell-state structure. Decoder adds architectural complexity.

**SEACells analogy**: Replaces both PCA and the Gaussian kernel in one step. Asks whether
a non-linear MRL AE can learn the cell-state manifold end-to-end.

---

### Paradigm Comparison

| Property | Supervised | Contrastive | Reconstruction AE |
|---|---|---|---|
| Labels needed | Yes (cell types) | No | No |
| Noise robustness | Via label smoothing | Built-in (augmentation) | Weak (MSE amplifies noise) |
| Novel state discovery | Poor (optimizes known types) | Good | Moderate |
| Interpretability | High (dims → cell types) | Medium | Low (dims → variance) |
| SEACells analogy | Post-hoc label assignment | Gaussian kernel (geometry) | PCA step |
| Implementation difficulty | Easy (existing code) | Medium | Medium |
| Primary risk | Misses unlabeled states | Augmentation collapse | Captures technical noise |

### Recommended Progression

```
MNIST testbed (supervised)
    → scRNA-seq supervised       [upper bound]
        → scRNA-seq contrastive  [fair SEACells comparison — key scientific claim]
            → ablation: does contrastive match supervised?
                → if yes: MRL without labels recovers cell states = strong result
```

---

## Experiment Design

### Data
- **Primary**: PBMC 10x Multiome public dataset (~10,000 cells, ~15 known cell types, RNA + ATAC)
- **Testbed first**: MNIST digits — treat each digit as a "cell type," hold out 10% as rare states;
  verify MRL prefix sweeps recover rare classes better than flat embeddings at small k.

### Step 1 — Train MRL encoder on scRNA-seq
- Input: log1p-normalized gene expression, top 2,000 highly variable genes (HVGs)
- Architecture: MLP encoder (input_dim=2000 → hidden → embed_dim=64), same as existing encoder.py
- Loss: supervised MRL with cell type labels (mode A, shared head)
- Config: EMBED_DIM=64, HIDDEN_DIM=256, EPOCHS=20, BATCH_SIZE=128

### Step 2 — Prefix sweep evaluation
At k = 1, 2, 4, 8, 16, 32, 64:
- Cluster cells in k-dim prefix space (k-means with n_clusters = expected cell types at that resolution)
- Compute **cell type purity** of clusters (frequency of dominant cell type per cluster)
- Compute **compactness** (mean intra-cluster distance in diffusion space)
- Compute **separation** (distance between nearest neighboring cluster centroids)
- Plot purity / compactness / separation vs prefix length

### Step 3 — Metacell assignment (no kernel)
- Use k-NN aggregation in MRL embedding space (e.g., k=5 nearest neighbors per seed cell)
- Aggregate raw gene counts within each metacell
- Compare to SEACells metacells from the official SEACells Python package

### Step 4 — Downstream recovery tests
Same benchmarks as SEACells paper:
- **Marker gene correlation**: TAL1 (erythroid), MPO (myeloid), IRF8 (dendritic), EBF1 (lymphoid)
  — compute Pearson/Spearman correlation between metacell-aggregated gene expression and ATAC
  accessibility (or expression of paired marker)
- **Trajectory recovery**: run Palantir on MRL metacells → compare pseudotime correlation
  with known differentiation order
- **NMI score**: normalized mutual information between metacell assignments and ground-truth
  cell type labels

### Step 5 — Interpretability / Privileged Basis Connection
- **Dimension alignment**: correlate individual MRL dimensions with known biological scores
  (GATA1 expression, MPO expression, stemness score)
- **Rotation sensitivity test**: apply random orthogonal rotations to MRL embeddings vs PCA
  embeddings; measure degradation in cell-type separability (decision tree accuracy) as a
  function of rotation magnitude — connects directly to the project's privileged basis
  hypothesis

## Evaluation Metrics

| Metric | Target | Comparison |
|---|---|---|
| Cell type purity | ≥ SEACells | SEACells paper Fig. 2a,b |
| Compactness (lower = better) | ≤ SEACells | SEACells paper Fig. 4c |
| Separation (higher = better) | ≥ SEACells | SEACells paper Fig. 4c |
| Marker gene correlation | ≥ single-cell baseline | SEACells paper Fig. 3a, 4b |
| NMI score | ≥ MetaCell / SuperCell | SEACells paper Methods |
| Rotation sensitivity gap | MRL > PCA | Project hypothesis |

## Expected Outputs

- `training_curves_{stamp}.png` — loss curves for MRL encoder training
- `prefix_purity_curve_{stamp}.png` — cell type purity vs prefix length k
- `prefix_compactness_separation_{stamp}.png` — compactness & separation vs k
- `marker_gene_correlation_{stamp}.png` — scatter plots (expression vs accessibility) per metacell method
- `rotation_sensitivity_{stamp}.png` — accuracy vs rotation magnitude for MRL vs PCA
- `experiment_description.log`, `results_summary.txt`, `runtime.txt`, `code_snapshot/`

## Implementation Notes

- Use existing `models/encoder.py`, `losses/mat_loss.py`, `training/trainer.py` infrastructure
- scRNA-seq data loading: add `data/scrna_loader.py` (scanpy-based; log1p + HVG selection)
- SEACells comparison: install SEACells package; run their pipeline on same data as reference
- Metacell aggregation utility: add `evaluation/metacell_eval.py` (purity, compactness, separation)
- Start with MNIST testbed to validate approach before moving to scRNA-seq data

## Open Questions

1. Does supervised MRL (with cell type labels) actually outperform SEACells, or does the
   kernel provide information labels cannot?
2. At what prefix length k does the purity plateau? Does this match the number of biologically
   meaningful cell states?
3. Does contrastive / unsupervised MRL maintain the advantage, or is the label signal essential?
4. Do MRL dimensions align with specific biological axes, or is the embedding rotated arbitrarily
   (i.e., do privileged bases exist in the biological domain)?

## Related Plans
- `exp8_dimension_importance_scoring.md` — per-dim importance scoring; directly relevant to Step 5
- `exp9_dense_prefix_multiseed.md` — dense prefix sweep methodology reused in Step 2
