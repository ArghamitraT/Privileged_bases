# Hierarchical Prefix-Change Analysis

## Purpose

Test whether the **spatial changes induced by extending an embedding prefix** resemble the **known coarse-to-mid and mid-to-fine changes** in the synthetic data, and inspect whether this behavior is consistent across five independently trained seeds.

Compare the standard AE and the MRL-AE. Work at **one fixed training epoch** at a time, initially for **one fixed test image**. Later, the same procedure can be repeated for other images and epochs.

**Scope:** What changes, where it changes, and whether the signed change resembles the data hierarchy. Do **not** measure pixelwise correction/improvement, use a classifier, or compute an extra scalar hierarchy-ordering score.

## Notation

- Image: \(x_i\in\mathbb R^{64\times64}\), with known nested labels \((c_i,m_i,f_i)\).
- Training epoch: \(t\).
- Model type: \(a\in\{\mathrm{AE},\mathrm{MRL}\}\).
- Independent model seed: \(s\in\{0,1,2,3,4\}\).
- Encoder and decoder at the checkpoint: \(E_{a,s,t},D_{a,s,t}\).
- Full latent: \(z=E_{a,s,t}(x_i)\in\mathbb R^{10}\).
- Reconstruction using the **cumulative prefix** \(1{:}k\):

  \[
  \hat x_{i,1:k}^{(a,s,t)}=D_{a,s,t}([z_1,\ldots,z_k,0,\ldots,0]),\qquad k=1,\ldots,10.
  \]

**Crucial:** \(1{:}k\) means *all of the first k coordinates together*, **not coordinate k alone**. Use the *same decoder* at every prefix. The encoder and decoder are frozen for this analysis.

**Empty prefix (decided 2026-11-03):** \(\hat x_{i,1:0}^{(a,s,t)}=D_{a,s,t}(\mathbf 0)\), the decode of the all-zero code. For MRL this approximates the global mean image \(G_0\). This makes the first step \(\varnothing\to1{:}1\) visible, where MRL does most of its work.

## Settings (decided 2026-11-03)

- **Run:** `exprmnt_2026_11_03__15_08_36` (lr 6e-5, no noise, weights for epochs 0–25), AE and MRL, seeds 0–4.
- **Epochs:** 2, 3, 4, 5, 10, 15, 20, 25.
- **Images (4):** the first test image of each of these leaves, all with fine \(f=0\):
  ring + stem / stem through (c1 m0 f0), ring + stem / stem slanted (c1 m2 f0),
  dumbbell / bowed (c0 m2 f0), boomerang / stepped (c2 m2 f0).
- **Caveat (AE zero-fill):** MRL is trained to decode zeroed suffixes, so for MRL a zero means "not known". The AE never saw zeros: a zero is a specific, possibly off-data code. AE change maps therefore show off-data decoding, not an AE hierarchy. A fairer AE variant (fill missing coordinates with their training mean) is noted, not done.
- **Caveat (AE coordinate order):** AE coordinates have no order and are not matched across seeds, so AE seed-mean curves over k are expected to be flat and wide by construction. This is a contrast, not evidence about AE hierarchy.

## Figure 1 — Signed spatial-change maps across prefixes and seeds

For each fixed image, epoch, model type and seed, define the incremental reconstruction change:

\[
\boxed{D_{i,k}^{(a,s,t)}=\hat x_{i,1:k}^{(a,s,t)}-\hat x_{i,1:(k-1)}^{(a,s,t)},\quad k=1,\ldots,10.}
\]

Each \(D_{i,k}\) is a **signed 64×64 image**. It represents exactly what changes in the reconstruction when the available prefix grows from \(1{:}(k-1)\) to \(1{:}k\). No absolute value is taken.

### Figure layout

- One figure per **model type** (or adjacent AE and MRL panels), for the **same test image** and **same epoch**.
- **Five rows:** seeds 0–4.
- **Ten columns:** \(\varnothing\to1{:}1\), \(1{:}1\to1{:}2\), ..., \(1{:}9\to1{:}10\) (\(D_1,\ldots,D_{10}\)).
- Show **only signed change maps** in grid cells, **not raw reconstructed images** or heatmap overlays. Show the original input separately for context if desired.
- **Red** = negative pixel change (darker); **blue** = positive change (brighter); **white** = zero change.
- Use a **symmetric diverging color scale centered at zero**, **one per model type** (decided 2026-11-03): shared across all ten columns and five seeds of that model, not shared between AE and MRL (AE's early changes are much larger and would wash MRL out). Label the color bar in pixel-intensity change units. Report each model's limit and avoid per-panel auto-normalization.

### Outputs

1. Figure(s) of signed change maps, by model type, five seeds × ten prefix extensions (\(\varnothing\to1{:}1\) through \(1{:}9\to1{:}10\)).
2. The underlying numerical maps \(D_{i,k}^{(a,s,t)}\) at full 64×64 resolution.
3. Minimal metadata: fixed image identifier and \((c_i,m_i,f_i)\), checkpoint epoch, model type, seed, prefix transitions, and color limits.

**Interpretation:** The figure shows both the **spatial pattern** of change and the **prefix order** in which the patterns occur. It does not, by itself, assign the changes to hierarchy levels or claim improved accuracy.

## Figure 2 — Alignment of model changes to ground-truth hierarchy

### Ground-truth conditional-average references

The dataset provides full images and nested labels, not explicit coarse-only renders. Define reference images directly from the **training set** (not the test set), using the labels of the selected test image:

\[
\begin{aligned}
G_0&=\mathbb E_{\mathrm{train}}[x],\\
G_C(i)&=\mathbb E_{\mathrm{train}}[x\mid c=c_i],\\
G_{CM}(i)&=\mathbb E_{\mathrm{train}}[x\mid c=c_i,m=m_i],\\
G_{CMF}(i)&=\mathbb E_{\mathrm{train}}[x\mid c=c_i,m=m_i,f=f_i].
\end{aligned}
\]

These are **conditional mean images**, not literal separate renderings of coarse/mid/fine components. For example, \(G_C\) averages all dumbbells over their different connector shapes and fine details; \(G_{CM}\) averages images with the specified connector over fine details.

For the main analysis, use three **signed** ground-truth transitions:

\[
\boxed{T_C(i)=G_C(i)-G_0,\qquad T_M(i)=G_{CM}(i)-G_C(i),\qquad T_F(i)=G_{CMF}(i)-G_{CM}(i).}
\]

- \(T_C\): what changes in the conditional average when **coarse identity** is known, beyond the global mean.
- \(T_M\): what changes in the conditional average when **mid identity is additionally known**, beyond coarse identity.
- \(T_F\): what changes when **fine identity is additionally known**, beyond coarse and mid.

\(T_C\) was added on 2026-11-03 together with the empty-prefix step \(D_1\). Every \(D_k\), including \(D_1\), is compared with all three references; \(D_1\) against \(T_M\) and \(T_F\) shows whether z1 already carries mid and fine changes.

A perfectly hierarchical MRL decoder (z1 = c, z1:2 adds m, ...) would give \(D_k = T_\ell\) exactly, so the conditional means are the right references under MSE.

The synthetic generator draws coarse and mid jointly; do **not** invent a coarse-only render. The separate no-fine render is an optional qualitative illustration of the mid-conditioned object, not the definition of \(G_{CM}\).

### Signed cosine alignment

Flatten each 64×64 map to a 4096-vector. For each model, seed, and prefix extension, compute:

\[
\boxed{
A_{\ell,k}^{(a,s,t)}(i)=
\frac{\langle D_{i,k}^{(a,s,t)},T_\ell(i)\rangle}
{\|D_{i,k}^{(a,s,t)}\|_2\,\|T_\ell(i)\|_2},
\qquad \ell\in\{C,M,F\},\ k=1,\ldots,10.
}
\]

- \(+1\): change follows the same signed pixel pattern as the GT transition.
- \(0\): orthogonal/no directional correspondence.
- \(-1\): opposite signed pattern.
- No near-zero cut-off for now (decided 2026-11-03): alignment is NaN only if a norm is exactly zero. Cosines on very small changes will be noisy; read them with that in mind.

**Figure layout:** For a selected model and seed at a fixed epoch, display \(T_C\), \(T_M\) and \(T_F\) with the same red-negative / blue-positive convention; beside or below them, plot **three alignment curves** across \(k=1,\ldots,10\) (coarse, mid and fine). Use a y-axis covering \([-1,1]\), and label x-axis as the prefix extension \(1{:}(k-1)\to1{:}k\) (first tick \(\varnothing\to1{:}1\)).

**Outputs:**

1. Three GT signed reference images, \(T_C\), \(T_M\) and \(T_F\).
2. Three alignment curves for the selected model/seed.
3. The **3×10 numerical alignment matrix** \([A_{\ell,k}]\), with undefined values recorded explicitly.

**Interpretation:** Check visually whether positive alignment with \(T_C\) comes first, then \(T_M\), then \(T_F\) in later extensions. **Do not add a weighted mean-prefix score, hierarchy-order score, or forced hard assignment per prefix.** Several dimensions may align with the same reference, both, or neither. Cosine similarity measures signed *pattern* alignment, not the amount of pixel change.

## Figure 3 — Seed distribution of hierarchy alignment (AE vs MRL)

For the **same image, epoch, and GT reference** \(T_\ell(i)\), compute \(A_{\ell,k}^{(a,s,t)}(i)\) for all five seeds.

### Figure layout

- Side-by-side panels: **AE** and **MRL-AE**.
- Horizontal axis: prefix extension \(k=1,\ldots,10\).
- Three series: \(A_{C,k}\), \(A_{M,k}\) and \(A_{F,k}\).
- Show all **five individual-seed curves** as faint lines, with a **mean curve** and a variability band (e.g., observed min–max across five seeds) for each hierarchy level.
- Keep the same signed-alignment y-axis \([-1,1]\) in both model panels.
- Show missing/undefined alignments as missing points, **not zeros**; report the valid-seed count at each \((\ell,k)\).

### Outputs

1. Cross-seed alignment figure for AE vs MRL at the same epoch.
2. Raw alignment values: model type × seed × hierarchy transition × prefix extension.
3. Per-prefix, per-transition summary (mean, min, max, number of valid seeds).

**Interpretation:** A high mean with a narrow spread suggests seeds produce consistently GT-aligned changes. A low or negative mean with a narrow spread is **not** evidence for hierarchy alignment. At late stages, prefix changes may become almost zero; undefined/unstable cosine values must not be mistaken for disagreement.

## Recommended order of work

1. Fix **one test image and one early training epoch** (then the 4 images and 8 epochs in Settings). Produce Figure 1 for AE and MRL across all five seeds.
2. Compute training-set conditional averages for the selected image's \((c,m,f)\); inspect \(T_M,T_F\).
3. Produce Figure 2 for one chosen seed; inspect the signed maps and alignment curves together.
4. Produce Figure 3 across all five seeds for both models.
5. Only after checking these figures, consider repeating across images or epochs. Those extensions are **outside this initial specification**.

## Decisions explicitly excluded

- No reconstruction classifier or linear probe in this analysis.
- No reconstruction-correction / pixel-improvement metric.
- No separate change-magnitude bar plot.
- No extra scalar ordering score; evaluate the observed ordering from curves.
- No hierarchy segmentation masks or exclusive pixel attribution to coarse/mid/fine.
- No assumption that one latent dimension corresponds to exactly one data-hierarchy level.
- No assumption that the final trained MRL must display nonzero changes at later prefixes: its first prefix may already encode most of the image.

## Runs, outputs and pages (record, 2026-11-05)

Results root: `/home/argha/Mat_embedding/files/results/`. Pages are private claude.ai artifacts.
Full list with all Experiment 1 folders: `NonLinearPBMeetingNotes/2026_11_05_hier_exp1_session_summary.md`.

| Run | What | Outputs (in the run folder) | Pages |
|---|---|---|---|
| `exprmnt_2026_11_03__15_08_36` | 10-d, lr 6e-5, AE + MRL, seeds 0–4 (the run in Settings) | `figures/prefix_change/` (data, alignment.csv, fig1–3, slider/, jaccard/, error_attribution/, pca_trajectory/, spatial_trajectory/), `figures/recon_epochs/` | MRL slider https://claude.ai/artifact/6bNqsjbGQLXFY8BYCWncMT · AE slider https://claude.ai/artifact/XafsLH5KZ6wUcsfnTkfaVh · Jaccard https://claude.ai/artifact/5SbJW89dUs7FJhMQhFWoUR · recon by epoch https://claude.ai/artifact/CdAxuLvf38MhC8P1QTbLCr |
| `exprmnt_2026_11_05__11_44_34` | same, **5-d latent** | `figures/prefix_change/` (data, alignment.csv, summary.csv, fig1–3, slider/, recon_cosine/, steps_seeds/) | MRL slider https://claude.ai/artifact/NdQWC4uPY7xCf25AHwpLM4 · AE slider https://claude.ai/artifact/4DPukzgddNkCwuz9ryRk45 · recon + cosine https://claude.ai/artifact/CBLfETePCXeAzk2o871mpG · every step × seed https://claude.ai/artifact/VqBMgRvyj737d7Rm2Ya1o9 |
| `exprmnt_2026_11_05__13_45_22` | 5-d, **pair objective**: codes {z1}, {z1, zj}; D_1 = x̂_{z1} − decode(0), D_j = x̂_{z1,zj} − x̂_{z1} | `figures/pair_change/` (data, alignment.csv, recon_cosine/, z1_seeds/) | pair vs MRL https://claude.ai/artifact/B1H2gKeuoVVCT3KDodojmL · z1 across seeds https://claude.ai/artifact/VspVwGPFhaqFaGPVaY5nve |
| `exprmnt_2026_11_05__20_43_41` | **CIFAR-100** 32×32 colour, 10-d, lr 6e-5, 25 epochs, seed 0: AE, MRL-prefix, MRL-pair (data `data/cifar100/v1`) | `figures/recon_epochs/` | recon by epoch https://claude.ai/artifact/N82GdA3ag5n31asTL6nYj8 |
| `exprmnt_2026_11_05__21_01_13` | CIFAR-100, lr 3e-4, 100 epochs (converged), AE / MRL-prefix / MRL-pair | checkpoints, training/ | — |
| `exprmnt_2026_11_02__19_43_27` | final run (lr 3e-4, 200 epochs), for reference | `figures/recon_epochs/` | https://claude.ai/artifact/6X1PEYsmhDD9utQHzTeDis |
| `exp1_report/` | compiled report | `index.html` | https://claude.ai/artifact/PtNCPeCVLaeQJs7kU8m33S |

Scripts (`NonLinearPBExperiments/scripts/`): `hier_exp1_prefix_change.py` (+ `_html.py`), `hier_exp1_seed_jaccard.py`,
`hier_exp1_error_attribution.py`, `hier_exp1_prefix_pca.py`, `hier_exp1_spatial_trajectory.py`, `hier_exp1_recon_cosine_html.py`,
`hier_exp1_pair_change.py`, `hier_exp1_z1_seeds_html.py`, `hier_exp1_steps_seeds_html.py`, `prepare_cifar100.py`, `cifar_recon_epochs_html.py`.
