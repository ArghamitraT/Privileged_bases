# NonLinearPB: Objective-Specific Privileged Bases in Nonlinear Representation Learning

**Location:** `/home/argha/Mat_embedding/code/NonLinearPB/`  
**Status:** Research program in development. This document records the current aim and scope; experiment plans contain the implementation decisions.  
**Shared project norms:** [MainREADME.md](../MainREADME.md).

## Code location

All code for this project goes in **`NonLinearPBExperiments/`** inside `NonLinearPB/`:

```text
/home/argha/Mat_embedding/code/NonLinearPB/NonLinearPBExperiments/
```

This includes experiment entry points, model implementations, data and evaluation helpers, training code, and tests. Documentation and plans remain in their existing locations; run outputs follow the results convention in the experiment plan.

## Professor meeting reports

Create NonLinearPB reports for meetings with Argha's professors in **`NonLinearPBMeetingNotes/`**, inside this project. Keep the questions, analyses, evidence/figures, and conclusions together in dated Markdown reports, using descriptive filenames such as `2026_10_14_vae_beta_vae_mrl_report.md`.

**Use very, very simple language. Be precise and to the point, as if explaining the work directly to someone. Nothing fancy.** Use short sentences and concrete examples. Explain what we did, how we did it, and what we see. Explain necessary technical terms when first used; avoid decorative wording and unnecessary jargon.

Every report follows this section order:

1. **Questions we are trying to answer** — scientific questions and the scope of the comparison.
2. **Analysis we did** — start with a **Training and evaluation settings** subsection, then analyses actually completed, supporting results/figures, and limitations; mark pending work explicitly. Distinguish current experiments from historical pilots/configurations.
3. **Summary** — what the evidence supports and what remains unresolved.
4. **Future directions** — curated by Argha; proposed ideas are not automatically commitments to new experiments.
5. **Technical details** — **always the final section**, containing experiment names and exact current/previous results-folder paths, environment/package versions, data, architecture/loss, hardware/tmux/logs, scripts, and reproducibility/provenance locations. The training/evaluation settings table belongs at the start of Analysis rather than here.

Put all technical subsections inside that final section; do not append another report section after it. Write mathematical expressions in LaTeX (`$...$` inline, `$$...$$` display); actual code remains in code blocks. Distinguish generated outputs from analyses already reviewed. Preserve previous reports when creating a new meeting version.

**Meeting-report references must not use hyperlinks.** Reports may be read where links cannot be accessed. Use plain text or code-formatted **full absolute paths** for files, figures, result directories, scripts, and sources. Do not replace the full location with a relative path or only a timestamped folder name. Explain parent experiment folders versus nested condition/seed runs, and verify listed locations from the manifest and filesystem. Include the necessary results and explanations in the report itself.

Initial report: `/home/argha/Mat_embedding/code/NonLinearPB/NonLinearPBMeetingNotes/2026_10_14_vae_beta_vae_mrl_report.md`. It includes the verified three-seed numerical summary, pending visual analysis, an unfilled user-curated future-directions section, and the full technical record at the end.

## Training and retraining: always use tmux

**For every training or retraining job, open a detached tmux session and launch the job there.** Keep persistent console logs and checkpoints. Training, subsequent evaluations, and figure generation should continue inside that session when SSH or the terminal disconnects. A machine reboot or process failure still requires checkpoint recovery; tmux protects against terminal disconnection.

For paired dSprites retraining, use:

```bash
# From code/; a fresh session name and results directory are printed.
bash NonLinearPB/NonLinearPBExperiments/scripts/run_dsprites_comparison_tmux.sh --profile repo

# Return to the existing paper-style defaults when needed.
bash NonLinearPB/NonLinearPBExperiments/scripts/run_dsprites_comparison_tmux.sh --profile paper
```

The [paired launcher](NonLinearPBExperiments/scripts/run_dsprites_comparison_tmux.sh) runs the [comparison workflow](NonLinearPBExperiments/scripts/run_dsprites_comparison.py) in tmux. The workflow trains one model per GPU, performs the existing metric/reconstruction/traversal evaluations, and automatically generates Figure 6 A/B after both finish. Defaults are seed 0, 100 epochs, batch 256, and a 2,000-epoch metric-classifier cap. `repo` uses Adam/Xavier/ReLU for both models, fixed β=1 for VAE, and capacity control for the candidate; `paper` uses Adagrad/default initialization/Tanh with fixed β=1 and β=4. `training-only` uses Adam/default initialization/Tanh with fixed β=1 and a capacity candidate.

Attach using the printed `tmux attach -t <session>` command. Detach with Ctrl-b, then d; do not press Ctrl-c in a running job. Closing the terminal leaves the detached job running. Inspect `workflow.log`, the model logs, and `status.json`; a session existing does not imply successful completion. Preserve old sessions and results, and use a new timestamped folder for retraining.

## Project aim

Determine whether Matryoshka Representation Learning (MRL) selects a reproducible, objective-specific privileged basis in nonlinear representations, and characterize when it fails to do so.

The starting point is our established result in the linear autoencoder setting: MRL can select an objective-specific privileged basis under the assumptions of that result. The next step is to investigate what survives when the encoder is nonlinear. The linear result is motivation, not a guarantee of nonlinear identifiability.

The central question is:

> Does an ordered learning objective select reproducible latent coordinates or ordered subspaces in a nonlinear model, and does changing the task systematically change which directions become privileged?

Once this is established, we want to explain what those directions represent and whether their changes matter for prediction, transfer, or distribution shift.

In plain language: a model may solve a task equally well using several different internal coordinate systems. We want to know whether MRL makes a particular coordinate system reproducible, whether the task determines that choice, and whether we can understand the features it selects.

## Why objective specificity matters

Feature relevance depends on the task. Distinguishing a zebra from a cow may emphasize stripes; distinguishing a zebra from a starfish may emphasize body shape. These are motivating examples, not measured results. The input can remain the same while the useful features change.

The scientific object is **basis selection under representation non-identifiability**. Feature importance asks which parts of an existing representation affect a prediction. This project asks why learning selects particular latent directions when alternative representations could support equivalent performance, and whether that selection follows the objective.

## Research questions

1. **Privileged-basis formation:** Under what conditions does MRL break representational symmetry and select reproducible coordinates or ordered subspaces in nonlinear models?
2. **Objective dependence:** Does changing the task change the selected basis more systematically than changing initialization or optimization randomness for the same task?
3. **Semantic structure:** What information do the selected directions contain? Do they respond selectively to known factors, or mix several factors?
4. **Functional significance:** Does the ordering agree with ablation or intervention effects, support useful short prefixes, or help explain transfer and distribution-shift behavior?

The first two questions are the core. Semantic analysis explains the selected directions. Functional analysis tests their practical significance.

## Distinctions we must preserve

- **Useful prefixes:** A small prefix supports good task performance.
- **Task-specific prioritization:** Early coordinates preferentially contain information useful for the chosen objective.
- **Privileged basis:** The objective distinguishes particular coordinates or subspaces from otherwise equivalent alternatives. We must specify which transformations remain equivalent.
- **Reproducibility / identifiability:** Independent solutions agree under a stated equivalence, such as sign changes or transformations within an unresolved subspace.
- **Factor selectivity / disentanglement:** A coordinate responds mainly to a particular factor rather than an arbitrary mixture.
- **Semantic hierarchy:** Features form a meaningful coarse-to-fine structure. An importance ordering alone does not establish this.
- **Functional importance:** Changing or removing a component changes model behavior. Probe accuracy alone does not establish this, and latent ablations alone do not establish real-world causal meaning.

These are separate claims. Strong prefix performance does not prove a unique basis, disentanglement, semantic hierarchy, or stability across seeds.

## Evidence and controls

Use the same data and architecture across objectives wherever possible. Separate the contributions of the data distribution, architecture, regularization, objective, and optimizer.

- **Across seeds:** Compare coordinates in their learned order, allowing sign changes where appropriate. Report matched-coordinate agreement separately so matching does not hide ordering instability. Use subspace comparisons when individual axes are unresolved; rotation-invariant similarity alone cannot establish coordinate agreement.
- **Across tasks:** Test whether task changes produce systematic changes beyond within-task seed variability. Define the comparison before examining final outcomes.
- **Under rotations:** Rotate learned representations and refit the relevant heads. Measure whether prefix utility changes; fixed-head degradation alone would confound representation changes with head mismatch.
- **Against factors:** Use known generative factors, probes, and controlled factor interventions to distinguish decodable information from selective coordinate responses.
- **Against alternatives:** Include ordinary task supervision without MRL. Add sparsity, regularization-based ordering, nested dropout, or other comparators as justified by the experiment.
- **Under optimization controls:** Compare optimizers where feasible, and distinguish objective-induced effects from optimization-induced coordinate preferences.

Sparse prefix schedules may constrain blocks or nested subspaces without resolving every coordinate. Full-prefix training is a separate condition to test; it is not assumed to guarantee nonlinear uniqueness. Tied task relevance, redundant coordinates, and unused dimensions may leave genuine ambiguities.

## Current experiment: task-specific MRL–β-VAE on dSprites

Detailed design: [dSprites task-specific MRL experiment plan](plans/dsprites_task_specific_mrl_experiment_plan_AEOLUS.md).

The immediate experiment uses the same dSprites images with different factor-prediction objectives. Start with X-position and Y-position; finalize rotation targets only after addressing periodicity, shape symmetry, and image-level ambiguity.

Compare VAE, β-VAE, supervised β-VAE with a full-latent task head, and β-VAE with nested task heads. Include validation-selected post-hoc coordinate sorting of the supervised baseline.

The primary question is whether changing the prediction objective changes which information is prioritized in early latent coordinates. Evaluate frozen encoders with prefix probes, coordinate–factor maps, and latent traversals, alongside seed variability and reconstruction/KL diagnostics.

**Scope of this experiment:** It can provide evidence for task-specific prioritization. It does not by itself certify global optimality or prove nonlinear basis identifiability. A single-factor task may concentrate sufficient information in one coordinate without ranking every remaining factor.

## Broader experimental progression

1. **Controlled nonlinear settings:** Use known factors and explicit degeneracies to test basis selection and its limits. Where possible, design objectives with analytically known lower bounds; claim global optimality only after establishing attainability and verifying the bound under the exact model constraints.
2. **Objective switching:** Change factor relevance or which factors define the task while holding inputs fixed. Predict the expected changes before measuring them.
3. **Semantic and functional analysis:** Combine factor probes, controlled interventions, ablations, and prefix curves. Extend to classification and contrastive objectives without assuming the same theory applies to both.
4. **Real-model validation:** Test whether reproducible bases make task comparisons more reliable on pretrained representations or fine-tuned models. A linear head on frozen features may remain within the linear setting and cannot alone establish the nonlinear claim.
5. **Transfer and shift:** Investigate whether shared privileged directions predict transfer, and whether re-ranking or reusing them provides a practical benefit.

## Longer-term method development

A possible extension is to learn task-adaptive ordering pressure rather than manually choosing dimension penalties or prefix weights. This is an open method-development direction, not an implemented contribution.

The current MRL experiment avoids prescribing which generative factor belongs in which coordinate, but it still imposes a prefix structure and chooses prefix sizes and weights. We should therefore distinguish **no predefined factor ranking** from **no predefined ordering pressure**. Learned weights would also need constraints against trivial solutions that neglect difficult prefixes.

Semantic hierarchy, biological state transitions, continual learning, diffusion, and question-dependent bases in larger models remain future directions until the core basis-selection evidence is established.

## Documentation and current artifacts

- Completed paired retraining: tmux `dsprites-retrain-2026_10_13__14_52_36`; [results root](/home/argha/Mat_embedding/files/results/exprmnt_2026_10_13__14_52_36/) and [workflow log](/home/argha/Mat_embedding/files/results/launch_logs/dsprites-retrain-2026_10_13__14_52_36/workflow.log). Repo-inspired Adam/Xavier/ReLU, 100 epochs, seed 0; fixed-beta1 VAE versus capacity candidate. Mean metric scores: **83.01% / 98.49%**. [Figure 6 A/B](/home/argha/Mat_embedding/files/results/exprmnt_2026_10_13__14_52_36/figure6/figure6_AB_2026_10_13__15_16_48.png) and all existing diagnostics completed automatically. Capacity reconstructions remain blurry and traversals mix factors; all six trained-model classifiers reached the 2000-epoch cap. See [the evaluation record](plans/beta_vae_reproduction_plan.md#completed-results-and-evaluation-review).
- [Detached one-seed pilot launcher](NonLinearPBExperiments/scripts/run_dsprites_seed0_tmux.sh): launches 100-epoch β=1/β=4 runs on GPUs 0/1 in tmux, with a 2,000-epoch metric-classifier cap and persistent console logs. Attach with `tmux attach -t dsprites-seed0-100ep`; detach with Ctrl-b, then d, and switch windows with Ctrl-b, then n. Closing the terminal leaves detached jobs running.
- [VAE / β-VAE reproduction entry point](NonLinearPBExperiments/experiments/dsprites_vae_reproduction.py): CONFIG, CLI, training, metric evaluation, diagnostics, and output management.
- [Reproduction test runner](NonLinearPBExperiments/tests/run_tests.py): subprocess-based correctness tests. Model/data/metric helpers are in `NonLinearPBExperiments/vae_reproduction/`.
- Optional VAE training switches: `--objective capacity`, `--optimizer adam`, `--initialization xavier`, and `--decoder-activation relu`; existing fixed-beta/Adagrad/Tanh defaults remain. See [implemented switches and commands](plans/beta_vae_reproduction_plan.md#implemented-switches--2026-10-13). Metric calculations and Figure 6 analysis are retained; checkpoint loading respects decoder options.
- [MainREADME.md](../MainREADME.md): shared project norms.
- [dSprites experiment plan](plans/dsprites_task_specific_mrl_experiment_plan_AEOLUS.md): current experimental proposal and unresolved design choices.
- [VAE / β-VAE reproduction plan](plans/beta_vae_reproduction_plan.md): implemented PyTorch baseline, completed run locations and scores, Figure 6 provenance, and proposed controlled comparisons.
- [Three-seed VAE / β-VAE versus reconstruction-Matryoshka plan](plans/dsprites_vae_beta_vae_matryoshka_three_seed_plan.md): implemented four-condition experiment, original Adagrad/Tanh settings, all ten reconstruction prefixes, one full KL term, 12 model trainings and 36 classifier fits without score filtering. MRL Figure 6 preserves original coordinate order for ranking/seed comparisons; ordinary figures retain variance sorting. Full training launched after successful two-epoch smoke validation.
- [Four-condition tmux launcher](NonLinearPBExperiments/scripts/run_dsprites_mrl_matrix_tmux.sh): use `--smoke --epochs 2` for reduced-data validation of the complete matrix. [Workflow](NonLinearPBExperiments/scripts/run_dsprites_mrl_matrix.py) defaults to the agreed full three-seed experiment when explicitly launched without `--smoke`; preserve that distinction.
- Completed MRL smoke validation: [results](/home/argha/Mat_embedding/files/results/test_runs/exprmnt_2026_10_13__20_59_08/), tmux `dsprites-mrl-2026_10_13__20_59_08`; exactly two epochs on 8192 images per model, 12 model runs, 36 classifier fits, six A/B figure sets, no score filtering. MRL columns retain native order. Runtime 205.924 seconds.
- **Completed full MRL matrix:** tmux `dsprites-mrl-2026_10_13__21_22_10`; [results](/home/argha/Mat_embedding/files/results/exprmnt_2026_10_13__21_22_10/) and [workflow log](/home/argha/Mat_embedding/files/results/launch_logs/dsprites-mrl-2026_10_13__21_22_10/workflow.log). All 12 models finished 100 epochs; 36 classifier fits and six Figure 6 pairs completed. Runtime 7h11m. Unfiltered means: VAE **80.48%**, β-VAE **77.22%**, MRL VAE **83.85%**, MRL β-VAE **78.23%**. No scores discarded. See [run record](plans/dsprites_vae_beta_vae_matryoshka_three_seed_plan.md#15-full-experiment-launch--2026-10-13); detailed cross-seed visual review remains pending.
- [disentanglement_lib reproduction plan](plans/CODEX_disentanglement_lib_CPU_reproduction_AEOLUS.md): reference implementation reproduction workflow.
- [Reproduction notes](REPRODUCTION_NOTES.md): environment, data, and reproduction record.
- `disentanglement_lib/`: existing reference library checkout.
- `disentanglement_lib_reproduction/`: reproduction helper scripts.

The supplied research notes contain candidate theory statements, literature leads, and proposed experiments. They are working material; theorem assumptions, citation accuracy, and novelty claims must be checked before being presented as established results.
