10. Prediction 2 has a built-in excuse. "If the opposite appears, lambda_sup is miscalibrated" means the prediction can never fail. Also, S_posX only needs about one dimension for its label, so supervision may protect just one unit. That makes the prediction weaker than it reads. Drop the excuse clause, or move calibration into an acceptance check instead.



means what?

There are two separate criticisms here.
1. What does "built-in excuse" mean?

Our Prediction 2 says:

    The number of active dimensions dactive
    dactive​ should change less with β for S_posX than for S_recon, because supervision protects useful dimensions.

Suppose we get these hypothetical results:

β
	

S_recon
	

S_posX

0
	

10
	

10

1
	

8
	

7

4
	

5
	

3

Decrease
	

5
	

7

Our prediction is wrong! S_posX lost more dimensions than S_recon.

But the original document says:

    If the opposite appears, λsup
    λsup​ is miscalibrated.

That creates a problem.

We could always blame λsup
λsup​ whenever the prediction fails, instead of accepting that our hypothesis might be incorrect.

A scientific prediction must be falsifiable. We must allow the experiment to prove us wrong.

So we should calibrate λsup
λsup​ before running the main experiments and then test the prediction without changing it afterward.
2. Why might supervision protect only one dimension?

Remember our supervised objective:
L=Lrecon+βLKL+λsupLsup
L=Lrecon​+βLKL​+λsup​Lsup​

For S_posX, the supervised task is to predict X position.

Suppose our encoder learns:
z1=f(X)
z1​=f(X)

If z1
z1​ already contains sufficient information to predict X accurately, the supervised objective has little reason to preserve additional coordinates.

For example:
z1⏟X information,z2,z3,…,z10
X information
z1​​​,z2​,z3​,…,z10​

Increasing β may cause coordinates z2,…,z10
z2​,…,z10​ to collapse, while supervision keeps z1
z1​ active.

This means supervision can successfully protect the information needed for X prediction without protecting many latent dimensions.

One nuance: a single scalar can encode 32 X positions, but a linear softmax classifier must also be able to decode them. This is possible if the scalar organizes the positions appropriately.
3. So what should we predict instead?

I would separate two hypotheses:

Prediction A — Activity: Increasing β reduces dactive
dactive​. Whether supervision reduces that decline is an empirical question, not something we should assume.

Prediction B — Information preservation: At high β, S_posX should retain stronger X-position accessibility in its early prefixes than S_recon, provided the supervision is effective.

Prediction B directly tests what supervision is intended to do.

For example:

At β = 4
	

S_recon
	

S_posX

Active dimensions
	

5
	

3

Prefix-1 X accuracy
	

35%
	

90%

Even though S_posX has fewer active dimensions, it has organized X information more effectively.

The key distinction:
Preserving task information≠Preserving more dimensions
Preserving task information=Preserving more dimensions​

And that is why the original Prediction 2 is weaker than it initially sounds.

For our research, I think where the task-relevant information is located matters more than how many dimensions survive.
Pasted markdown(9).mdDocument
got so The revision is much stronger than the earlier version. Reusing the MLP β-VAE, nesting only the reconstruction term, keeping the KL un-nested, and crossing objective with β are all sound decisions. Your reason for keeping the KL un-nested (a nested KL would penalize coordinate 1 ten times and create the ordering artificially) is correct and worth keeping in the write-up. These are the issues I'd still fix:

Should fix

1. The lambda_sup grid is probably too small to matter. Reconstruction BCE is summed over 4096 pixels, so on dSprites it's roughly tens of nats per image. Cross-entropy is at most log 32 ≈ 3.5 nats for X position and log 3 ≈ 1.1 for shape. At λ = 1–30 the supervised term may barely affect training. Check the loss magnitudes in one short run before fixing the grid; you may need values in the hundreds. Also say what happens if no value in the grid reaches 80%.

2. A single λ shared across objectives doesn't give equal pressure. For the same λ, the 32-class X loss is about 3× larger than the 3-class shape loss. "Same λ" therefore isn't a controlled comparison. Either normalize by log C, or tune λ for each objective with a criterion matched to chance level (e.g. a fixed fraction of the gap between chance and full-probe accuracy).

3. β = 0: option (b) doesn't work, so commit to (a). Without a KL term, the encoder can make μ arbitrarily large instead of shrinking σ, so σ may never fall below 0.01 in absolute terms even when the noise is irrelevant. The meaningful quantity is σ relative to std(μ). Option (a), setting z = μ, is cleaner. Also note that at β = 0 the scale of each coordinate is arbitrary. That affects §5.4 and the whitened/unwhitened choice downstream.

4. The null based on d_active is wrong unless the collapsed units come last. max(k, j)/d_active assumes the first k coordinates are all active. If a dead unit sits at position 3, prefix 3 contains only 2 active coordinates. The correct null is k_active(k) · j / d_active / min(k, j), where k_active(k) is the number of active units among the first k. MRL should push active units to the front, but that should be checked, not assumed.

5. KL-based d_active can't be computed at β = 0. Setting it to 10 by hand is a guess, because a deterministic autoencoder can also leave coordinates unused. Use the variance-based active-unit count from Burda et al. (Var_x(μ_i) > 0.01, after normalizing scale) for all 12 cells. That gives one definition across the whole grid, and the KL_i vector can be kept as an extra field.

6. The condition number (§5.4) will be dominated by collapsed units. At β > 0, dead units have near-zero variance in μ, so the condition number is huge by construction and tells you nothing. Compute it on the active subspace only. The same applies downstream: whitening must not include dead coordinates, or their noise gets blown up.

7. Figure 0 scores aren't comparable across factors. Raw macro accuracy has chance at 1/3 for shape and 1/32 for X, so the heatmap colors mean different things in different columns. Use normalized accuracy, (acc − chance)/(1 − chance), or mutual information. Also, logistic regression on a single scalar can only cut the line into ordered intervals. Orientation is circular, and squares and ellipses look the same under some rotations, so that column will look weak everywhere regardless of the encoder.

Worth reconsidering

8. X and Y are tied in S_recon. They explain the same pixel variance, so coordinate 1 can be any mix of them, with the mix changing from seed to seed. Of all the cells, S_recon at β = 0 has the least pressure to align with the axes, so it's the least identifiable, not the "cleanest." Prediction 3 and check 7 should allow position to be split across coordinates 1–2.

9. Training one cell differently breaks the factorial design. Keeping S_recon at β = 4 at 100 epochs while the other cells get fewer makes that cell incomparable with its row and column. Either train everything for 100 epochs, or add a separate continuity run outside the grid.

10. Prediction 2 has a built-in excuse. "If the opposite appears, lambda_sup is miscalibrated" means the prediction can never fail. Also, S_posX only needs about one dimension for its label, so supervision may protect just one unit. That makes the prediction weaker than it reads. Drop the excuse clause, or move calibration into an acceptance check instead.

11. Acceptance check 1 needs a tolerance. Beyond d_active, the reconstruction curve is flat, and small fluctuations will break strict monotonicity. Use something like "non-increasing within 0.1% of the loss."

Minor

- Repository layout. §9 introduces src/, but the existing code lives in NonLinearPBExperiments/vae_reproduction/ and mrl_vae_loss already exists there. Extend that code in place, adding the heads to the sequential-backward path, rather than starting a parallel tree.
- Compute budget. Tuning (3 β × 4 λ) plus 36 runs × 100 epochs × 10 decoder passes per step adds up to about 48 long runs. Add a wall-clock estimate from your existing MRL runs.
- Old text. The previous Tier 1 doc is still at the bottom inside an HTML comment, including the old K = [1, 2, 3, 4, 6, 8, 10] and the conv net. Move it to its own file so the two specs don't contradict each other.

Issues 1–5 are the ones to settle before writing predictions.md. They change either what gets trained or how every downstream null is computed.

✻ Worked for 53s · done 11:02 AM

※ recap: You're designing pretraining for the privileged-basis experiment, and I reviewed your edited spec without changing it. Next, settle the top five fixes, starting with the supervised-loss weight grid and a single shared active-unit count, before writing predictions.md. (disable recaps in /config)

❯ wait one by one, precise, short:
  1. what is lambda,? is nes of nats? what is nats? Cross-entropy is at most log 32 ≈ 3.5 nats for X position and log 3 ≈ 1.1 for shape. how did you come up with this?
  2. I agree, what do you want to normalize by logc?
  3. what is option B?  Also note that at β = 0 the scale of each coordinate is arbitrary. That affects §5.4 and the whitened/unwhitened choice downstream?? means what?
     talk abot these first

  Searched for 2 patterns, ran 1 shell command

1. What λ is, what nats are, and where those numbers came from

- λ is lambda_sup, the weight on the supervised loss in L = L_recon + β·L_KL + λ·L_sup.
- Nats are the unit a loss is in when it uses the natural log (ln). Bits are the same idea with log₂. PyTorch's BCE and cross-entropy both use ln, so all your losses are in nats.
- log C is the cross-entropy of a classifier that guesses uniformly over C classes: −ln(1/C) = ln C. So ln 32 ≈ 3.47 for X position and ln 3 ≈ 1.10 for shape. I was wrong to say "at most." That's the starting, chance-level value; a confidently wrong classifier can go higher. It's the right size to compare against, though.
- Reconstruction scale. I checked your existing runs at epoch 100. The prefix-averaged reconstruction loss is about 88 nats at β=1 and about 118 nats at β=4, because BCE is summed over 4096 pixels. So with λ=1 the supervised term is roughly 3.5 vs 100, which is too small to steer the encoder. λ probably needs to be around 30–300.

2. Normalizing by log C

Divide each task's cross-entropy by ln C:

L_sup = mean_k  CE_k / ln(C)

Then every task's supervised loss starts at 1.0 at chance and goes to 0 when solved, whether C = 3, 6 or 32. One value of λ then means the same pressure for S_posX, S_shape and S_scale, so you can tune it once per β and reuse it fairly across objectives.

3. Option (b), and why arbitrary scale matters

Option (b) is from your §4.2: at β=0, keep sampling z = μ + σε and assert that the mean σ is below 0.01 by the end of training. The problem is that without a KL term the encoder can make μ larger instead of making σ smaller. If μ spans ±1000, a σ of 0.5 is effectively zero noise, yet the assertion fails. Option (a) avoids this by setting z = μ when β = 0.

Arbitrary scale means that without a KL term nothing fixes how large each coordinate is. Multiply coordinate 1 by 100 and divide the decoder's first-layer weights for it by 100, and the loss is identical. So at β=0 the variance of each μᵢ is an accident of training: it can differ by orders of magnitude between coordinates and between seeds. At β>0 the KL pulls each active coordinate toward N(0,1), which pins the scale.

Two consequences:
- §5.4, condition number: at β=0 it measures those arbitrary scales, not real structure.
- Downstream alignment: the unwhitened between-class scatter's top direction gets pulled toward whichever coordinates happen to be large. The alignment number would then partly reflect scale, not geometry. The whitened variant, or standardizing each coordinate before the analysis, removes this. For β=0 cells, standardize or whiten.

Those logs also bear on a later item: the β=4 run's prefix reconstruction goes 101.31 → 101.42 after prefix 5. It rises slightly, so a strict "monotone decreasing" check (§6 check 1) would already fail on your existing model. It needs a tolerance.

✻ Churned for 32s · done 11:11 AM

※ recap: We're reviewing your 12-encoder dSprites MRL pretraining plan, and I've explained points 1–3: what λ is, normalizing cross-entropy by log C, and fixing coordinate scale at β=0. Next, I'll go through issues 4–11 the same way, one at a time. (disable recaps in /config)

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
❯
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ auto mode on (shift+tab to cycle)


make necessary changes
…

I've revised the pretraining specification and separated the outdated Tier 1 experiment.

Revised MRL Pretraining Specification

Updated Markdown · 12 encoder configurations

Download revised specification

Archived Tier 1 Specification

Original version, preserved separately

Download archived specification
Main changes

    Supervised loss: Normalize by ln⁡C
    lnC, but calibrate λsup
    λsup​ separately for each task and β.

    β = 0: Use deterministic z=μ
    z=μ.

    Active dimensions: Replace KL-only counting with a scale-relative variance diagnostic and track the positions of inactive coordinates.

    Whitening: Exclude inactive dimensions and preserve the original MRL coordinate basis for alignment comparisons.

    Predictions: Remove the unfalsifiable supervision prediction and account for X/Y entanglement.

    Training: Keep the same epoch budget across all 12 configurations.

    Acceptance checks: Allow small reconstruction fluctuations and report failed calibration explicitly.

I also corrected the claim that log⁡C
logC is a maximum cross-entropy and removed the assumption that supervised models necessarily retain more active dimensions.

One important point: The variance-based active-unit threshold and supervision calibration floors still need empirical calibration before the predictions are frozen. I've made those decisions explicit in the document rather than inventing values.
MRL_dSprites_pretraining_revised.mdDocument
Tier1_original_archived.mdDocument

That works, and it drops the right one. 18 runs, which with prefix sampling and 40 epochs is roughly two original run-equivalents.

Dropping S_shape costs little. It was the awkward cell anyway: 3 classes means S_B has rank ≤ 2, so the closed-form estimator only gives you 2 directions and A(k) is undefined past k=2. S_posX (32 classes, rank 9) and S_scale (6 classes, rank 5) both give you nearly the full latent dimension to work with. And T_shape stays as a target task, which is where you actually need it — the source encoder S_shape was only ever a calibration anchor.

Two things to fix about 2 seeds.

You can't compute a standard deviation from 2 points, only a range. For the headline figure that's acceptable, because you're comparing measured A(1) against an analytic line rather than testing a difference between conditions — the prediction is cos²α and either the points land on it or they don't. But set a disagreement rule in advance: if the two seeds of a cell differ by more than 0.15 in A(1) or more than 1 in d_active, run a third seed for that cell. Otherwise you'll have two numbers and no way to tell which is the outlier.

Second: the 5 seeds for the learned Q_T are separate from encoder seeds and should stay at 5. They cost no encoder retraining — same frozen encoder, 5 restarts of a small rotation fit. That's the identifiability check, and it's the one place cutting seeds would actually break a measurement.

Keep the staging. Run the beta = 0 row first: 3 objectives × 2 seeds = 6 runs. That gets you Figure 0, the cos²α curve, and the random-rotation control. If the headline relationship doesn't appear there, you stop at 6 runs instead of 18. The beta = 1 and 4 rows are the basis-versus-order extension, and they're the part a deadline would cut.

Want me to update the spec file to 18 runs with the staging and prefix sampling?

Yes, but it can be much cheaper than the 12 runs the spec asks for.

Normalize the supervised loss first, then you only tune once. Right now L_sup for S_posX starts at ln(32) ≈ 3.47 and for S_scale at ln(6) ≈ 1.79, so the same lambda_sup applies roughly twice the pressure in one cell as the other. Divide the cross-entropy by ln(C):

L_sup = mean over k of  CrossEntropy(head_k(mask_k(z)), y) / log(C)

Now both start near 1.0 and a single lambda means the same thing in both. That removes one axis of the sweep for free.

Tune at beta = 4 only. It's the worst case — KL is competing hardest for capacity, so a lambda that clears the bar there clears it at beta = 0 and 1. If you want insurance, spot-check the chosen value at beta = 0 with one run.

Tune at 10 epochs. Prefix-1 accuracy on the supervised factor saturates early; you're looking for "is the supervision strong enough to claim coordinate 1," not final quality.

That gives you: 4 lambdas × 1 objective × 1 beta × 10 epochs ≈ 0.4 run-equivalents. Negligible against 18. And the winning config can be continued to 40 epochs rather than restarted.

One problem with the acceptance bar as written. "Prefix-1 X accuracy > 80%" over 32 classes is a hard target for a single coordinate — you're asking one scalar to resolve 32 ordered bins exactly. Use an ordinal criterion instead, either of:

    R² of a linear regression from mu_1 to X position > 0.9, or
    within-±1-bin accuracy > 80%.

Both measure "coordinate 1 carries X position" without penalizing the model for ordinary quantization error. The exact-match bar would make you crank lambda up until supervision dominates reconstruction, which distorts the very ordering you're trying to study.
…

Updated the pretraining specification to incorporate the cheaper experimental design.

Download updated MRL pretraining specification (.md)
Revised experiment

Setting
	

Updated design

Source objectives
	

S_recon, S_posX, S_scale

β values
	

0, 1, 4

Encoder seeds
	

2 initially

Training
	

40 epochs, uniform prefix sampling

Total
	

18 initial encoder runs

Learned rotation QTQT​
	

Retain 5 optimization restarts

Staging: Run the six β = 0 models first. Evaluate the factor heatmap, the alignment curve against cos⁡2α
cos2α, and the random-rotation control. Proceed to β = 1 and 4 only if the initial results justify it.

I also added the third-seed disagreement rule, cheaper supervised-weight tuning, ordinal X-position evaluation using R2
R2, and calibration-failure handling.
Three technical corrections worth retaining

    Two seeds can produce a sample standard deviation. It is simply a very unreliable estimate. Reporting both observations and triggering a third seed when they disagree is more informative.

    Normalizing CE by ln⁡C
    lnC does not guarantee one λ works for both tasks. The updated procedure starts with one shared λ but checks it on both objectives and across β.

    The X-position between-class scatter can have rank 10, not 9, in a 10-dimensional latent space. Shape remains limited to rank 2.

The specification is updated, but the calibration thresholds and geometric predictions still need to be frozen before confirmatory training.
MRL_dSprites_pretraining_revised.mdDocument
Failed to download file. Please try again later.

The file is available. I've verified it exists.

Download the updated MRL dSprites Pretraining Specification (.md)

This is the latest version with 18 runs, 40 epochs, prefix sampling, staged training, and updated supervision calibration.
MRL_dSprites_pretraining_revised.mdDocument
annot download

Sorry about the repeated download errors. I've created a fresh copy with a new filename and a ZIP backup.

Try the ZIP version first:

Download ZIP file

Download Markdown file (.md)

I've verified that both files exist and that the Markdown contains the updated 18-run experimental design.
dsprites_18run_spec_v2.mdDocument
Pretraining Specification: 9 MRL Source Configurations on dSprites

Status: Revised experimental specification, not yet preregistered. Final numeric calibration thresholds must be frozen before the confirmatory runs.

Scope: Pretrain 9 encoder configurations (3 objectives × 3 β values), each with 2 initial seeds, freeze them, and measure representation structure. Downstream target tasks, alignment estimators, and transfer tests belong to the separate Tier 1 specification. The superseded Tier 1 text was removed from this file and archived in Tier1_original_archived.md; do not implement its old conv net, different prefix set, or old null without further review.

Main question: Does changing an encoder's pretraining objective change where task-relevant information becomes accessible in an MRL-ordered latent representation? Vary β independently to examine its effect on learned representation geometry. Neither ordered coordinates nor disentanglement/identifiability is assumed in advance.
0. Notation
Symbol	Meaning
x	Binary 64 × 64 dSprites image (4096 flattened pixels)
d	10 latent coordinates
mu, logvar	Encoder outputs, sigma = exp(0.5 * logvar)
z	At β > 0 during training: mu + sigma * eps, eps ~ N(0,I); at β = 0 during training: mu; at all evaluations: mu
K	All prefixes [1, 2, ..., 10]
beta	KL weight in {0, 1, 4}
lambda_sup	Shared starting weight on class-count-normalized supervised loss; tuned at β=4 and checked for both tasks and remaining β values
C	Number of supervised classes (X: 32, scale: 6)
a_i	Indicator that original latent coordinate i is active
d_active	Sum of a_i, measured per trained model
k_active(k)	Number of active coordinates among the first k original coordinates
1. Data

    dSprites file: dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz, 737,280 binary 64 × 64 images.
    Use the integer latents_classes columns: color 0 (constant, ignore), shape 1 (3 classes), scale 2 (6), orientation 3 (40), X 4 (32), Y 5 (32).
    Use a single saved random train/validation/test split of 80/10/10 with split_seed=0, shared by every condition. Preserve full factor coverage; do not hold out an entire class or factor interval. The split tests unseen factor combinations, but does not by itself rule out all forms of memorization.
    Fit supervised heads on the training split only; select hyperparameters with validation data. Keep test data untouched until final evaluation.

2. Model and continuity

Reuse the existing MLP β-VAE under NonLinearPBExperiments/vae_reproduction/ (existing model.py and mrl_vae_loss implementation). Extend this code in place, including its existing sequential-backward training path; do not introduce a second competing src/ implementation.
Encoder: 4096 → 1200 → 1200 → mu (10), logvar (10)
Decoder: 10 → 1200 → 1200 → 1200 → 4096
One shared decoder, fed zero-masked 10-dimensional prefixes.
Ten additional linear supervised heads, one R^k → R^C per k ∈ K (supervised objectives only).

Preserve activation, BCE reduction (sum over pixels, averaged over examples), optimizer (Adagrad 0.01), and other existing code defaults. Add supervised heads only for the supervised objectives; train jointly with encoder and decoder, then discard heads for frozen-source artifacts. For β = 0, set z = mu explicitly and do not use or train a sampling path for the loss.
3. Objective and relative loss scales

Nested reconstruction (uniform prefix sampling during training):
L_recon = E_{k ~ Uniform(K)} BCE_sum_pixels(decoder(mask_k(z)), x)

Implementation: Each training minibatch samples one k uniformly from 1..10, applies that prefix mask to both decoder and (when present) supervised head, and computes one unbiased Monte Carlo estimate of the prefix-averaged objective. The KL remains calculated once on the full latent. This reduces decoder/head passes but increases gradient variance; monitor convergence and validate using all ten prefixes, not sampled prefixes. Keep shared decoder and train every prefix head over successive batches.

One full-latent KL, not nested:
L_KL = KL(q(z|x) || N(0,I))

The KL must be computed once on the full latent, not on each prefix: nesting it would repeatedly penalize early coordinates (coordinate 1 appears in ten prefixes, coordinate 10 once), imposing an artificial positional KL penalty. When β = 0 the KL contributes nothing and need not be evaluated during optimization.

Nested supervised loss (use the same sampled k as reconstruction):
L_sup_raw = E_{k ~ Uniform(K)} CrossEntropy(head_k(z[:k]), label)
L_sup_norm = L_sup_raw / ln(C)
L_total = L_recon + beta * L_KL + lambda_sup * L_sup_norm

Use ln(C) normalization as a common baseline convention: uniform predicted class probabilities give normalized cross-entropy 1; perfect prediction gives 0. CE has no finite maximum and can exceed ln(C) when the correct label is assigned low probability. Normalization does not guarantee equal gradients, difficulty, or supervisory pressure; λ may still need task- or β-specific adjustment. Both BCE and CE use natural logs, giving loss values in nats; λ itself is dimensionless.

Existing end-of-training prefix-averaged reconstruction losses have been reported as approximately 88 nats (β = 1) and 118 nats (β = 4). Verify these are comparable under the current reduction and log actual loss and gradient magnitudes in a short pilot before selecting λ. These are reference observations, not universal loss magnitudes.
4. Three objectives × three β values
Source	Supervised label	C	Total loss
S_recon	None	—	L_recon + β L_KL
S_posX	X position	32	L_recon + β L_KL + λ L_sup_norm
S_scale	Scale	6	L_recon + β L_KL + λ L_sup_norm

S_shape is removed as a source, but shape remains a downstream target (T_shape) and a probe factor. The target's between-class scatter has rank ≤2, so a j-dimensional task-direction estimator is limited to j<=2; prefix k may still exceed 2 when evaluating O(k,j). For X with 32 classes and d=10, the rank limit is min(31,10)=10, not 9; scale has rank ≤5.

Reconstruction remains in every objective. Values of β are {0,1,4}. For β=0 use z=mu deterministically; do not rely on a sigma assertion. Latent scale is not pinned by the KL at β=0, so report both covariance-adjusted and unadjusted geometry.
4.1 Cheap supervision calibration (before confirmatory runs)

    Normalize supervised CE by ln(C) as in §3. This sets uniform-prediction normalized CE to 1, but does not ensure equal gradient strength or task difficulty. Use a common λ only if validation confirms it works for both S_posX and S_scale.
    Pilot at β=4 using S_posX, one training seed, 10 epochs, and four prespecified λ candidates (initial suggestion {10,30,100,300}, subject to an initial logged loss/gradient-scale sanity check). Evaluate prefix-1 X-position linear-regression R², exact-match and ±1-bin classification accuracy, as well as full-probe performance and reconstruction. Do not assume 10 epochs gives final performance: it is a screen, not a replacement for 40-epoch validation.
    Use the smallest λ that gives R²(mu_1 → X) >= 0.90 on validation without unacceptable reconstruction deterioration (log the tradeoff; set an explicit reconstruction cap before selecting). R² is the proposed primary ordinal X-position manipulation check; also report ±1-bin accuracy, with a provisional 80% target, and exact-match accuracy as a secondary metric. Fit the regression on training data and score on validation. A single scalar need not achieve 80% exact 32-class accuracy.
    Check λ on S_scale, β=4 using a short run and an appropriate validation measure for the six-level scale label (e.g., linear-regression R² on the integer scale factor plus full-probe accuracy). Also spot-check both supervised objectives at β=0 and β=1 early in staging / on short runs. Stronger KL at β=4 does not mathematically guarantee a λ works at lower β. If checks fail, permit a prespecified task/β-specific retune, record the exception, and do not claim that one shared λ equalizes supervision pressure.
    If no candidate meets the prespecified performance–reconstruction requirements, record calibration failure and all candidate results; do not tune indefinitely or weaken the target post hoc. Freeze any chosen λ, criteria, and exceptions before using the affected configuration for confirmatory analysis. A successful 10-epoch pilot may be continued to 40 epochs, retaining optimizer state, but clearly label pilot-seed reuse when estimating variability.

Archive every pilot in results/lambda_sup.json, including train/validation losses, performance, reconstruction tradeoffs, and whether each objective×β passes the selected manipulation check.
4.2 Staged runs and compute budget

    Core design: 9 configurations × 2 encoder seeds = 18 initial training runs, each 40 epochs, batch 256, original optimizer Adagrad 0.01, and one uniformly sampled prefix per minibatch. Use identical training schedules in all factorial cells.
    Stage 1 (gate): β=0 for S_recon, S_posX, S_scale: 3 objectives × 2 seeds = 6 runs. Compute Figure 0, prefix probing, the downstream α-sweep alignment against cos²α, and the random-rotation control. If those geometric checks do not support the main hypothesis, report the negative result and stop before spending on β=1 or β=4.
    Stage 2 (optional extension): β=1 and β=4, 12 additional runs. This isolates whether KL changes factor selectivity and ordering; a deadline can stop after Stage 1 without relabeling Stage 2 results as completed.
    Disagreement rule fixed before seeing confirmatory results: if the two encoder seeds for a configuration differ by >0.15 in headline A(1) on a prespecified reference target/angle, or by >1 in d_active, run a third encoder seed for that configuration. Report both individual seed values; with two values a sample standard deviation can be computed, but it is unstable and cannot reliably diagnose an outlier. For α-sweep points, report seedwise traces instead of overconfident error bars.
    Learned target rotation Q_T: retain five independent optimization restarts for each frozen encoder/target pair. These are not encoder retrainings; keep the rotation-seed identifiability diagnostic.
    Prefix sampling changes compute accounting. Compared with a 100-epoch all-ten-prefix reference, naive decoder-pass equivalence is 18×40/(100×10)=0.72 reference decoder-training equivalents before extra work, calibration, diagnostics, optimizer/encoder overhead, and possible third seeds. A claim of “~2 originals” requires a measured wall-clock benchmark; record the real elapsed times of one current 40-epoch sampled-prefix run and one historical full-prefix run.

5. Required diagnostics (validation set; always on mu)

Training uses reparameterized samples for β > 0; evaluation uses the posterior mean for every condition to avoid injection of sampling variance into probing and between-class covariance. At β = 0, training and evaluation both use mu.
5.1 Active-unit mask and count

A KL-based count is not comparable to β = 0 (and KL alone does not establish usefulness). Use a variance-based activity mask for all 9 conditions, along with supplementary KL diagnostics when β > 0.

For each model, compute raw per-coordinate validation variance v_i = Var_x(mu_i(x)). At β = 0, raw scales are arbitrary, so the threshold v_i > 0.01 is not invariant to coordinate rescaling; simply standardizing each coordinate to variance 1 would make that test tautological. Choose and preregister a shared global scale normalization, e.g. v_i / median_positive(v) within a model using a documented robust reference, then flag coordinates above the fixed threshold 0.01; record a threshold-sensitivity analysis. Treat results as operational activity, not exact intrinsic dimension. A small-variance axis can still encode useful information; report full-prefix probe results and sensitivity to the activity rule.
v_i = Var_x(mu_i(x))
reference_scale = median of non-negligible v_i (fixed numerical floor preregistered)
relative_v_i = v_i / reference_scale
a_i = 1(relative_v_i > 0.01)
d_active = sum_i a_i
k_active(k) = sum_{i=1..k} a_i

Save the entire v_i, relative variances, a_i, d_active, and k_active(k) curve for each trained model. Also store mean KL_i = E_x[0.5(mu_i² + sigma_i² − ln(sigma_i²) − 1)] for β > 0 (diagnostic, not the common activity definition). At β = 0, KL_i should be null/not applicable, not used to infer activity. Invariance and threshold sensitivity are especially important for β = 0.

Downstream random-alignment null (conditional on activity): If M_j is a uniformly random orthonormal j-frame inside the d_active-dimensional active coordinate subspace, and original prefix projector P_k includes k_active(k) active coordinates, then for the original alignment definition O(k,j) = ||P_k M_j||_F² / min(k,j):

E[O(k,j)] = k_active(k) * j / [d_active * min(k,j)].

Use this formula only when 1 <= j <= d_active, d_active > 0, and directions are constructed inside that subspace. If k_active(k)=0, the null and true overlap are zero. If task directions include inactive coordinates, the assumptions fail; compute an appropriate full-space or empirical random-rotation null instead. Do not assume dead coordinates sit at the end of the MRL ordering.
5.2 Figure 0 — factor-by-coordinate information

For each trained encoder, fit held-out single-coordinate linear probes for each of the five factors and each of mu_1,...,mu_10. Keep raw balanced/macro accuracy, but display chance-normalized accuracy:

A_norm = (A_macro − 1/C)/(1 − 1/C).

Chance is 1/3 (shape), 1/6 (scale), 1/40 (orientation), and 1/32 (X and Y), assuming balanced classes. Values can be negative if below chance; do not clip silently. The matrix is 10 × 5 per encoder (and across seeds); include raw accuracy in supporting data.

Interpretation caveat: A scalar multinomial logistic probe has limited class-boundary expressivity; circular orientation and rotational symmetries of shapes can make orientation look weak even when encoded. Complement it with full-prefix probes and, if needed, a documented nonlinear probe; do not interpret a weak orientation column as proof that orientation is absent. A high cell indicates decodability, not a uniquely disentangled causal factor.
5.3 Prefix accessibility

Fit matched linear probes on mu[:k] for k=1..10 for each factor. Report macro accuracy, normalized accuracy, full-probe accuracy, and k* = smallest k reaching 95% of the k=10 accuracy. Also report a chance-adjusted k* (smallest k with (A(k)−chance) >= 0.95*(A(10)−chance)) as a sensitivity analysis; raw 95% can be misleading near chance. Do not require each added coordinate to strictly increase probe accuracy on finite data.
5.4 Covariance conditioning; whitened and unwhitened geometry

Compute Sigma_active = Cov_x(mu_active(x)) using only active original coordinates. Report its eigenvalues, numerical rank/tolerance and condition_number = max_eigenvalue/min_positive_eigenvalue if stable. No meaningful condition number exists if d_active < 2 or the active covariance remains singular; flag these cases rather than returning infinity as a scientific finding. Log raw per-coordinate scales, especially for β = 0.

For downstream covariance-adjusted directions, whiten only the active subspace, using fixed numerical regularization as needed. Whitening can rotate/mix coordinates, and changes the geometry being assessed; express the resulting directions back in the original active coordinate frame, QR-orthonormalize there if computing Euclidean overlap, and state that the resulting metric is not invariant to all rescalings or readout choices. Report the unwhitened counterpart alongside it. There is no basis for claiming they must disagree whenever the condition number is large.
5.5 Prefix reconstruction

Measure validation BCE at each prefix k=1..10. Expect an overall decreasing reconstruction trend, not strict pointwise monotonicity from a shared nonlinear decoder. Record every adjacent increase and relative increase (L_{k+1}-L_k)/L_k. A proposed acceptance tolerance is 0.1% per adjacent step, set before the run; larger deviations trigger investigation, not an automatic declaration that masking is misimplemented. Existing observed results include an approximately 101.31 → 101.42 increase (≈0.109%) after prefix 5 in a β=4 run, so even 0.1% may be slightly too tight; use a calibrated, preregistered tolerance or explicitly accept this existing counterexample as diagnostic evidence.
6. Acceptance and interpretation checks
#	Check	Required response
1	Recon versus prefix	Log all violations; investigate sustained increases or violations above the predeclared tolerance; no strict monotonicity claim
2	Supervised feasibility	Each supervised × β cell meets its preregistered calibration criterion and full-probe performance floor, or is explicitly marked calibration-failed
3	β=0 implementation	Training and evaluation use z=mu, no sigma assertion
4	Active units	Nonzero, plausible activity mask; report threshold sensitivity, ordering and k_active curve for every seed
5	Seed reproducibility	Report across-seed variation in activity, probe accuracy, prefix curves, and task-direction alignment; trigger a third encoder seed if the two initial seeds differ by >1 active unit; also trigger for >0.15 discrepancy in prespecified headline A(1)
6	Objective manipulation	Compare each supervised target's early-prefix accuracy against S_recon at the same β; failures are reportable observations, not automatically wrong λ
7	Figure 0 interpretation	Report whether early coordinates encode X/Y position jointly or separately; don't require z1 alone to represent X or Y in S_recon

For the confirmatory scientific comparison, do not change λ after inspecting results. Calibration and failure criteria are determined up front; unexpected results may falsify hypotheses.
7. Pre-registered predictions (freeze before final runs)

These statements are falsifiable hypotheses, not guarantees.

    Activity versus β: Within a given source objective, increasing β is expected to reduce the number of active coordinates, under the common scale-relative activity rule. Counterexamples count against this prediction.
    Task information versus count: At matched β, manipulation-checked S_posX is expected to make X position more accessible from early prefixes than S_recon. Do not predict that S_posX necessarily preserves more active dimensions: one scalar coordinate may suffice to encode the supervised X label. If this accessibility improvement fails despite successful preregistered calibration, report it; do not blame λ post hoc.
    Reconstruction ordering: S_recon is expected to place useful position information in its first few coordinates (especially prefix k<=2), rather than necessarily identifying X in coordinate 1 and Y in coordinate 2. X/Y may mix or swap across seeds. Compare early-prefix position probing against later factors; exact order of shape and scale is empirical.
    β and factor alignment: S_recon, β=4 may have stronger single-coordinate factor selectivity than S_recon, β=0, but this is a testable trend, not an identifiability theorem. β-VAE can remain entangled; MRL encourages ordered prefix usefulness, not uniquely determined latent axes.

7.2 Reference cells

    S_recon, β=0: deterministic nested-reconstruction-only reference; isolates the effect of nested reconstruction from the KL term, but may have the least identifiable or most seed-dependent axes. It is not necessarily the geometrically cleanest encoder.
    S_recon, β=4: continuity reference to existing MRL β-VAE work. Compare it at the same training duration as all other grid cells; keep an extra full-100-epoch continuity run outside the grid if necessary.

8. Artifacts

For each run, save under results/pretrain/{objective}_beta{beta}_seed{seed}/:
encoder.pt              frozen encoder weights (and optionally full training checkpoint)
config.yaml             optimizer, epochs, K, beta, lambda, normalization, seeds
loss_history.csv        separate per-term weighted/unweighted train/validation losses
activity.json           v_i, relative_v_i, activity_mask, d_active, k_active, KL_i if applicable
probes.json             factor-wise raw and normalized probe accuracy, prefix curves, k* variants
figure0.npy             10x5 normalized single-coordinate probe matrix
covariance.json         active covariance eigenvalues, rank, condition number, regularization
reconstruction.json     per-prefix validation BCE, adjacent violations and tolerance
mu_val.npy              optional cached posterior means; align rows with saved validation indices

At project level:
results/lambda_sup.json       all pilot candidates, losses, floors, selected shared lambda and any documented objective×beta exceptions
results/pretrain_summary.csv  one row per seed×objective×beta with diagnostic outcomes
predictions.md               copy of Section 7, frozen **before confirmatory runs**
Tier1_original_archived.md   archived superseded Tier 1 text (not an active implementation spec)

Frozen downstream analysis must not update encoder weights. Save split indices, probe fitting details, software versions and exact selection rules for reproducibility.
9. Repository layout and execution

Extend the existing directory rather than creating a parallel code tree:
NonLinearPBExperiments/
  vae_reproduction/
    model.py                  existing MLP encoder / decoder
    ...                       existing mrl_vae_loss and sequential-backward code
    train_pretrain.py         objective, beta, lambda, seed, deterministic beta=0 path
    tune_lambda.py            beta=4 X pilot; scale and lower-beta checks; failure reporting
    diagnostics.py            activity masks, factor probes, recon curves, covariance
    check_pretrain.py         quality/feasibility checks
  configs/pretrain/           9 configuration templates
  results/                    source artifacts and pilot logs

Execution order: (1) Save shared splits; (2) benchmark sampled-prefix training and log loss magnitudes; (3) short β=4 X λ pilot, scale check, and lower-β spot checks; (4) freeze λ, acceptance criteria and directional predictions; (5) run the six β=0 encoder models and the geometric gate; (6) if justified, run 12 β=1/4 encoder models; (7) invoke third seeds only under the frozen disagreement rule; (8) compute all-prefix diagnostics, freeze encoders and perform downstream Tier 1. Keep the separate five-restart learned-rotation experiment. Do not silently replace failed cells.

Analysis rule: every training job analyzes itself as soon as it finishes, in the same process, into `<job>/analysis_<stamp>/` (`dsprites_pretrain.py` calls `analyze_job`). The same analysis must also run standalone on saved checkpoints (`scripts/analyze_pretrain.py`), including combined cross-run plots (`--combine-only`). Never wait for the whole grid before analyzing. See `NonLinearPB/NonLinearPBREADME.md`, "Analysis: after every job, and standalone".



<!-- # Pretraining Specification: 12 MRL Source Encoders on dSprites

**Scope.** This document covers pretraining only. It produces 12 frozen encoders plus
the diagnostics needed to interpret them. No target tasks, no rotations, no alignment
metrics — those live in the Tier 1 document and consume the artifacts produced here.

**Design.** A 4 x 3 factorial: four training objectives crossed with three values of
beta. The objective controls *what the encoder is trained to predict*; beta controls
*how much axis-alignment pressure* it is under. Crossing them lets the downstream
analysis separate two mechanisms that would otherwise be confounded.

**Continuity requirement.** This reuses the existing MLP beta-VAE in
`vae_reproduction/model.py`. Architecture, prefix masking, reconstruction loss and
optimizer stay exactly as they are. The only additions are the supervised MRL heads and
the beta sweep. This keeps the new encoders comparable with results already reported.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| `d` | latent dimensionality. **Fixed at 10.** |
| `x` | a dSprites image, 64x64 binary, flattened to 4096 |
| `mu`, `logvar` | encoder outputs |
| `z` | `mu + sigma * eps` during training; **`mu` at evaluation time** |
| `K` | MRL nesting list. **`[1, 2, ..., 10]`**, i.e. every prefix, as in the existing code |
| `k` | a prefix length |
| `beta` | KL weight. Swept over `{0, 1, 4}` |
| `lambda_sup` | weight on the supervised MRL loss. Tuned per beta, see §4 |
| `d_active` | number of non-collapsed latent units. Defined in §5.1. **Measured per model.** |

---

## 1. Data

**Dataset:** dSprites, `dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`.
737,280 images, 64x64 binary.

The npz contains `imgs` (737280, 64, 64), `latents_values` and `latents_classes`,
both (737280, 6). Use **`latents_classes`** (integer indices) for all labels.

| Column | Factor | Range |
|---|---|---|
| 0 | color | 0 (constant, ignore) |
| 1 | shape | 0–2 |
| 2 | scale | 0–5 |
| 3 | orientation | 0–39 |
| 4 | X position | 0–31 |
| 5 | Y position | 0–31 |

All five non-trivial factors are independent by construction — the dataset is the full
Cartesian product, each combination appearing exactly once. Do not subsample in any way
that breaks this.

**Splits.** Random 80 / 10 / 10 train / val / test over all 737,280 images with a fixed
`split_seed = 0`. Write the index arrays to disk and load them in every run; do not
regenerate per run.

> Because every factor combination appears exactly once, a random split means test
> images are *combinations the encoder has never seen*. This rules out memorization by
> construction, and it is why a random split is correct here. Do **not** hold out a
> contiguous slice of a factor — that would delete entire label bins from the downstream
> target tasks.

**All 12 encoders train on the same train split.** Nothing differs between them except
the loss.

---

## 2. Model

Unchanged from `vae_reproduction/model.py`:

```
Encoder:  4096 -> 1200 -> 1200 -> (mu, logvar), each of dim 10
Decoder:  10 -> 1200 -> 1200 -> 1200 -> 4096
```

Keep the existing activation and reconstruction loss exactly as implemented. Do not
switch to a conv net. The architecture is not load-bearing for this experiment, and
changing it would break comparability with results already reported.

### 2.1 Prefix masking

Unchanged: one shared decoder, `z` zero-masked beyond the prefix (`masked_prefix`).

One shared decoder rather than `len(K)` separate decoders is required — separate
decoders would let capacity differ by prefix, which confounds the ordering.

### 2.2 Supervised heads (new)

For each prefix `k` in `K`, a separate linear head `R^k -> R^C`, applied to the
zero-masked `z`. These are the only new parameters. Heads are trained jointly with the
encoder and discarded afterwards — they are not part of the frozen artifact.

---

## 3. Losses

### 3.1 Reconstruction (nested)

```
L_recon = mean over k in K of  Recon( decoder( mask_k(z) ), x )
```

Mean over prefixes, as in the existing code. (Sum versus mean only rescales the loss;
keep mean for continuity.)

### 3.2 KL (NOT nested)

```
L_KL = KL( q(z|x) || N(0, I) )          # computed once, on the full 10-dim z
```

**The KL term must not be nested.** If it were, coordinate 1 would be penalized once
per prefix containing it — ten times under `K = [1..10]` — while coordinate 10 is
penalized once. That is a monotone capacity penalty that would manufacture exactly the
ordering the downstream experiment is trying to measure. Ordering pressure must come
from the reconstruction term alone.

### 3.3 Supervised (nested)

```
L_sup = mean over k in K of  CrossEntropy( head_k( mask_k(z) ), y )
```

### 3.4 Total

```
L = L_recon + beta * L_KL + lambda_sup * L_sup
```

with `lambda_sup = 0` for the `S_recon` objective.

---

## 4. The 4 x 3 grid

### 4.1 Objectives

| Name | Supervision label | Classes | `lambda_sup` |
|---|---|---|---|
| `S_recon` | none | — | 0 |
| `S_posX` | X position (column 4) | 32 | tuned, §4.3 |
| `S_shape` | shape (column 1) | 3 | tuned, §4.3 |
| `S_scale` | scale (column 2) | 6 | tuned, §4.3 |

The reconstruction term is present in **all four**, including the supervised ones.
Without it a supervised encoder collapses to roughly `C - 1` useful dimensions and the
ordering past that point is arbitrary noise.

### 4.2 Beta values

`beta` in `{0, 1, 4}`.

**Handling beta = 0.** With no KL term nothing constrains `logvar`, which will drift to
large negative values and leave you with a deterministic autoencoder carrying unused
parameters. That is the intended arm, but make it explicit rather than accidental.
Choose one and record which:

- (a) when `beta == 0`, bypass sampling and set `z = mu`; or
- (b) keep sampling and **assert** that mean `sigma` falls below 0.01 by end of training.

Do not leave it ambiguous — a half-stochastic model is uninterpretable.

### 4.3 Tuning `lambda_sup`

`lambda_sup` must be tuned **once per beta value**, not once globally. A fixed weight
means different things at beta = 0 and beta = 4, because KL competes for the same
capacity.

Procedure: for each beta, train `S_posX` with `lambda_sup` in `{1, 3, 10, 30}` and pick
the smallest value where prefix-1 X-position accuracy (validation, probe on `mu`)
exceeds 80%. Freeze that value and reuse it for `S_shape` and `S_scale` at the same
beta.

Record all three chosen weights in `results/lambda_sup.json`. If they differ
substantially across beta, that is itself a finding about how much capacity KL consumes,
and it belongs in the writeup.

### 4.4 Runs

12 configurations x 3 seeds = **36 training runs**.

Training: keep the existing optimizer and schedule (Adagrad at 0.01, 100 epochs, batch
256). If total wall-clock is a problem, reduce epochs uniformly across all 36 and verify
on the validation reconstruction curve that every cell has converged — but keep
`S_recon` at beta = 4 at the full 100 epochs regardless, since that is the continuity
cell (§7.2).

---

## 5. Required diagnostics

Computed after training, on the **validation split**, using `mu` and not sampled `z`.

> **Evaluate on `mu` everywhere.** Train with the reparametrized sample so the KL term
> stays meaningful, but compute every diagnostic and every downstream measurement on the
> posterior mean. Sampling noise inflates within-class covariance and would contaminate
> all of it.

### 5.1 `d_active` — per-dimension KL

For each latent unit `i`, compute mean KL in nats over the validation set:

```
KL_i = mean over x of  0.5 * ( mu_i^2 + sigma_i^2 - log(sigma_i^2) - 1 )
```

```
d_active = count of units with KL_i > 0.01
```

Store the full `KL_i` vector and `d_active` in the results file as first-class fields.

**Why this matters downstream.** Collapsed units carry no information, so the null for
every alignment statistic is `max(k, j) / d_active`, not `max(k, j) / d`. If a model
retains 5 active units, the k=1 null is 0.2, not 0.1 — double. Every significance call
in the Tier 1 analysis reads `d_active` from here.

`d_active` must be recorded **per model, not per beta row.** Supervision keeps
dimensions alive: a coordinate carrying X-position signal resists collapse even under
heavy KL, because the supervised loss pays for it. So at beta = 4, `S_posX` may retain
substantially more active units than `S_recon`.

At beta = 0, KL is unregularized and this diagnostic is meaningless. Set
`d_active = 10` and note it.

### 5.2 Figure 0 — factor-by-coordinate heatmap

For each of the 12 models, and each of the five factors, train a **single-coordinate**
probe: logistic regression on `mu_i` alone, for `i = 1..10`. Report macro-averaged
accuracy as a 10 x 5 matrix.

This is the direct check on what the MRL ordering actually did. The table in §4.1 says
`S_recon` should order "position first, then scale, then shape/orientation" — that is a
**hypothesis about reconstruction pressure, not an established property.** Figure 0
tests it, costs almost nothing, and is the thing to look at before any alignment
measurement is interpretable.

### 5.3 Prefix accuracy curves

For each model and each of the five factors, probe on `mu[:k]` for `k = 1..10`, macro
accuracy. Report the curve and `k*` = smallest `k` reaching 95% of the `k = 10` value.

### 5.4 Covariance conditioning

Report the condition number of the covariance of `mu` on the validation split.

A VAE's latent covariance is anisotropic by construction — active dimensions carry large
variance in `mu`, collapsed ones near zero. If the condition number is large, the
whitened and unwhitened variants of the downstream scatter analysis will disagree, and
both must be reported. Log it here so that decision is made on evidence.

### 5.5 Reconstruction quality

Validation reconstruction loss at each prefix `k = 1..10`. The curve must be monotone
decreasing in `k` — if it is not, prefix masking is misimplemented.

---

## 6. Acceptance checks

Run all of these before declaring pretraining done. Each one can invalidate downstream
work.

| # | Check | Required outcome |
|---|---|---|
| 1 | Prefix reconstruction monotonicity (§5.5) | monotone decreasing in `k` for all 12 |
| 2 | `S_posX` prefix-1 X accuracy | > 80% at every beta, after §4.3 tuning |
| 3 | `S_shape` prefix-1 shape accuracy | clearly above the 1/3 chance baseline at every beta |
| 4 | `S_scale` prefix-1 scale accuracy | clearly above the 1/6 chance baseline at every beta |
| 5 | beta = 0 determinism | per §4.2, either bypassed or mean sigma < 0.01 |
| 6 | Seed stability | `d_active` varies by at most 1 across the 3 seeds of a cell |
| 7 | Figure 0 sanity | each supervised model's target factor peaks at or near coordinate 1 |

If check 2, 3 or 4 fails, `lambda_sup` is wrong for that beta — retune before
proceeding. If check 6 fails, the cell is not reproducible and needs more seeds or a
longer schedule before any downstream number from it means anything.

---

## 7. Pre-registered predictions

Commit these before the first run.

### 7.1 Directional predictions

1. **`d_active` decreases with beta**, for every objective. Standard KL pruning.
2. **The spread of `d_active` across beta is largest for `S_recon` and smallest for
   `S_posX`.** Supervision resists collapse, so it should blunt the beta effect. If the
   opposite appears, `lambda_sup` is miscalibrated.
3. **For `S_recon`, Figure 0 should show position factors peaking at early coordinates
   and shape late.** Reconstruction pressure follows pixel variance: changing scale or
   position moves many pixels, changing shape at fixed scale and position moves few.
4. **`S_recon` at beta = 4 should show more axis-aligned factor-to-coordinate structure
   in Figure 0 than at beta = 0**, since KL with a diagonal posterior encourages
   factorized latents. Beta produces a privileged *basis*; MRL produces a privileged
   *order*. The grid separates the two.

### 7.2 Two reference cells

Name these explicitly in the writeup:

- **`S_recon`, beta = 0** — the pure-ordering arm. No label information, no axis
  pressure, so ordering comes from nested reconstruction alone. This is the cleanest
  cell for interpreting downstream alignment results and controls.
- **`S_recon`, beta = 4** — the continuity cell, closest to the existing MRL beta-VAE.
  Its numbers should be comparable to what has already been reported.

---

## 8. Artifacts

Every run writes to `results/pretrain/{objective}_beta{beta}_seed{seed}/`:

```
encoder.pt            frozen encoder weights (decoder and heads optional, not required downstream)
config.yaml           full config including the resolved lambda_sup
diagnostics.json      KL_i vector, d_active, condition number, prefix recon curve,
                      per-factor prefix accuracy curves, k* per factor
figure0.npy           the 10 x 5 factor-by-coordinate probe matrix
mu_val.npy            posterior means on the validation split  (optional but recommended;
                      caching these makes the whole Tier 1 analysis a CPU job)
```

Plus, at the top level:

```
results/lambda_sup.json      the three tuned weights
results/pretrain_summary.csv one row per run: objective, beta, seed, d_active,
                             condition number, prefix-1 accuracy on the supervised
                             factor, final val recon loss
predictions.md               a copy of §7, committed BEFORE the first run
```

**The `encoder.pt` files are frozen artifacts.** Nothing downstream updates them.

---

## 9. Repository layout

```
src/
  data.py              dsprites loading, split generation, factor extraction
  model.py             EXISTING - add supervised heads only
  losses.py            nested recon, un-nested KL, nested sup
  train_pretrain.py    one run; takes objective, beta, seed
  tune_lambda.py       the §4.3 sweep
  diagnostics.py       d_active, figure 0, prefix curves, condition number
  check.py             the §6 acceptance checks, exits nonzero on failure
configs/
  pretrain/            12 yaml files
results/
  pretrain/
```

**Execution order:**

```
1. generate splits (once, seed 0)
2. tune_lambda.py for each beta in {0, 1, 4}        ->  lambda_sup.json
3. train_pretrain.py for all 12 configs x 3 seeds   ->  36 runs
4. diagnostics.py over all runs
5. check.py                                          ->  must pass before Tier 1
```

Step 2 must complete before step 3. Step 5 gates everything downstream.


<!-- # Tier 1 Experiment: Privileged-Basis Alignment on dSprites

**Purpose.** Test whether a cheap, frozen-encoder measurement of geometric alignment
recovers a *known* ground-truth relationship between a source encoder's privileged
ordering and a target task's discriminative directions.

**Scope.** No fine-tuning anywhere in this document. Frozen encoders only. If this
experiment fails, nothing downstream is worth running.

**Expected cost.** ~30 encoder trainings at dSprites scale, single GPU, 1–2 days wall
clock. No step requires more than 20 minutes of training.

---

## 0. Notation

Every symbol used below, defined once.

| Symbol | Meaning |
|---|---|
| `d` | latent dimensionality. **Fixed at 10 throughout.** |
| `x` | a dSprites image, 64x64, binary |
| `phi_S` | a source encoder; maps `x` to `z` in R^d. Frozen after pretraining. |
| `z` | latent vector, `z = phi_S(x)`, length `d` |
| `K` | the MRL nesting list: prefix lengths at which losses are applied. **Fixed at `[1, 2, 3, 4, 6, 8, 10]`.** |
| `k` | a prefix length, integer in `1..d` |
| `P_k` | projection keeping the first `k` coordinates, zeroing the rest (a `d x d` diagonal 0/1 matrix) |
| `T` | a target task (a labelling of the same images) |
| `C` | number of classes in target task `T` |
| `M_j` | a `d x j` matrix whose columns are orthonormal **task-relevant directions**, expressed in source latent coordinates |
| `Q_T` | a `d x d` orthogonal matrix learned for task `T`; its columns are task directions in source coordinates |
| `O(k, j)` | overlap between the source's first `k` coordinates and the task's first `j` directions. Defined in Section 4. |
| `A(k)` | shorthand for `O(k, k)`. The headline alignment curve. |
| `k*` | smallest prefix length reaching 95% of full-probe macro accuracy. Defined in Section 5. |
| `S_B` | between-class scatter matrix, `d x d`. Defined in Section 4.1. |
| `alpha` | the sweep angle defining Family A target tasks. Degrees. |
| `null(k, j)` | expected value of `O(k, j)` under a uniformly random orthogonal transform: `max(k, j) / d` |

---

## 1. Data

**Dataset:** dSprites (`dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`).
737,280 images, 64x64, binary. Download from the deepmind/dsprites-dataset GitHub repo.

**Factors**, all independent by construction (full Cartesian product, each combination
appears exactly once):

| Factor | Values | Index in latents array |
|---|---|---|
| color | 1 (constant) | 0 |
| shape | 3 (square, ellipse, heart) | 1 |
| scale | 6 | 2 |
| orientation | 40 (circular) | 3 |
| X position | 32 | 4 |
| Y position | 32 | 5 |

Factor independence is the property that makes this testbed valid. Do not subsample in
any way that breaks it.

**Splits:** random 80 / 10 / 10 train / val / test over the full 737,280 images, with a
fixed seed (`split_seed = 0`) shared by every run in the project.

> **Do not** hold out a contiguous slice of a factor (e.g. "all images with X in the top
> quarter"). That would delete entire label bins from the Family A tasks and make the
> alpha sweep incomparable across angles. Random split is correct here.

---

## 2. Source encoders

Four source encoders. **All four see exactly the same images.** Only the training
objective differs. The objective is what induces a different privileged ordering — that
is the whole point.

### 2.1 Architecture (identical for all sources)

```
Encoder:  Conv(1->32, 4, stride 2) -> ReLU      # 64 -> 32
          Conv(32->32, 4, stride 2) -> ReLU     # 32 -> 16
          Conv(32->64, 4, stride 2) -> ReLU     # 16 -> 8
          Conv(64->64, 4, stride 2) -> ReLU     # 8 -> 4
          Flatten -> Linear(1024 -> 128) -> ReLU
          Linear(128 -> 10)                     # z, d = 10

Decoder:  mirror of the above, transposed convs, sigmoid output
```

Deterministic autoencoder, not a VAE. A KL term adds a second source of basis pressure
and confounds the measurement. Keep it out of Tier 1.

### 2.2 MRL loss

For nesting list `K = [1, 2, 3, 4, 6, 8, 10]`, the MRL reconstruction loss is

```
L_recon = sum over k in K of  BCE( decoder_k( z[:k] ) , x )
```

where `decoder_k` zero-pads `z[:k]` back to length `d` and feeds the shared decoder.
Zero-padding (one shared decoder) rather than `len(K)` separate decoders — otherwise
capacity differences between prefixes confound the ordering.

Supervised MRL loss, for a label `y`:

```
L_sup = sum over k in K of  CrossEntropy( head_k( z[:k] ) , y )
```

with `head_k` a separate linear layer `R^k -> R^C`.

### 2.3 The four sources

| Name | Loss | Intended privileged ordering |
|---|---|---|
| `S_recon` | `L_recon` | ordered by pixel variance explained: position first, then scale, then shape/orientation |
| `S_posX` | `L_recon + 10 * L_sup(X position, 32 classes)` | X position first |
| `S_shape` | `L_recon + 10 * L_sup(shape, 3 classes)` | shape first |
| `S_scale` | `L_recon + 10 * L_sup(scale, 6 classes)` | scale first |

The reconstruction term in the supervised sources is **mandatory**. Without it the
representation collapses to `C - 1` useful dimensions and the ordering past that point
is arbitrary noise, which makes alignment at larger `k` meaningless.

The weight `10` on `L_sup` is a starting point. Tune it once, on `S_posX`, so that
prefix-1 X-position accuracy exceeds 80%, then freeze the same weight for all sources.

**Training:** Adam, lr 1e-3, batch 256, 15 epochs, 3 seeds per source.
**Freeze after training.** Nothing in Sections 3–7 updates encoder weights.

### 2.4 Acceptance check before proceeding

For each source, train a linear probe on the full 10 dims for each of the five factors.
`S_posX` must reach high X-position accuracy at prefix 1; `S_shape` must reach high
shape accuracy at prefix 1. If not, the supervision weight is wrong — fix it before
moving on.

---

## 3. Target tasks

All targets reuse the same images. **Only labels change.** This holds `P(x)` fixed, so
any difference between tasks is label-induced by construction.

### 3.1 Family A — the alpha sweep (the core of the experiment)

For `alpha` in `{0, 15, 30, 45, 60, 75, 90}` degrees:

```
s(x)     = cos(alpha) * X(x) + sin(alpha) * Y(x)
label(x) = quantile_bin(s, n_bins = 6)
```

Geometrically: slice the 32x32 position grid into 6 parallel stripes perpendicular to
the direction `(cos alpha, sin alpha)`. `alpha = 0` is pure X position; `alpha = 90` is
pure Y; `alpha = 45` is distance along the main diagonal. Shape, scale and orientation
are ignored.

**Use equal-frequency (quantile) bins, not equal-width.** At `alpha = 45` the
distribution of `X + Y` is triangular, so equal-width bins would put far more samples in
the middle stripes. Label entropy would then vary with `alpha` and you would be measuring
task difficulty rather than geometry.

Verify: label entropy must be within 1% of `log(6)` for every `alpha`.

Why this family exists: the true angle between the task at `alpha_1` and the task at
`alpha_2` is known exactly — it is `alpha_2 - alpha_1`. That gives a calibration curve
rather than a handful of unrelated points.

### 3.2 Family B — off-manifold discrete targets

| Name | Labels | Classes |
|---|---|---|
| `T_shape` | shape | 3 |
| `T_scale` | scale | 6 |
| `T_orient` | orientation binned into 6 equal arcs | 6 |
| `T_composite` | `shape == heart AND scale > 0.75` | 2 |

**Subsample to equal class counts** for all Family B tasks. Class imbalance is a Tier 2
question; keeping it out here means the balanced and frequency-weighted variants of
Section 4.1 coincide, removing one degree of freedom.

Total: 11 target tasks, 4 source encoders = 44 source-target pairs.

---

## 4. Obtaining task-relevant directions

Two estimators of the same object. Compute both; they are a cross-check on each other.

### 4.1 Estimator 1 — closed form (primary)

No training. For target task `T`, on the frozen source latents of the training split:

```
mu_c = mean of z over examples of class c
mu   = mean of z over all examples

S_B_bal  = sum over classes c of  (mu_c - mu)(mu_c - mu)^T      # balanced, unweighted
S_B_freq = sum over classes c of  n_c * (mu_c - mu)(mu_c - mu)^T  # frequency-weighted
```

Use **`S_B_bal`** as primary. The `n_c` weighting lets frequent classes dominate the
leading eigenvectors; with Family B balanced by construction the two coincide here, but
keep the balanced version so the code carries over to Tier 2 unchanged.

`M_j` = top `j` eigenvectors of `S_B_bal`. These are orthonormal automatically, since
`S_B_bal` is symmetric PSD. Rank is at most `min(C - 1, d)`, so `j <= C - 1`.

**Whitened variant (secondary).** Let `Sigma` be the total covariance of `z`. Compute
eigenvectors of `Sigma^(-1/2) S_B_bal Sigma^(-1/2)`, map back via `Sigma^(-1/2)`, then
re-orthonormalize with QR in the original coordinate frame.

> The two variants answer different questions. Unwhitened asks "where do class means
> separate in the latent space as the encoder laid it out." Whitened asks "where do they
> separate relative to the noise." They agree when the latent covariance is near
> isotropic. **Log the condition number of `Sigma` for every source encoder** — if it is
> large, report both variants side by side and discuss, because the gap is itself a
> finding about what MRL ordering is doing to scale.

### 4.2 Estimator 2 — learned rotation (secondary, validation only)

Parametrize `Q_T = matrix_exp(W - W^T)` with `W` a free `d x d` parameter. This
guarantees orthogonality exactly. **Do not** use Gram-Schmidt on a free matrix — it is
numerically unstable and its gradients are poorly conditioned.

Train `Q_T` plus one linear head per prefix, encoder frozen:

```
L = sum over k in K of  CrossEntropy( head_k( (Q_T^T z)[:k] ), y )
```

Adam, lr 1e-3, 5 epochs, **5 seeds**.

**Identifiability check — run this before trusting any learned-Q number.** Report the
standard deviation across the 5 seeds of `A(1)`. `Q_T` is only constrained on the prefix
the task actually uses, so its action on later coordinates is free and will vary. If the
seed spread of `A(1)` exceeds 0.1, the learned estimator is not identified for that task;
report the closed-form result and note the instability rather than averaging over
something undefined.

Optional: add `lambda * ||Q_T - I||_F^2` and sweep `lambda` over
`{0, 0.01, 0.1, 1, 10}` to trace an accuracy-versus-rotation curve. The curve is more
informative than any single `lambda`.

`M_j` = first `j` columns of `Q_T`.

---

## 5. Measurements

### 5.1 Alignment

For source prefix length `k` and task direction count `j`:

```
O(k, j) = || P_k M_j ||_F^2 / min(k, j)
```

`|| . ||_F^2` is the sum of squared entries. In words: take the first `j` task
directions, keep only their components along the source's first `k` coordinates, sum the
squares, normalize.

- `A(k) := O(k, k)`, the headline curve, reported for `k = 1..d`.
- `A(1) = cos^2(theta)` where `theta` is the angle between the top task direction and the
  source's first axis. This is the single number used in the headline figure.
- **Null:** `null(k, j) = max(k, j) / d`. For `k = j`, that is `k / d`. For `d = 10`,
  `null(1,1) = 0.1`.
- `A(d) = 1` always, for any orthogonal `Q_T`. **This is a unit test, not a result.**

Report every alignment value alongside its null. Raw degrees are not interpretable: a
random direction in R^10 sits about 72 degrees from `e_1`, so "75 degrees" means
*nothing is happening*, not "large rotation."

### 5.2 Accessibility (measured separately, do not merge with alignment)

Linear probes on frozen `z`, **macro-averaged accuracy** throughout:

- `full_probe(T)` — probe on all `d` coordinates. Answers: is the information present at
  all?
- `prefix_acc(T, k)` for `k = 1..d` — probe on `z[:k]`, original ordering, no rotation.
- `rotated_prefix_acc(T, k)` — probe on `(Q_T^T z)[:k]`.
- `k*(T)` — smallest `k` with `prefix_acc(T, k) >= 0.95 * full_probe(T)`.

`A(k)` says *where* the information sits. `k*` says *how deep* you must go. They are
different measurements and a task can score well on one and badly on the other —
see the prediction table, row `S_posX / alpha=30`.

---

## 6. Controls

Non-negotiable. Each one is cheap and each one can invalidate the metric.

### 6.1 Random-rotation control

Draw a fixed random orthogonal `R` (`d x d`, Haar). Define `S_posX_rot` with latents
`R z`. Full-probe accuracy is **unchanged by construction** — `R` is invertible — but the
privileged ordering is destroyed.

**Required outcome:** every `A(k)` collapses to `null(k, k) = k / d`, while
`full_probe` is identical to `S_posX` to within numerical tolerance.

If `A(k)` stays high on `S_posX_rot`, the metric is reading off task geometry and has
nothing to do with privileged bases. That falsifies the framing. This is the single most
important control in the document.

### 6.2 Random-label control

Shuffle target labels within each Family A task.

**Required outcome:** `A(k)` at the null for all `k`. If not, you are measuring an
artifact of fitting `Q_T` (or of finite-sample noise in `S_B`) rather than task
structure. With 737k images the finite-sample floor should be negligible; if it is not,
report it as the effective null.

### 6.3 Oracle target encoder (the geometric ground truth for Family B)

For each of the 11 targets, train an MRL encoder **from scratch on that target**, same
architecture, same `K`. Call its basis `B_T`. By construction this is the ideal
privileged ordering for `T`.

Two checks:

1. **Is the estimator right?** Principal angles between `M_j` (from the frozen source)
   and the first `j` directions of `B_T`. If the learned `Q_T` does not approximate
   `B_T`, the estimator is broken regardless of any correlation it produces.
2. **Is the hypothesis right?** Alignment between the source basis and `B_T` is the
   ground-truth geometric compatibility. The cheap frozen `A(k)` should track it across
   all 11 targets.

This is what supplies ground truth for Family B, where there is no analytic angle to
compute.

---

## 7. Pre-registered predictions

**Write these to a file before running anything.** Pre-registration is what separates a
test from a post-hoc story.

### 7.1 Analytic point prediction (Family A, source `S_posX`)

The target direction in factor space is `(cos alpha, sin alpha)` in the (X, Y) plane;
the source's first axis is X. Therefore:

```
A(1) = cos^2(alpha)
```

| alpha | predicted `A(1)` |
|---|---|
| 0 | 1.00 |
| 15 | 0.93 |
| 30 | 0.75 |
| 45 | 0.50 |
| 60 | 0.25 |
| 75 | 0.07 |
| 90 | 0.00 |

This is a **point prediction, not a trend**. Plot measured `A(1)` against `cos^2(alpha)`
and the ideal result is the identity line. Note that `A(1)` at `alpha = 90` is *below*
the null of 0.1 — actively orthogonal, not merely unrelated.

Note the curve is flat near `alpha = 0` and steepest at 45 degrees, so the 0–30 region is
where the measurement is least sensitive and noise will hurt most.

### 7.2 Ordinal predictions

| Source | Target | Prediction | Reason |
|---|---|---|---|
| `S_posX` | alpha = 0 | `A(1) ~ 1` | same direction |
| `S_posX` | alpha = 45 | `A(1) ~ 0.5` | analytic |
| `S_posX` | alpha = 90 | `A(1) ~ 0` | orthogonal in factor space |
| `S_posX` | alpha = 30 | `A(1) ~ 0.75`, but `k*` possibly >> 1 | needs little rotation, may still need a deep prefix if Y sits late |
| `S_posX` | `T_shape` | `A(1) ~ null` | unrelated factor |
| `S_shape` | `T_shape` | `A(1) ~ 1` | same |
| `S_shape` | alpha = 0 | `A(1) ~ null` | unrelated |
| `S_recon` | alpha = 0 | `A(1)` high | position dominates pixel variance |
| `S_recon` | `T_orient` | `A(k) ~ null` for all `k`, low `full_probe` | known failure mode of reconstruction-trained encoders on orientation |
| any | any | `A(d) = 1` | unit test |
| `S_posX_rot` | any | `A(k) = k/d` | control 6.1 |

---

## 8. Deliverables

**Figure 1 (headline).** Measured `A(1)` on y, `cos^2(alpha)` on x, one line per source
encoder, with the identity line and the null (0.1) drawn in. Error bars over 3 encoder
seeds. Success looks like: `S_posX` on the diagonal, `S_shape` flat at the null.

**Figure 2.** Full `A(k)` curves, `k = 1..10`, for the four Family B targets, one panel
per source, null line `k/d` drawn in. This is where the "present but late" pattern should
appear if it exists.

**Figure 3.** `A(1)` versus `k*` scatter over all 44 pairs. These should *not* be
redundant — if they correlate above about 0.9, one of the two measurements is not adding
anything and that needs explaining.

**Table 1.** All 44 pairs: `A(1)`, `A(3)`, `k*`, `full_probe`, seed std of learned-`Q`
`A(1)`.

**Table 2.** Controls: `S_posX_rot` and random-label results against their nulls.

**results.json** — every number above, so downstream analysis never re-runs training.

---

## 9. Falsification criteria

Check these before writing anything up.

1. If `A(k)` on `S_posX_rot` does not collapse to `k/d`, the metric does not measure
   privileged-basis alignment. **Stop.**
2. If measured `A(1)` on `S_posX` does not track `cos^2(alpha)` with rank correlation
   above 0.9, the geometric story is wrong on the easiest possible case. **Stop.**
3. If `A(1)` correlates with `full_probe` above `r = 0.8` across the 44 pairs, alignment
   is a restatement of linear accessibility and the geometric framing adds nothing new.
   Report this honestly; it is a real negative result and worth knowing before Tier 2.
4. If the learned-`Q` seed spread of `A(1)` exceeds 0.1 on most tasks, drop the learned
   estimator from the paper and present the closed-form one only.

---

## 10. Suggested repository layout

```
data/            dsprites npz, split indices
src/
  data.py        loading, splits, Family A / B label construction
  models.py      encoder, decoder, MRL heads
  train_source.py
  train_oracle.py
  directions.py  S_B_bal, S_B_freq, whitened variant, learned Q_T
  metrics.py     O(k,j), A(k), nulls, k*, probes, principal angles
  controls.py    random rotation, random labels
  analyze.py     figures and tables
configs/         one yaml per source and per target
results/         results.json, figures/
predictions.md   copy of Section 7, committed BEFORE the first run
```

**Order of execution:**
`train_source` -> acceptance check 2.4 -> `directions` (closed form) -> `metrics` ->
controls 6.1 and 6.2 -> falsification checks 1 and 2 -> only then `train_oracle` and the
learned `Q_T`.

Checks 1 and 2 use only the closed-form estimator and cost almost nothing. Run them
before spending compute on the oracle encoders. --> -->