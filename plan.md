# plan.md — `LDA_hyperplane_check`

Design document for an experiment and the software that runs it.
Companion to `CLAUDE.md` (which covers *how* we work). Read that first.

---

## 1. The question

A widely-held assumption in the lab:

> Train an LDA to discriminate the **two extremes** of a latent control
> variable. The LDA's decision value on intermediate data then varies
> **linearly** with that latent variable.

This is treated as licence to read decision values as a graded measurement of
the latent variable. Nobody has tested it. We will, in a simulation where the
latent variable is under our control and the mapping from latent variable to
observable pixels is known exactly.

### 1.1 Ground truth

**The ground truth is the latent variable `c` itself.**

We are not comparing the LDA to another estimator or to a closed-form optimum.
We generated `c`; we know it exactly on every trial. Everything downstream —
pixel noise widening the distributions, zones amplifying / deadening /
inverting the coupling, geometric jitter moving zones from trial to trial,
patch placement deciding what the observer even sees — is a **transformation
chain applied to `c`**. The experiment asks how faithfully the decision value
`f` inverts that chain, and *which link* in the chain breaks the faithfulness.

So the design has two obligations:

1. Every transformation is **individually switchable and parameterised**, so
   distortion can be attributed to a specific link rather than to "the
   simulation".
2. Every trial carries its true `c` *and* the realised nuisance state (per-zone
   jitter offsets, clip fraction), so distortion can be regressed onto its
   cause. Aggregation happens at analysis time, never at generation time.

### 1.2 Falsifiable hypotheses

| ID | Hypothesis | Rejected when |
|----|-----------|---------------|
| H1 | `f` is **monotone** in `c`. | `rho` significantly below 1, or a sign reversal anywhere in the grid. |
| H2 | `f` is **affine** in `c`: `f = β₀ + β₁c`. | `kappa > 0.15`, or `kappa ≥ 0.05` with a significant nested curvature term. |
| H3 | Trial-to-trial spread of `f` is **homoscedastic** across `c`. | `sd_ratio ≥ 1.5` — which breaks the use of `f` as a measurement of constant precision even if the mean curve is straight. |
| H4 | Affinity is **robust to spatial heterogeneity** (heat / dead zones). | Curvature or heteroscedasticity grows with zone heterogeneity. |
| H5 | Affinity survives **anti-correlated and random zones**. | Same, for the adversarial configs. |
| H6 | Where affinity fails, it fails because **no linear readout could do better** (a limit of the data, not of LDA). | The oracle readout (§7.3) is affine where LDA is not. |

H2 is the load-bearing one. H1 is a weaker claim that may well hold while H2
fails — and the distinction matters, because monotone-but-curved still supports
ordinal use of `f` while forbidding metric use. H3 is scored separately from H2:
a straight mean curve with `c`-dependent spread is a different failure from a
bent curve, and collapsing them into one verdict hides that.

### 1.3 Pre-registered decision rule

Fixed before looking at results, so "significant" is not decided post hoc.

Quantities (defined in §7.2):

- `kappa` = `|β₂| / |β₁|` from an orthogonal-polynomial fit on the standardised
  contrast grid — an **effect size**, and the primary criterion.
- Nested F-test, affine vs. affine + quadratic, at α = 0.01.
- `sd_ratio` = max SD(f‖c) / min SD(f‖c), computed over levels with **nonzero**
  SD; `n_zero_variance_levels` reports how many were excluded. Bernoulli pixels
  have zero variance at `theta ∈ {0,1}`, so without this the ratio is `inf` for
  even the cleanest run and H3 is untestable (docs/decisions.md D5).
- `slope_t` = `|β₁| / SE(β₁)` — the degeneracy gate, threshold 10. **Revised from
  the original "95% CI of β₁ covers 0"**, which flags a slope on pure noise 5% of
  the time by construction and would break Phase-1 exit criterion 2 at random
  (measured: 10 false alarms in 200 null runs; docs/decisions.md D4). At the
  default 4200 trials, `slope_t = 10` corresponds to `|r| ≈ 0.155`.
- Minimum 200 trials per contrast level, ≥ 5 master seeds per condition.

**Verdict for H2** (evaluated in this order):

| Verdict | Condition |
|---------|-----------|
| `degenerate` | **`slope_t` = |β₁| / SE(β₁) < 10** ⇒ the slope is unresolved and every slope-normalised quantity is undefined (0/0): `kappa`, calibration, effective-range-use, max-local-slope-ratio. Applies by design to the null controls in §7.5. |
| `affine` | `kappa < 0.05` |
| `nonlinear` | `kappa > 0.15`, **or** `kappa ≥ 0.05` **and** nested F significant at α = 0.01 |
| `equivocal` | otherwise (i.e. `0.05 ≤ kappa ≤ 0.15` with a non-significant curvature term) |

`kappa` takes precedence over the p-value deliberately: with 4200 evaluation
trials the F-test detects curvature far too small to matter, so its role is only
to confirm that mid-band curvature is not sampling noise (`CLAUDE.md` §10).

**Verdict for H3**: `homoscedastic = sd_ratio < 1.5`, reported alongside but
never folded into the H2 verdict.

---

## 2. Scope and priorities

**The deliverable is an interactive app** (`streamlit run app.py`) in which the lab
can build worlds, place the observer's patches, and see whether the readout comes
out straight. It is a serious toy: everything is simulated, but the ground truth is
exact and the diagnostics do not let a bent curve pass as a straight one.

The engineering order was still headless-core-first, and that was right — the app
is a thin layer over `core/`, which stays separately testable and scriptable. What
the first draft got wrong was mistaking the engine for the product
(`docs/decisions.md` D8, D10).

So there are two front doors, and both are supported:

- **the app** — exploration, hypothesis generation, showing a colleague what a dead
  zone does to a readout;
- **`scripts/run_experiment.py` and `scripts/run_matrix.py`** — the careful run
  across many seeds, once exploration has found something worth pinning down.

The app defaults to a **fast preset** that returns in well under a second and is
*below* the pre-registered trial minimum of §1.3. It labels every verdict it
produces accordingly: nothing quotable comes out of the fast preset.

---

## 3. System overview

```
                    ┌──────────────────────────────────────────┐
                    │              CONFIGS (pydantic)           │
                    │  DataConfig   SamplingConfig              │
                    │  TrainingConfig   EvalConfig              │
                    └────────┬─────────────────┬───────────────┘
                             │                 │
        ┌────────────────────▼──────────┐      │   INVARIANT I1:
        │        core/data.py            │      │   these two paths
        │  zones ──> gain field (a, b)   │      │   never read each
        │  c ──> theta ──> pixel model   │      │   other's config
        │  ──> field (H, W)              │      │
        └────────────────┬───────────────┘      │
                         │ field                │
                         ▼                      ▼
                 ┌───────────────────────────────────────┐
                 │        core/sampling.py                │
                 │  patch layout ──> extract 5x5 ──> x    │
                 └───────────────────┬───────────────────┘
                                     │ x  (+ true c, + nuisance state)
                                     ▼
                       ┌──────────────────────────┐
                       │     core/dataset.py       │
                       │  TrialSet(X, c, y, meta)  │
                       │  train = {c_lo, c_hi}     │
                       │  eval  = contrast grid    │
                       └────────┬─────────┬───────┘
                                │         │
                    train ──────▼         ▼────── eval (fresh, never fitted on)
                     ┌────────────────┐   ┌──────────────────────────────┐
                     │  core/model.py │──>│    core/evaluation.py         │
                     │  scaler + LDA  │ f │  f(c) curve, linearity report │
                     └────────────────┘   │  weight attribution, oracle   │
                             │            └───────────┬──────────────────┘
                             ▼                        ▼
                     storage/ (joblib, json)   results/ (parquet + figures)
                             │                        │
                             └──────────┬─────────────┘
                                        ▼
                              ui/ (Streamlit, Phase 2+)
```

---

## 4. Generative model

### 4.1 Layer structure

```
c  (latent scalar in [0,1])
│
├─ zones + jitter  ──>  gain field:  a(H,W), b(H,W), random mask   [per trial]
│
├─ coupling:            theta = pivot + a*(c - pivot) + b          [per pixel]
├─ clip:                theta = clip(theta, 0, 1)                  [opt-in artefact]
├─ eps clamp:           theta = clip(theta, eps, 1-eps)            [Beta model only]
│
└─ pixel model:         field ~ P(theta)      [Bernoulli | clipped Gaussian | Beta]
   random-masked pixels ~ contrast-independent distribution
```

### 4.2 Coupling: gain field instead of additive zones

**This is a deliberate change from `docs/original_spec.md`**, which had
`mu_i(c) = c + Σ_k Z_k(i)` with clipping. Additive zone contributions saturate
as soon as two zones overlap, and that saturation is a nonlinearity we would
then be measuring *instead of* the effect we care about. The original spec's own
pitfalls section and open-risks section name this problem; the reformulation
removes it.

Instead, each pixel gets an **affine coupling to `c`**:

```
theta_i(c) = pivot + a_i * (c - pivot) + b_i
```

- `a_i` — coupling gain. How strongly pixel *i* tracks the latent variable.
- `b_i` — contrast-independent offset. Zero by default.
- `pivot` — the contrast around which gains rotate. Default `0.5`.

Properties that make this the right choice:

- **Background is exactly the simple scenario.** `a = 1, b = 0` ⇒ `theta = c`.
- **Clipping becomes opt-in.** For `pivot = 0.5`, `b = 0` and `|a| ≤ 1` on every
  trial, `theta ∈ [0,1]` for all `c ∈ [0,1]` (the bound is attained exactly at
  `a = ±1`, `c ∈ {0,1}`), so clipping is unreachable. It is switched on
  deliberately by exactly four things, and nothing else: a gain magnitude above
  1 (including `background_gain`), a nonzero offset, a `pivot` other than 0.5
  (e.g. `pivot = 0.2, a = −1` gives `theta(1) = −0.6` with every other
  precondition satisfied), or a nonzero `jitter.gain_sigma` (which can push
  `|a|` past 1 on individual trials even when configured gains are in bounds).
  Saturation thus becomes an experimental condition rather than a confound
  present in every run.
- **Zones differ in slope, not in brightness.** At `c = pivot` every pixel has
  the same expected value regardless of gain, so a zone is invisible at mid
  contrast and shows up only as a different rate of change. That is exactly the
  "strength of the relationship varies across the plane" idea, cleanly isolated.
- **Anti-correlation is just `a < 0`.** No special case.

`clip_fraction` is returned by `coupling_to_theta` and recorded per trial, so no
result is ever read without knowing whether saturation was active.

### 4.3 Zone types

`DataConfig.background_gain` (default `1.0`) sets `a` outside all zones. It is a
real experimental knob, not boilerplate — see the `strong` row.

| Type | Gain `a` | Offset `b` | Pixel behaviour | Purpose |
|------|----------|-----------|-----------------|---------|
| *(background)* | `background_gain`, default `1.0` | 0 | tracks `c` at the baseline rate | the simple scenario |
| `heat` | `gain > 1` (typ. 1.5–3) | 0 | amplified tracking; **saturates** | strong-relationship region *and* the saturation probe |
| `strong` | `background_gain < gain ≤ 1` | 0 | strong relative tracking, **never saturates** | strong region with the clipping artefact removed — the control for `heat` |
| `dead` | `gain ≈ 0–0.2` | 0 | weak / no tracking; noise dominates | dead zone |
| `anti` | `gain < 0` (typ. −0.5 … −1) | 0 | inverted tracking | scenario 3 |
| `random` | ignored | ignored | value drawn from a **contrast-independent** distribution; masks coupling entirely | pure nuisance variance |

`heat` and `strong` exist as a pair on purpose, and the pair only works because
`background_gain` is configurable. What "strong coupling" means here is
*relative* to the background:

- a `heat` zone at `gain = 2.0` with `background_gain = 1.0` gives
  `theta = 2c − 0.5`, which tracks `c` twice as steeply as its surround and
  leaves `[0,1]` for `c ≤ 0.20` and `c ≥ 0.80` — **10 of the 21 default grid
  points (48%)** clip, half the continuous range;
- a `strong` zone at `gain = 1.0` with `background_gain = 0.4` gives
  `theta = c` exactly, tracks 2.5× as steeply as its surround, and never clips.

Comparing the two isolates saturation from heterogeneity.

Limitation to state in the write-up: the pair matches the *ratio* of zone to
background coupling, not absolute gain, so the comparison controls for
heterogeneity rather than for signal amplitude. Validators (only when at least
one `strong` zone is present — otherwise these fire on every default config)
warn when a `strong` zone's gain does not exceed `background_gain` (it would
then be a `dead` zone by another name) or when `background_gain ≥ 1` makes
clip-free strong coupling impossible.

### 4.4 Zone geometry, edges, and overlap

Each zone is a circle: `center=(x, y)`, `radius`, plus a radial **falloff**
`hard | smoothstep | gaussian` with a width parameter.

- Falloff produces a weight `w_k(i) ∈ [0,1]` per pixel.
- `hard` edges create high-spatial-frequency discontinuities that a 5×5 patch
  straddles; a real effect worth studying, but it should be a choice rather than
  the only option.
- A radius larger than the plane is legal, and is how the whole-plane null
  controls in §7.5 are expressed.

**Overlap composition** — two modes, because summing gains is what explodes:

- `blend` (**default**): partition-of-unity weighted average, with the
  background as an explicit contributor.
  ```
  w_bg = max(0, 1 - Σ_k w_k)
  a    = (w_bg * background_gain + Σ_k w_k * a_k) / (w_bg + Σ_k w_k)
  ```
  The denominator is `1` when `Σ w_k ≤ 1` and `Σ w_k` otherwise — never zero,
  and continuous at `Σ w_k = 1`. The result is a convex combination of
  **`background_gain` together with the contributing zone gains**, so it is
  bounded by
  ```
  min(background_gain, min_k a_k)  ≤  a  ≤  max(background_gain, max_k a_k)
  ```
  Note this is *not* "never larger than the largest zone gain": a `dead` zone
  (gain 0) at edge weight 0.5 yields `a = 0.5`, pulled up by the background. The
  bound above is the one to assert in tests.
- `replace`: highest-`priority` zone wins at each pixel (PowerPoint-style
  z-order paint semantics — matches how users will think about the canvas).

`random` zones are applied last as a **mask**, in both modes: a pixel inside a
random zone ignores `c` entirely. Random is a masking operation, not a gain. The
mask threshold is the zone's falloff weight ≥ 0.5, so soft-edged random zones
still produce a definite mask.

### 4.5 Noise sources — kept strictly separate

| Source | Config | Character | Removed by averaging trials? |
|--------|--------|-----------|------------------------------|
| Pixel noise | `noise.sigma` (clipped Gaussian) / `noise.beta_concentration` (Beta) / intrinsic (Bernoulli) | i.i.d. per pixel per trial | yes |
| Geometric jitter | `jitter.center_sigma_px`, `jitter.radius_sigma_px`, `jitter.gain_sigma` | **coherent within a trial**, resampled per trial, independently per zone | no — changes the shape of `f(c)`, not just its spread |
| Patch layout jitter | `SamplingConfig.layout_jitter_px` | fixed per model (drawn once from the `patches` stream) | n/a |

Jitter is the subtle one: it perturbs whole zones together, so on any given
trial the observer's patches sit in systematically wrong places. This produces
`c`-dependent variance (a zone edge crossing a patch matters more at high gain),
i.e. a candidate mechanism for H3 failure. Keep it independently switchable and
record the realised per-zone offsets per trial (§7.1).

### 4.6 Pixel models (switchable — agreed)

All three implement one protocol: `sample(theta, rng) -> field`.

| Model | Definition | Notes |
|-------|-----------|-------|
| `bernoulli` | `field ~ Bernoulli(theta)` | Matches the verbal description ("contrast = chance a pixel is black or white"). No clipping bias, and no noise parameter — variance is set by `theta` itself. **Zero variance at `theta ∈ {0,1}`** ⇒ singular covariance if trained at exact extremes (§6.2). |
| `clipped_gaussian` | `field = clip(N(theta, sigma²), 0, 1)` | Matches the original spec. Continuous. Clipping biases the mean toward the interior near the ends — classic censored-distribution bias. |
| `beta` | `field ~ Beta(theta·nu, (1−theta)·nu)`, mean `theta`, precision `nu` | Continuous, bounded, no clipping needed. **Support is open**: undefined at `theta ∈ {0,1}`, which the default contrast grid includes, so `theta` is clamped into `[theta_eps, 1−theta_eps]` for this model only. The clamp is a documented artefact, and `theta_eps` defaults to `1e-3` (its effect on the mean is ~1e-3, three orders below the noise). |

`nu` is written `nu`, never `kappa` — `kappa` is reserved for the curvature
index (`CLAUDE.md` §5.2).

Switching is itself an experimental factor: if `f(c)` is affine under `beta` but
curved under `clipped_gaussian`, the curvature is the censoring, not the method.

---

## 5. Observer: patch sampling and features

### 5.1 Patch layouts

Patches are `patch_size × patch_size` squares (default 5×5), stored in
`SamplingConfig`. **They are entirely separate from data zones (I1).**

- `uniform_grid` — regular grid with `grid_step`. The "somewhat uniform"
  baseline. `grid_step == patch_size` tiles without overlap; smaller overlaps.
- `jittered_grid` — grid plus per-patch positional jitter, optionally snapped to
  a grid for UI tidiness.
- `random_uniform` — `n_patches` uniformly placed, overlap allowed.
- `manual` — explicit list, produced by the Phase-2 canvas.

The layout is drawn once from the `patches` stream and stored on the model, so
the observer has a fixed receptive-field layout across trials (see §12, question 1).

Overlapping patches duplicate pixels in the feature vector, inflating apparent
dimensionality without adding information. Report unique pixels sampled
alongside `n_features`.

### 5.2 Feature modes

- `pixels` (default) — flatten and concatenate all patches.
  `n_features = n_patches * patch_size²`.
- `patch_mean` — one value per patch. Lower dimensional, higher SNR per feature;
  a useful contrast because within-patch averaging changes the covariance
  structure LDA has to estimate.
- `patch_mean_std` — mean and SD per patch. Lets the model exploit *variance*
  cues, which matters for `random`/`dead` zones where the mean is uninformative
  but the variance is not.

### 5.3 Sampled-area composition — always reported

For every `SamplingConfig` × `DataConfig` pair, compute what the observer
actually sees: the weight-weighted fraction of sampled pixels attributable to
each zone kind (using the per-zone weight maps, so `blend` overlaps are handled
honestly), plus the mean and SD of the true gain `a` over sampled pixels. Patch
placement can single-handedly decide the result (`CLAUDE.md` §11, trap 6), so this
table is printed next to every `f(c)` curve, not buried in metadata.

---

## 6. Training

### 6.1 Protocol

1. Draw `n_per_class` trials at `c = c_lo` (label 0) and `n_per_class` at
   `c = c_hi` (label 1), using the `field_train` and `jitter_train` streams.
2. Extract features → `X_train (2·n_per_class, n_features)`, `y_train`.
3. If `shuffle_labels` (the null control), permute `y_train` using the `labels`
   stream.
4. Fit `StandardScaler` on `X_train` **only** (I4).
5. Fit `LinearDiscriminantAnalysis`.
6. Persist: LDA, scaler, patch layout, both config hashes, seeds, package
   versions, git commit, and the diagnostics in §6.4.

No intermediate contrast level is ever seen during training (I5).

### 6.2 Extremes are configurable, and that matters

`c_lo`/`c_hi` default to **0.05 / 0.95**, not 0 / 1. Under the Bernoulli model
`theta = 0` gives an all-black field with exactly zero variance, so the pooled
within-class covariance is singular and the solution is arbitrary in the null
directions. Slightly interior extremes keep the problem well posed. `c_lo = 0.0`
remains available as an explicit condition, and when used the code emits its own
warning rather than letting scikit-learn's collinearity warning pass unnoticed.

### 6.3 Regularisation is an experimental factor, not a detail

With `pixels` features we easily reach `n_features = 2500` against a few hundred
trials. Solver choice then dominates:

- `svd` — scikit-learn's default; unregularised, unstable here, and **rejects
  any shrinkage value**, so a config validator forces `shrinkage=None` with it.
- `lsqr` / `eigen` with `shrinkage="auto"` (Ledoit–Wolf) — **recommended
  default**.
- explicit shrinkage values, swept.

Both live in `TrainingConfig` and appear in the report. A finding that only
holds at one shrinkage level is a finding about shrinkage.

### 6.4 Training diagnostics recorded per model

Training accuracy and cross-validated accuracy; `d'` between the two classes;
covariance condition number and effective rank; `‖w‖`; and the fraction of `w`'s
squared mass on its top 1% of features (a concentration measure — a model
leaning on a handful of pixels is fragile).

---

## 7. Evaluation and diagnostics

### 7.1 Protocol and the long-form trial table

For each `c` in `EvalConfig.contrast_grid` (default 21 points, 0.0 → 1.0 step
0.05), generate `n_per_contrast` **fresh** trials (default 200) from the
`field_eval` / `jitter_eval` streams, extract features with the *same* patch
layout and the *same* fitted scaler, and compute
`f = model.decision_function(X_scaled)`.

Store long form, **one row per trial** — aggregation happens at analysis time,
or the spread information H3 needs is gone:

```
run_id | model_id | data_config_id | c | trial_index | f | f_oracle |
clip_fraction | jit_<zone_id>_dx | jit_<zone_id>_dy | jit_<zone_id>_dr | jit_<zone_id>_dg | ...
```

Jitter columns are **per zone**, one group of four per `ZoneSpec.id`, because
jitter perturbs each zone independently (§4.5) and §1.1 obligation 2 requires
the realised nuisance state to be regressable onto the distortion. Zone counts
are small (typically < 10), so wide columns are fine; the column set is recorded
in the run's sidecar metadata.

Identifiers, all defined in `storage/io.py`:

- `data_config_id` = `config_hash(DataConfig)`, first 12 hex chars.
- `model_id` = hash of (`data_config_hash`, `sampling_config_hash`,
  `training_config`, `master_seed`), first 12 hex chars.
- `run_id` = hash of (`model_id`, `eval_config`, `master_seed`) — deterministic,
  so re-running the same experiment overwrites rather than accumulating.

### 7.2 Diagnostics — the actual deliverable

Reported as a `LinearityReport` per (model, data config) pair:

| Diagnostic | Definition | Speaks to |
|-----------|-----------|-----------|
| Mean curve | mean of `f` per `c`, with CI | descriptive |
| Pearson `r` | `corr(c, f)` over trials | H2 (weakly — **never reported alone**) |
| `rho` | Spearman rank correlation | H1 |
| Affine fit | OLS `f ~ c`; β₀, β₁ with CI, R² | H2 |
| **`kappa`** | `|β₂|/|β₁|` from an orthogonal-polynomial fit up to cubic | **H2 — the headline number** |
| Nested-model test | F-test, affine vs. affine+quadratic(+cubic) | H2 |
| Max local-slope ratio | max/min of finite-difference slope of the mean curve | H2; localises *where* it bends |
| Residual pattern | mean residual per `c` from the affine fit, with CI | H2; shows the *shape* of the failure |
| SD profile, `sd_ratio` | trial SD of `f` per `c`, and its max/min ratio | H3 |
| Calibration | `ĉ = (f − β₀)/β₁`; bias and RMSE per `c` | practical: how wrong is `f` used as a measurement |
| Effective range use | fraction of the total `f` range spanned by the middle 50% of `c` | detects end compression / S-shape |
| Monotonicity violations | count of grid steps where the mean curve reverses | H1 |
| Clip activity | mean `clip_fraction` per `c` | attributes curvature to saturation |

Two reporting rules, both from `CLAUDE.md` §10:

1. `r` never appears without `kappa` and the residual pattern. A curve can be
   visibly S-shaped at `r = 0.99`.
2. When the 95% CI of β₁ covers zero — which is the *expected* outcome for the
   null controls — every slope-normalised quantity is undefined (0/0) and is
   returned as `None` with verdict `degenerate`: `curvature_index`,
   `calibration`, `effective_range_use` **and** `max_local_slope_ratio`.
   Emitting infinities there is a bug. These are exactly the four `| None`
   fields of `LinearityReport` (§8); the lists must stay in step.

### 7.3 Oracle readout — separating "LDA failed" from "impossible"

Because we know the true gain field `a`, we can build the best gain-aligned
linear readout by construction: weight each sampled pixel proportionally to its
true coupling gain, L2-normalised.

The weights are **not** mean-centred. Centring looks harmless and destroys the
simple scenario outright: there every pixel has gain 1, so centred weights are
identically zero and the "best possible readout" would be a constant. When all
pixels track `c` equally the right readout is to average them all
(docs/decisions.md D3).

- Weights are computed on the **jitter-free** gain field (the configured zone
  geometry), because an oracle that saw each trial's realised jitter would be
  omniscient rather than merely well-informed.
- They are applied to the **same scaled features** the LDA sees, so `f` and
  `f_oracle` share units and can be plotted on one axis.
- Feature-mode mapping: `pixels` → gain per sampled pixel; `patch_mean` → mean
  gain per patch; `patch_mean_std` → mean gain on the mean features and **zero
  on the SD features**. That last case is a stated limitation: variance cues
  carry information about `c` that a gain-proportional oracle cannot express, so
  under `patch_mean_std` the oracle is a lower bound on the achievable linear
  readout, not a ceiling.

`f_oracle` is evaluated on the same trials as `f`, giving:

- a **noise ceiling**: how affine the best gain-aligned linear readout is under
  this data config;
- the test for H6: if `f_oracle` is affine and `f` is not, the distortion belongs
  to LDA's estimation (finite samples, covariance whitening) — a criticism of
  the method. If both curve, the distortion is baked into the generative chain
  (saturation, random zones, jitter) and no linear readout can fix it.

This is not an analytic derivation of LDA's solution; it is a reference derived
from the ground-truth latent structure, consistent with §1.1.

### 7.4 Weight attribution — what did the LDA latch onto?

Map `w` back into pixel space (feature index → patch → pixel coordinates) and
compare against the true gain field:

- spatial correlation between `|w_pixels|` and `|a|` over sampled pixels;
- mean signed weight per zone kind — does it correctly assign negative weight
  inside `anti` zones and near-zero inside `dead` and `random` zones?
- sign-agreement rate between `sign(w_pixels)` and `sign(a)`;
- a `w_pixels` heat map rendered over the plane with zone outlines, beside the
  true gain field.

This diagnostic explains *why* a mapping curved, and it is the most likely thing
to surprise the lab. It also catches whole classes of bug: large weight inside a
`random` zone means either overfitting to nuisance variance or a broken
patch↔feature index mapping.

### 7.5 Null and control conditions

Every experiment batch runs these alongside the substantive configs. Each is
expressible in the config schema — noted, because a control you cannot write
down is a control you will not run.

| Control | How to express it | Expected result |
|---------|-------------------|-----------------|
| 1. Simple scenario | no zones, low noise | `verdict == "affine"`, R² > 0.99. If not, the bug is ours, not the assumption's. |
| 2. All-dead | `background_gain = 0.0`, no zones (or one `dead` zone with radius > plane) | `verdict == "degenerate"`; β₁ CI covers 0 |
| 3. All-random | one `random` zone centred on the plane with `radius > plane diagonal` | `degenerate` |
| 4. Label-shuffle | `TrainingConfig.shuffle_labels = True` on any substantive config | `degenerate` |
| 5. Sampling ablation | same `DataConfig`, three `SamplingConfig`s: uniform / heat-zone-only / dead-and-random-only patches | bounds how much of any result is really about patch placement |

Controls 2–4 are the leakage detectors: a non-flat `f(c)` there means
information is arriving through a path we have not modelled. They are also
exactly the cases where slope-normalised diagnostics are undefined, which is why
`degenerate` is a first-class verdict (§1.3).

### 7.6 Planned experiment matrix (Phase 1 output)

Factors, crossed selectively rather than fully:

- **Scenario**: `simple`, `heat_dead`, `heat_dead_anti`, `heat_dead_anti_random`,
  plus `strong_dead` (the clip-free counterpart to `heat_dead`)
- **Pixel model × noise level** — the noise knob differs per model, and
  `bernoulli` has none, so this is a 7-cell factor, not 3×3:

  | Level | `clipped_gaussian` `sigma` | `beta` `nu` | `bernoulli` |
  |-------|---------------------------|-------------|-------------|
  | low | 0.05 | 100 | — |
  | medium | 0.15 | 25 | — |
  | high | 0.30 | 8 | — |
  | intrinsic | — | — | single cell (variance set by `theta`) |

- **Jitter**: off / on (`center_sigma_px = 2.0`, `radius_sigma_px = 1.0`,
  `gain_sigma = 0.0`; a second on-level with `gain_sigma = 0.1` probes
  jitter-induced saturation)
- **Feature mode**: `pixels`, `patch_mean`
- **Shrinkage**: `lsqr`+`auto`, `svd`+`None`
- **Seeds**: ≥ 5 master seeds per cell, so between-seed spread is reported
  (`CLAUDE.md` §10)

Deliverable: one table of `kappa`, `rho`, `sd_ratio`, calibration RMSE and
verdict per cell, with between-seed spread, plus curve panels for the notable
cells.

---

## 8. Module specification

Signatures are the contract: change them here before changing them in code. `...`
below marks an elided body, never an elided type — `CLAUDE.md` §5.1 requires
full annotations in code.

Shared imports assumed: `numpy.typing as npt`, `Generator = np.random.Generator`,
`FloatArray = npt.NDArray[np.float64]`.

### `core/config.py`

```python
class ZoneKind(StrEnum):
    HEAT = "heat"; STRONG = "strong"; DEAD = "dead"
    ANTI = "anti"; RANDOM = "random"

class Falloff(StrEnum):
    HARD = "hard"; SMOOTHSTEP = "smoothstep"; GAUSSIAN = "gaussian"

class ZoneSpec(BaseModel):
    id: str                        # stable; used for per-zone jitter columns
    kind: ZoneKind
    center_x: float                # pixel coords, float so the canvas stays smooth
    center_y: float
    radius: float                  # may exceed the plane (whole-plane controls)
    gain: float = 1.0              # 'a' inside the zone; sign carries anti-correlation.
                                   # Ignored for kind == RANDOM (validator warns if set).
    offset: float = 0.0            # 'b'; nonzero enables clipping — use deliberately
    falloff: Falloff = Falloff.SMOOTHSTEP
    falloff_width: float = 2.0     # pixels
    priority: int = 0              # used by overlap_mode == "replace"

class JitterSpec(BaseModel):
    enabled: bool = False
    center_sigma_px: float = 0.0
    radius_sigma_px: float = 0.0
    gain_sigma: float = 0.0        # NOTE: > 0 can push |a| past 1 -> enables clipping

class NoiseSpec(BaseModel):
    pixel_model: Literal["bernoulli", "clipped_gaussian", "beta"] = "bernoulli"
    sigma: float = 0.1                       # clipped_gaussian only
    beta_concentration: float = 20.0         # 'nu'; beta only
    theta_eps: float = 1e-3                  # beta only: open-support clamp (§4.6)
    random_zone_dist: Literal["bernoulli_half", "uniform"] = "bernoulli_half"

class PlaneSpec(BaseModel):
    height: int = 100
    width: int = 100

class DataConfig(BaseModel):
    name: str
    plane: PlaneSpec = PlaneSpec()
    pivot: float = 0.5
    background_gain: float = 1.0             # 'a' outside all zones (§4.3)
    overlap_mode: Literal["blend", "replace"] = "blend"
    zones: list[ZoneSpec] = Field(default_factory=list)
    jitter: JitterSpec = JitterSpec()
    noise: NoiseSpec = NoiseSpec()
    # validators, all gated on `any(z.kind is ZoneKind.STRONG for z in zones)`
    # so they never fire on a config with no STRONG zone (background_gain
    # defaults to 1.0, so an ungated check would warn on every default config):
    #   - warn if a STRONG zone's gain <= background_gain
    #   - warn if background_gain >= 1 (clip-free strong coupling impossible)
    # Ungated: warn if any RANDOM zone sets a non-default gain or offset.

class SamplingConfig(BaseModel):
    name: str
    patch_size: int = 5
    layout: Literal["uniform_grid", "jittered_grid", "random_uniform", "manual"]
    grid_step: int = 10
    n_patches: int | None = None             # random_uniform only
    layout_jitter_px: float = 0.0            # jittered_grid only
    snap_to_grid: int | None = None
    feature_mode: Literal["pixels", "patch_mean", "patch_mean_std"] = "pixels"
    patches: list[tuple[int, int]] = Field(default_factory=list)  # manual: (row, col)

class TrainingConfig(BaseModel):
    c_lo: float = 0.05
    c_hi: float = 0.95
    n_per_class: int = 400
    solver: Literal["svd", "lsqr", "eigen"] = "lsqr"
    shrinkage: float | Literal["auto"] | None = "auto"
    standardize: bool = True
    shuffle_labels: bool = False             # null control (§7.5)
    cv_folds: int = 5
    # validator: solver == "svd" requires shrinkage is None (sklearn raises otherwise)

class EvalConfig(BaseModel):
    contrast_grid: list[float] = Field(
        default_factory=lambda: np.linspace(0.0, 1.0, 21).tolist()
    )
    n_per_contrast: int = 200
    compute_oracle: bool = True

class RunConfig(BaseModel):
    """Everything needed to reproduce one experiment, plus one master seed."""
    data: DataConfig
    sampling: SamplingConfig
    training: TrainingConfig
    evaluation: EvalConfig
    master_seed: int
```

`RunConfig` composes the four, but they remain separate types and are serialised
separately as well — I1 holds.

### `core/rng.py`

```python
STREAM_NAMES: tuple[str, ...] = (
    "field_train", "jitter_train", "field_eval", "jitter_eval", "patches", "labels",
)   # APPEND only — a stream's identity is its position (CLAUDE.md §7)

@dataclass(frozen=True)
class RngBundle:
    """The generators one dataset build is allowed to touch."""
    field: Generator
    jitter: Generator

def spawn_streams(master_seed: int) -> dict[str, Generator]: ...

def bundle_for(streams: dict[str, Generator],
               role: Literal["train", "eval"]) -> RngBundle: ...
    # role "train" -> (field_train, jitter_train); "eval" -> (field_eval, jitter_eval)
```

### `core/data.py`

```python
@dataclass(frozen=True)
class GainField:
    """The realised per-pixel coupling for ONE trial (jitter already applied)."""
    a: FloatArray                       # (H, W) coupling gain
    b: FloatArray                       # (H, W) offset
    random_mask: npt.NDArray[np.bool_]  # (H, W) True where c is ignored
    zone_weights: FloatArray            # (n_zones, H, W) falloff weight per zone
    zone_ids: tuple[str, ...]           # length n_zones, parallel to zone_weights
    zone_kinds: tuple[ZoneKind, ...]    # length n_zones
    jitter_realised: dict[str, dict[str, float]]  # zone_id -> {dx, dy, dr, dg}

    def dominant_zone(self) -> npt.NDArray[np.int16]:
        """(H, W) argmax over `zone_weights`; -1 where every weight is 0.

        Only meaningful for overlap_mode == "replace". Under the default
        "blend" mode there IS no dominant zone — the point is a mixture — so
        composition and attribution use `zone_weights` directly (§5.3, §7.4)
        and this helper is for display only.
        """

def zone_weight_map(zone: ZoneSpec, plane: PlaneSpec) -> FloatArray: ...
    # (H, W) in [0, 1] from the radial falloff

def build_gain_field(cfg: DataConfig, rng_jitter: Generator) -> GainField: ...
    # applies per-zone jitter, then composes per cfg.overlap_mode (§4.4)

def coupling_to_theta(c: float, gf: GainField,
                      cfg: DataConfig) -> tuple[FloatArray, float]: ...
    # theta = pivot + a*(c - pivot) + b, clipped. Returns (theta, clip_fraction):
    # clipping and its diagnostic must live in the same function (CLAUDE.md §6.3).

def sample_field(theta: FloatArray, gf: GainField, noise: NoiseSpec,
                 rng_field: Generator) -> FloatArray: ...
    # applies the pixel model, then overwrites random-masked pixels with draws
    # from noise.random_zone_dist. Beta's theta_eps clamp happens here.

def generate_field(cfg: DataConfig, c: float, rngs: RngBundle
                   ) -> tuple[FloatArray, GainField, TrialDiagnostics]: ...
    # one trial end-to-end. Returns the realised gain field too, because jitter
    # makes it trial-specific and the diagnostics need it.

@dataclass(frozen=True)
class TrialDiagnostics:
    clip_fraction: float
    jitter_realised: dict[str, dict[str, float]]
```

### `core/sampling.py`

```python
@dataclass(frozen=True)
class Patch:
    row: int; col: int; size: int

def build_patches(cfg: SamplingConfig, plane: PlaneSpec,
                  rng_patches: Generator) -> list[Patch]: ...

def patch_pixel_indices(patches: Sequence[Patch],
                        plane: PlaneSpec) -> npt.NDArray[np.int64]: ...
    # (n_patches * patch_size**2,) flat indices into the (H*W) plane, in feature
    # order. REQUIRED for weight attribution (§7.4) — build once, reuse.

def n_features(cfg: SamplingConfig, n_patches: int) -> int: ...

def extract_features(field: FloatArray, patches: Sequence[Patch],
                     feature_mode: str) -> FloatArray: ...
    # (n_features,)

def sampled_area_composition(patches: Sequence[Patch],
                             gf: GainField) -> pd.DataFrame: ...
    # weight-weighted fraction of sampled pixels per zone kind, plus mean/SD of
    # the true gain over sampled pixels (§5.3)
```

### `core/dataset.py`

```python
@dataclass
class TrialSet:
    X: FloatArray                    # (n_trials, n_features)
    c: FloatArray                    # (n_trials,) true latent value — ground truth
    y: npt.NDArray[np.int64] | None  # (n_trials,) labels; None for evaluation sets
    meta: pd.DataFrame               # per-trial: c, trial_index, clip_fraction,
                                     # jit_<zone_id>_{dx,dy,dr,dg}

def generate_trials(data: DataConfig, sampling: SamplingConfig,
                    patches: Sequence[Patch], contrasts: Sequence[float],
                    n_per_level: int, rngs: RngBundle) -> TrialSet: ...

def build_training_set(data: DataConfig, sampling: SamplingConfig,
                       training: TrainingConfig, patches: Sequence[Patch],
                       rngs: RngBundle,
                       rng_labels: Generator) -> TrialSet: ...
    # only c_lo / c_hi, labelled; applies training.shuffle_labels

def build_evaluation_set(data: DataConfig, sampling: SamplingConfig,
                         evaluation: EvalConfig, patches: Sequence[Patch],
                         rngs: RngBundle) -> TrialSet: ...
    # full grid, unlabelled, freshly generated (I4)
```

### `core/model.py`

```python
@dataclass
class TrainingDiagnostics:
    train_accuracy: float
    cv_accuracy: float
    cv_accuracy_sd: float
    d_prime: float
    cov_condition_number: float
    effective_rank: int
    weight_norm: float
    top1pct_weight_mass: float

@dataclass
class TrainedModel:
    lda: LinearDiscriminantAnalysis
    scaler: StandardScaler | None
    patches: list[Patch]
    sampling_config: SamplingConfig
    training_config: TrainingConfig
    data_config_hash: str
    sampling_config_hash: str
    master_seed: int
    diagnostics: TrainingDiagnostics
    provenance: dict[str, str]        # git commit, package versions, timestamp

def train_lda(train: TrialSet, training: TrainingConfig,
              sampling: SamplingConfig, patches: Sequence[Patch],
              data_config_hash: str, master_seed: int) -> TrainedModel: ...

def decision_values(model: TrainedModel, X: FloatArray) -> FloatArray: ...
    # applies the stored scaler, then decision_function. Single path — no caller
    # ever scales by hand.

def scaled_features(model: TrainedModel, X: FloatArray) -> FloatArray: ...
    # exposed so the oracle readout acts on exactly the same representation

def weights_in_pixel_space(model: TrainedModel,
                           plane: PlaneSpec) -> FloatArray: ...
    # (H, W), NaN where unsampled; overlapping patches sum their contributions
```

### `core/evaluation.py`

```python
@dataclass
class LinearityReport:
    n_trials: int
    mean_by_c: pd.DataFrame                  # c, mean_f, ci_lo, ci_hi
    sd_by_c: pd.DataFrame                    # c, sd_f
    clip_by_c: pd.DataFrame                  # c, mean_clip_fraction
    residual_by_c: pd.DataFrame              # c, mean_residual, ci_lo, ci_hi
    calibration: pd.DataFrame | None         # c, bias, rmse; None if degenerate
    pearson_r: float
    spearman_rho: float
    beta0: float
    beta1: float
    beta1_ci: tuple[float, float]
    slope_t: float                           # |beta1| / SE(beta1); gates `degenerate`
    r2: float
    poly_coefs: dict[str, float]             # orthogonal-polynomial coefficients
    curvature_index: float | None            # kappa; None if degenerate
    nested_f: dict[str, float]               # F, p, df_num, df_den per added order
    max_local_slope_ratio: float | None
    effective_range_use: float | None
    sd_ratio: float
    n_zero_variance_levels: int              # levels excluded from sd_ratio (D5)
    homoscedastic: bool                      # H3, scored separately (§1.3)
    monotonicity_violations: int
    verdict: Literal["affine", "equivocal", "nonlinear", "degenerate"]

@dataclass
class WeightAttribution:
    abs_correlation: float                   # corr(|w_pixels|, |a|) over sampled px
    sign_agreement: float                    # fraction of sampled px where signs match
    mean_signed_weight_by_kind: pd.DataFrame # zone kind -> weight-weighted mean w
    w_pixels: FloatArray                     # (H, W), NaN where unsampled
    true_gain: FloatArray                    # (H, W) jitter-free gain field

@dataclass
class ExperimentResult:
    config: RunConfig
    run_id: str
    model_id: str
    model: TrainedModel
    trials: pd.DataFrame                     # long form, §7.1
    report: LinearityReport
    oracle_report: LinearityReport | None
    attribution: WeightAttribution
    composition: pd.DataFrame                # §5.3
    provenance: dict[str, str]

def linearity_report(trials: pd.DataFrame, f_column: str = "f") -> LinearityReport: ...
    # needs the whole table, not just (c, f): clip_by_c comes from the
    # clip_fraction column (§7.2)

def oracle_weights(cfg: DataConfig, patches: Sequence[Patch],
                   feature_mode: str) -> FloatArray: ...
    # (n_features,) gain-proportional, mean-centred, from the JITTER-FREE gain
    # field (§7.3)

def oracle_decision_values(model: TrainedModel, X: FloatArray,
                           w_oracle: FloatArray) -> FloatArray: ...
    # w_oracle applied to scaled_features(model, X), so f and f_oracle share units

def weight_attribution(model: TrainedModel, cfg: DataConfig,
                       patches: Sequence[Patch]) -> WeightAttribution: ...

def evaluate_model(model: TrainedModel, data: DataConfig,
                   evaluation: EvalConfig, rngs: RngBundle) -> pd.DataFrame: ...
    # returns the long-form trial table incl. f, f_oracle, clip_fraction, jitter

def run_experiment(cfg: RunConfig) -> ExperimentResult: ...
    # train -> evaluate -> diagnose -> return everything, SAVE NOTHING.
    # Persistence is the caller's job: core/ stays free of I/O side effects
    # (CLAUDE.md §3, layout comment on core/), and free of UI imports (I2).
```

### `storage/`

- `io.py` — `config_hash`, `data_config_id`, `model_id`, `run_id`,
  `save_config`, `load_config`, `save_model`, `load_model`, `save_result`,
  `load_result`. Configs as JSON (pydantic round-trip), models as joblib,
  trial tables as parquet with a JSON sidecar carrying provenance and the
  per-zone jitter column list.
- `registry.py` — one JSON index of the configs and models under `configs/` and
  `models/`, keyed by hash with human-readable names, so the UI never scans the
  filesystem on every rerun.

### `ui/` and `app.py` (Phase 2+)

- `ui/controls.py` — config forms, canvas → config translation, registry widgets.
- `ui/plots.py` — Plotly figures: plane preview with zone/patch overlays, `f(c)`
  curve with spread band and `f_oracle` overlay, per-trial scatter, residual
  panel, `w_pixels` heat map beside the true gain field.
- `app.py` (repo root) — layout only. Toolbar (zone/patch tools) · left model
  registry · right data configs · centre canvas · bottom global controls and
  analysis panel, per the original sketch. All computation delegated to `core/`
  and cached with `st.cache_data` keyed on config hashes.

---

## 9. Phases and exit criteria

### Phase 1 — Headless pipeline and the actual answer *(priority)*

Deliverables: `core/` complete, `scripts/run_experiment.py`, JSON configs for
the five controls and the five scenarios, the diagnostics suite, tests.

Exit criteria:

1. Simple scenario (no zones, low noise) gives `verdict == "affine"` with
   `kappa < 0.05` and R² > 0.99 — the pipeline can detect affinity when present.
2. All-dead, all-random and label-shuffle controls all return
   `verdict == "degenerate"` with β₁ CI covering zero — the pipeline is not
   leaking, and the degenerate branch works.
3. Same master seed ⇒ bit-identical results across runs; training and
   evaluation trials at the same `c` are demonstrably different draws.
4. The experiment matrix (§7.6) runs end-to-end and produces the
   `kappa` / `rho` / `sd_ratio` / calibration table with between-seed spread.
5. Weight attribution reproduces known structure: negative mean weight in
   `anti` zones, near-zero in `dead` and `random`.
6. No clipping occurs in any config whose gains satisfy the §4.2 bound — checked
   by asserting `clip_fraction == 0` in those runs.

**At the end of Phase 1 the research question is answered.** Everything after is
exploration and communication.

### Phase 2 — the app  *(done)*

Interactive zone and patch placement, live plane preview, one-click train and
evaluate, a model registry and a comparison view. Clicks are captured with
`st.plotly_chart(on_select="rerun")` rather than `streamlit-drawable-canvas`
(`docs/decisions.md` D10): no extra dependency, and no canvas-pixel rescaling.

Exit criteria met: the app runs headless in CI through
`streamlit.testing.v1.AppTest` (`tests/test_app.py`), a config saved from the canvas
round-trips through `configs/` and reproduces its results, and the all-dead null
control driven through the UI returns `degenerate`.

Still open: dragging (placement is click-then-slider) and cross-seed comparison
inside the app.

### Phase 3 — Registry and persistence

Model/config registry with provenance display, load-and-compare, deletion.
Exit: a model trained in a previous session reloads and reproduces its stored
diagnostics exactly.

### Phase 4 — Batch exploration and reporting

Sweep runner over the factor grid, comparison views across models and configs,
export of a summary report. Exit: one command reproduces every figure in the
write-up from configs alone.

---

## 10. Risk register

| Risk | How it shows up | Detection | Mitigation |
|------|----------------|-----------|------------|
| Saturation mistaken for a property of LDA | curved `f(c)` under `clipped_gaussian` or `heat` zones | `clip_by_c` in every report | gain-field formulation makes clipping opt-in (§4.2); `beta` and `strong` controls |
| Beta model undefined at grid endpoints | `ValueError: a <= 0` at `c ∈ {0,1}` | test that the `beta` model runs the full default grid | `theta_eps` clamp, documented as an artefact (§4.6) |
| Singular covariance at extremes | collinearity warning; wild `w` | condition number and effective rank (§6.4) | interior `c_lo`/`c_hi` defaults; shrinkage on by default; explicit warning |
| `n_features` ≫ `n_trials` overfitting | high train accuracy, unstable `w` across seeds | between-seed `w` correlation; `top1pct_weight_mass` | shrinkage as a swept factor; `patch_mean` as a low-dim control |
| Invalid solver/shrinkage pair | sklearn raises mid-sweep | config validator | `solver="svd"` forces `shrinkage=None` (§6.3) |
| Standardisation / evaluation leakage | suspiciously clean results | leakage tests (`CLAUDE.md` §8, test class 5); label-shuffle control | scaler fitted on training only; single `decision_values` path |
| Null controls crashing the diagnostics | `inf`/`nan` in `kappa`, calibration | β₁ CI check before normalising | `degenerate` verdict, `None` fields (§7.2) |
| Patch placement drives the result | conclusions flip between layouts | sampled-area composition reported everywhere | sampling ablation (§7.5, control 5) is mandatory |
| Zone overlap producing runaway gain | extreme `a`, heavy clipping | assert `min(background_gain, min gains) ≤ a ≤ max(background_gain, max gains)` | `blend` partition-of-unity composition is bounded by construction (§4.4) |
| Jitter conflated with pixel noise | "noise" changes curve shape, not just spread | separate RNG streams; jitter-off condition | separate config sections and separate streams (I3) |
| Per-trial nuisance state unrecoverable | cannot regress distortion onto its cause | per-zone jitter columns present in the parquet | wide per-zone columns keyed by `ZoneSpec.id` (§7.1) |
| Streamlit state loss / accidental retraining | results change on a widget click | config-hash-keyed caching; `model_id` in every figure title | `st.session_state`; explicit train button |
| Comparing `f` across models with different scales | apparent differences that are pure units | scale-free diagnostics only | never compare raw `f` across models |
| Reading a high `r` as linearity | overclaimed conclusion | `kappa` and residual pattern reported with every `r` | reporting rule in `CLAUDE.md` §10 |

---

## 11. What would actually falsify the lab's assumption

Stated in advance so the conclusion is not written after the fact.

**The assumption survives** if, across the scenario range including heat/dead
zones, moderate noise and jitter: `rho ≈ 1`, `verdict == "affine"`,
`homoscedastic == True`, and calibration RMSE small and roughly constant in `c`.

**The assumption is qualified** if affinity holds in the simple and mildly
heterogeneous cases but the verdict turns `nonlinear` under anti-correlated /
random zones, saturating gains, or high jitter. The honest statement is then:
*monotone in general, metric only under conditions X* — and the deliverable is
the list of conditions.

**The assumption fails** if the verdict is `nonlinear` (or `homoscedastic` is
False) even in mildly heterogeneous configs *and* the oracle readout is affine
there (H6) — meaning a linear readout of `c` existed and extremes-trained LDA
did not find it.

Whichever it is, the write-up reports `kappa`, `rho`, `sd_ratio` and calibration
RMSE per condition, with between-seed spread, and states plainly which regimes
were tested and which were not.

---

## 12. Open design questions

Flagged rather than silently resolved. Revisit before Phase 1 exit.

1. **Should patch layout be resampled per trial?** Currently fixed per model
   (the observer has a fixed receptive-field layout). Resampling per trial would
   model a wandering observer — a different question, possibly worth a factor.
2. **Non-uniform sampling.** You mentioned relaxing the uniform assumption
   later. `random_uniform` and `manual` cover it mechanically, but a principled
   density model (e.g. foveated) may be worth adding.
3. **Should we test alternative readouts?** Logistic regression, ridge, or a
   `c`-regressed linear model would show whether observed curvature is
   LDA-specific or generic to extremes-trained linear classifiers. Cheap once
   the pipeline exists, and directly relevant to how the lab will react.
4. **Multiple latent variables.** The current design has one. A second nuisance
   latent (e.g. global luminance) would test whether `f` confounds them —
   probably the most practically important follow-up.
5. **Plane size vs. patch size scaling.** 100×100 with 5×5 patches is fixed by
   the brief; whether conclusions depend on that ratio is untested.
6. **Is `strong` the right clip-free control?** It matches the zone-to-background
   coupling *ratio* but not absolute gain (§4.3). An alternative is to keep
   `background_gain = 1` and narrow the contrast grid so `heat` never saturates
   on the tested range — cheaper, but then the grid differs between conditions,
   which contaminates the comparison differently.
