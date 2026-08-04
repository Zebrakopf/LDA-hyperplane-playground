# CLAUDE.md — Working agreement for `LDA_hyperplane_check`

This file tells any AI or human contributor **how to work in this repository**.
`plan.md` tells you **what to build**. Read both before writing code.

---

## 1. What this project is (and why the code must stay readable)

We are testing an assumption that is treated as truth but hardly verified with complex data:

> *If you train an LDA to discriminate the two extremes of a latent control
> variable, the LDA decision value on intermediate data varies linearly with
> that latent variable.*

This is a **methods audit**, not a product. The output of this repo is a claim
about a statistical method that informs possible experiments in the future. That has two
consequences for how we write code:

1. **Every result must be traceable to the line that produced it.** A reader
   must be able to follow the path from the latent variable `c` to a number in
   a figure without guessing.
2. **A silent bug here does not crash — it produces a plausible curve.** There
   is no exception to catch, no test that obviously fails. Readability and
   explicit assumptions *are* the error-detection mechanism.

Write code as if a reviewer will ask "how do you know that?" about every plot.

---

## 2. Non-negotiable invariants

Violating any of these invalidates results, so they are checked in review and,
where possible, asserted in code.

| # | Invariant | Why |
|---|-----------|-----|
| I1 | **`DataConfig` and `SamplingConfig` are never merged, nested, or cross-read.** Data zones (circles) shape the *world*; sampling patches (squares) shape the *observer*. | Conflating them means the model is peeking at the generative process. This is the easiest way to fake a positive result. |
| I2 | **`core/` and `storage/` never import `streamlit`.** | The scientific pipeline must run headless in a script, a test, and a cluster job. |
| I3 | **All randomness flows through an explicitly passed `numpy.random.Generator`.** No `np.random.seed`, no module-level global RNG, no implicit default. | Reproducibility, and independence between generation / jitter / sampling streams. |
| I4 | **Evaluation data is always freshly generated and never used to fit anything** — not the LDA, not a scaler, not a threshold. | Otherwise the linearity we "find" is partly memorised. |
| I5 | **Training uses only the two extreme contrast levels.** Intermediate levels exist only at evaluation time. | That *is* the assumption under test. Any leakage of intermediate labels answers a different question. |
| I6 | **The latent variable `c` is the ground truth.** Every diagnostic is ultimately "how faithfully does the decision value track `c`?" | Decided explicitly: there is no external reference we trust more than the variable we controlled. |

If you believe an invariant must be broken, stop and write the reason in
`docs/decisions.md` first. Do not break it silently.

---

## 3. Repository layout

```
LDA_hyperplane_check/
├── CLAUDE.md                  # this file — how to work
├── plan.md                    # what to build
├── environment.yml            # conda env: lda_patch_lab
├── app.py                     # Streamlit entrypoint (Phase 2+; thin, root level)
│
├── core/                      # pure science. No UI, no I/O side effects.
│   ├── config.py              # pydantic models for every config object
│   ├── rng.py                 # named RNG streams (see §7)
│   ├── data.py                # gain field + zones + pixel models
│   ├── sampling.py            # patch layouts + feature extraction
│   ├── dataset.py             # trial generation (train / eval sets)
│   ├── model.py               # LDA training + decision values
│   └── evaluation.py          # diagnostics, linearity tests, experiment runner
│
├── storage/                   # persistence only. No science.
│   ├── io.py                  # JSON configs, joblib models, run artefacts, ids
│   └── registry.py            # on-disk index of models/configs
│
├── ui/                        # Streamlit-only. No science.
│   ├── controls.py
│   └── plots.py
│
├── scripts/                   # headless CLI entrypoints (Phase 1 deliverable)
│   └── run_experiment.py
│
├── docs/
│   ├── decisions.md           # short decision log (context / decision / consequence)
│   └── original_spec.md       # the original written brief, committed verbatim
│
├── tests/                     # pytest
├── models/                    # saved .joblib (git-ignored)
├── configs/                   # saved .json configs (committed when canonical)
└── results/                   # run outputs: parquet/csv + figures (git-ignored)
```

`app.py` lives at the **repository root**, not in `ui/` — Streamlit convention,
and it keeps `ui/` importable by tests without launching a server.

`core/config.py`, `core/rng.py`, `scripts/`, `docs/`, `tests/` and `results/`
are deliberate additions to the original sketch, needed by the headless-first
plan. Everything else matches the agreed structure; do not invent further
top-level directories without saying so.

**Dependency direction is one-way:**
`ui/` → `storage/` → `core/`. Never the reverse. `core/` imports nothing from
the other three.

---

## 4. Environment

Use the conda environment defined in `environment.yml` (`lda_patch_lab`).

- **Do not add a dependency without flagging it.** If something genuinely needs
  a new package, say so, explain what it replaces, and add it to
  `environment.yml` in the same change.
- `streamlit-drawable-canvas` is **not used**. Canvas clicks come from
  `st.plotly_chart(on_select="rerun")`, which needs no extra dependency and gives
  exact plane coordinates (`docs/decisions.md` D10). Do not add it back without
  recording a reason there.
- `pyarrow` (parquet), `streamlit>=1.35`, `plotly` and `matplotlib` are all in
  `environment.yml`. The streamlit floor is load-bearing: `on_select` arrived in
  1.35 and the canvas is dead without it.
- Target Python 3.11. Use `numpy` ≥ 1.24 style (`np.random.default_rng`).

---

## 5. Code style

### 5.1 Mechanics

- **Type hints on every public function** — including return types. `...` in a
  signature is acceptable in `plan.md` prose only; never in code.
- **Array shapes in the docstring, always**, e.g. `(n_trials, n_features)`.
  Shape confusion is the most common failure mode in this kind of code.
- **Dataclasses / pydantic models over dicts.** A `dict[str, Any]` return type
  is acceptable only for genuinely open-ended provenance metadata. Diagnostics
  that anyone will plot get a dataclass.
- **No magic numbers.** Any constant a reviewer might question becomes a named
  config field with a documented default. `0.5` appearing in three files is a
  bug waiting to happen; `cfg.pivot` is not.
- **Functions do one thing.** If a function both generates data and fits a
  model, split it. Target ≤ 40 lines; longer is allowed for a single vectorised
  numerical routine that reads top-to-bottom.
- **Pure functions where possible.** Given the same inputs and RNG, output is
  identical. Mutating an input array in place requires a comment saying so.
- Formatting: 4-space indent, ~88 column soft limit, `snake_case` for functions
  and variables, `PascalCase` for classes, `UPPER_CASE` for module constants.

### 5.2 Naming glossary — use these names, do not invent synonyms

Latent and generative quantities:

| Name | Meaning |
|------|---------|
| `c` | latent contrast, scalar in `[0, 1]`. **The ground truth.** |
| `theta` | per-pixel distribution parameter *before* the pixel model (Bernoulli `p`, Gaussian mean, or Beta mean). Shape `(H, W)`. |
| `a` | per-pixel **coupling gain** — how strongly that pixel tracks `c`. Shape `(H, W)`. A zone's contribution to `a` is the field `ZoneSpec.gain`; in prose, always write "gain", never a bare `g`. |
| `b` | per-pixel **offset**: contrast-independent brightness shift. Shape `(H, W)`. Zero by default. |
| `pivot` | the contrast value around which gains rotate (`DataConfig.pivot`, default `0.5`). |
| `nu` | Beta-model concentration (`NoiseSpec.beta_concentration`). Never write this as `kappa` — see below. |
| `field` | one realised pixel image. Shape `(H, W)` from `PlaneSpec` (default 100×100, configurable). |
| `zone` | a circular region that modifies `a` and/or `b` — except `random` zones, which modify neither and instead *mask* pixels so they ignore `c` entirely. Belongs to `DataConfig`. |
| `jitter` | per-trial random perturbation of zone geometry. A coherent nuisance latent, not i.i.d. noise. |

Observer and model quantities:

| Name | Meaning |
|------|---------|
| `patch` | one sampling window, `SamplingConfig.patch_size` square (default 5×5, configurable). |
| `X` | feature matrix, shape `(n_trials, n_features)`. |
| `y` | binary training labels, shape `(n_trials,)`. |
| `w` | LDA weight vector in feature space; `w_pixels` when mapped back onto the plane. |
| `f` | LDA decision value(s) — `w·x + b_lda`. Never call this `y_pred`. |
| `f_oracle` | decision values from the ground-truth-derived oracle readout (`plan.md` §7.3). |
| `trial` | one generated field + its extracted feature vector + its true `c`. |

Analysis quantities:

| Name | Meaning |
|------|---------|
| `beta0`, `beta1`, `beta2` | intercept, linear and quadratic coefficients of the fit of `f` on `c` (written β₀, β₁, β₂ in prose). `beta1` is the slope; the LDA intercept is `b_lda`, never `b`. |
| `kappa` | **curvature index** `|β₂|/|β₁|` — the headline diagnostic. Reserved exclusively for this. |
| `rho` | Spearman rank correlation between `c` and `f`. |
| `r` | Pearson correlation between `c` and `f`. |

Noise field names, spelled out because two of the three do not follow a common
pattern: pixel noise is `noise.sigma`; geometric jitter magnitudes are
`jitter.center_sigma_px`, `jitter.radius_sigma_px` and `jitter.gain_sigma`.
Never use a bare `sigma` for a jitter magnitude.

---

## 6. Commenting etiquette

This is the part that matters most here. The rule is:

> **Comments explain intent, assumptions and consequences. The code explains
> mechanism.**

### 6.1 Module docstrings

Every module opens with a docstring covering: what it is responsible for, what
it deliberately does *not* do, and its place in the pipeline.

```python
"""Generation of pixel fields from the latent contrast variable.

Responsibility
--------------
Turn a `DataConfig` plus a contrast value `c` into a realised (H, W) field.

Explicitly NOT this module's job
--------------------------------
Patch sampling, feature extraction, anything model-related. Those live in
`core/sampling.py` and `core/model.py` respectively.

Pipeline position
-----------------
    c  ->  [this module]  ->  field  ->  sampling  ->  X
"""
```

### 6.2 Every statistical transformation names its assumption and its artefact

Mandatory. Whenever code changes the distribution of the data, the comment
states (a) what is assumed and (b) what artefact it can introduce. The
artefacts *are* the research object; hiding them defeats the purpose.

```python
# Clip theta into [0, 1] so it is a valid Bernoulli parameter / pixel value.
#
# ASSUMPTION: values outside [0, 1] are meaningless for this display model.
# ARTEFACT:   clipping is a saturating nonlinearity. Wherever it bites, the
#             pixel's response to c flattens, so f(c) compresses at that end of
#             the range. With pivot=0.5 and b == 0 and |a| <= 1 on every trial
#             this branch is unreachable, so clipping is an *opt-in* condition —
#             activated only by a gain magnitude above 1 (including
#             background_gain), a nonzero offset, a pivot other than 0.5, or a
#             nonzero jitter.gain_sigma (which can push |a| past 1 on individual
#             trials even when the configured gains are all within bounds).
#             `clip_fraction` is returned alongside theta and recorded per trial,
#             so no result is ever read without knowing whether saturation fired.
theta = np.clip(theta, 0.0, 1.0)
```

### 6.3 Docstrings carry the maths

Where a function implements an equation, the equation goes in the docstring in
plain-text notation, with every symbol tied to the glossary in §5.2.

```python
def coupling_to_theta(
    c: float, gf: GainField, cfg: DataConfig
) -> tuple[npt.NDArray[np.float64], float]:
    """Map latent contrast to per-pixel parameters.

        theta_i(c) = pivot + a_i * (c - pivot) + b_i        # then clipped

    Rotating around `pivot` rather than adding to a baseline keeps every zone
    type on a common footing: at c == pivot all pixels agree regardless of
    gain, so zones differ in *slope*, not in mean brightness. See plan.md §4.2
    for why this replaces the additive formulation in docs/original_spec.md.

    Returns
    -------
    theta : (H, W) float array in [0, 1]
    clip_fraction : float
        Fraction of pixels whose parameter was clipped. Clipping and its
        diagnostic live in one function on purpose — a caller receiving an
        already-clipped `theta` cannot recover this number.
    """
```

### 6.4 What not to comment

- No restating the obvious: `# increment i` above `i += 1`.
- No commented-out code. Delete it; git remembers.
- No `# TODO` without an owner and a concrete condition:
  `# TODO(phase-3): revisit once the registry stores gain fields.`

### 6.5 Deviations from the spec must be justified inline

Where the implementation departs from `plan.md` or from `docs/original_spec.md`,
say so at the point of departure, with a one-line reason and a pointer to the
section that explains it. A silent deviation is the worst outcome, because the
spec is what reviewers will read.

---

## 7. Determinism and reproducibility

One master seed per run. Independent child streams are derived **by name**, so
adding a stream never shifts existing ones. `core/rng.py` owns this:

```python
# Order matters for reproducibility: APPEND new names, never insert or reorder,
# because a stream's identity is its position in this tuple.
STREAM_NAMES: tuple[str, ...] = (
    "field_train",    # pixel noise for training trials
    "jitter_train",   # zone geometry jitter for training trials
    "field_eval",     # pixel noise for evaluation trials
    "jitter_eval",    # zone geometry jitter for evaluation trials
    "patches",        # patch layout placement (fixed per model)
    "labels",         # label-shuffle control only; unused otherwise
)
```

- Training and evaluation draw from **separate** field and jitter streams. This
  is the independence that matters: it guarantees no evaluation trial is a
  replay of a training trial.
- There is no train/test *split* stream, because there is no split — evaluation
  trials are generated fresh (I4).
- `RngBundle` (see `plan.md` §8) hands a role-appropriate pair of generators to
  the dataset builders; nothing else constructs generators.
- Every saved artefact records: master seed, full config objects, package
  versions, git commit, and a content hash of each config. A run that cannot be
  reproduced from its own metadata is a failed run.
- Re-running `scripts/run_experiment.py` with the same config and seed must
  produce **bit-identical** output. If it does not, that is a P0 bug — find the
  unseeded stream.

---

## 8. Testing expectations

Tests are not for coverage numbers. They are for the assertions we would
otherwise be trusting by eye.

Required test classes:

1. **Shape and dtype tests** — cheap, catch most integration mistakes.
2. **Determinism tests** — same seed ⇒ identical arrays; different seed ⇒
   different arrays; training and evaluation trials at the same `c` are not
   identical (separate streams).
3. **Analytic sanity tests** — cases where the answer is known independently:
   - uniform plane, no zones, no noise: `mean(field) ≈ c`
   - a pure `anti` zone: pixel means decrease in `c`
   - a `dead` zone with gain 0: `corr(pixel, c) ≈ 0`
   - a `random` zone: mean is contrast-independent
   - Beta model: sample mean over many draws ≈ `theta`
4. **Invariant tests**:
   - no clipping when `pivot == 0.5`, all `|gain| ≤ 1`,
     `|background_gain| ≤ 1`, all `offset == 0`, **and**
     `jitter.gain_sigma == 0` (note `background_gain` is an unbounded float —
     the all-dead control sets it to 0.0 and nothing forbids a negative value,
     so the absolute value is load-bearing)
   - the blend composition satisfies
     `min(background_gain, min_k gain_k) ≤ a ≤ max(background_gain, max_k gain_k)`
   - a serialised `DataConfig` contains no `SamplingConfig` field (I1)
5. **Leakage tests** — the scaler's fitted statistics are a function of the
   training features only (refit on a permuted evaluation set leaves the stored
   scaler unchanged); the label-shuffle control yields a slope whose CI covers
   zero.

A statistical test needs a fixed seed and a tolerance chosen from the expected
standard error, with that reasoning in a comment. "Flaky test, increased
tolerance" is not acceptable; work out the SE.

---

## 9. Workflow

1. **Plan before coding.** For anything beyond a small edit, state which
   `plan.md` section is being implemented and what the exit criterion is.
2. **Work in phase order** (`plan.md` §9), but keep sight of the deliverable: the
   app is how the lab uses this, and `core/` is the engine under it. The headless
   scripts exist for the eventual careful run, not as the product.
3. **Small, reviewable changes**, one concern each. Commit messages:
   `area: imperative summary`, e.g. `core/data: add radial falloff to zones`.
4. **When behaviour is ambiguous, ask.** Do not resolve a scientific ambiguity
   by picking whichever option is easier to code. Ambiguities here change what
   the experiment means.
5. **Update `plan.md` when the design changes.** A stale plan is worse than no
   plan. Record non-obvious choices in `docs/decisions.md` (one short entry:
   context, decision, consequence).

### Definition of done

- [ ] Runs in the `lda_patch_lab` environment with no new dependencies (or the
      new one is justified and added to `environment.yml`).
- [ ] Public functions have full type hints and docstrings with array shapes.
- [ ] Every new statistical transformation names its assumption and artefact.
- [ ] Randomness goes through a passed-in `Generator` from a named stream.
- [ ] Tests added for the new behaviour, including a determinism check.
- [ ] `core/` still imports no UI code; `DataConfig`/`SamplingConfig` still
      separate.
- [ ] Any spec deviation documented inline and in `plan.md`.

---

## 10. How to talk about results

The scientific value of this repo depends on not overclaiming. In comments,
docstrings, commit messages and chat replies, keep three registers distinct and
label them:

- **Observation** — what the numbers say, with the run that produced them.
  *"Under config `heat_dead_v1` (seed set A, 5 seeds): ρ = 0.98, but κ = 0.21
  and the nested quadratic term is significant (p < 1e-4); SD(f|c) rises 3×
  from c = 0.5 to c = 1."*
- **Interpretation** — what that plausibly means, hedged.
- **Speculation** — flagged as such, or omitted.

Specific prohibitions:

- **Never call the mapping "linear" on the strength of a correlation
  coefficient.** A high Pearson `r` is compatible with obvious systematic
  curvature. Report `kappa` and the residual pattern alongside `r`, every time.
- **`kappa` is an effect size and takes precedence over the p-value.** With
  4200 evaluation trials at the default settings the nested F-test will detect
  curvature far too small to matter. Its role is to confirm that mid-band curvature is not sampling
  noise, not to define the verdict.
- **Never report a mean without its spread**, and never a spread without saying
  whether it is across trials, across seeds, or across configs.
- **Distinguish "our LDA failed" from "no linear readout exists".** The oracle
  readout derived from the true gain field (`plan.md` §7.3) is what separates
  those two; cite it when claiming a limitation of the method rather than of
  the data.
- A negative result is a result. If the assumption fails in some regimes, say
  so plainly and characterise *which* regimes.

---

## 11. Known traps in this specific project

Read these before debugging something surprising.

1. **Degenerate covariance at the extremes.** With Bernoulli pixels at
   `theta = 0` or `1` the within-class variance is exactly zero and the
   covariance estimate is singular. Train at `c_lo`/`c_hi` slightly inside the
   range (defaults 0.05 / 0.95) and/or use a shrinkage solver. Never silently
   accept scikit-learn's collinearity warning.
2. **The Beta model has open support.** `Beta(theta·nu, (1-theta)·nu)` is
   undefined at `theta ∈ {0, 1}`, which the default contrast grid *includes*.
   `theta` is therefore clamped into `[eps, 1-eps]` for that model only — a
   documented artefact (`NoiseSpec.theta_eps`), not a silent fix.
3. **`n_features` ≫ `n_trials`.** 100 patches × 25 pixels = 2500 features
   against a few hundred trials: the covariance estimate is rank-deficient and
   `svd`/`lsqr` behave very differently. Solver and shrinkage are experimental
   factors, not implementation details. Note `solver="svd"` rejects any
   shrinkage — the config validator enforces that combination away.
4. **Standardisation leakage.** Fit the scaler on training data only, then
   apply. Fitting on the pooled set quietly injects evaluation statistics.
5. **Jitter is a latent nuisance, not i.i.d. noise.** It perturbs whole zones
   coherently within a trial, so averaging over trials does not remove its
   effect on the *shape* of `f(c)`. Keep it separately switchable, and record
   the realised per-zone offsets per trial.
6. **Patch placement can decide the answer.** Patches concentrated on heat
   zones give a near-perfect mapping; patches on dead/random zones give noise.
   Always report sampled-area composition per zone type next to any `f(c)`
   curve.
7. **Streamlit reruns the whole script.** Anything expensive is cached with
   `st.cache_data` keyed on a config hash; anything persistent lives in
   `st.session_state`. Never let a widget interaction silently retrain a model.
8. **Feature scale vs. decision-value scale.** `decision_function` has
   arbitrary units. Comparing raw `f` across models is meaningless; compare
   shape after affine alignment, or use the scale-free diagnostics (`kappa`,
   `rho`, R²).
9. **Slope-normalised diagnostics blow up on the null controls.** `kappa` and
   the calibration metrics divide by β₁, which is ≈ 0 by design in the
   all-dead, all-random and label-shuffle controls. The report must return
   verdict `"degenerate"` and leave those fields `None` rather than emitting
   infinities.
