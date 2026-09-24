# Details

The longer companion to [`README.md`](README.md). Everything here is reference —
read it when a control confuses you, or before quoting a number.

---

## What's on screen

**Sidebar — every setting.** Speed preset (shows `custom` once you change any number
it controls), then World (plane, background gain, pivot, overlap, pixel model and its
noise, jitter), Observer (patch size, grid step, features), Training & evaluation
(extremes, trial counts, levels, solver, shrinkage, CV folds, seed, controls) and
Configs (load a committed or saved config; save the current one to `configs/user/`).

**Above the canvas.** The tool bar — which zone kind a click places, `＋ patch`,
`− patch`, or `look` — the preview contrast, and display toggles. **gain map** draws
each pixel's coupling gain instead of one noisy realisation: blue tracks c, gray does
not, red runs backwards. It is what the zones actually mean.

**Canvas.** Circles in kind-specific colours are **data zones** — the world. Yellow
squares are **sampling patches** — what the model sees. They are never drawn alike;
conflating the world with the observer is the one mistake that would invalidate
everything here. Clicks place; they never pan or zoom.

**Beside the canvas.** Numeric placement (exact x/y, or the fallback if a click ever
misses), undo, grid/clear buttons, and the zone list, where each zone's position,
radius, gain, offset and edge can be fine-tuned.

**Results.** A model picker (every model trained this session, newest first, with
load / delete / save), the verdict with a one-line meaning, six headline numbers, and
five tabs: the mapping, precision & residuals, what the model used, model detail, and
a comparison view.

---

## Reading the numbers, in full

**`verdict`** — the pre-registered answer to "is the mapping affine?". Thresholds
were fixed before any result was looked at (`plan.md` §1.3), so it is not a judgement
made after seeing the curve.

**κ, the curvature index** — `√(κ₂² + κ₃²)` from an orthogonal-polynomial fit to
cubic, where κ₂ = |β₂|/|β₁| measures a one-sided bend and κ₃ = |β₃|/|β₁| an S-shape.
Below 0.05 is affine, above 0.15 a clear violation. Both parts are shown in the κ
tooltip. The cubic part matters most here: saturation around the pivot bends *both*
ends and makes an S, which the quadratic term alone scores as zero (D14). For a feel
for the scale:

| one-sided bend `f = c + γc²` | γ = 0.1 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| κ | 0.025 | 0.054 | 0.090 | 0.135 | 0.180 |

| S-curve `f = logistic(s·(c − 0.5))` | s = 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| κ | 0.022 | 0.047 | 0.075 | 0.136 | 0.189 | 0.264 |

**Range use** — the share of the output range spent on the middle half of `c`. A
straight line is exactly 0.50 on any grid; an S-curve spends more (0.86 for the
heat-zone-only ablation), an end-expanded curve less.

**`r` is never enough on its own.** A visibly S-shaped curve sits happily at
r = 0.99. That is why κ sits next to it and why the **precision & residuals**
tab exists: the residual is shown in units of c (how far off you would be reading f
through a straight line), and a flat line at zero is genuinely affine, a smile is a
one-sided bend, an S is saturation.

**SD max/min** — how much the *precision* of the readout varies across the range. A
perfectly straight mean curve with a 20× swing in spread still breaks the use of
decision values as measurements with constant error bars. Scored separately from the
verdict, because it is a different failure.

**`degenerate` is not an error.** The slope is not resolved above noise
(|β₁|/SE < 10), so κ and the calibration numbers are 0/0 and show as n/a. It is the
*correct* answer for a world where nothing tracks `c`.

**`fit failed` is different.** The LDA had nothing to learn — every feature was
constant within each training class, typically because both extremes saturate
completely. The app says why and what to change. It is not a finding about the world.

**Fresh vs in-sample d′.** The model tab shows the separation of the two extremes on
fresh trials next to the in-sample figure. A large gap means the model memorised
nuisance variance (the all-random control: 3.6 in-sample, ~0 fresh).

**The oracle readout** is a linear readout built from the true gain field. It
separates "the LDA failed" from "no linear readout was possible here": if the oracle
comes out affine where the LDA does not, the problem is the method; if both bend, the
distortion is baked into the world.

**Fast-preset verdicts are for exploring, not quoting.** The app says so under the
metrics whenever trials per level fall below the pre-registered minimum of 200.

---

## The world model

```
theta_i(c) = pivot + a_i * (c - pivot) + b_i        then clipped to [0, 1]
```

`a_i` is the **coupling gain** of pixel `i`: how strongly it tracks `c`. Zones set
it; `background_gain` sets it everywhere else. Gains *rotate around the pivot* rather
than adding to a baseline, so a zone is invisible at mid contrast and differs only in
its rate of change — which is what "the relationship is stronger here" should mean.

The useful consequence: with pivot 0.5, all `|gain| ≤ 1`, offsets 0 and no gain
jitter, clipping is **mathematically impossible**. Saturation is something you switch
on deliberately, not a confound in every run. There are exactly four switches — a
gain magnitude above 1 (including background gain), a nonzero offset, a pivot other
than 0.5, or nonzero jitter gain σ — and the canvas caption reports the clip fraction
so you always know whether it fired.

### Zone kinds

| kind | gain | what it does |
|------|------|--------------|
| `heat` | `> 1` | tracks `c` more steeply than its surround; **saturates** |
| `strong` | `background_gain < gain ≤ 1` | strong tracking, never saturates. Needs background gain below 1 |
| `dead` | `≈ 0` | barely responds; noise dominates |
| `anti` | `< 0` | runs backwards in `c` |
| `random` | ignored | masks the pixel entirely: it ignores `c` |

`heat` and `strong` are a matched pair. `heat` at gain 2 with background 1 and
`strong` at gain 1 with background 0.5 both track exactly twice as steeply as their
surround; heat clips at 10 of the 21 grid points, strong never does. Comparing them separates
*strong coupling* from *coupling strong enough to saturate*.

Overlapping zones **average** (partition-of-unity blend), bounded by the background
gain and the contributing zone gains — so overlap can never produce a runaway gain.
The alternative `replace` mode gives shared pixels to the highest-priority zone, like
z-order in a drawing program.

### Pixel models

| model | notes |
|-------|-------|
| `bernoulli` | contrast *is* the chance a pixel is white. No clipping bias, but variance vanishes at the ends |
| `clipped_gaussian` | continuous, carries the classic censoring bias. The closest to constant precision |
| `beta` | bounded and never clipped — yet this is where the LDA bends most (F3). Not the "no-artefact reference" it was designed as |

---

## Gotchas

- **Zones are invisible at c = 0.5.** By construction — that is the pivot. Slide the
  preview contrast away from the middle, or switch on **gain map**.
- **Clicking places things; it does not pan or zoom.** Drag, scroll-zoom and the
  Plotly toolbar are switched off over the canvas, because they fought the click and
  made coordinates meaningless. If a click ever fails to register, the **x / y +
  place** button does the same thing numerically, and **undo last** reverses it.
- **Clicking the active tool again keeps it.** (A segmented control would normally
  deselect it; the app restores it.)
- **Placing a patch exactly on an existing one is refused**, with a message:
  an identical patch duplicates features without adding information.
- **Shrinking the plane or growing the patch size** moves any patch that no longer
  fits back onto the plane and merges exact duplicates, with a notice.
- **Training extremes default to 0.05 / 0.95, not 0 / 1.** At exactly 0 the Bernoulli
  world gives an all-black field with zero variance, so the covariance is singular
  and the fit is arbitrary. Set them to 0 and 1 to watch that happen; the code warns.
- **`features` drives speed more than anything else.** `pixels` on a step-10 grid is
  2500 features and a ~5 s fit; `patch_mean` is 100 features and ~0.3 s.
- **"covariance: singular" is expected**, not a bug, whenever features outnumber
  training trials. It is why shrinkage is on by default. Note `svd` cannot take
  shrinkage at all — the config validator rejects that pairing.
- **Patch placement can decide the answer.** Patches on heat zones give a nearly
  perfect mapping; patches on dead zones give noise. The **what the model used** tab
  shows what your patches actually cover; `ablation_*` configs demonstrate it.
- **Edit the canvas after training and the app warns you.** The figures describe the
  model, not what is now on screen. Press train again.
- **Never compare raw decision values between models.** Arbitrary units. The
  comparison tab rescales every curve for exactly this reason.

---

## Full CLI

The app is the front door, but the engine is scriptable — which is what you want for
a careful run across many seeds.

```bash
pytest                                    # 103 tests, ~30 s. Run after any change.

# one config, with a four-panel diagnostic PNG
python -m scripts.run_experiment configs/scenario_heat_dead.json --figures

# the whole battery: 4 controls, 5 scenarios, 2 ablations   (~5–15 min)
python -m scripts.run_experiment configs/control_*.json configs/scenario_*.json \
                                 configs/ablation_*.json --figures

# five seeds, so between-seed spread can be reported        (~30–60 min)
python -m scripts.run_experiment configs/scenario_*.json --seeds 0 1 2 3 4

# smoke test — fast, NOT valid for reporting
python -m scripts.run_experiment configs/control_simple.json --quick --no-save

# sweep one factor at a time
python -m scripts.run_matrix --factors pixel_noise solver features --seeds 0 1 2 3 4 --no-cv
python -m scripts.run_matrix --dry-run          # list the cells, run none

# regenerate the committed configs (only if you edit make_configs.py)
python -m scripts.make_configs
```

Results go to `results/` (trial tables as parquet, summaries as CSV/JSON, figures as
PNG) and `models/`. `summary_table.csv` accumulates across invocations (a re-run of
the same run id replaces its own row). Both are git-ignored — reproducible from a config plus a seed.
Re-running the same config and seed gives bit-identical output; if it ever does not,
there is an unseeded random stream and every result is suspect.

---

## Findings so far

11 configs × 5 seeds at the careful (pre-registered) settings, plus a pixel-model,
solver and feature sweep on `heat_dead` (5 seeds each). Spread is across seeds. Full
write-up in `docs/decisions.md` F3 and F4; F1 and F2 are withdrawn.

**The controls behave.** All-dead, all-random and label-shuffle are `degenerate` in
5/5 seeds (label-shuffle max |t| 1.9 against a threshold of 10); the simple world is
`affine`, κ 0.0003.

**Linearity holds in the simple regime** — `affine` in 5/5 seeds, κ < 0.01 — with
Bernoulli pixels for the simple world, heat + dead, heat + dead + anti, the same plus
random zones, the clip-free strong + dead pair, and dead-zone-only sampling. The
saturating zones leave a faint S (κ almost entirely cubic, at most ~0.005 in units of
c). The LDA reads out *straighter* than the gain-weighted oracle (κ ≈ 0.049): it
down-weights pixels that clip.

**It fails in four regimes:**

| condition | LDA | oracle | reading |
|---|---|---|---|
| Beta pixel noise, all three levels (heat + dead world) | nonlinear 15/15, κ 0.20–0.21 | affine, κ 0.045 | **the method bends where a straight readout exists** |
| clipped-Gaussian noise, all three levels | nonlinear 15/15, κ 0.07–0.08 | nonlinear, κ 0.06–0.09 | mostly the censoring itself |
| patches only on saturating zones | nonlinear 5/5, κ 0.19, range use 0.86 | nonlinear, κ 0.21 | baked into what is observed |
| zones jitter between trials | nonlinear 3/5, κ 0.049–0.062 | κ 0.048 | on the threshold, ~7× the no-jitter κ |

Readout choices move κ too, though all stayed affine: `patch_mean` features 0.034 vs
`pixels` 0.0068; `svd` 0.0007 vs `lsqr` + shrinkage 0.0068.

The Beta row is the headline: an oracle built from the true gains is affine, the
extremes-trained LDA on the same trials is not. It needs both ingredients: the
*uniform* world under Beta is affine in 15/15 runs (3 noise levels × 5 seeds,
κ 0.0006–0.0008), so it is the combination of Beta noise with saturating zones that
bends the readout.
Why Beta and not Bernoulli is open.

**Precision varies everywhere.** SD max/min exceeded the 1.5 threshold in every
non-null condition, including the simple world (2.2–2.5): Bernoulli worlds 3.0–53,
Beta 16–24, clipped Gaussian 1.6–2.0. The readout shares the blame — on the same world
and seeds, `svd` gives 7.2 against 48 for `lsqr` + shrinkage.

Practical reading: in the regimes where the mean readout is straight, the error bar on
a single trial still changes across the range; and in several realistic regimes the
mean readout is not straight at all.

---

## Code layout

```
app.py              Streamlit layout and event handling. No computation.
ui/controls.py      session state, config assembly, caching
ui/plots.py         Plotly figures

core/config.py      pydantic models — the single source of every default
core/rng.py         one master seed -> six named streams
core/data.py        zones -> gain field -> theta -> pixels
core/sampling.py    patch layout -> feature vector
core/dataset.py     trials, with true c and realised nuisance state attached
core/model.py       scaler + LDA; the only path from features to decision values
core/evaluation.py  diagnostics, the oracle, weight attribution, run_experiment

storage/            config hashing, joblib models, parquet trials, JSON sidecars
scripts/            CLI entrypoints and static matplotlib figures
tests/              103 tests: analytic sanity, determinism, leakage, verdict, app, scripts
configs/            committed experiment definitions (configs/user/: saved from the app)
```

Dependencies point one way: `app.py` → `ui/` → `storage/` → `core/`. `core/` imports
no UI and does no file I/O, so the pipeline runs in a test or a cluster job with no
display.

Before changing anything, read [`CLAUDE.md`](CLAUDE.md) — six invariants, commenting
etiquette, and how to talk about results without overclaiming. The two you are most
likely to trip over: `DataConfig` and `SamplingConfig` are never merged or
cross-read, and all randomness flows through an explicitly passed generator.

---

## Not built yet

- **Dragging.** Zones are placed by clicking and adjusted with sliders; no drag
  handle (`docs/decisions.md` D10, D12).
- **Cross-seed comparison in the app.** The comparison tab overlays models trained in
  the session; seed sweeps are a script for now.
- **Non-uniform sampling** beyond manual placement — a foveated density model is the
  natural next step.
- **Why Beta noise bends the LDA** when the oracle stays straight — the most
  pressing open question (F3).
- **Other readouts** — whether logistic regression or ridge, trained the same way,
  bend where the LDA does not. Cheap to add, and it would help answer the Beta
  question.
- **A second latent variable**, to test whether the decision value confounds them.
  Probably the most useful follow-up.
