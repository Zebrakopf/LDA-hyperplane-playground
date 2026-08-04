# Details

The longer companion to [`README.md`](README.md). Everything here is reference —
read it when a control confuses you, or before quoting a number.

---

## What's on screen

**Toolbar.** What a canvas click does (place a zone, add or remove a sampling patch,
or look only), the speed preset, which contrast the canvas previews, display
toggles. `show gain field` draws the *coupling gain* rather than a pixel
realisation, which is what the zones actually mean — a single realisation at mid
contrast can look like plain noise.

**Canvas (centre).** One realised pixel field. Circles in kind-specific colours are
**data zones** — the world. Yellow squares are **sampling patches** — what the model
sees. They are deliberately never drawn alike; conflating the world with the observer
is the one mistake that would invalidate everything here.

**Left: trained models.** Every model trained this session stays listed with its
verdict. Load one's config back onto the canvas to edit and re-run, or save to disk.

**Right: configs and the zone list.** Load a committed config, save your own, and
fine-tune each zone (position, radius, gain, edge profile) with sliders. Use the zone
list when a click got you close and you want exact numbers.

**Below: world, observer, training.** Plane size, background gain, pivot, pixel
model, noise, jitter; patch size and feature mode; where the training extremes sit,
trial counts, solver and shrinkage, the seed.

**Analysis panel.** Six headline numbers, then five tabs: the mapping, spread and
residuals, what the model used, model internals, and a comparison view.

---

## Reading the numbers, in full

**`verdict`** — the pre-registered answer to "is the mapping affine?". Thresholds
were fixed before any result was looked at (`plan.md` §1.3), so it is not a judgement
made after seeing the curve.

**κ, the curvature index** — `|β₂|/|β₁|` from an orthogonal-polynomial fit. Below
0.05 is affine, above 0.15 a clear violation. For a feel for the scale,
`f = c + γc²` gives:

| γ | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|-----|------|-----|-----|-----|
| κ | 0.024 | 0.054 | 0.090 | 0.135 | 0.180 |

**`r` is never enough on its own.** A visibly S-shaped curve sits happily at
r = 0.99. That is why κ sits next to it and why the **spread & residuals** tab
exists: a flat residual line centred on zero means genuinely affine, while a smile or
an S is curvature a correlation coefficient will not show you.

**SD max/min** — how much the *precision* of the readout varies across the range. A
perfectly straight mean curve with a 20× swing in spread still breaks the use of
decision values as measurements with constant error bars. Scored separately from the
verdict, because it is a different failure.

**`degenerate` is not an error.** The slope is not resolved above noise, so κ and the
calibration numbers are 0/0 and show as n/a. It is the *correct* answer for a world
where nothing tracks `c`.

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

`heat` and `strong` are a matched pair. `heat` at gain 2 with background 1 tracks
twice as steeply and clips at 10 of the 21 grid points; `strong` at gain 1 with
background 0.4 tracks 2.5× as steeply and never clips. Comparing them separates
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
| `beta` | bounded, needs no clipping at all — the no-artefact reference for the other two |

---

## Gotchas

- **Zones are invisible at c = 0.5.** By construction — that is the pivot. Slide the
  preview contrast away from the middle, or switch on `show gain field`.
- **Clicking places things; it does not pan or zoom.** Drag, scroll-zoom and the
  Plotly toolbar are switched off over the canvas, because they fought the click and
  made coordinates meaningless. If a click ever fails to register, the **x / y +
  place here** row does the same thing numerically, and **undo last** reverses it.
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
pytest                                    # 62 tests, ~8 s. Run after any change.

# one config, with a four-panel diagnostic PNG
python -m scripts.run_experiment configs/scenario_heat_dead.json --figures

# the whole battery: 4 controls, 5 scenarios, 2 ablations   (~4.5 min)
python -m scripts.run_experiment configs/control_*.json configs/scenario_*.json \
                                 configs/ablation_*.json --figures

# five seeds, so between-seed spread can be reported        (~25 min)
python -m scripts.run_experiment configs/scenario_*.json --seeds 0 1 2 3 4

# smoke test — fast, NOT valid for reporting
python -m scripts.run_experiment configs/control_simple.json --quick --no-save

# sweep one factor at a time
python -m scripts.run_matrix --factors pixel_noise --seeds 0 --no-cv
python -m scripts.run_matrix --dry-run          # list the cells, run none

# regenerate the committed configs (only if you edit make_configs.py)
python -m scripts.make_configs
```

Results go to `results/` (trial tables as parquet, summaries as CSV/JSON, figures as
PNG) and `models/`. Both are git-ignored — reproducible from a config plus a seed.
Re-running the same config and seed gives bit-identical output; if it ever does not,
there is an unseeded random stream and every result is suspect.

---

## Findings so far

Full battery at seed 0, plus a seven-cell pixel-model × noise sweep. Also in
`docs/decisions.md` (F1, F2).

**The controls behave.** All three null controls came out `degenerate`; the simple
world came out `affine` with R² = 0.999. The pipeline can detect affinity when it is
there and does not manufacture it when it is not.

**The linearity assumption held everywhere tested.** Every non-degenerate run
returned `affine` with κ between 0.0000 and 0.0021 — two orders of magnitude inside
the threshold. That held with saturating heat zones active (4.2% of pixels clipping),
with anti-correlated and random zones, under geometric jitter, under all three pixel
models, and whether patches sat on heat zones or dead zones.

**The precision assumption did not.** SD max/min ran from 1.67 to 46.98. The pixel
model sweep says why: Bernoulli (47) and Beta (16–24) both have mean-dependent
variance `theta·(1−theta)`, while clipped Gaussian (1.7–1.9) is nearly constant. The
varying precision is a property of the observation model, not of the LDA.

Practical reading: **the mean decision value tracks the latent variable faithfully,
but the error bar on a single trial can differ by more than an order of magnitude
across the range.**

Caveat: **one seed.** Between-seed spread is not characterised yet — run the
five-seed command above before quoting any of this.

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
tests/              62 tests: analytic sanity, determinism, leakage, app
configs/            committed experiment definitions
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
- **Other readouts** — whether logistic regression or ridge, trained the same way,
  bend where the LDA does not. Cheap to add, directly relevant.
- **A second latent variable**, to test whether the decision value confounds them.
  Probably the most useful follow-up.
