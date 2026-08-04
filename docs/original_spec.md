# Original written brief (committed verbatim)

Kept in the repository so that `plan.md` and the code can cite the source of a
deviation by filename instead of by a dangling section number. Do not edit this
file to reflect later decisions — that is what `plan.md` and
`docs/decisions.md` are for.

Two things in this brief were deliberately superseded; both are recorded in
`docs/decisions.md`:

- the **additive zone model** of §3 (`mu_i(c) = c + sum_k Z_k(i)`), replaced by
  the affine gain field of `plan.md` §4.2 — see D1;
- the **development phase ordering** of §13, reordered so the headless pipeline
  and the diagnostics precede any UI work — see `plan.md` §2.

---

## 1. Objective

Build an interactive, browser-based Python application to:

* Simulate pixel fields with contrast-driven statistics
* Introduce spatial heterogeneity (zones)
* Allow interactive placement of data zones (affecting pixel generation) and LDA
  sampling patches (feature extraction)
* Train multiple LDA models on extreme contrast conditions
* Evaluate models on continuous contrast ranges
* Quantify and visualise the mapping between LDA decision output and contrast,
  and their correlation

## 2. Technology stack

Streamlit for the UI; `streamlit-drawable-canvas` for drag-and-drop placement of
circles (zones) and rectangles (sampling patches); NumPy and SciPy for
generation; scikit-learn's `LinearDiscriminantAnalysis`; Plotly for interactive
plots with matplotlib as a static fallback; Pydantic for config validation;
joblib for model serialisation; JSON for configs.

Noted limitation: the drawable canvas is not object-oriented, so object metadata
must be managed manually.

## 3. Conceptual model

Each pixel `x_i ~ N(mu_i(c), sigma_i^2)` with `c` in `[0,1]`, `mu_i(c)` in
`[0,1]`. To keep Gaussian mass from exceeding 1, use a clipped Gaussian:
`x_i = clip(N(mu_i, sigma_i), 0, 1)`.

Base mean mapping: `mu_i(c) = c`.

Zone contributions are **additive**: `mu_i(c) = c + sum_k Z_k(i)`, where

| Type | Effect |
| --- | --- |
| Heat zone | `+alpha * c` |
| Dead zone | `+alpha * epsilon` (weak dependence) |
| Anticorrelated | `-alpha * c` |
| Random zone | ignores `c`, replaces with noise |

After summation, `mu_i(c) = clip(mu_i(c), 0, 1)`.

Flagged pitfall: additive zones can saturate values, producing nonlinear
distortion. (This is D1's starting point.)

## 4. Interaction model

Top toolbar with patch tools (add LDA patch — square; add zone — circle) and a
zone-type selector (heat / dead / anti / random). Left panel: model registry
listing trained models with their training and sampling configs, with select and
delete actions. Right panel: saved data configs with a `+` to create a new one,
each storing zones, global noise and plane size. Centre: the pixel-field canvas
showing the generated field with zone and patch overlays. Bottom panel: global
data controls (plane size, global noise level, zone jitter, random seed). Bottom
analysis panel: select model(s) and data config(s), define a contrast grid and
repetitions, run evaluation.

## 5. Patch systems

Data zones affect data generation, are circles, and live in the `DataConfig`. LDA
sampling patches affect feature extraction, are squares, and live in the
`SamplingConfig`. Pitfall: these must never be conflated; keep separate object
registries. (This became invariant I1.)

## 6. Feature extraction

Generate the pixel field, extract patches (overlap allowed, user-defined
placement), flatten each patch, concatenate into a feature vector.

## 7. Training

Training data at `c = 0` (label 0) and `c = 1` (label 1). For each sample:
generate a plane, apply jitter to zones, sample patches, extract features. Fit
`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`, giving a weight
vector `w` and bias `b`.

## 8. Evaluation

For each `c` in `[0,1]`: generate multiple samples, extract features, compute the
decision function `f(x) = w^T x + b`, optionally its logistic transform. Output
metrics: mean LDA output per contrast, variance, and the Pearson correlation
`r = corr(c, f(x))`.

## 9. Visualisation

Required: LDA output vs contrast with a mean line and variance shading; a scatter
plot of raw samples. Optional: a histogram per contrast level.

## 10. State management

Use Streamlit session state for `data_configs`, `sampling_configs`, `models`,
`active_model`, `canvas_objects`. Pitfall: Streamlit reruns the entire script, so
state must be persisted carefully.

## 11. Persistence

Models as `.joblib` including LDA weights and training metadata; configs as
`.json` including zones, sampling patches and global parameters.

## 12. Critical pitfalls

**12.1 Statistical.** Additive zones lead to saturation and hence nonlinear
effects; clipping introduces bias; LDA assumes Gaussian data, which is violated.

**12.2 UI / interaction.** The drawable canvas does not natively track object
types, so types must be mapped manually; dragging precision issues suggest grid
snapping.

**12.3 Sampling bias.** User-placed patches can over-represent zones or miss
signal regions.

**12.4 Numerical stability.** High noise destabilises the LDA covariance
estimate; small sample sizes cause overfitting.

**12.5 Performance.** Repeated simulation is expensive; cache results with
`st.cache_data` and vectorise generation.

## 13. Suggested development phases

1. Static configs, no canvas (manual input), full pipeline working
2. Add canvas interaction, patch and zone placement
3. Model registry and persistence
4. Advanced visualisation and batch evaluation

(Superseded: `plan.md` §2 puts the headless pipeline **and the diagnostics
suite** in Phase 1, on the grounds that the research question is answerable from
a script and the UI is for exploring results we can already compute.)

## 14. Open design risks

* Clipping vs truncated Gaussian — affects interpretability
* Zone overlap handling — additive may explode
* UI complexity — may outgrow Streamlit

## 15. Deliverable expectations

Implement a modular backend with core logic separated; use Streamlit for the UI
and the drawable canvas for interaction; maintain strict separation of
`DataConfig`, `SamplingConfig` and the model; implement persistence; provide
reproducibility via seeds.
