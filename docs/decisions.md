# Decision log

One entry per non-obvious choice: context, decision, consequence. Newest last.
Required by `CLAUDE.md` §9. Keep entries short; the reasoning that belongs next
to code goes in the code.

---

## D1 — Coupling is an affine gain field, not additive zone contributions

**Context.** `docs/original_spec.md` defines `mu_i(c) = c + sum_k Z_k(i)` with a
clip to `[0, 1]`. Its own pitfalls section flags that overlapping zones then
saturate, and saturation is a nonlinearity — the very thing under study.

**Decision.** Replace it with `theta_i(c) = pivot + a_i * (c - pivot) + b_i`
(`plan.md` §4.2). Overlap composes by partition-of-unity averaging, not summation.

**Consequence.** Clipping becomes opt-in: with `pivot = 0.5`, `|a| <= 1`,
`b = 0` and `gain_sigma = 0` it is mathematically unreachable, so saturation is
an experimental condition rather than a confound in every run. Cost: a zone's
nominal gain is only reached in its core, because soft edges average toward the
background.

---

## D2 — `background_gain` is configurable

**Context.** With the background fixed at gain 1, a zone cannot track `c` more
strongly than its surround without exceeding `|a| = 1` and therefore clipping. So
"strong coupling" and "saturating coupling" could not be separated.

**Decision.** `DataConfig.background_gain` (default 1.0), plus a `strong` zone
kind meaningful when the background is below 1.

**Consequence.** `heat` (gain 2.0, background 1.0, saturates) and `strong`
(gain 1.0, background 0.5, never saturates) are both exactly 2x their surround,
isolating saturation. *(Corrected in D14: this entry originally said background
0.4 and "the same 2.5x ratio" — that pair was 2x vs 2.5x.)* It also makes the all-dead control expressible
(`background_gain = 0`). The pair matches coupling *ratio*, not absolute gain —
`plan.md` §12, question 6.

---

## D3 — Oracle weights are NOT mean-centred

**Context.** `plan.md` §7.3 originally specified mean-centred, gain-proportional
oracle weights. In the simple scenario every pixel has gain 1, so centring makes
the weights identically zero and the "best possible linear readout" a constant.

**Decision.** Drop the centring; L2-normalise instead. The affine fit absorbs any
offset.

**Consequence.** The oracle is informative in the uniform case (it averages every
pixel, which is correct there) and still returns all-zero weights — hence
`degenerate` — when nothing tracks `c`, which is also correct. `plan.md` §7.3 was
updated.

---

## D4 — Degeneracy is decided by a slope-to-noise margin, not by a CI covering zero

**Context.** `plan.md` §1.3 originally declared a report `degenerate` when the 95%
CI of `beta1` covers zero. With pure noise that test flags a "significant" slope
5% of the time by construction. Measured over 200 simulated null runs: 10 false
alarms. Across four null controls x five seeds, Phase-1 exit criterion 2 would
therefore fail at random roughly once per batch.

**Decision.** `degenerate` when `|beta1| / SE(beta1) < 10` (`SLOPE_T_MIN` in
`core/evaluation.py`). Over the same 200 null runs max `|t|` was about 2.6
*(corrected in D14; this entry originally said 0.13, which contradicts the 10
false alarms above)*, while a real mapping reaches `|t|` in the hundreds. A perfectly constant `f` is
handled explicitly as `t = 0`, not `0/0`.

**Consequence.** The null controls are reliably degenerate. At the default 4200
evaluation trials the threshold corresponds to `|r| ~ 0.155` (`|r| ~ 0.36` at the
fast preset's 660), so a mapping whose
linear component is buried in trial noise is also reported as degenerate rather
than getting a meaningless `kappa` — `kappa` divides by exactly that component.
`plan.md` §1.3 and the `LinearityReport` (new `slope_t` field) were updated.

---

## D5 — `sd_ratio` excludes contrast levels with exactly zero variance

**Context.** Under the Bernoulli pixel model `theta = 0` or `1` makes every pixel
deterministic, so `SD(f|c)` vanishes at the grid endpoints and the raw max/min
ratio is `inf` for even the cleanest simple scenario. H3 would be untestable for
the default pixel model.

**Decision.** Compute `sd_ratio` over levels with nonzero SD, and report
`n_zero_variance_levels` alongside it so the exclusion is never invisible.

**Consequence.** H3 asks the question actually of interest — does precision vary
across the range where there *is* variance — at the cost of one more number to
read. It does not rescue the Bernoulli model from being heteroscedastic; see F2.

---

## D6 — Zone-kind composition shares are normalised by the blend denominator

**Context.** Reporting each kind's raw summed falloff weight lets two overlapping
zones each claim full credit for the shared pixels, so the "fractions" sum above
1 and read like a coverage bug.

**Decision.** Normalise by the same denominator the blend composition uses, so
coupling kinds plus background partition the sampled area exactly. `random` is
reported separately as a plain fraction of masked pixels, because a masked pixel
leaves the coupling composition entirely.

**Consequence.** The composition table sums to 1 by construction (asserted in
`tests/test_pipeline.py`), and the `random` row is deliberately outside that sum.

---

## D7 — Phase-1 figures use matplotlib in `scripts/`, not `ui/plots.py`

**Context.** Results need to be checkable by eye during Phase 1, but `ui/` is
Streamlit/Plotly territory and importing that stack into a batch run is both slow
and a step toward breaking invariant I2.

**Decision.** `scripts/figures.py` renders static PNGs with the `Agg` backend.
`ui/plots.py` (Plotly, interactive) arrives with Phase 2.

**Consequence.** Two plotting code paths eventually. Accepted: they serve
different consumers, and the headless one must keep working in CI without a
browser.

---

## D8 — `app.py` was absent from the first draft — SUPERSEDED by D10

**Context.** `CLAUDE.md` §3 lists `app.py` in the target layout, and `plan.md` §2
said not to start Streamlit work while the headless pipeline was incomplete.

**Decision at the time.** Ship no `app.py` rather than a stub that looks like a
feature.

**Superseded.** The phase ordering was right about engineering order and wrong
about the deliverable. The point of this project is a tool the lab can poke at;
the headless pipeline is the engine, not the product. `app.py` now exists — D10.

---

## D9 — `cv_folds = 0` disables cross-validation

**Context.** With 2500 features and Ledoit-Wolf shrinkage a single LDA fit takes
~4.7 s, so 5-fold CV multiplies the cost of every run by six and dominates a
sweep entirely.

**Decision.** Keep 5 folds as the default for single runs; `cv_folds = 0` skips
CV and reports `cv_accuracy` as NaN. `scripts/run_matrix.py --no-cv` uses it.

**Consequence.** Sweep cells carry no CV accuracy. Acceptable: a sweep asks about
the `c -> f` mapping, and CV accuracy on two extreme classes is near-ceiling in
every condition observed so far.

---

## D10 — The app is the front door; canvas clicks use native Plotly events

**Context.** The headless-first plan produced a correct pipeline that nobody in the
lab would ever open. What was actually wanted is an explorable toy: place zones and
patches, press a button, see whether the readout bends.

**Decision.** `app.py` plus `ui/controls.py` and `ui/plots.py`, run with
`streamlit run app.py`. Three sub-decisions worth recording:

1. **No `streamlit-drawable-canvas`.** The original spec named it for drag-and-drop
   placement. Instead `st.plotly_chart(on_select="rerun")` captures clicks on the
   pixel field directly. One fewer dependency, one fewer thing to break on a
   Streamlit upgrade, and it yields exact plane coordinates rather than canvas
   pixels needing rescaling. Cost: you place by clicking and then adjust with
   sliders, rather than dragging. The better trade for a tool whose zone geometry
   needs numeric precision anyway.
2. **Speed presets, defaulting to `fast (explore)`.** A run at the pre-registered
   settings takes ~6 s, which is unusable per slider nudge. The fast preset
   (`patch_mean` features, 150 train / 60 per level, 11 levels) returns in ~0.35 s.
   It is *below* the pre-registered minimum, so the app labels every verdict it
   produces and names the preset to switch to before quoting anything.
3. **Every cache is keyed on the JSON of a validated config**, and the app warns
   loudly when the canvas has been edited since the displayed model was trained.
   A cached Streamlit app silently showing figures for a superseded config is the
   most misleading state this tool could be in.

**Consequence.** Two plotting paths (Plotly for the app, matplotlib for batch runs
— D7) and a UI that must be kept working: `tests/test_app.py` drives the real
script through `streamlit.testing.v1.AppTest`, including the null control and the
stale-model warning.

---

## D11 — `init_state` seeds the patch grid once, via an explicit flag

**Context.** `init_state` originally refilled the patch grid whenever the list was
empty. Found by `test_clearing_patches_warns`: that silently breaks the "clear
patches" button, because Streamlit reruns the script after the click, `init_state`
sees an empty list and refills it. The observer could never be emptied.

**Decision.** An `initialised` flag; the grid is seeded exactly once per session.
An empty patch list is then a legitimate state, so `app.py` handles it explicitly —
a warning and a live canvas, rather than the `SamplingConfig` validation error that
a manual layout with no patches correctly raises.

**Consequence.** Two code paths guard on "are there patches yet". Worth it: being
able to look at a world with no observer is genuinely instructive.

---

## D12 — The canvas needs an invisible scatter lattice to be clickable at all

**Context.** Click-to-place shipped broken. The canvas was a bare Plotly `Heatmap`,
and Plotly's point-selection API fires **only for scatter-like traces** — a heatmap
is not selectable. So `st.plotly_chart(on_select="rerun")` returned nothing however
hard anyone clicked, and the only behaviour left was drag and zoom. Nothing raised
and no test failed: `AppTest` cannot deliver a Plotly selection event, so the six
app tests all passed against a feature that did not work.

**Decision.** Overlay an invisible `Scattergl` lattice at every pixel centre
(stride 2 above a 160-pixel plane), marker size 9 for a forgiving hit area and
opacity 0 in all three states — `marker`, `selected` and `unselected`, because
Plotly otherwise reveals the lattice as a grey smudge around the last click. Chart
`config` disables `scrollZoom`, `doubleClick` and the modebar; layout keeps
`dragmode=False` with `clickmode="event+select"` so a press is always a selection.

Three supporting changes:

- The widget's selection state is **cleared after being acted on**, rather than
  de-duplicated by comparing coordinates. The coordinate guard silently swallowed a
  genuine second click on the same pixel, so two patches could never be stacked.
- `place_with_current_tool` is the single entry point for both the canvas click and
  a new numeric "place here" button, so the two routes cannot drift. The numeric
  route exists as a path that cannot depend on Plotly events firing, and is also
  how you hit an exact coordinate.
- `add_patch` centres with `round(coord) - size // 2`, not `round(coord - size/2)`.
  The latter hits Python's banker's rounding on the .5 cases, so a click at row 21
  landed at 18 while row 22 landed at 20 — an off-by-one for some clicks only.

**Consequence.** ~10,000 invisible markers on a 100×100 plane; `Scattergl` handles
that without visible cost. The real lesson is about verification: a headless harness
proved the app *runs*, not that it *works*. `tests/test_app.py` now guards the
structural preconditions it CAN check — that a selectable trace exists, that it
spans the plane, that it is invisible in every state, that drag is off — and the
click path itself is verified by driving Chromium against a live server. Anything
that depends on a browser event needs that second kind of check.

---

## D13 — Every widget carries a tooltip, and a test enforces it

**Context.** The app exposes a lot of knobs whose meaning is not guessable from the
label — `pivot`, `background gain`, `shrinkage`, `d′`, `top 1% weight mass`,
`effective rank`. It is aimed at colleagues who did not write it, so an unexplained
control is a real defect rather than missing polish. An audit found 49 of 73
helpable widgets with no `help=`.

**Decision.** A tooltip on all 73, written to say what the control *means for the
experiment* rather than restating its label: what the knob does, what it costs, and
where it can mislead. Where a value is routinely misread, the tooltip says so
outright — `train accuracy` near 1.0 "is normal and means little", `singular`
covariance "is EXPECTED, not a failure", `degenerate` "is the CORRECT answer when
nothing tracks the contrast".

Two tests guard it, both **static** rather than through `AppTest`: coverage has to
hold for widgets inside branches a given run never reaches (the per-zone editors,
the pixel-model-specific noise sliders, the comparison tab that needs two trained
models), and a runtime check would silently pass on the branches it did not visit.

- `test_every_widget_has_a_tooltip` parses `app.py` and fails on any helpable
  widget call without `help=`.
- `test_tooltips_are_substantive` rejects tooltips under 30 characters, and — the
  check that actually matters, since length is easy to game — rejects any tooltip
  that contains every substantial word of its own label without being much longer
  than it. That is the `help="the plane height"` failure mode.

**Consequence.** Adding a widget now fails the suite until it is explained. The
30-character floor is deliberately low: a handful of controls genuinely need one
short sentence, and the echo check is what carries the quality bar.

---

## F1 — First finding: H2 (affinity) survives every condition tested — WITHDRAWN (see F3)

> **Withdrawn.** Measured with a curvature index that could not see S-shaped
> curves (D14). Re-measured, several of the conditions below are nonlinear.

Across the four controls, five scenarios, two sampling ablations and seven
pixel-model x noise cells at seed 0: every non-degenerate run returned
`verdict = affine` with `kappa` between 0.0000 and 0.0021, far below the
pre-registered 0.05. This holds with saturating `heat` zones active (mean clip
fraction 0.042), with anti-correlated and random zones present, under geometric
jitter, and under all three pixel models. Single seed, so between-seed spread is
not yet characterised — that is the remaining Phase-1 exit criterion.

## F2 — First finding: H3 (constant precision) fails, structurally — REVISED (see F4)

> **Revised.** The failure stands, but the causal claim ("not the LDA") does not:
> the solver alone changes the spread ratio from 48 to 7 (F4).

`sd_ratio` ranged from 1.67 to 46.98 and only the clipped-Gaussian cells came
close to homoscedastic. The pixel-model sweep locates the cause: Bernoulli
(47) and Beta (16-24) both have mean-dependent variance, `theta*(1-theta)`, while
clipped Gaussian (1.7-1.9) has nearly constant variance. So the varying precision
is a property of the observation model, not of extremes-trained LDA. Practical
reading: the *mean* of `f` tracks `c` affinely, but the *error bar on a single
trial's* `f` can differ by more than an order of magnitude across the contrast
range — which matters for anyone running statistics on decision values as if they
were equally precise measurements.

---

## D14 — The September review: the verdict could not see saturation

**Context.** A review in September 2026 reproduced a set of defects, several of
which changed the scientific conclusions. The most serious: `kappa = |β₂|/|β₁|`
measures only the quadratic term. Saturation here is symmetric around the pivot
(0.5) — clipping flattens *both* ends — so the curve it produces is S-shaped and
odd-symmetric: β₂ ≈ 0 while β₃ is large. A logistic with slope 12 scored
κ = 0.001 and a symmetric hard clip 0.0001; both were called `affine`. The cubic
F-test flagged them (p ≈ 0) but its result was never used. F1's claim that
linearity survived "even with saturating zones" was therefore measured by a test
that could not fail on the saturating case.

**Decisions.**

1. `kappa = √(κ₂² + κ₃²)` with κ₂ = |β₂|/|β₁| and κ₃ = |β₃|/|β₁|, both reported.
   It reduces to the old index when β₃ = 0, so one-sided bends score as before.
   The `nonlinear` branch now fires on the cubic nested test as well as the
   quadratic one. Thresholds (0.05 / 0.15) unchanged. On the new scale a logistic
   of slope 4 scores 0.075, slope 8 scores 0.19.
2. `sd_ratio` treats a level as zero-variance when its SD is below 1e-9 of the
   largest, not when it is exactly 0. A deterministic level comes out of pandas as
   ~1e-12, so the smoke test had printed `SD max/min = 71,557,154,339,403`.
3. `effective_range_use` interpolates the mean curve at the quarter points. It had
   selected grid points between the quartiles, so a straight line read 0.40 on the
   fast preset's 11-point grid.
4. A fit with nothing to learn (every feature constant within each training class —
   e.g. both extremes fully clipped) is now `fit_failed` with a warning, not
   `degenerate`. It had been indistinguishable from "nothing tracks c" while the
   oracle on the same data had a strong slope. d′ is NaN for 0/0, not `inf`.
5. Attribution and coverage give random-masked pixels gain 0 and exclude them from
   the background share. The all-random control had reported a mean sampled gain
   of 1.0 and coverage fractions summing to 2.0.
6. The label-shuffle control relabels exactly half of each true class. A plain
   permutation left the shuffled classes unbalanced and leaked a real slope
   (|t| up to 9.3 against the threshold of 10 on the fast preset; now ≤ 1.9 over
   5 careful-preset seeds).
7. In-sample d′ and accuracy are labelled as such; a fresh d′ on evaluation trials
   at the grid levels nearest c_lo / c_hi is reported alongside (the all-random
   control reads 3.6 in-sample, ~0 fresh).
8. Configs forbid unknown keys and re-validate on assignment: a typo like
   `"n_per_clas"` used to be dropped silently, and svd + shrinkage could be set
   after load.
9. `scenario_strong_dead` uses background 0.5, so heat and strong are both 2x their
   surround (D2 corrected). Effective rank now applies numpy's tolerance to
   singular values, as its comment always claimed.

**Consequence.** F1 is withdrawn and F2 revised; F3 and F4 replace them. The
suite went from 62 to 103 tests; 14 of the new ones fail on the pre-review code,
each on one of the defects above.

---

## D15 — App state: every widget is key-bound, every mutation is a callback

**Context.** The first app did `state[k] = st.slider(label, …, state[k])` with no
key. That looked equivalent to binding but was not. The canvas and configs were
built from state *before* the widgets further down the script wrote their new
values, so every setting took effect one interaction late. A keyless widget's
identity includes its value, so it was rebuilt after each change and an arrow-key
nudge worked only once. Loading a config kept an old zone's slider values whenever
ids matched; "clear patches" after training crashed the results panel; a config
holding only `heat2` produced a second `heat2` and a DuplicateElementKey crash;
models were keyed on `model_id`, which ignores evaluation settings, so re-running
with more trials overwrote the earlier result; and shrinking the plane silently
collapsed 100 patches onto 36 positions.

**Decision.** Widgets bind by `key=` and are created without `value=`
(`init_state` seeds the defaults). Anything that changes settings — load, preset,
clear, undo, remove — runs in an `on_click` / `on_change` callback, which Streamlit
executes before the script body, so no key is ever written after its widget is
drawn. Zone editors use `zone:<id>:<field>` keys, synced into the zone dicts at the
top of each run and dropped whenever the zone list is replaced. Patches are
clamped and de-duplicated every run, with a notice. Zone ids skip any id already
present. Models are keyed on `run_id`. The canvas is re-keyed after each handled
click, which is the supported way to clear a Plotly selection (the first version
wrote to the widget's own state, which Streamlit forbids, and fell back to a guard
that was never read). A segmented control deselects on a second click, so
re-clicking the active tool used to drop it silently; `on_tool_change` restores it.

**Consequence.** `tests/test_app.py` now drives the real widgets and buttons by key
and reproduces each bug above. The streamlit floor is **1.55**: verified by running
the app tests against 1.53, 1.54 (fail — segmented-control values) and 1.55, 1.56,
1.59 and 1.64 (pass).

---

## D16 — Visual rules for the app

**Context.** The first app used ad-hoc colours and two dual-axis charts (oracle on
a second y-axis; clip fraction on another). Two scales on one plot invite reading
a crossing as meaningful.

**Decision.** One validated categorical palette, assigned by meaning and never
cycled (zones: heat orange, strong aqua, dead blue, anti violet, random red;
patches yellow, which no zone uses). The gain map is diverging blue ↔ red with a
neutral gray at 0 ("does not track c"). One y-axis per chart: LDA and oracle are
both shown scaled so their own affine fit runs 0 → 1, where a perfectly affine
readout lies on the diagonal, and clip fraction gets its own panel. Verdicts use
status colours with an icon and a word, never colour alone. The weight map's colour
limit is the 98th percentile of |w|, not the maximum, so a few extreme pixels do
not wash every other patch out. Settings live in the sidebar so the canvas and the
results get the main area. `scripts/figures.py` follows the same one-axis rule.

**Consequence.** A test asserts no figure uses a secondary y-axis.

---

## F3 — Linearity depends on the regime (replaces F1)

Full battery re-run with the corrected verdict: 11 configs × 5 seeds at the
careful (pre-registered) settings, plus a pixel-model / solver / feature sweep on
`heat_dead` (5 seeds each, no CV). Spread is across seeds.

**Holds** — `affine` in 5/5 seeds, κ < 0.01 — under the Bernoulli pixel model for
the simple world (κ 0.0003), `heat_dead` (0.0068), `heat_dead_anti` (0.0070),
`heat_dead_anti_random` (0.0075), `strong_dead` (0.0007) and dead-zone-only
sampling (0.0020). The saturating heat zones do leave a faint S — κ is almost
entirely cubic, and the residual panel shows it — but it is small: at most ~0.005
in units of c. Notably the LDA reads out *straighter* than the gain-weighted
oracle (κ ≈ 0.049): it learns to down-weight the clipping pixels (mean weight in
heat zones 2.2 vs 10.4 in the background, `heat_dead_anti` seed 0).
*Interpretation:* shrinkage LDA on Bernoulli pixels partly protects itself from
saturation.

**Fails** in four regimes:

| condition | LDA verdict | LDA κ | oracle κ | reading |
|---|---|---|---|---|
| patches only on the saturating zones (`ablation_heat_only`) | nonlinear 5/5 | 0.19 | 0.21 | baked into what the observer sees |
| **Beta pixels** in the `heat_dead` world (all three noise levels) | nonlinear 15/15 | 0.20–0.21 | 0.045 (affine) | **the method bends where a linear readout exists** |
| clipped-Gaussian pixels (all three noise levels) | nonlinear 15/15 | 0.07–0.08 | 0.06–0.09 | mostly the censoring itself |
| geometric jitter (`heat_dead_jitter`) | nonlinear 3/5 | 0.049–0.062 | 0.048 | on the threshold, ~7× the no-jitter κ |

Features and solver also move κ on the same world: `patch_mean` 0.034 vs `pixels`
0.0068; `svd` 0.0007 vs `lsqr` + shrinkage 0.0068. All five remain `affine`, but a
5× swing from a readout choice is itself a caution.

The Beta row is the most important result so far: it is the H6 pattern, where an
oracle built from the true gains is affine and the extremes-trained LDA on the same
trials is not. It needs the saturating zones as well: the uniform world under Beta is affine in
15/15 runs (ν = 8, 25, 100 × 5 seeds, pre-registered trial counts, no CV;
κ 0.0006–0.0008, oracle identical). Why Beta and not Bernoulli bends once zones
saturate is not yet understood. *Speculation:* the
extremes' covariance, which the LDA whitens with, is least representative of the
middle of the range under Beta noise.

## F4 — Precision varies everywhere, and the readout shares the blame (replaces F2)

SD max/min exceeded the 1.5 homoscedasticity threshold in **every** non-null
condition, including the simple uniform world (2.2–2.5). Bernoulli worlds 3.0–53,
Beta 16–24, clipped Gaussian 1.6–2.0. The pixel model matters, but so does the
readout: on the same `heat_dead` world and seeds, `svd` gives 7.2 against 48 for
`lsqr` + shrinkage, and `patch_mean` features 38. F2's claim that the varying
precision is "a property of the observation model, not of the LDA" is withdrawn.

Practical reading, unchanged in spirit: even where the mean readout is straight,
the error bar on a single trial's decision value is not constant across the range,
so statistics that treat decision values as equally precise measurements rest on
an assumption this simulation does not support.
