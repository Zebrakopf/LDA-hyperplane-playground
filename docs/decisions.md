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
(gain 1.0, background 0.4, never saturates) have the same 2.5x zone-to-background
ratio, isolating saturation. It also makes the all-dead control expressible
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
`core/evaluation.py`). Over the same 200 null runs `|t|` never exceeded 0.13,
while a real mapping reaches `|t|` in the hundreds. A perfectly constant `f` is
handled explicitly as `t = 0`, not `0/0`.

**Consequence.** The null controls are reliably degenerate. At the default 4200
evaluation trials the threshold corresponds to `|r| ~ 0.155`, so a mapping whose
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

## F1 — First finding: H2 (affinity) survives every condition tested

Across the four controls, five scenarios, two sampling ablations and seven
pixel-model x noise cells at seed 0: every non-degenerate run returned
`verdict = affine` with `kappa` between 0.0000 and 0.0021, far below the
pre-registered 0.05. This holds with saturating `heat` zones active (mean clip
fraction 0.042), with anti-correlated and random zones present, under geometric
jitter, and under all three pixel models. Single seed, so between-seed spread is
not yet characterised — that is the remaining Phase-1 exit criterion.

## F2 — First finding: H3 (constant precision) fails, structurally

`sd_ratio` ranged from 1.67 to 46.98 and only the clipped-Gaussian cells came
close to homoscedastic. The pixel-model sweep locates the cause: Bernoulli
(47) and Beta (16-24) both have mean-dependent variance, `theta*(1-theta)`, while
clipped Gaussian (1.7-1.9) has nearly constant variance. So the varying precision
is a property of the observation model, not of extremes-trained LDA. Practical
reading: the *mean* of `f` tracks `c` affinely, but the *error bar on a single
trial's* `f` can differ by more than an order of magnitude across the contrast
range — which matters for anyone running statistics on decision values as if they
were equally precise measurements.
