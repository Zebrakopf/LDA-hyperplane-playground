# LDA hyperplane check

An interactive sandbox for a belief that gets repeated in the lab but rarely tested:

> Train an LDA on the **two extremes** of a latent variable, and its decision value
> varies **linearly** with that variable in between.

You build a world where you decide how the latent variable drives the pixels — heat
zones, dead zones, anti-correlated zones, random zones, jitter — choose where the
model gets to look, press a button, and see whether the readout comes out
straight. Everything is simulated, so the ground truth is exact and the diagnostics
won't let a bent curve pass as a straight one.

## Install and run

```bash
conda env create -f environment.yml
conda activate lda_patch_lab
streamlit run app.py
```

Python 3.11, Streamlit ≥ 1.55. Runs entirely on your machine — nothing is deployed.

## Five-minute tour

1. **Press ▶ Train & evaluate.** The default world is uniform: every pixel tracks
   the latent contrast `c` equally, so you should get **✓ affine** with κ ≈ 0.
   That's the sanity check.
2. **Pick `heat` above the canvas and click on it** — a region that tracks `c`
   twice as steeply as its surround. Slide **preview at contrast c** away from 0.5
   to see it, or switch on **gain map**. Train again.
3. **Add a `dead` zone.** Now part of the plane barely responds. The **What the
   model used** tab shows where the LDA put its weight.
4. **Break it on purpose.** Set **background gain** to 0 in the sidebar — nothing
   tracks `c`. You should get **○ degenerate**, not a plausible line. That control
   is what proves the tool isn't manufacturing structure.
5. **Find a real failure.** Load `scenario_heat_dead` from the sidebar's Configs,
   switch the pixel model to `beta` and train — see below. (Beta on the plain
   uniform world stays straight; it takes the saturating zones as well.)

Every control has a tooltip. The speed preset starts on `fast (explore)`, which
returns in under a second; use `careful` before believing a number.

## Reading the output

| what | means |
|------|-------|
| verdict | ✓ affine · ~ equivocal · ✗ nonlinear · ○ degenerate · ! fit failed — thresholds fixed before any result was seen |
| **κ** | curvature, `√(κ₂² + κ₃²)`: quadratic (one-sided bend) plus cubic (S-shape) part. Below 0.05 affine, above 0.15 a clear bend. **The headline number.** |
| range use | share of the output spent on the middle half of `c`. A straight line is 0.50; an S-curve more |
| `r` | Pearson. Never read alone — an S-curve sits happily at r = 0.99 |
| SD max/min | how much the *precision* varies across the range — a separate failure from a bent curve |
| oracle | a readout built from the true gains — separates "the LDA failed" from "no linear readout existed" |

## What we found

11 configs × 5 seeds at the pre-registered settings, plus a pixel-model, solver
and feature sweep. Details in [`details.md`](details.md) and
[`docs/decisions.md`](docs/decisions.md) (F3, F4).

**Linearity holds in the simple regime and fails in several realistic ones.** With
Bernoulli pixels the readout stays affine (κ < 0.01) even with saturating, dead
and anti-correlated zones. The LDA is actually straighter than the ideal
gain-weighted readout, because it learns to down-weight pixels that clip. It
**bends** when:

| condition | LDA κ | oracle κ | meaning |
|---|---|---|---|
| Beta pixel noise + saturating zones | **0.20** (5/5 seeds) | 0.045 (affine) | the method bends where a straight readout exists |
| clipped-Gaussian noise | 0.07–0.08 | 0.06–0.09 | mostly the censoring itself |
| patches only on saturating zones | 0.19 | 0.21 | baked into what the observer sees |
| zones that jitter between trials | 0.05–0.06 | 0.048 | on the threshold, ~7× the no-jitter value |

**Precision varies everywhere.** SD max/min exceeded the threshold in every
non-null condition, including the uniform world, and the readout contributes:
switching the solver alone moves it from 48 to 7. Statistics that treat decision
values as equally precise measurements rest on an assumption this simulation does
not support.

> An earlier version of this README said linearity held everywhere. That was
> measured with a curvature index that could not see S-shaped curves — the shape
> saturation produces. See D14.

## Without the app

```bash
pytest                                                        # 103 tests, ~30 s
python -m scripts.run_experiment configs/scenario_heat_dead.json --figures
python -m scripts.run_matrix --factors pixel_noise --seeds 0 1 2 3 4 --no-cv
```

Results land in `results/`, models in `models/` — both git-ignored and
reproducible from a config plus a seed. Configs saved from the app go to
`configs/user/`.

## More

- **[`details.md`](details.md)** — the world model, zone kinds, pixel models,
  gotchas, full CLI, findings, code layout.
- **[`plan.md`](plan.md)** — experiment design, hypotheses, architecture.
- **[`CLAUDE.md`](CLAUDE.md)** — working agreement: invariants, commenting rules,
  how to report results without overclaiming.
- **[`docs/decisions.md`](docs/decisions.md)** — decision log, including what was
  got wrong first.
