# LDA hyperplane check

An interactive sandbox for to explore the following assumption:

> Train an LDA on the **two extremes** of a latent variable, and its decision value
> varies **linearly** with that variable in between.

You build a world where you decide how the latent variable drives the pixels — heat
zones, dead zones, anti-correlated zones, random zones, jitter — choose where the
model gets to look, press a button, and see whether the readout actually comes out
straight. Everything is simulated, so the ground truth is exact and the diagnostics
won't let a bent curve pass as a straight one.

## Install and run

```bash
conda env create -f environment.yml
conda activate lda_patch_lab
streamlit run app.py
```

Python 3.11. Runs entirely on your machine — nothing is deployed.

## Five-minute tour

1. **Press ▶ Train & evaluate.** The default world is uniform: every pixel tracks
   the latent contrast `c` equally, so you should get `verdict: affine` with κ ≈ 0.
   That's the sanity check.
2. **Click the canvas** with `zone: heat` selected — a region that tracks `c` twice
   as steeply as its surround. Slide **preview contrast** away from 0.5 to see it.
   Train again.
3. **Add a `zone: dead`.** Now part of the plane barely responds. Check the **what
   the model used** tab: weight should land on the heat zone, not the dead one.
4. **Break it on purpose.** Set **background gain** to 0 — nothing tracks `c`. You
   should get `degenerate`, not a plausible line. That control is what proves the
   tool isn't manufacturing structure.
5. **Compare worlds.** Train, switch the pixel model from `bernoulli` to
   `clipped_gaussian`, train again, open **compare models**.

Every control has a tooltip. The **speed preset** starts on `fast (explore)`, which
returns in under a second; switch to `careful` before believing a number.

## Reading the output

| what | means |
|------|-------|
| `verdict` | `affine` / `equivocal` / `nonlinear` / `degenerate`, by thresholds fixed before any result was seen |
| **κ** | curvature index, `\|β₂\|/\|β₁\|`. Below 0.05 affine, above 0.15 a clear bend. **The headline number.** |
| `r` | Pearson. Never read alone — an S-shaped curve sits happily at r = 0.99 |
| SD max/min | how much the *precision* varies across the range (a separate failure from a bent curve) |
| `degenerate` | the slope isn't resolved above noise. The **correct** answer when nothing tracks `c` |
| oracle | a readout built from the true gain field — separates "the LDA failed" from "no linear readout existed" |

## Ready-made worlds

`configs/` holds eleven configs, loadable from the right-hand panel. The controls
matter most — they're what makes the tool trustworthy:

- `control_simple` → must come out `affine`
- `control_all_dead`, `control_all_random`, `control_label_shuffle` → must come out
  `degenerate`
- `scenario_heat_dead*` → heterogeneous worlds, with variants adding anti-correlated
  regions, random regions and jitter
- `ablation_heat_only` / `ablation_dead_only` → the same world, sampled only where
  the signal is / isn't

## Without the app

```bash
pytest                                                        # 62 tests, ~8 s
python -m scripts.run_experiment configs/scenario_heat_dead.json --figures
python -m scripts.run_matrix --factors pixel_noise --seeds 0 --no-cv
```

Results land in `results/`, models in `models/` — both git-ignored and reproducible
from a config plus a seed.

## What the first pass found

The linearity assumption **held** everywhere tested: κ between 0.0000 and 0.0021,
including with saturating zones, anti-correlated regions, jitter and all three pixel
models. The *precision* assumption did not — SD max/min ranged 1.67 to 46.98, driven
by the observation model rather than the LDA.

So the mean decision value tracks the latent variable faithfully, but the error bar
on a single trial can differ by more than an order of magnitude across the range.
If anyone is doing statistics on decision values as equally precise measurements,
that's the assumption worth worrying about — not the linearity one it usually
travels with. One seed so far; see `details.md`.

## More

- **[`details.md`](details.md)** — the generative model, zone kinds, pixel models,
  gotchas, full CLI, findings, code layout.
- **[`plan.md`](plan.md)** — experiment design, hypotheses, architecture.
- **[`CLAUDE.md`](CLAUDE.md)** — working agreement: invariants, commenting rules,
  how to report results without overclaiming.
- **[`docs/decisions.md`](docs/decisions.md)** — decision log, including what was
  got wrong first.
