"""The observer: which pixels are sampled, and as what features.

Responsibility
--------------
Build a patch layout from a `SamplingConfig`, extract a feature vector from a
field, and report what the observer's patches actually cover.

Explicitly NOT this module's job
--------------------------------
Anything about the world. This module reads `SamplingConfig`; it reads a
`GainField` only for the composition report, and never to decide where patches
go (invariant I1 — patch placement must not depend on zone placement).

Pipeline position
-----------------
    field  ->  [this module]  ->  x  ->  dataset  ->  X

Feature ordering is a contract
------------------------------
`patch_pixel_indices` must return pixels in exactly the order
`extract_features` flattens them, because weight attribution (plan.md §7.4)
inverts that mapping. Both are driven by the same nested loop over patches.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.random import Generator

from core.config import PlaneSpec, SamplingConfig, ZoneKind
from core.data import GainField

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Patch:
    """One square sampling window, addressed by its top-left corner."""

    row: int
    col: int
    size: int


def _grid_origins(plane: PlaneSpec, patch_size: int, step: int
                  ) -> list[tuple[int, int]]:
    """Top-left corners of a regular grid that stays inside the plane."""
    rows = range(0, plane.height - patch_size + 1, step)
    cols = range(0, plane.width - patch_size + 1, step)
    return [(r, c) for r in rows for c in cols]


def build_patches(cfg: SamplingConfig, plane: PlaneSpec,
                  rng_patches: Generator) -> list[Patch]:
    """Place the observer's sampling windows.

    Drawn ONCE per model from the `patches` stream, then stored on the trained
    model: the observer has a fixed receptive-field layout across trials. Whether
    it should instead wander is plan.md §12, question 1.
    """
    size = cfg.patch_size
    max_row, max_col = plane.height - size, plane.width - size
    if max_row < 0 or max_col < 0:
        raise ValueError(f"patch_size {size} does not fit in plane {plane}")

    if cfg.layout == "uniform_grid":
        origins = _grid_origins(plane, size, cfg.grid_step)
    elif cfg.layout == "jittered_grid":
        origins = []
        for row, col in _grid_origins(plane, size, cfg.grid_step):
            offset = rng_patches.normal(0.0, cfg.layout_jitter_px, size=2)
            origins.append((int(round(row + offset[0])), int(round(col + offset[1]))))
    elif cfg.layout == "random_uniform":
        assert cfg.n_patches is not None      # guaranteed by the config validator
        rows = rng_patches.integers(0, max_row + 1, size=cfg.n_patches)
        cols = rng_patches.integers(0, max_col + 1, size=cfg.n_patches)
        origins = list(zip(rows.tolist(), cols.tolist(), strict=True))
    elif cfg.layout == "manual":
        origins = [(int(r), int(c)) for r, c in cfg.patches]
    else:
        raise ValueError(f"unhandled layout {cfg.layout!r}")

    if cfg.snap_to_grid:
        snap = cfg.snap_to_grid
        origins = [(int(round(r / snap) * snap), int(round(c / snap) * snap))
                   for r, c in origins]

    # Clamp rather than reject: jitter and snapping can push a patch off the
    # edge, and silently dropping patches would change n_features between trials.
    return [Patch(row=int(np.clip(r, 0, max_row)),
                  col=int(np.clip(c, 0, max_col)),
                  size=size)
            for r, c in origins]


def patch_pixel_indices(patches: Sequence[Patch],
                        plane: PlaneSpec) -> npt.NDArray[np.int64]:
    """Flat indices into the (H*W) plane, in `pixels` feature order.

    Returns
    -------
    (n_patches * patch_size**2,) int array.

    REQUIRED for weight attribution (plan.md §7.4) — build once, reuse. Repeated
    indices are expected and meaningful: overlapping patches genuinely duplicate
    pixels in the feature vector.
    """
    blocks = []
    for patch in patches:
        rows = np.arange(patch.row, patch.row + patch.size)
        cols = np.arange(patch.col, patch.col + patch.size)
        blocks.append((rows[:, None] * plane.width + cols[None, :]).ravel())
    if not blocks:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(blocks).astype(np.int64)


def n_features(cfg: SamplingConfig, n_patches: int) -> int:
    """Length of the feature vector this sampling config produces."""
    if cfg.feature_mode == "pixels":
        return n_patches * cfg.patch_size ** 2
    if cfg.feature_mode == "patch_mean":
        return n_patches
    if cfg.feature_mode == "patch_mean_std":
        return 2 * n_patches
    raise ValueError(f"unhandled feature_mode {cfg.feature_mode!r}")


def unique_pixels_sampled(patches: Sequence[Patch], plane: PlaneSpec) -> int:
    """How many distinct pixels the layout covers.

    Reported next to `n_features` because overlapping patches inflate apparent
    dimensionality without adding information (plan.md §5.1).
    """
    return int(np.unique(patch_pixel_indices(patches, plane)).size)


def extract_features(field: FloatArray, patches: Sequence[Patch],
                     feature_mode: str) -> FloatArray:
    """Flatten the sampled windows into one feature vector.

    Returns
    -------
    (n_features,) float array.

    Ordering contract: for `patch_mean_std` the vector is ALL means followed by
    ALL standard deviations, not interleaved. The oracle readout relies on this
    (plan.md §7.3).
    """
    blocks = [field[p.row:p.row + p.size, p.col:p.col + p.size] for p in patches]
    if feature_mode == "pixels":
        return np.concatenate([b.ravel() for b in blocks])
    means = np.array([b.mean() for b in blocks], dtype=np.float64)
    if feature_mode == "patch_mean":
        return means
    if feature_mode == "patch_mean_std":
        stds = np.array([b.std() for b in blocks], dtype=np.float64)
        return np.concatenate([means, stds])
    raise ValueError(f"unhandled feature_mode {feature_mode!r}")


def sampled_area_composition(patches: Sequence[Patch], gf: GainField,
                             plane: PlaneSpec) -> pd.DataFrame:
    """What the observer actually sees, broken down by zone kind.

    Uses the per-zone weight maps rather than a dominant-zone label, so `blend`
    overlaps are reported honestly: a pixel half-inside a dead zone contributes
    0.5 to `dead` and 0.5 to `background`.

    Returns
    -------
    DataFrame with one row per zone kind present (plus `background`), columns:
    `kind`, `weight_fraction`, `mean_gain`, `sd_gain`.

    Printed next to every f(c) curve, never buried in metadata: patch placement
    can single-handedly decide the result (CLAUDE.md §11, trap 6).
    """
    indices = np.unique(patch_pixel_indices(patches, plane))
    if indices.size == 0:
        return pd.DataFrame(columns=["kind", "weight_fraction", "mean_gain", "sd_gain"])

    masked_sampled = gf.random_mask.ravel()[indices]
    # Effective gain: a masked pixel ignores c, so it carries gain 0 here rather
    # than the composed `a` underneath the mask (which reads ~1 and made the
    # all-random control report a mean sampled gain of 1.0).
    gain_sampled = np.where(masked_sampled, 0.0, gf.a.ravel()[indices])
    rows: list[dict[str, object]] = []

    def _gain_stats(inside: npt.NDArray[np.bool_]) -> tuple[float, float]:
        if not inside.any():
            return float("nan"), float("nan")
        return float(gain_sampled[inside].mean()), float(gain_sampled[inside].std())

    if gf.zone_weights.size:
        flat_weights = gf.zone_weights.reshape(len(gf.zone_ids), -1)[:, indices]
        coupling = np.array([k is not ZoneKind.RANDOM for k in gf.zone_kinds])

        # Shares are normalised by the SAME denominator the blend composition
        # uses, so the coupling kinds and the background partition the sampled
        # area exactly. Without this, two overlapping zones each claim full
        # credit for the shared pixels and the "fractions" sum above 1 —
        # double counting that reads as a coverage error in the report.
        coupling_sum = (flat_weights[coupling].sum(axis=0) if coupling.any()
                        else np.zeros(indices.size))
        background_weight = np.maximum(0.0, 1.0 - coupling_sum)
        denominator = background_weight + coupling_sum
        denominator[denominator == 0.0] = 1.0     # fully masked pixels: see below

        for kind in dict.fromkeys(gf.zone_kinds):
            selector = np.array([k is kind for k in gf.zone_kinds])
            weight = flat_weights[selector].sum(axis=0)
            if kind is ZoneKind.RANDOM:
                # A masked pixel leaves the coupling partition entirely, so its
                # share is reported as a plain fraction of sampled pixels and is
                # deliberately NOT part of the sum-to-one group.
                masked = gf.random_mask.ravel()[indices]
                mean_gain, sd_gain = _gain_stats(masked)
                rows.append({"kind": str(kind),
                             "weight_fraction": float(masked.mean()),
                             "mean_gain": mean_gain, "sd_gain": sd_gain})
                continue
            share = np.where(masked_sampled, 0.0, weight / denominator)
            mean_gain, sd_gain = _gain_stats(share >= 0.5)
            rows.append({"kind": str(kind), "weight_fraction": float(share.mean()),
                         "mean_gain": mean_gain, "sd_gain": sd_gain})
        background_share = background_weight / denominator
    else:
        background_share = np.ones(indices.size)
    # Masked pixels belong to the `random` row only; counting them as background
    # too made the all-random control's fractions add up to 2.0.
    background_share = np.where(masked_sampled, 0.0, background_share)

    background_mean, background_sd = _gain_stats(background_share >= 0.5)
    rows.append({"kind": "background",
                 "weight_fraction": float(background_share.mean()),
                 "mean_gain": background_mean, "sd_gain": background_sd})
    rows.append({
        "kind": "ALL_SAMPLED",
        "weight_fraction": 1.0,
        "mean_gain": float(gain_sampled.mean()),
        "sd_gain": float(gain_sampled.std()),
    })
    return pd.DataFrame(rows)
