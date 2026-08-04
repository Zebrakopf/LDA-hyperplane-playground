"""Shape, dtype and analytic-sanity tests for field generation.

Test classes 1 and 3 of CLAUDE.md §8: cheap structural checks, plus the cases
where the right answer is known independently of the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import DataConfig, Falloff, NoiseSpec, PlaneSpec, ZoneKind
from core.data import (
    build_gain_field,
    coupling_to_theta,
    generate_field,
    jitter_free_gain_field,
    sample_field,
    zone_weight_map,
)
from tests.conftest import SMALL_PLANE, zone


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #

def test_gain_field_shapes_and_dtypes(uniform_data: DataConfig) -> None:
    gf = build_gain_field(uniform_data, rng())
    shape = (SMALL_PLANE.height, SMALL_PLANE.width)
    assert gf.a.shape == shape and gf.a.dtype == np.float64
    assert gf.b.shape == shape
    assert gf.random_mask.shape == shape and gf.random_mask.dtype == np.bool_
    assert gf.zone_weights.shape == (0, *shape)


def test_field_values_stay_in_unit_interval(uniform_data: DataConfig) -> None:
    for pixel_model in ("bernoulli", "clipped_gaussian", "beta"):
        cfg = uniform_data.model_copy(
            update={"noise": NoiseSpec(pixel_model=pixel_model)})
        for c in (0.0, 0.5, 1.0):
            field, _, _ = generate_field(cfg, c, rng(1), rng(2))
            assert field.min() >= 0.0 and field.max() <= 1.0, pixel_model


# --------------------------------------------------------------------------- #
# Analytic sanity — the answer is known without running the code
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("c", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_uniform_plane_mean_equals_contrast(uniform_data: DataConfig, c: float) -> None:
    """With no zones, E[pixel] == c exactly.

    Tolerance: 400 pixels x 40 trials = 16000 Bernoulli draws. The worst-case SE
    is 0.5/sqrt(16000) = 0.004, so 4 SE = 0.016 is a ~1-in-16000 false-failure
    rate per parameter value.
    """
    generator = rng(3)
    means = [generate_field(uniform_data, c, generator, generator)[0].mean()
             for _ in range(40)]
    assert abs(float(np.mean(means)) - c) < 0.016


def test_anti_zone_pixels_decrease_with_contrast(uniform_data: DataConfig) -> None:
    """A pure anti-correlated zone must run downhill in c while the background
    runs uphill. Signs are exact predictions of the coupling model, not estimates.
    """
    cfg = uniform_data.model_copy(
        update={"zones": [zone(ZoneKind.ANTI, gain=-1.0)]})
    gf = jitter_free_gain_field(cfg)
    inside = gf.zone_weights[0] >= 0.5
    generator = rng(4)
    low = np.mean([generate_field(cfg, 0.1, generator, generator)[0][inside].mean()
                   for _ in range(30)])
    high = np.mean([generate_field(cfg, 0.9, generator, generator)[0][inside].mean()
                    for _ in range(30)])
    assert high < low
    outside_low = np.mean([generate_field(cfg, 0.1, generator, generator)[0][~inside].mean()
                           for _ in range(30)])
    outside_high = np.mean([generate_field(cfg, 0.9, generator, generator)[0][~inside].mean()
                            for _ in range(30)])
    assert outside_high > outside_low


def test_dead_zone_pixels_are_uncorrelated_with_contrast(
        uniform_data: DataConfig) -> None:
    """Gain exactly 0 means the pixel's expectation is `pivot` for every c."""
    cfg = uniform_data.model_copy(update={"zones": [zone(ZoneKind.DEAD, gain=0.0)]})
    gf = jitter_free_gain_field(cfg)
    inside = gf.zone_weights[0] >= 0.5
    generator = rng(5)
    contrasts = np.linspace(0.0, 1.0, 11)
    means = [np.mean([generate_field(cfg, float(c), generator, generator)[0][inside].mean()
                      for _ in range(20)]) for c in contrasts]
    # Tolerance: ~113 pixels x 20 trials = 2260 draws per level, SE ~ 0.011.
    # A correlation of |r| < 0.35 across 11 noisy level means is consistent with
    # zero; the substantive check is that the SLOPE is ~0, tested next.
    slope = np.polyfit(contrasts, means, 1)[0]
    assert abs(slope) < 0.05
    assert abs(float(np.mean(means)) - cfg.pivot) < 0.03


def test_random_zone_is_contrast_independent(uniform_data: DataConfig) -> None:
    """A random zone masks the pixel: its mean must not depend on c at all."""
    cfg = uniform_data.model_copy(update={"zones": [zone(ZoneKind.RANDOM, gain=1.0)]})
    gf = jitter_free_gain_field(cfg)
    masked = gf.random_mask
    assert masked.any()
    generator = rng(6)
    low = np.mean([generate_field(cfg, 0.0, generator, generator)[0][masked].mean()
                   for _ in range(40)])
    high = np.mean([generate_field(cfg, 1.0, generator, generator)[0][masked].mean()
                    for _ in range(40)])
    # bernoulli_half draws: mean 0.5 either way. SE over ~113*40 draws is 0.0074,
    # so 4 SE = 0.03 for each and 0.04 for the difference.
    assert abs(low - 0.5) < 0.03 and abs(high - 0.5) < 0.03
    assert abs(low - high) < 0.04


def test_beta_model_mean_matches_theta(uniform_data: DataConfig) -> None:
    """Beta(theta*nu, (1-theta)*nu) has mean theta — checked at c = 0.3."""
    cfg = uniform_data.model_copy(
        update={"noise": NoiseSpec(pixel_model="beta", beta_concentration=20.0)})
    generator = rng(7)
    field, _, _ = generate_field(cfg, 0.3, generator, generator)
    # 400 pixels, Beta variance = 0.3*0.7/21 = 0.01 => SD 0.1, SE 0.005. 4 SE = 0.02.
    assert abs(field.mean() - 0.3) < 0.02


def test_beta_model_survives_grid_endpoints(uniform_data: DataConfig) -> None:
    """theta in {0, 1} is outside Beta's open support; the eps clamp must handle it.

    This is the defect the spec review caught: the default grid includes c = 0
    and c = 1, where an unclamped Beta draw raises ValueError.
    """
    cfg = uniform_data.model_copy(update={"noise": NoiseSpec(pixel_model="beta")})
    for c in (0.0, 1.0):
        field, _, _ = generate_field(cfg, c, rng(8), rng(9))
        assert np.isfinite(field).all()


# --------------------------------------------------------------------------- #
# Zone geometry and composition
# --------------------------------------------------------------------------- #

def test_falloff_profiles_agree_in_the_hard_limit() -> None:
    """All three edge profiles collapse to the hard edge as width -> 0."""
    base = zone(ZoneKind.HEAT, gain=2.0, falloff=Falloff.HARD)
    hard = zone_weight_map(base, SMALL_PLANE)
    for falloff in (Falloff.SMOOTHSTEP, Falloff.GAUSSIAN):
        variant = base.model_copy(update={"falloff": falloff, "falloff_width": 0.0})
        assert np.array_equal(zone_weight_map(variant, SMALL_PLANE), hard)


def test_blend_composition_is_bounded_by_background_and_zone_gains(
        uniform_data: DataConfig) -> None:
    """The convex-hull bound from plan.md §4.4.

    Note this is NOT "never larger than the largest zone gain": a dead zone at
    edge weight 0.5 is pulled UP by the background. The bound must include
    `background_gain` as a contributor — the defect the spec review caught.
    """
    cfg = uniform_data.model_copy(update={
        "background_gain": 1.0,
        "zones": [
            zone(ZoneKind.DEAD, gain=0.0, radius=6.0, center=(8.0, 8.0),
                 falloff=Falloff.SMOOTHSTEP, zone_id="dead1"),
            zone(ZoneKind.HEAT, gain=3.0, radius=5.0, center=(12.0, 12.0),
                 falloff=Falloff.SMOOTHSTEP, zone_id="heat1"),
        ],
    })
    # Soft edges are needed for a partial-weight pixel to exist at all.
    cfg.zones[0].falloff_width = 3.0
    cfg.zones[1].falloff_width = 3.0
    gf = build_gain_field(cfg, rng())
    gains = [z.gain for z in cfg.zones] + [cfg.background_gain]
    assert gf.a.min() >= min(gains) - 1e-12
    assert gf.a.max() <= max(gains) + 1e-12
    # And the pull-up really happens: some pixel inside the dead zone's skirt
    # ends up with a gain strictly between 0 and 1.
    assert ((gf.a > 0.05) & (gf.a < 0.95)).any()


def test_replace_mode_gives_the_pixel_to_the_highest_priority_zone(
        uniform_data: DataConfig) -> None:
    low = zone(ZoneKind.DEAD, gain=0.0, radius=8.0, zone_id="low")
    high = zone(ZoneKind.HEAT, gain=2.0, radius=8.0, zone_id="high")
    high.priority = 5
    cfg = uniform_data.model_copy(
        update={"overlap_mode": "replace", "zones": [low, high]})
    gf = build_gain_field(cfg, rng())
    overlap = (gf.zone_weights[0] >= 0.5) & (gf.zone_weights[1] >= 0.5)
    assert overlap.any()
    assert np.allclose(gf.a[overlap], 2.0)


# --------------------------------------------------------------------------- #
# The clipping invariant (CLAUDE.md §8, test class 4)
# --------------------------------------------------------------------------- #

def test_no_clipping_when_the_bound_is_satisfied(uniform_data: DataConfig) -> None:
    """pivot 0.5, |gain| <= 1, |background_gain| <= 1, offset 0, gain_sigma 0.

    Under those preconditions clipping is mathematically unreachable, so
    clip_fraction must be exactly 0 at every contrast — including the endpoints,
    where the bound is attained with equality.
    """
    cfg = uniform_data.model_copy(update={
        "background_gain": 0.4,
        "zones": [zone(ZoneKind.STRONG, gain=1.0), zone(ZoneKind.ANTI, gain=-1.0,
                                                        center=(4.0, 4.0), radius=3.0,
                                                        zone_id="anti1")],
    })
    assert cfg.clipping_is_reachable() is False
    gf = build_gain_field(cfg, rng())
    for c in np.linspace(0.0, 1.0, 21):
        _theta, clip_fraction = coupling_to_theta(float(c), gf, cfg)
        assert clip_fraction == 0.0


def test_clipping_fires_when_gain_exceeds_one(uniform_data: DataConfig) -> None:
    cfg = uniform_data.model_copy(update={"zones": [zone(ZoneKind.HEAT, gain=2.0)]})
    assert cfg.clipping_is_reachable() is True
    gf = build_gain_field(cfg, rng())
    _theta, at_end = coupling_to_theta(0.0, gf, cfg)
    _theta, at_middle = coupling_to_theta(0.5, gf, cfg)
    assert at_end > 0.0          # theta = 0.5 + 2*(0 - 0.5) = -0.5 inside the zone
    assert at_middle == 0.0      # every gain agrees at the pivot


def test_pivot_other_than_half_is_a_clipping_path(uniform_data: DataConfig) -> None:
    """The fourth clipping path, which the first draft of the spec missed."""
    cfg = uniform_data.model_copy(update={
        "pivot": 0.2, "zones": [zone(ZoneKind.ANTI, gain=-1.0)]})
    assert cfg.clipping_is_reachable() is True
    gf = build_gain_field(cfg, rng())
    _theta, clip_fraction = coupling_to_theta(1.0, gf, cfg)
    assert clip_fraction > 0.0


def test_cached_gain_field_is_rejected_when_jitter_is_enabled(
        uniform_data: DataConfig, jittered) -> None:
    """Reusing a gain field under jitter would silently freeze the nuisance."""
    cfg = uniform_data.model_copy(update={
        "jitter": jittered, "zones": [zone(ZoneKind.HEAT, gain=2.0)]})
    gf = build_gain_field(cfg, rng())
    with pytest.raises(ValueError, match="cached gain field"):
        generate_field(cfg, 0.5, rng(), rng(), gain_field=gf)
