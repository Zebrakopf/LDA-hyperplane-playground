"""Tests for the defects found in the September review (docs/decisions.md D14).

Each test here pins a failure that the original suite let through: the suite
passed with every one of these bugs present. They are grouped by the diagnostic
they protect, and each states the specific wrong answer it guards against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from core.config import (
    DataConfig,
    EvalConfig,
    Falloff,
    JitterSpec,
    RunConfig,
    SamplingConfig,
    TrainingConfig,
    ZoneKind,
    ZoneSpec,
)
from core.data import jitter_free_gain_field, zone_weight_map
from core.dataset import _balanced_shuffle
from core.evaluation import linearity_report, run_experiment
from core.sampling import build_patches, sampled_area_composition
from tests.conftest import SMALL_PLANE, zone

GRID_21 = np.linspace(0.0, 1.0, 21)
GRID_11 = np.linspace(0.0, 1.0, 11)


def _report(f_of_c, grid=GRID_21, per_level=200, noise=0.02, seed=0):
    """Linearity report for a known curve with small additive noise."""
    rng = np.random.default_rng(seed)
    c = np.repeat(grid, per_level)
    f = f_of_c(c) + rng.normal(0.0, noise, c.size)
    return linearity_report(pd.DataFrame({"c": c, "f": f, "clip_fraction": 0.0}))


# --------------------------------------------------------------------------- #
# The curvature index must see S-shapes (the headline defect)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,curve", [
    ("logistic k=12", lambda c: 1.0 / (1.0 + np.exp(-12.0 * (c - 0.5)))),
    ("logistic k=6", lambda c: 1.0 / (1.0 + np.exp(-6.0 * (c - 0.5)))),
    ("symmetric hard clip", lambda c: np.clip(0.5 + 2.0 * (c - 0.5), 0.0, 1.0)),
])
def test_s_shaped_curves_are_nonlinear(name: str, curve) -> None:
    """Saturation around the pivot is odd-symmetric: beta2 ~ 0, beta3 large.

    The original |beta2|/|beta1| index scored these ~0.001 and called them
    `affine`. That is precisely the case clipping produces, so the original
    "linearity held even with saturating zones" finding could not have failed.
    """
    report = _report(curve)
    assert report.verdict == "nonlinear", (name, report.curvature_index)
    assert report.kappa_quadratic < 0.05          # the old index really was blind
    assert report.kappa_cubic > report.kappa_quadratic


def test_one_sided_bend_still_detected_by_the_quadratic_part() -> None:
    report = _report(lambda c: c + 2.0 * c ** 2)
    assert report.verdict == "nonlinear"
    assert report.kappa_quadratic > report.kappa_cubic


def test_straight_line_is_affine_on_both_grids() -> None:
    for grid in (GRID_11, GRID_21):
        report = _report(lambda c: 3.0 * c - 1.0, grid=grid)
        assert report.verdict == "affine"
        assert report.curvature_index < 0.01


def test_kappa_reduces_to_the_quadratic_index_without_a_cubic_part() -> None:
    """Continuity with the pre-registered rule for purely one-sided bends."""
    report = _report(lambda c: c + 0.5 * c ** 2, noise=0.0)
    assert report.kappa_cubic < 1e-9
    assert report.curvature_index == pytest.approx(report.kappa_quadratic)


# --------------------------------------------------------------------------- #
# Effective range use must not depend on the grid
# --------------------------------------------------------------------------- #

def test_effective_range_use_is_one_half_for_a_line_on_any_grid() -> None:
    """The original read 0.40 on 11 levels and 0.50 on 21 for the same line."""
    for grid in (GRID_11, GRID_21, np.linspace(0.0, 1.0, 7)):
        report = _report(lambda c: c, grid=grid, noise=0.0, per_level=3)
        assert report.effective_range_use == pytest.approx(0.5, abs=1e-9)


def test_effective_range_use_exceeds_one_half_for_an_s_curve() -> None:
    report = _report(lambda c: 1.0 / (1.0 + np.exp(-12.0 * (c - 0.5))), noise=0.0,
                     per_level=3)
    assert report.effective_range_use > 0.8


# --------------------------------------------------------------------------- #
# H3: the spread ratio was entirely untested
# --------------------------------------------------------------------------- #

def _spread_report(sd_of_c, grid=GRID_21, per_level=400, seed=1):
    rng = np.random.default_rng(seed)
    c = np.repeat(grid, per_level)
    f = c + rng.normal(0.0, 1.0, c.size) * sd_of_c(c)
    return linearity_report(pd.DataFrame({"c": c, "f": f, "clip_fraction": 0.0}))


def test_constant_spread_is_homoscedastic() -> None:
    """With 400 trials per level the sample-SD ratio across 21 levels sits
    near 1 + ~4 x (1/sqrt(2*400)) ~ 1.15, well inside the 1.5 threshold."""
    report = _spread_report(lambda c: np.full_like(c, 0.05))
    assert report.homoscedastic
    assert report.sd_ratio < 1.3


def test_known_spread_ratio_is_recovered() -> None:
    """SD rising linearly from 0.02 to 0.10 is a true ratio of 5."""
    report = _spread_report(lambda c: 0.02 + 0.08 * c)
    assert not report.homoscedastic
    assert report.sd_ratio == pytest.approx(5.0, rel=0.15)


def test_near_zero_spread_levels_are_excluded_not_divided_by() -> None:
    """A deterministic level comes out of pandas as ~1e-12, not 0.

    The original exact `> 0` test kept it and the smoke test printed
    SD max/min = 71,557,154,339,403.
    """
    rng = np.random.default_rng(2)
    c = np.repeat(GRID_11, 50)
    f = c + rng.normal(0.0, 0.05, c.size)
    endpoint = c == 1.0
    f[endpoint] = 1.0 + 1e-13 * rng.normal(size=endpoint.sum())
    report = linearity_report(pd.DataFrame({"c": c, "f": f, "clip_fraction": 0.0}))
    assert report.n_zero_variance_levels == 1
    assert report.sd_ratio < 2.0


# --------------------------------------------------------------------------- #
# A fit with nothing to learn must not read as a null result
# --------------------------------------------------------------------------- #

def _tiny_run(data: DataConfig, **training) -> RunConfig:
    return RunConfig(
        data=data,
        sampling=SamplingConfig(name="g", layout="uniform_grid", patch_size=5,
                                grid_step=5, feature_mode="pixels"),
        training=TrainingConfig(n_per_class=40, cv_folds=0, **training),
        evaluation=EvalConfig(contrast_grid=GRID_11.tolist(), n_per_contrast=30),
        master_seed=5,
    )


def test_fully_saturated_training_is_fit_failed_not_degenerate() -> None:
    """background gain 2 clips every pixel at both extremes: zero variance.

    Originally: zero weights, verdict `degenerate` (same as "nothing tracks c"),
    d' reported as inf, no warning — while the oracle on the same data showed a
    strong slope.
    """
    data = DataConfig(name="saturated", plane=SMALL_PLANE, background_gain=2.0)
    with pytest.warns(UserWarning, match="LDA fit failed"):
        result = run_experiment(_tiny_run(data))
    assert result.model.diagnostics.fit_failed
    assert result.report.verdict == "fit_failed"
    assert np.isnan(result.model.diagnostics.d_prime)
    assert result.oracle_report.verdict != "degenerate"


def test_fresh_d_prime_exposes_in_sample_overfitting() -> None:
    """In-sample d' is optimistic when features outnumber trials; fresh is not."""
    data = DataConfig(name="all_random", plane=SMALL_PLANE,
                      zones=[zone(ZoneKind.RANDOM, gain=1.0, radius=200.0)])
    result = run_experiment(_tiny_run(data))
    assert result.fresh_d_prime < 1.0
    assert result.model.diagnostics.d_prime > result.fresh_d_prime


# --------------------------------------------------------------------------- #
# Label-shuffle control
# --------------------------------------------------------------------------- #

def test_balanced_shuffle_makes_labels_independent_of_the_true_class() -> None:
    y = np.repeat([0, 1], 150)
    shuffled = _balanced_shuffle(y, np.random.default_rng(0))
    assert shuffled.sum() == y.sum()
    for true_class in (0, 1):
        members = shuffled[y == true_class]
        # exactly half of each true class relabelled 1 (the rest 0)
        assert members.sum() == members.size - members.size // 2


def test_label_shuffle_stays_degenerate_across_many_seeds() -> None:
    """With a plain permutation, fast-preset seeds reached |t| = 9.3 of 10."""
    from tests.conftest import SMALL_PLANE as plane
    data = DataConfig(name="shuffle", plane=plane)
    worst = 0.0
    for seed in range(12):
        cfg = _tiny_run(data, shuffle_labels=True)
        cfg.master_seed = seed
        report = run_experiment(cfg).report
        worst = max(worst, report.slope_t)
        assert report.verdict == "degenerate", (seed, report.slope_t)
    assert worst < 6.0


# --------------------------------------------------------------------------- #
# Masked pixels carry no coupling
# --------------------------------------------------------------------------- #

def test_random_zone_pixels_have_zero_effective_gain_in_the_reports() -> None:
    data = DataConfig(name="all_random", plane=SMALL_PLANE,
                      zones=[zone(ZoneKind.RANDOM, gain=1.0, radius=200.0)])
    patches = build_patches(SamplingConfig(name="g", layout="uniform_grid",
                                           grid_step=5),
                            SMALL_PLANE, np.random.default_rng(0))
    table = sampled_area_composition(patches, jitter_free_gain_field(data),
                                     SMALL_PLANE)
    rows = table.set_index("kind")
    assert rows.loc["random", "weight_fraction"] == pytest.approx(1.0)
    assert rows.loc["background", "weight_fraction"] == pytest.approx(0.0)
    assert rows.loc["ALL_SAMPLED", "mean_gain"] == pytest.approx(0.0)
    # The whole table, random included, now partitions the sampled area.
    assert table[table["kind"] != "ALL_SAMPLED"]["weight_fraction"].sum() == \
        pytest.approx(1.0)


def test_attribution_true_gain_is_zero_under_the_mask() -> None:
    data = DataConfig(name="half_random", plane=SMALL_PLANE,
                      zones=[zone(ZoneKind.RANDOM, gain=1.0, radius=6.0)])
    result = run_experiment(_tiny_run(data))
    mask = jitter_free_gain_field(data).random_mask
    assert np.all(result.attribution.true_gain[mask] == 0.0)
    assert np.all(result.attribution.true_gain[~mask] == 1.0)


# --------------------------------------------------------------------------- #
# Outputs of run_experiment that nothing checked
# --------------------------------------------------------------------------- #

def test_oracle_report_is_computed_from_the_oracle_column() -> None:
    data = DataConfig(name="heat", plane=SMALL_PLANE,
                      zones=[zone(ZoneKind.HEAT, gain=2.0, radius=6.0)])
    result = run_experiment(_tiny_run(data))
    expected = result.trials.groupby("c")["f_oracle"].mean().to_numpy()
    np.testing.assert_allclose(result.oracle_report.mean_by_c["mean_f"], expected)
    assert not np.allclose(result.oracle_report.mean_by_c["mean_f"],
                           result.report.mean_by_c["mean_f"])


def test_calibration_is_exact_for_a_noiseless_line() -> None:
    report = _report(lambda c: 4.0 * c + 2.0, noise=0.0, per_level=3)
    assert report.calibration["rmse"].max() < 1e-9


def test_sign_agreement_matches_its_definition() -> None:
    data = DataConfig(name="anti", plane=SMALL_PLANE,
                      zones=[zone(ZoneKind.ANTI, gain=-1.0, radius=6.0)])
    result = run_experiment(_tiny_run(data))
    w, a = result.attribution.w_pixels, result.attribution.true_gain
    sampled = ~np.isnan(w)
    expected = float(np.mean(np.sign(w[sampled]) == np.sign(a[sampled])))
    assert result.attribution.sign_agreement == pytest.approx(expected)
    assert expected < 1.0 or (a[sampled] < 0).any()


# --------------------------------------------------------------------------- #
# Determinism with jitter on (the unseeded-jitter mutant passed everything)
# --------------------------------------------------------------------------- #

def test_jittered_run_with_zones_is_bit_identical() -> None:
    data = DataConfig(
        name="jitter", plane=SMALL_PLANE,
        zones=[zone(ZoneKind.HEAT, gain=2.0, radius=5.0),
               zone(ZoneKind.DEAD, gain=0.0, radius=4.0, center=(4.0, 15.0),
                    zone_id="dead1")],
        jitter=JitterSpec(enabled=True, center_sigma_px=1.5, radius_sigma_px=0.5,
                          gain_sigma=0.05))
    first, second = run_experiment(_tiny_run(data)), run_experiment(_tiny_run(data))
    np.testing.assert_array_equal(first.trials["f"], second.trials["f"])
    np.testing.assert_array_equal(first.trials.filter(like="jit_").to_numpy(),
                                  second.trials.filter(like="jit_").to_numpy())
    assert first.trials["jit_heat1_dx"].std() > 0          # jitter really ran


# --------------------------------------------------------------------------- #
# Geometry details the old tests could not fail on
# --------------------------------------------------------------------------- #

def test_soft_falloff_profiles_have_their_documented_shape() -> None:
    """Checked at known distances with width > 0 (the old test used width 0,
    where every profile short-circuits to the hard edge)."""
    plane = SMALL_PLANE
    base = ZoneSpec(id="z", kind=ZoneKind.HEAT, center_x=10.0, center_y=10.0,
                    radius=6.0, gain=2.0, falloff_width=4.0)
    smooth = zone_weight_map(base.model_copy(update={"falloff": Falloff.SMOOTHSTEP}),
                             plane)
    gauss = zone_weight_map(base.model_copy(update={"falloff": Falloff.GAUSSIAN}),
                            plane)
    # Row 10, column 10 + d is distance d from the centre.
    at = lambda m, d: m[10, 10 + d]
    assert at(smooth, 0) == 1.0 and at(smooth, 2) == 1.0      # core: radius - width
    assert 0.0 < at(smooth, 4) < 1.0                          # inside the ramp
    assert at(smooth, 4) == pytest.approx(0.5)                # smoothstep midpoint
    assert at(smooth, 6) == 0.0                               # zone ends at radius
    assert at(gauss, 6) == 1.0                                # flat to the radius
    assert at(gauss, 8) == pytest.approx(np.exp(-0.5 * (2 / 4) ** 2))


def test_replace_mode_follows_priority_not_list_order() -> None:
    """High-priority zone listed FIRST, so config order would give the wrong answer."""
    high = zone(ZoneKind.HEAT, gain=2.0, radius=8.0, zone_id="high")
    high.priority = 5
    low = zone(ZoneKind.DEAD, gain=0.0, radius=8.0, zone_id="low")
    data = DataConfig(name="replace", plane=SMALL_PLANE, overlap_mode="replace",
                      zones=[high, low])
    gf = jitter_free_gain_field(data)
    overlap = (gf.zone_weights[0] >= 0.5) & (gf.zone_weights[1] >= 0.5)
    assert overlap.any()
    assert np.allclose(gf.a[overlap], 2.0)


# --------------------------------------------------------------------------- #
# Config strictness
# --------------------------------------------------------------------------- #

def test_config_files_reject_unknown_keys() -> None:
    """A typo used to be dropped silently and the default used instead."""
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate({"n_per_clas": 5})


def test_editing_a_loaded_config_is_revalidated() -> None:
    training = TrainingConfig()
    with pytest.raises(ValidationError):
        training.solver = "svd"                       # shrinkage is still "auto"
