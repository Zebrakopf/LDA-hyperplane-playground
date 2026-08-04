"""Sampling, invariant and leakage tests, plus the end-to-end null controls.

Covers CLAUDE.md §8 test classes 1, 4 and 5, and the Phase-1 exit criteria that
can be checked at test scale.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from core.config import (
    DataConfig,
    EvalConfig,
    RunConfig,
    SamplingConfig,
    TrainingConfig,
    ZoneKind,
)
from core.data import jitter_free_gain_field
from core.evaluation import linearity_report, oracle_weights, run_experiment
from core.rng import bundle_for, spawn_streams
from core.sampling import (
    build_patches,
    extract_features,
    n_features,
    patch_pixel_indices,
    sampled_area_composition,
    unique_pixels_sampled,
)
from tests.conftest import SMALL_PLANE, zone


# --------------------------------------------------------------------------- #
# Sampling: the feature-order contract
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("feature_mode,expected_factor",
                         [("pixels", 25), ("patch_mean", 1), ("patch_mean_std", 2)])
def test_feature_length_matches_declared_size(uniform_data: DataConfig,
                                              sampling: SamplingConfig,
                                              feature_mode: str,
                                              expected_factor: int) -> None:
    cfg = sampling.model_copy(update={"feature_mode": feature_mode})
    patches = build_patches(cfg, SMALL_PLANE, np.random.default_rng(0))
    field = np.random.default_rng(1).random((SMALL_PLANE.height, SMALL_PLANE.width))
    features = extract_features(field, patches, feature_mode)
    assert features.size == len(patches) * expected_factor
    assert features.size == n_features(cfg, len(patches))


def test_pixel_features_follow_the_index_contract(uniform_data: DataConfig,
                                                  sampling: SamplingConfig) -> None:
    """`patch_pixel_indices` must return pixels in `extract_features` order.

    Weight attribution inverts this mapping, so a mismatch would silently
    scatter the learned weights across the wrong pixels — a bug that produces a
    plausible-looking figure rather than an error.
    """
    patches = build_patches(sampling, SMALL_PLANE, np.random.default_rng(0))
    field = np.random.default_rng(2).random((SMALL_PLANE.height, SMALL_PLANE.width))
    features = extract_features(field, patches, "pixels")
    indices = patch_pixel_indices(patches, SMALL_PLANE)
    np.testing.assert_array_equal(features, field.ravel()[indices])


def test_overlapping_patches_duplicate_pixels(uniform_data: DataConfig) -> None:
    overlapping = SamplingConfig(name="overlap", layout="uniform_grid", patch_size=5,
                                 grid_step=2, feature_mode="pixels")
    patches = build_patches(overlapping, SMALL_PLANE, np.random.default_rng(0))
    indices = patch_pixel_indices(patches, SMALL_PLANE)
    assert unique_pixels_sampled(patches, SMALL_PLANE) < indices.size


def test_sampled_area_composition_partitions_the_sampled_area(
        uniform_data: DataConfig, sampling: SamplingConfig) -> None:
    """Coupling-zone shares plus background must sum to exactly 1.

    Tested with OVERLAPPING zones on purpose: that is the case where naive
    per-kind weight sums exceed 1, because each zone claims full credit for the
    shared pixels. `random` is excluded from the partition by design — a masked
    pixel leaves the coupling composition entirely.
    """
    cfg = uniform_data.model_copy(update={
        "zones": [zone(ZoneKind.HEAT, gain=2.0, radius=6.0),
                  zone(ZoneKind.DEAD, gain=0.0, radius=4.0, center=(6.0, 12.0),
                       zone_id="dead1")]})
    gf = jitter_free_gain_field(cfg)
    # Confirm the zones really do overlap, or the test proves nothing.
    assert ((gf.zone_weights[0] >= 0.5) & (gf.zone_weights[1] >= 0.5)).any()
    patches = build_patches(sampling, SMALL_PLANE, np.random.default_rng(0))
    table = sampled_area_composition(patches, gf, SMALL_PLANE)
    partition = table[~table["kind"].isin(["ALL_SAMPLED", "random"])]
    assert abs(partition["weight_fraction"].sum() - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #

def test_data_config_never_serialises_sampling_fields(uniform_data: DataConfig,
                                                      sampling: SamplingConfig) -> None:
    """Invariant I1, checked on the wire format rather than the type."""
    payload = json.loads(uniform_data.model_dump_json())
    for forbidden in ("patch_size", "layout", "feature_mode", "grid_step", "patches"):
        assert forbidden not in payload
    sampling_payload = json.loads(sampling.model_dump_json())
    for forbidden in ("zones", "pivot", "background_gain", "noise", "jitter"):
        assert forbidden not in sampling_payload


def test_core_does_not_import_streamlit() -> None:
    """Invariant I2, checked by source inspection rather than by import order."""
    from pathlib import Path
    core_dir = Path(__file__).resolve().parent.parent / "core"
    for module in core_dir.glob("*.py"):
        text = module.read_text(encoding="utf-8")
        assert "import streamlit" not in text, module.name
        assert "from streamlit" not in text, module.name


def test_svd_solver_rejects_shrinkage() -> None:
    """scikit-learn raises for this combination; reject it at config time."""
    with pytest.raises(ValidationError):
        TrainingConfig(solver="svd", shrinkage="auto")
    TrainingConfig(solver="svd", shrinkage=None)      # the legal pairing


def test_strong_zone_warns_only_when_a_strong_zone_exists(
        uniform_data: DataConfig) -> None:
    """The validator must not fire on every default config (background_gain = 1)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # any warning becomes a failure
        DataConfig(name="plain", plane=SMALL_PLANE)
        DataConfig(name="heat", plane=SMALL_PLANE,
                   zones=[zone(ZoneKind.HEAT, gain=2.0)])
    with pytest.warns(UserWarning, match="background_gain"):
        DataConfig(name="bad_strong", plane=SMALL_PLANE,
                   zones=[zone(ZoneKind.STRONG, gain=1.0)])


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #

def test_scaler_is_fitted_on_training_features_only(small_run: RunConfig) -> None:
    """Invariant I4, checked by recomputing the scaler from the training set.

    If the scaler had ever seen evaluation data its stored means would not match
    a scaler fitted on the training features alone.
    """
    from sklearn.preprocessing import StandardScaler

    from core.dataset import build_training_set
    from core.sampling import build_patches as build

    result = run_experiment(small_run)
    streams = spawn_streams(small_run.master_seed)
    patches = build(small_run.sampling, small_run.data.plane, streams["patches"])
    train = build_training_set(small_run.data, small_run.sampling, small_run.training,
                              patches, bundle_for(streams, "train"), streams["labels"])
    expected = StandardScaler().fit(train.X)
    np.testing.assert_allclose(result.model.scaler.mean_, expected.mean_)
    np.testing.assert_allclose(result.model.scaler.scale_, expected.scale_)


def test_training_set_contains_only_the_two_extremes(small_run: RunConfig) -> None:
    """Invariant I5: no intermediate contrast level is ever seen during training."""
    from core.dataset import build_training_set
    streams = spawn_streams(small_run.master_seed)
    patches = build_patches(small_run.sampling, small_run.data.plane, streams["patches"])
    train = build_training_set(small_run.data, small_run.sampling, small_run.training,
                              patches, bundle_for(streams, "train"), streams["labels"])
    assert set(np.unique(train.c)) == {small_run.training.c_lo, small_run.training.c_hi}


# --------------------------------------------------------------------------- #
# Diagnostics behaviour
# --------------------------------------------------------------------------- #

def test_perfectly_affine_input_is_reported_as_affine() -> None:
    """A straight line with noise must give kappa ~ 0 and verdict 'affine'."""
    generator = np.random.default_rng(0)
    c = np.repeat(np.linspace(0.0, 1.0, 21), 200)
    trials = pd.DataFrame({"c": c,
                           "f": 3.0 * c + 1.0 + generator.normal(0, 0.1, c.size),
                           "clip_fraction": 0.0})
    report = linearity_report(trials)
    assert report.verdict == "affine"
    assert report.curvature_index < 0.05
    # A straight line spends exactly half its output range on the middle half of
    # its input. Anything far from 0.5 means end compression.
    assert abs(report.effective_range_use - 0.5) < 0.05


def test_quadratic_input_is_reported_as_nonlinear() -> None:
    generator = np.random.default_rng(1)
    c = np.repeat(np.linspace(0.0, 1.0, 21), 200)
    trials = pd.DataFrame({"c": c,
                           "f": 3.0 * c + 2.0 * c ** 2 + generator.normal(0, 0.1, c.size),
                           "clip_fraction": 0.0})
    report = linearity_report(trials)
    assert report.verdict == "nonlinear"
    # kappa is a ratio of ORTHOGONAL-polynomial components, so it does not equal
    # the raw quadratic coefficient. Empirically f = c + gamma*c^2 gives
    # kappa ~ 0.024 at gamma=0.1, 0.054 at 0.25, 0.135 at 1.0, 0.180 at 2.0.
    # f = 3c + 2c^2 lands at 0.106 -- above the affine threshold and escalated to
    # `nonlinear` by the nested F-test, which is the pre-registered rule working
    # exactly as designed.
    assert report.curvature_index > 0.05


def test_strong_curvature_exceeds_the_nonlinear_threshold() -> None:
    """A clearly bent curve must clear KAPPA_NONLINEAR on its effect size alone."""
    generator = np.random.default_rng(11)
    c = np.repeat(np.linspace(0.0, 1.0, 21), 200)
    trials = pd.DataFrame({"c": c,
                           "f": c + 2.0 * c ** 2 + generator.normal(0, 0.05, c.size),
                           "clip_fraction": 0.0})
    report = linearity_report(trials)
    assert report.curvature_index > 0.15
    assert report.verdict == "nonlinear"


def test_flat_input_is_reported_as_degenerate_without_infinities() -> None:
    """The null controls land here. Slope-normalised fields must be None, not inf.

    This is the defect the spec review caught: kappa and the calibration metrics
    divide by a slope that is zero BY DESIGN in the all-dead, all-random and
    label-shuffle controls.
    """
    # Twenty independent null realisations, not one. The rule this replaced --
    # "the 95% CI of beta1 covers zero" -- fails 5% of the time BY CONSTRUCTION,
    # so a single-seed version of this test passed or failed at random and the
    # null controls would intermittently break Phase-1 exit criterion 2.
    for seed in range(20):
        generator = np.random.default_rng(seed)
        c = np.repeat(np.linspace(0.0, 1.0, 21), 200)
        trials = pd.DataFrame({"c": c, "f": generator.normal(0, 1.0, c.size),
                               "clip_fraction": 0.0})
        report = linearity_report(trials)
        assert report.verdict == "degenerate", f"seed {seed}: {report.slope_t=}"
        assert report.curvature_index is None
        assert report.calibration is None
        assert report.effective_range_use is None
        assert report.max_local_slope_ratio is None


def test_oracle_weights_are_not_degenerate_in_the_simple_scenario(
        uniform_data: DataConfig, sampling: SamplingConfig) -> None:
    """With a uniform gain field the right readout is "average everything".

    Mean-centring the oracle weights would make them identically zero here, which
    would report the best possible linear readout as a constant.
    """
    patches = build_patches(sampling, SMALL_PLANE, np.random.default_rng(0))
    weights = oracle_weights(uniform_data, patches, "pixels")
    assert np.linalg.norm(weights) > 0
    assert np.allclose(weights, weights[0])          # uniform gain => uniform weights


def test_oracle_gives_zero_weight_to_masked_pixels(uniform_data: DataConfig,
                                                   sampling: SamplingConfig) -> None:
    cfg = uniform_data.model_copy(update={"zones": [zone(ZoneKind.RANDOM, gain=1.0)]})
    patches = build_patches(sampling, SMALL_PLANE, np.random.default_rng(0))
    weights = oracle_weights(cfg, patches, "pixels")
    gf = jitter_free_gain_field(cfg)
    masked = gf.random_mask.ravel()[patch_pixel_indices(patches, SMALL_PLANE)]
    assert np.allclose(weights[masked], 0.0)


# --------------------------------------------------------------------------- #
# End-to-end null controls (Phase-1 exit criteria 1 and 2, at test scale)
# --------------------------------------------------------------------------- #

def _small(data: DataConfig, **training_overrides) -> RunConfig:
    return RunConfig(
        data=data,
        sampling=SamplingConfig(name="grid", layout="uniform_grid", patch_size=5,
                                grid_step=5, feature_mode="pixels"),
        training=TrainingConfig(n_per_class=60, cv_folds=0, **training_overrides),
        evaluation=EvalConfig(contrast_grid=np.linspace(0.0, 1.0, 11).tolist(),
                              n_per_contrast=40),
        master_seed=3,
    )


def test_simple_scenario_is_affine(uniform_data: DataConfig) -> None:
    """Exit criterion 1: the pipeline can detect affinity when it is there."""
    result = run_experiment(_small(uniform_data))
    assert result.report.verdict == "affine"
    assert result.report.r2 > 0.98


def test_all_dead_control_is_degenerate(uniform_data: DataConfig) -> None:
    """Exit criterion 2, part 1: nothing tracks c, so nothing can be read out."""
    data = uniform_data.model_copy(update={"name": "all_dead", "background_gain": 0.0})
    result = run_experiment(_small(data))
    assert result.report.verdict == "degenerate"


def test_all_random_control_is_degenerate(uniform_data: DataConfig) -> None:
    """Exit criterion 2, part 2: a whole-plane random zone masks every pixel."""
    data = uniform_data.model_copy(update={
        "name": "all_random",
        "zones": [zone(ZoneKind.RANDOM, gain=1.0, radius=200.0)]})
    assert jitter_free_gain_field(data).random_mask.all()
    result = run_experiment(_small(data))
    assert result.report.verdict == "degenerate"


def test_label_shuffle_control_is_degenerate(uniform_data: DataConfig) -> None:
    """Exit criterion 2, part 3: permuted labels leave no usable direction."""
    result = run_experiment(_small(uniform_data, shuffle_labels=True))
    assert result.report.verdict == "degenerate"
