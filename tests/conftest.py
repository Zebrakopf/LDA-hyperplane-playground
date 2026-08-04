"""Shared fixtures. Small planes and low trial counts keep the suite fast.

Tolerances throughout the suite are derived from the expected standard error and
that reasoning is stated in a comment at each assertion, per CLAUDE.md §8.
"Flaky test, increased tolerance" is not an acceptable fix here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (
    DataConfig,
    Falloff,
    JitterSpec,
    NoiseSpec,
    PlaneSpec,
    RunConfig,
    SamplingConfig,
    TrainingConfig,
    EvalConfig,
    ZoneKind,
    ZoneSpec,
)

SMALL_PLANE = PlaneSpec(height=20, width=20)


@pytest.fixture
def plane() -> PlaneSpec:
    return SMALL_PLANE


@pytest.fixture
def uniform_data() -> DataConfig:
    """The simple scenario: every pixel tracks c exactly, no zones."""
    return DataConfig(name="uniform", plane=SMALL_PLANE,
                      noise=NoiseSpec(pixel_model="bernoulli"))


@pytest.fixture
def sampling() -> SamplingConfig:
    return SamplingConfig(name="grid", layout="uniform_grid", patch_size=5,
                          grid_step=5, feature_mode="pixels")


def zone(kind: ZoneKind, gain: float, *, radius: float = 6.0,
         center: tuple[float, float] = (10.0, 10.0),
         falloff: Falloff = Falloff.HARD, zone_id: str | None = None) -> ZoneSpec:
    """Build one zone with a hard edge, so membership is unambiguous in tests."""
    return ZoneSpec(id=zone_id or f"{kind.value}1", kind=kind, center_x=center[0],
                    center_y=center[1], radius=radius, gain=gain, falloff=falloff,
                    falloff_width=0.0)


@pytest.fixture
def small_run(uniform_data: DataConfig, sampling: SamplingConfig) -> RunConfig:
    """A complete but tiny run, fast enough for the pipeline-level tests."""
    return RunConfig(
        data=uniform_data,
        sampling=sampling,
        training=TrainingConfig(n_per_class=30, cv_folds=0),
        evaluation=EvalConfig(contrast_grid=[0.1, 0.3, 0.5, 0.7, 0.9],
                              n_per_contrast=30),
        master_seed=7,
    )


@pytest.fixture
def jittered() -> JitterSpec:
    return JitterSpec(enabled=True, center_sigma_px=1.5, radius_sigma_px=0.5,
                      gain_sigma=0.0)
