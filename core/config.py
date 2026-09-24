"""Validated configuration objects for every part of the pipeline.

Responsibility
--------------
Define, validate and document every parameter the experiment can take. These
pydantic models are the single source of truth for defaults; no default value
may be duplicated elsewhere in the codebase (CLAUDE.md §5.1, "no magic
numbers").

Explicitly NOT this module's job
--------------------------------
Any computation. Validators may warn or reject, but nothing here generates
data, fits models or touches the filesystem.

Pipeline position
-----------------
    [this module]  ->  everything else

Invariant I1
------------
`DataConfig` (the world) and `SamplingConfig` (the observer) are separate
types. `RunConfig` composes them for convenience but they are serialised
separately and neither ever reads the other's fields.
"""

from __future__ import annotations

import warnings
from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Number of points on the default evaluation contrast grid. Named because both
# the grid default and the documented trial count (21 * n_per_contrast) depend
# on it.
DEFAULT_GRID_POINTS = 21


class StrictModel(BaseModel):
    """Base for every config object.

    `extra="forbid"`: a typo in a config file (`"n_per_clas": 5`) is an error,
    not a silently ignored key that leaves the default in place.
    `validate_assignment=True`: editing a loaded config (the scripts do, for
    `--quick` and `--seeds`) re-runs validation, so an illegal combination such
    as svd + shrinkage cannot be smuggled in after load.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ZoneKind(StrEnum):
    """The five zone behaviours described in plan.md §4.3."""

    HEAT = "heat"          # gain > 1: amplified tracking, saturates
    STRONG = "strong"      # background_gain < gain <= 1: strong, never saturates
    DEAD = "dead"          # gain ~ 0: weak or no tracking
    ANTI = "anti"          # gain < 0: inverted tracking
    RANDOM = "random"      # masks the pixel: it ignores c entirely


class Falloff(StrEnum):
    """Radial edge profile of a zone (plan.md §4.4)."""

    HARD = "hard"
    SMOOTHSTEP = "smoothstep"
    GAUSSIAN = "gaussian"


class ZoneSpec(StrictModel):
    """One circular region that modifies the coupling between `c` and pixels.

    A zone is part of the WORLD, not of the observer. See invariant I1.
    """

    id: str                                   # stable; keys the per-zone jitter columns
    kind: ZoneKind
    center_x: float                           # pixel coords; float so a canvas stays smooth
    center_y: float
    radius: float                             # may exceed the plane (whole-plane controls)
    gain: float = 1.0                         # 'a' inside the zone; sign carries anti-correlation
    offset: float = 0.0                       # 'b'; nonzero enables clipping — use deliberately
    falloff: Falloff = Falloff.SMOOTHSTEP
    falloff_width: float = 2.0                # pixels over which the weight decays
    priority: int = 0                         # used by overlap_mode == "replace"


class JitterSpec(StrictModel):
    """Per-trial perturbation of zone geometry — a coherent nuisance latent.

    Unlike pixel noise this is NOT i.i.d.: it moves whole zones together within
    a trial, so averaging over trials does not remove its effect on the shape of
    f(c) (plan.md §4.5).
    """

    enabled: bool = False
    center_sigma_px: float = 0.0
    radius_sigma_px: float = 0.0
    gain_sigma: float = 0.0                   # > 0 can push |a| past 1 => enables clipping


class NoiseSpec(StrictModel):
    """Pixel-level observation model and its noise parameters (plan.md §4.6)."""

    pixel_model: Literal["bernoulli", "clipped_gaussian", "beta"] = "bernoulli"
    sigma: float = 0.1                        # clipped_gaussian only
    beta_concentration: float = 20.0          # 'nu'; beta only. NEVER call this kappa.
    theta_eps: float = 1e-3                   # beta only: open-support clamp
    random_zone_dist: Literal["bernoulli_half", "uniform"] = "bernoulli_half"


class PlaneSpec(StrictModel):
    """Size of the pixel plane. Configurable; 100x100 is only the default."""

    height: int = 100
    width: int = 100


class DataConfig(StrictModel):
    """The WORLD: how the latent contrast maps onto pixel statistics.

    Never merged with, nested in, or read alongside `SamplingConfig` (I1).
    """

    name: str
    plane: PlaneSpec = PlaneSpec()
    pivot: float = 0.5                        # contrast around which gains rotate
    background_gain: float = 1.0              # 'a' outside all zones (plan.md §4.3)
    overlap_mode: Literal["blend", "replace"] = "blend"
    zones: list[ZoneSpec] = Field(default_factory=list)
    jitter: JitterSpec = JitterSpec()
    noise: NoiseSpec = NoiseSpec()

    @model_validator(mode="after")
    def _warn_on_suspicious_combinations(self) -> DataConfig:
        """Warn about configurations that are legal but probably not intended.

        The two `strong`-zone checks are GATED on a strong zone actually being
        present. `background_gain` defaults to 1.0, so an ungated check would
        fire on every default config including the null controls.
        """
        has_strong = any(z.kind is ZoneKind.STRONG for z in self.zones)
        if has_strong:
            for zone in self.zones:
                if zone.kind is ZoneKind.STRONG and zone.gain <= self.background_gain:
                    warnings.warn(
                        f"strong zone {zone.id!r} has gain {zone.gain} <= "
                        f"background_gain {self.background_gain}: it is a dead zone "
                        "by another name.",
                        stacklevel=2,
                    )
            if self.background_gain >= 1.0:
                warnings.warn(
                    f"background_gain is {self.background_gain} >= 1, so a strong "
                    "zone cannot exceed it without clipping. Lower background_gain "
                    "(plan.md §4.3).",
                    stacklevel=2,
                )
        for zone in self.zones:
            if zone.kind is ZoneKind.RANDOM and (zone.gain != 1.0 or zone.offset != 0.0):
                warnings.warn(
                    f"random zone {zone.id!r} sets gain/offset; both are ignored "
                    "because a random zone masks the pixel (plan.md §4.4).",
                    stacklevel=2,
                )
        return self

    def clipping_is_reachable(self) -> bool:
        """True if this config can clip theta at some c in [0, 1].

        The four (and only four) paths to clipping, per plan.md §4.2:
        a gain magnitude above 1 including `background_gain`, a nonzero offset,
        a pivot other than 0.5, or a nonzero `jitter.gain_sigma`.
        """
        if self.pivot != 0.5:
            return True
        if abs(self.background_gain) > 1.0:
            return True
        if self.jitter.enabled and self.jitter.gain_sigma != 0.0:
            return True
        for zone in self.zones:
            if zone.kind is ZoneKind.RANDOM:
                continue          # masked pixels never use theta
            if abs(zone.gain) > 1.0 or zone.offset != 0.0:
                return True
        return False


class SamplingConfig(StrictModel):
    """The OBSERVER: which pixels the model gets to see, and as what features.

    Never merged with, nested in, or read alongside `DataConfig` (I1).
    """

    name: str
    patch_size: int = 5
    layout: Literal["uniform_grid", "jittered_grid", "random_uniform", "manual"]
    grid_step: int = 10                       # uniform_grid / jittered_grid
    n_patches: int | None = None              # random_uniform only
    layout_jitter_px: float = 0.0             # jittered_grid only
    snap_to_grid: int | None = None
    feature_mode: Literal["pixels", "patch_mean", "patch_mean_std"] = "pixels"
    patches: list[tuple[int, int]] = Field(default_factory=list)  # manual: (row, col)

    @model_validator(mode="after")
    def _check_layout_requirements(self) -> SamplingConfig:
        if self.layout == "random_uniform" and not self.n_patches:
            raise ValueError("layout='random_uniform' requires n_patches")
        if self.layout == "manual" and not self.patches:
            raise ValueError("layout='manual' requires a non-empty patches list")
        return self


class TrainingConfig(StrictModel):
    """How the LDA is trained. Only the two extremes are ever seen (I5)."""

    c_lo: float = 0.05                        # interior by default: see plan.md §6.2
    c_hi: float = 0.95
    n_per_class: int = 400
    solver: Literal["svd", "lsqr", "eigen"] = "lsqr"
    shrinkage: float | Literal["auto"] | None = "auto"
    standardize: bool = True
    shuffle_labels: bool = False              # null control (plan.md §7.5)
    cv_folds: int = 5                          # 0 disables cross-validation

    @model_validator(mode="after")
    def _check_solver_shrinkage(self) -> TrainingConfig:
        """scikit-learn raises for svd + shrinkage; reject it at config time."""
        if self.solver == "svd" and self.shrinkage is not None:
            raise ValueError(
                "solver='svd' does not support shrinkage; set shrinkage=null "
                "(plan.md §6.3)."
            )
        return self


class EvalConfig(StrictModel):
    """The contrast grid on which the trained model is probed."""

    contrast_grid: list[float] = Field(
        default_factory=lambda: np.linspace(0.0, 1.0, DEFAULT_GRID_POINTS).tolist()
    )
    n_per_contrast: int = 200
    compute_oracle: bool = True


class RunConfig(StrictModel):
    """Everything needed to reproduce one experiment, plus one master seed."""

    data: DataConfig
    sampling: SamplingConfig
    training: TrainingConfig = TrainingConfig()
    evaluation: EvalConfig = EvalConfig()
    master_seed: int = 0
