"""Generation of pixel fields from the latent contrast variable.

Responsibility
--------------
Turn a `DataConfig` plus a contrast value `c` into a realised (H, W) field, and
expose the per-trial coupling structure that the diagnostics need.

Explicitly NOT this module's job
--------------------------------
Patch sampling, feature extraction, anything model-related. Those live in
`core/sampling.py` and `core/model.py`.

Pipeline position
-----------------
    c  ->  [this module]  ->  field  ->  sampling  ->  X

The coupling model (plan.md §4.2)
---------------------------------
    theta_i(c) = pivot + a_i * (c - pivot) + b_i

`a` is the per-pixel coupling gain, `b` a contrast-independent offset. This
replaces the additive `mu_i(c) = c + sum_k Z_k(i)` of docs/original_spec.md,
because additive zone contributions saturate on overlap and that saturation is a
nonlinearity we would then measure instead of the effect under study.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from core.config import DataConfig, Falloff, NoiseSpec, PlaneSpec, ZoneKind, ZoneSpec

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# A random zone masks a pixel when its falloff weight reaches this threshold.
# Chosen so a soft-edged random zone still produces a definite mask instead of a
# probabilistic halo (plan.md §4.4).
RANDOM_MASK_THRESHOLD = 0.5

# In "replace" overlap mode a zone owns a pixel once its weight reaches this.
REPLACE_OWNERSHIP_THRESHOLD = 0.5


@dataclass(frozen=True)
class TrialDiagnostics:
    """Nuisance state realised on one trial, recorded so distortion can be
    regressed onto its cause (plan.md §1.1, obligation 2)."""

    clip_fraction: float
    jitter_realised: dict[str, dict[str, float]] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class GainField:
    """The realised per-pixel coupling for ONE trial (jitter already applied)."""

    a: FloatArray                     # (H, W) coupling gain
    b: FloatArray                     # (H, W) offset
    random_mask: BoolArray            # (H, W) True where the pixel ignores c
    zone_weights: FloatArray          # (n_zones, H, W) falloff weight per zone
    zone_ids: tuple[str, ...]         # length n_zones, parallel to zone_weights
    zone_kinds: tuple[ZoneKind, ...]  # length n_zones
    jitter_realised: dict[str, dict[str, float]]   # zone_id -> {dx, dy, dr, dg}

    def dominant_zone(self) -> npt.NDArray[np.int16]:
        """(H, W) argmax over `zone_weights`; -1 where every weight is 0.

        Only meaningful for overlap_mode == "replace". Under the default "blend"
        mode there IS no dominant zone — the point is a mixture — so composition
        and attribution use `zone_weights` directly (plan.md §5.3, §7.4) and this
        helper is for display only.
        """
        if self.zone_weights.size == 0:
            return np.full(self.a.shape, -1, dtype=np.int16)
        winner = np.argmax(self.zone_weights, axis=0).astype(np.int16)
        winner[self.zone_weights.max(axis=0) <= 0.0] = -1
        return winner


def zone_weight_map(zone: ZoneSpec, plane: PlaneSpec, *, dx: float = 0.0,
                    dy: float = 0.0, dr: float = 0.0) -> FloatArray:
    """Radial membership weight of every pixel in `zone`.

    Parameters
    ----------
    dx, dy, dr : float
        Jitter offsets applied to the zone's centre and radius on this trial.

    Returns
    -------
    (H, W) float array in [0, 1].

    The three falloff profiles agree in the limit `falloff_width -> 0`, so a
    config can be moved between them without changing the zone's nominal extent.

    ASSUMPTION: membership is isotropic and depends only on distance to centre.
    ARTEFACT:   `hard` edges are a step in the coupling field. A 5x5 patch
                straddling one averages across two very different couplings,
                which shows up as extra between-trial variance under jitter. That
                is a real effect worth studying, which is why the edge profile is
                a choice rather than a constant.
    """
    rows = np.arange(plane.height, dtype=np.float64)[:, None]
    cols = np.arange(plane.width, dtype=np.float64)[None, :]
    distance = np.sqrt((cols - (zone.center_x + dx)) ** 2
                       + (rows - (zone.center_y + dy)) ** 2)
    radius = max(zone.radius + dr, 0.0)
    width = max(zone.falloff_width, 0.0)

    if zone.falloff is Falloff.HARD or width == 0.0:
        return (distance <= radius).astype(np.float64)

    if zone.falloff is Falloff.SMOOTHSTEP:
        # Cubic smoothstep from 1 at (radius - width) down to 0 at radius.
        t = np.clip((radius - distance) / width, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    if zone.falloff is Falloff.GAUSSIAN:
        # Flat 1 inside the radius, Gaussian shoulder outside it.
        outside = np.maximum(distance - radius, 0.0)
        return np.exp(-0.5 * (outside / width) ** 2)

    raise ValueError(f"unhandled falloff {zone.falloff!r}")


def _draw_jitter(cfg: DataConfig, rng_jitter: Generator
                 ) -> dict[str, dict[str, float]]:
    """Draw per-zone geometric jitter for one trial.

    Each zone is perturbed INDEPENDENTLY but the perturbation is coherent across
    that zone's pixels — this is what makes jitter a nuisance latent rather than
    i.i.d. noise (plan.md §4.5).

    When jitter is disabled no draws are consumed at all, which is what lets
    `core.dataset` cache the gain field without changing any random stream.
    """
    jitter = cfg.jitter
    if not jitter.enabled:
        return {zone.id: {"dx": 0.0, "dy": 0.0, "dr": 0.0, "dg": 0.0}
                for zone in cfg.zones}
    realised: dict[str, dict[str, float]] = {}
    for zone in cfg.zones:
        realised[zone.id] = {
            "dx": float(rng_jitter.normal(0.0, jitter.center_sigma_px)),
            "dy": float(rng_jitter.normal(0.0, jitter.center_sigma_px)),
            "dr": float(rng_jitter.normal(0.0, jitter.radius_sigma_px)),
            "dg": float(rng_jitter.normal(0.0, jitter.gain_sigma)),
        }
    return realised


def build_gain_field(cfg: DataConfig, rng_jitter: Generator) -> GainField:
    """Compose zones into the per-pixel coupling field for one trial.

    Overlap composition (plan.md §4.4)
    ----------------------------------
    `blend` (default) is a partition-of-unity weighted average in which the
    BACKGROUND is an explicit contributor:

        w_bg = max(0, 1 - sum_k w_k)
        a    = (w_bg * background_gain + sum_k w_k * a_k) / (w_bg + sum_k w_k)

    The denominator is 1 when sum_k w_k <= 1 and sum_k w_k otherwise, so it is
    never zero and the expression is continuous at sum_k w_k == 1. The result is
    a convex combination of `background_gain` together with the contributing zone
    gains, hence bounded by

        min(background_gain, min_k a_k) <= a <= max(background_gain, max_k a_k)

    Note this is NOT "never larger than the largest zone gain": a dead zone
    (gain 0) at edge weight 0.5 yields a = 0.5, pulled up by the background.

    ASSUMPTION: overlapping influences average rather than accumulate.
    ARTEFACT:   averaging dilutes a zone's nominal gain wherever its weight is
                below 1, i.e. throughout every soft edge. A `heat` zone with
                gain 3 and a smoothstep edge reaches gain 3 only in its core.
                This is deliberate — the alternative (summation) is unbounded —
                but it means "the gain in zone k" is a core value, not a
                zone-wide one.

    `replace` instead gives each pixel to the highest-`priority` zone whose
    weight reaches `REPLACE_OWNERSHIP_THRESHOLD` (PowerPoint-style z-order paint
    semantics), leaving soft edges as a tie-break region rather than a mixture.
    """
    plane = cfg.plane
    shape = (plane.height, plane.width)
    jitter_realised = _draw_jitter(cfg, rng_jitter)

    if not cfg.zones:
        return GainField(
            a=np.full(shape, cfg.background_gain, dtype=np.float64),
            b=np.zeros(shape, dtype=np.float64),
            random_mask=np.zeros(shape, dtype=bool),
            zone_weights=np.zeros((0, *shape), dtype=np.float64),
            zone_ids=(),
            zone_kinds=(),
            jitter_realised=jitter_realised,
        )

    weights = np.stack([
        zone_weight_map(zone, plane, **{k: jitter_realised[zone.id][k]
                                       for k in ("dx", "dy", "dr")})
        for zone in cfg.zones
    ])
    gains = np.array([zone.gain + jitter_realised[zone.id]["dg"] for zone in cfg.zones])
    offsets = np.array([zone.offset for zone in cfg.zones])
    is_random = np.array([zone.kind is ZoneKind.RANDOM for zone in cfg.zones])

    # Random zones are a MASK, not a gain: they are excluded from composition
    # entirely and applied afterwards (plan.md §4.4).
    random_mask = np.zeros(shape, dtype=bool)
    if is_random.any():
        random_mask = (weights[is_random] >= RANDOM_MASK_THRESHOLD).any(axis=0)

    coupling_weights = weights[~is_random]
    coupling_gains = gains[~is_random]
    coupling_offsets = offsets[~is_random]

    if coupling_weights.size == 0:
        a = np.full(shape, cfg.background_gain, dtype=np.float64)
        b = np.zeros(shape, dtype=np.float64)
    elif cfg.overlap_mode == "blend":
        weight_sum = coupling_weights.sum(axis=0)
        w_bg = np.maximum(0.0, 1.0 - weight_sum)
        denominator = w_bg + weight_sum
        a = (w_bg * cfg.background_gain
             + np.tensordot(coupling_gains, coupling_weights, axes=(0, 0))) / denominator
        b = (np.tensordot(coupling_offsets, coupling_weights, axes=(0, 0))) / denominator
    elif cfg.overlap_mode == "replace":
        a = np.full(shape, cfg.background_gain, dtype=np.float64)
        b = np.zeros(shape, dtype=np.float64)
        priorities = np.array([zone.priority for zone in cfg.zones])[~is_random]
        # Ascending priority so later writes win; ties broken by config order.
        for index in np.argsort(priorities, kind="stable"):
            owned = coupling_weights[index] >= REPLACE_OWNERSHIP_THRESHOLD
            a[owned] = coupling_gains[index]
            b[owned] = coupling_offsets[index]
    else:
        raise ValueError(f"unhandled overlap_mode {cfg.overlap_mode!r}")

    return GainField(
        a=a,
        b=b,
        random_mask=random_mask,
        zone_weights=weights,
        zone_ids=tuple(zone.id for zone in cfg.zones),
        zone_kinds=tuple(zone.kind for zone in cfg.zones),
        jitter_realised=jitter_realised,
    )


def coupling_to_theta(c: float, gf: GainField,
                      cfg: DataConfig) -> tuple[FloatArray, float]:
    """Map latent contrast to per-pixel parameters.

        theta_i(c) = pivot + a_i * (c - pivot) + b_i        # then clipped

    Rotating around `pivot` rather than adding to a baseline keeps every zone
    type on a common footing: at c == pivot all pixels agree regardless of gain,
    so zones differ in *slope*, not in mean brightness. See plan.md §4.2 for why
    this replaces the additive formulation in docs/original_spec.md.

    Returns
    -------
    theta : (H, W) float array in [0, 1]
    clip_fraction : float
        Fraction of NON-masked pixels whose parameter was clipped. Clipping and
        its diagnostic live in one function on purpose — a caller receiving an
        already-clipped `theta` cannot recover this number.
    """
    theta = cfg.pivot + gf.a * (c - cfg.pivot) + gf.b

    # Clip theta into [0, 1] so it is a valid Bernoulli parameter / pixel value.
    #
    # ASSUMPTION: values outside [0, 1] are meaningless for this display model.
    # ARTEFACT:   clipping is a saturating nonlinearity. Wherever it bites, the
    #             pixel's response to c flattens, so f(c) compresses at that end
    #             of the range. With pivot == 0.5 and b == 0 and |a| <= 1 on
    #             every trial this is unreachable, so clipping is an *opt-in*
    #             condition — activated only by a gain magnitude above 1
    #             (including background_gain), a nonzero offset, a pivot other
    #             than 0.5, or a nonzero jitter.gain_sigma (which can push |a|
    #             past 1 on individual trials even when configured gains are in
    #             bounds). `DataConfig.clipping_is_reachable()` enumerates
    #             exactly those four paths, and `clip_fraction` below records
    #             whether saturation actually fired on this trial, so no result
    #             is ever read without knowing.
    out_of_range = (theta < 0.0) | (theta > 1.0)
    considered = ~gf.random_mask          # masked pixels never use theta
    n_considered = int(considered.sum())
    clip_fraction = (float((out_of_range & considered).sum()) / n_considered
                     if n_considered else 0.0)
    return np.clip(theta, 0.0, 1.0), clip_fraction


def sample_field(theta: FloatArray, gf: GainField, noise: NoiseSpec,
                 rng_field: Generator) -> FloatArray:
    """Draw one realised field from the per-pixel parameters.

    Returns
    -------
    (H, W) float array of pixel values in [0, 1].

    Random-masked pixels are overwritten last with draws from
    `noise.random_zone_dist`, making them contrast-independent nuisance variance.
    """
    if noise.pixel_model == "bernoulli":
        # ASSUMPTION: theta is the probability the pixel is white.
        # ARTEFACT:   variance is theta*(1-theta), so it VANISHES at theta in
        #             {0, 1}. Training at exact extremes then gives a singular
        #             within-class covariance (plan.md §6.2). No noise parameter
        #             exists for this model — the variance is set by theta alone.
        field = (rng_field.random(theta.shape) < theta).astype(np.float64)
    elif noise.pixel_model == "clipped_gaussian":
        # ASSUMPTION: additive homoscedastic pixel noise, then censoring.
        # ARTEFACT:   censoring biases the mean toward the interior near 0 and 1
        #             even when theta itself is unclipped, so E[pixel] != theta
        #             at the ends. This is the classic censored-distribution
        #             bias, and it curves f(c) independently of any zone
        #             structure — which is why the `beta` model exists as a
        #             control.
        field = np.clip(rng_field.normal(theta, noise.sigma), 0.0, 1.0)
    elif noise.pixel_model == "beta":
        # ASSUMPTION: bounded continuous pixels with mean theta and precision nu.
        # ARTEFACT:   Beta support is OPEN, so theta in {0, 1} is undefined — and
        #             the default contrast grid includes c = 0 and c = 1. theta
        #             is therefore clamped into [eps, 1-eps]. The clamp shifts
        #             the mean by at most eps (1e-3 by default, three orders
        #             below the noise), but it IS an artefact and is named here
        #             rather than hidden.
        eps = noise.theta_eps
        theta_open = np.clip(theta, eps, 1.0 - eps)
        nu = noise.beta_concentration
        field = rng_field.beta(theta_open * nu, (1.0 - theta_open) * nu)
    else:
        raise ValueError(f"unhandled pixel_model {noise.pixel_model!r}")

    if gf.random_mask.any():
        n_masked = int(gf.random_mask.sum())
        if noise.random_zone_dist == "bernoulli_half":
            replacement = (rng_field.random(n_masked) < 0.5).astype(np.float64)
        elif noise.random_zone_dist == "uniform":
            replacement = rng_field.random(n_masked)
        else:
            raise ValueError(f"unhandled random_zone_dist {noise.random_zone_dist!r}")
        field = field.copy()
        field[gf.random_mask] = replacement

    return field


def generate_field(cfg: DataConfig, c: float, rng_field: Generator,
                   rng_jitter: Generator,
                   gain_field: GainField | None = None
                   ) -> tuple[FloatArray, GainField, TrialDiagnostics]:
    """Generate one trial end-to-end: c -> gain field -> theta -> pixels.

    Parameters
    ----------
    gain_field : GainField, optional
        Reuse an already-built field. Only legal when `cfg.jitter.enabled` is
        False, in which case building it is deterministic and consumes no
        randomness — see `core.dataset` for why that matters for speed.

    Returns
    -------
    field : (H, W) pixel values
    gf : GainField
        The realised coupling. Returned because jitter makes it trial-specific
        and the diagnostics need it.
    diagnostics : TrialDiagnostics
    """
    if gain_field is not None and cfg.jitter.enabled:
        raise ValueError(
            "a cached gain field cannot be reused when jitter is enabled: the "
            "coupling is trial-specific by design (plan.md §4.5)."
        )
    gf = gain_field if gain_field is not None else build_gain_field(cfg, rng_jitter)
    theta, clip_fraction = coupling_to_theta(c, gf, cfg)
    field = sample_field(theta, gf, cfg.noise, rng_field)
    return field, gf, TrialDiagnostics(clip_fraction=clip_fraction,
                                       jitter_realised=gf.jitter_realised)


def jitter_free_gain_field(cfg: DataConfig) -> GainField:
    """The gain field implied by the configured geometry, with no jitter.

    Used by the oracle readout and by weight attribution, both of which must be
    blind to any individual trial's realised jitter (plan.md §7.3).
    """
    disabled = cfg.model_copy(deep=True)
    disabled.jitter.enabled = False
    # Passing a generator that must not be used: with jitter disabled,
    # `_draw_jitter` consumes nothing, so this is unreachable state, not a seed.
    return build_gain_field(disabled, np.random.default_rng(0))
