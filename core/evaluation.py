"""Diagnostics: how faithfully does the decision value track the latent variable?

Responsibility
--------------
Probe a trained model across the contrast grid and quantify the mapping from `c`
to `f` — including the ways a high correlation can hide a bent curve.

Explicitly NOT this module's job
--------------------------------
Persistence. `run_experiment` returns everything and SAVES NOTHING, keeping
`core/` free of I/O side effects (CLAUDE.md §3) and of UI imports (I2).

Pipeline position
-----------------
    model + fresh trials  ->  [this module]  ->  LinearityReport / ExperimentResult

Ground truth (plan.md §1.1)
---------------------------
`c` itself. Every quantity here is some way of asking "how far is f from an
affine function of c, and where does it go wrong?"
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from core.config import DataConfig, EvalConfig, RunConfig
from core.data import GainField, jitter_free_gain_field
from core.dataset import build_evaluation_set, build_training_set
from core.model import (
    TrainedModel,
    decision_values,
    separation_d_prime,
    scaled_features,
    train_lda,
    weights_in_pixel_space,
)
from core.rng import RngBundle, bundle_for, spawn_streams
from core.sampling import (
    Patch,
    build_patches,
    patch_pixel_indices,
    sampled_area_composition,
    unique_pixels_sampled,
)

FloatArray = npt.NDArray[np.float64]

# Pre-registered thresholds (plan.md §1.3). Fixed BEFORE looking at results, so
# "significant" is not decided post hoc. Changing these is a scientific decision
# and belongs in docs/decisions.md, not in a bugfix.
KAPPA_AFFINE = 0.05          # kappa below this counts as affine
KAPPA_NONLINEAR = 0.15       # kappa above this counts as a clear violation
NESTED_ALPHA = 0.01          # nested F-test significance level
SD_RATIO_HOMOSCEDASTIC = 1.5  # max/min SD(f|c) below this counts as homoscedastic
CI_LEVEL = 0.95

# A contrast level whose SD is below this fraction of the largest level's SD is
# treated as structurally zero-variance. Needed because a deterministic level
# (Bernoulli at theta in {0, 1}) comes out of pandas as ~1e-12, not 0.0, and an
# exact `> 0` test then let it through and produced ratios like 7e13.
ZERO_SD_RELATIVE_TOL = 1e-9

# A slope must beat its own standard error by this margin before any
# slope-normalised diagnostic is computed.
#
# Why a margin of 10 rather than the obvious "95% CI covers zero": with pure
# noise that test flags a significant slope 5% of the time BY CONSTRUCTION, so
# across four null controls x five seeds it would wrongly declare roughly one run
# non-degenerate every batch, and Phase-1 exit criterion 2 would fail at random.
# Measured over 200 simulated N(0,1) null runs at the default grid, max |t| was
# about 2.6 (10 of 200 exceeded the 5% critical value, as they must), while a
# real mapping reaches |t| in the hundreds. At the default 4200 evaluation trials
# t = 10 corresponds to |r| ~ 0.155 (|r| ~ 0.36 at the fast preset's 660), i.e.
# this also refuses to
# report a curvature ratio when the linear component is buried in trial noise —
# which is the honest answer, since kappa divides by that component.
# Superseded the CI-covers-zero rule of plan.md §1.3; see docs/decisions.md.
SLOPE_T_MIN = 10.0


@dataclass
class LinearityReport:
    """Everything needed to judge H1-H3 for one (model, data config) pair.

    The four `| None` fields are the slope-normalised quantities. They divide by
    the fitted slope, which is ~0 by design in the null controls, so they are
    `None` exactly when `verdict == "degenerate"` (plan.md §7.2, rule 2). The
    lists in that rule and the `| None` fields here must stay in step.
    """

    n_trials: int
    mean_by_c: pd.DataFrame                # c, mean_f, sd_f, ci_lo, ci_hi
    sd_by_c: pd.DataFrame                  # c, sd_f
    clip_by_c: pd.DataFrame                # c, mean_clip_fraction
    residual_by_c: pd.DataFrame            # c, mean_residual, ci_lo, ci_hi
    calibration: pd.DataFrame | None       # c, bias, rmse
    pearson_r: float
    spearman_rho: float
    beta0: float
    beta1: float
    beta1_ci: tuple[float, float]
    slope_t: float                         # |beta1| / SE(beta1); drives `degenerate`
    r2: float
    poly_coefs: dict[str, float]
    curvature_index: float | None          # kappa — the headline number
    kappa_quadratic: float | None          # |beta2|/|beta1|: one-sided bend
    kappa_cubic: float | None              # |beta3|/|beta1|: S-shape / saturation
    nested_f: dict[str, float]
    max_local_slope_ratio: float | None
    effective_range_use: float | None
    sd_ratio: float
    n_zero_variance_levels: int
    homoscedastic: bool
    monotonicity_violations: int
    verdict: Literal["affine", "equivocal", "nonlinear", "degenerate", "fit_failed"]

    def summary_row(self) -> dict[str, object]:
        """One flat row for the cross-condition table (plan.md §7.6)."""
        return {
            "verdict": self.verdict,
            "kappa": self.curvature_index,
            "kappa_quadratic": self.kappa_quadratic,
            "kappa_cubic": self.kappa_cubic,
            "spearman_rho": self.spearman_rho,
            "pearson_r": self.pearson_r,
            "r2": self.r2,
            "beta1": self.beta1,
            "slope_t": self.slope_t,
            "sd_ratio": self.sd_ratio,
            "n_zero_variance_levels": self.n_zero_variance_levels,
            "homoscedastic": self.homoscedastic,
            "calibration_rmse": (float(self.calibration["rmse"].mean())
                                 if self.calibration is not None else None),
            "nested_p_quadratic": self.nested_f.get("p_quadratic"),
            "nested_p_cubic": self.nested_f.get("p_cubic"),
            "effective_range_use": self.effective_range_use,
            "monotonicity_violations": self.monotonicity_violations,
            "mean_clip_fraction": float(self.clip_by_c["mean_clip_fraction"].mean()),
        }


@dataclass
class WeightAttribution:
    """Did the LDA put its weight where the coupling actually is?

    Explains WHY a mapping curved, and catches whole classes of bug: large weight
    inside a random zone means either overfitting to nuisance variance or a broken
    patch-to-feature index mapping (plan.md §7.4).
    """

    abs_correlation: float
    sign_agreement: float
    mean_signed_weight_by_kind: pd.DataFrame
    w_pixels: FloatArray                   # (H, W), NaN where unsampled
    true_gain: FloatArray                  # (H, W) jitter-free gain field


@dataclass
class ExperimentResult:
    """The complete outcome of one run. Returned, never saved, by run_experiment."""

    config: RunConfig
    run_id: str
    model_id: str
    model: TrainedModel
    trials: pd.DataFrame
    report: LinearityReport
    oracle_report: LinearityReport | None
    attribution: WeightAttribution
    composition: pd.DataFrame
    provenance: dict[str, str]
    # d' between the two TRAINING extremes, measured on FRESH evaluation trials
    # at the grid levels nearest c_lo and c_hi. The in-sample d' on the model
    # overstates separation badly when features outnumber trials: the all-random
    # control reads 3.6 in-sample and ~0.03 here.
    fresh_d_prime: float = float("nan")
    fresh_d_prime_levels: tuple[float, float] = (float("nan"), float("nan"))


def fresh_separation(trials: pd.DataFrame, c_lo: float,
                     c_hi: float) -> tuple[float, tuple[float, float]]:
    """d' on held-out trials at the grid levels closest to the training extremes.

    Returns
    -------
    (d_prime, (level_lo, level_hi)). NaN when the grid has fewer than two levels.
    """
    levels = np.unique(trials["c"].to_numpy())
    if levels.size < 2:
        return float("nan"), (float("nan"), float("nan"))
    level_lo = float(levels[np.argmin(np.abs(levels - c_lo))])
    level_hi = float(levels[np.argmin(np.abs(levels - c_hi))])
    if level_lo == level_hi:
        return float("nan"), (level_lo, level_hi)
    f_lo = trials.loc[trials["c"] == level_lo, "f"].to_numpy()
    f_hi = trials.loc[trials["c"] == level_hi, "f"].to_numpy()
    return separation_d_prime(f_lo, f_hi), (level_lo, level_hi)


def _orthogonal_polynomial_basis(c: FloatArray, degree: int) -> FloatArray:
    """Orthonormal polynomial basis on `c`, like R's `poly()`.

    Returns
    -------
    (n, degree + 1) array whose first column is constant.

    Using an ORTHOGONAL basis matters: with a raw Vandermonde design the linear
    and quadratic coefficients are correlated, so |beta2|/|beta1| would depend on
    where the contrast grid happens to be centred rather than on the curvature.
    """
    z = (c - c.mean()) / (c.std() if c.std() > 0 else 1.0)
    vandermonde = np.vander(z, degree + 1, increasing=True)
    q, _ = np.linalg.qr(vandermonde)
    # Fix the sign convention so coefficients are comparable across runs.
    signs = np.sign(np.sum(q * vandermonde, axis=0))
    signs[signs == 0] = 1.0
    return q * signs


def _nested_f_tests(c: FloatArray, f: FloatArray) -> dict[str, float]:
    """F-tests for adding a quadratic, then a cubic term to the affine fit.

    ASSUMPTION: homoscedastic residuals, which H3 explicitly questions.
    ARTEFACT:   under heteroscedasticity the F-test is anti-conservative, so a
                small p-value alone is weak evidence. This is one of two reasons
                `kappa` (an effect size) is the primary criterion; the other is
                that at 4200 trials the test detects curvature far too small to
                matter (CLAUDE.md §10).
    """
    results: dict[str, float] = {}
    residual_sums: list[float] = []
    for degree in (1, 2, 3):
        basis = _orthogonal_polynomial_basis(c, degree)
        coefficients, *_ = np.linalg.lstsq(basis, f, rcond=None)
        residual_sums.append(float(np.sum((f - basis @ coefficients) ** 2)))

    n = f.size
    for name, (index_small, index_large, params_large) in {
        "quadratic": (0, 1, 3),
        "cubic": (1, 2, 4),
    }.items():
        rss_small, rss_large = residual_sums[index_small], residual_sums[index_large]
        df_denominator = n - params_large
        if rss_large <= 0 or df_denominator <= 0:
            results[f"F_{name}"] = float("nan")
            results[f"p_{name}"] = float("nan")
            continue
        f_statistic = ((rss_small - rss_large) / 1.0) / (rss_large / df_denominator)
        results[f"F_{name}"] = float(f_statistic)
        results[f"p_{name}"] = float(stats.f.sf(f_statistic, 1, df_denominator))
    results["df_denominator"] = float(n - 4)
    return results


def linearity_report(trials: pd.DataFrame, f_column: str = "f") -> LinearityReport:
    """Full diagnostic suite for one set of evaluation trials.

    Parameters
    ----------
    trials : DataFrame
        Long form, one row per trial, with columns `c`, `clip_fraction` and
        `f_column`. The whole table is needed rather than just (c, f): clip
        activity comes from `clip_fraction` and cannot be recovered afterwards.

    Returns
    -------
    LinearityReport, with the H2 verdict decided by the pre-registered rule in
    plan.md §1.3 and the H3 judgement reported separately.
    """
    c = trials["c"].to_numpy(dtype=np.float64)
    f = trials[f_column].to_numpy(dtype=np.float64)
    n = f.size

    # --- affine fit with a CI on the slope -------------------------------------
    design = np.column_stack([np.ones_like(c), c])
    (beta0, beta1), *_ = np.linalg.lstsq(design, f, rcond=None)
    residuals = f - design @ np.array([beta0, beta1])
    df_residual = n - 2
    mse = float(np.sum(residuals ** 2) / df_residual)
    slope_se = float(np.sqrt(mse / np.sum((c - c.mean()) ** 2)))
    t_critical = float(stats.t.ppf(0.5 + CI_LEVEL / 2.0, df_residual))
    beta1_ci = (beta1 - t_critical * slope_se, beta1 + t_critical * slope_se)
    # A perfectly constant `f` — the all-zero oracle readout on a plane where
    # nothing tracks c — has zero residual variance AND zero slope, so the naive
    # ratio is 0/0. It must read as an unresolved slope (t = 0), not a perfect
    # fit (t = inf); otherwise the degenerate branch is skipped and the
    # calibration step divides by a zero slope.
    if f.std() == 0.0:
        slope_t = 0.0
    elif slope_se > 0:
        slope_t = float(abs(beta1) / slope_se)
    else:
        slope_t = float("inf")
    total_ss = float(np.sum((f - f.mean()) ** 2))
    r2 = float(1.0 - np.sum(residuals ** 2) / total_ss) if total_ss > 0 else 0.0

    # An unresolved slope makes every slope-normalised quantity 0/0. That is the
    # EXPECTED outcome for the null controls, not a failure.
    degenerate = slope_t < SLOPE_T_MIN

    # --- per-contrast aggregates ----------------------------------------------
    grouped = trials.groupby("c", sort=True)
    mean_f = grouped[f_column].mean()
    sd_f = grouped[f_column].std(ddof=1)
    count = grouped[f_column].count()
    half_width = stats.t.ppf(0.5 + CI_LEVEL / 2.0, count - 1) * sd_f / np.sqrt(count)
    mean_by_c = pd.DataFrame({
        "c": mean_f.index.to_numpy(),
        "mean_f": mean_f.to_numpy(),
        "sd_f": sd_f.to_numpy(),
        "ci_lo": (mean_f - half_width).to_numpy(),
        "ci_hi": (mean_f + half_width).to_numpy(),
        "n": count.to_numpy(),
    })
    sd_by_c = mean_by_c[["c", "sd_f"]].copy()
    clip_by_c = (grouped["clip_fraction"].mean().rename("mean_clip_fraction")
                 .reset_index())

    residual_frame = trials.assign(_residual=residuals).groupby("c", sort=True)
    residual_mean = residual_frame["_residual"].mean()
    residual_sd = residual_frame["_residual"].std(ddof=1)
    residual_half = (stats.t.ppf(0.5 + CI_LEVEL / 2.0, count - 1)
                     * residual_sd / np.sqrt(count))
    residual_by_c = pd.DataFrame({
        "c": residual_mean.index.to_numpy(),
        "mean_residual": residual_mean.to_numpy(),
        "ci_lo": (residual_mean - residual_half).to_numpy(),
        "ci_hi": (residual_mean + residual_half).to_numpy(),
    })

    # --- curvature ------------------------------------------------------------
    basis = _orthogonal_polynomial_basis(c, 3)
    poly_coefficients, *_ = np.linalg.lstsq(basis, f, rcond=None)
    poly_coefs = {f"beta{i}": float(v) for i, v in enumerate(poly_coefficients)}
    linear_term = abs(poly_coefficients[1])
    # kappa combines the quadratic AND cubic orthogonal components.
    #
    # The first version used |beta2| / |beta1| alone, and that was blind to the
    # most important failure this project exists to detect. Saturation here is
    # symmetric about the pivot (0.5): clipping flattens BOTH ends, producing an
    # S-shaped curve that is odd-symmetric, so beta2 ~ 0 while beta3 is large. A
    # logistic curve with slope 12 scored kappa = 0.001 and was called "affine"
    # (docs/decisions.md D14). The quadratic term catches one-sided bends, the
    # cubic term catches S-shapes; the Euclidean combination is invariant to
    # which of the two a curve has, and reduces to the old index when beta3 = 0.
    if degenerate or linear_term == 0.0:
        curvature_index = kappa_quadratic = kappa_cubic = None
    else:
        kappa_quadratic = float(abs(poly_coefficients[2]) / linear_term)
        kappa_cubic = float(abs(poly_coefficients[3]) / linear_term)
        curvature_index = float(np.hypot(kappa_quadratic, kappa_cubic))
    nested_f = _nested_f_tests(c, f)

    # --- shape of the mean curve ---------------------------------------------
    contrasts = mean_by_c["c"].to_numpy()
    curve = mean_by_c["mean_f"].to_numpy()
    local_slopes = np.diff(curve) / np.diff(contrasts)
    overall_direction = np.sign(curve[-1] - curve[0])
    monotonicity_violations = int(np.sum(np.sign(local_slopes) == -overall_direction)) \
        if overall_direction != 0 else int(local_slopes.size)

    if degenerate:
        max_local_slope_ratio = None
        effective_range_use = None
        calibration: pd.DataFrame | None = None
    else:
        magnitudes = np.abs(local_slopes)
        smallest = float(magnitudes.min())
        max_local_slope_ratio = (float(magnitudes.max() / smallest) if smallest > 0
                                 else float("inf"))
        # How much of the output range is spent on the middle half of the input?
        # A straight line spends exactly 0.5; an S-shaped curve spends more, an
        # end-expanded one less.
        #
        # The mean curve is INTERPOLATED at the quarter points of the c range.
        # Selecting the grid points that happen to fall between the quartiles
        # made the answer depend on the grid: a perfectly straight line read 0.40
        # on the fast preset's 11-point grid and 0.50 on the 21-point grid.
        c_min, c_max = float(contrasts.min()), float(contrasts.max())
        q1 = c_min + 0.25 * (c_max - c_min)
        q3 = c_min + 0.75 * (c_max - c_min)
        total_span = float(curve.max() - curve.min())
        effective_range_use = (
            float(abs(np.interp(q3, contrasts, curve) - np.interp(q1, contrasts, curve))
                  / total_span)
            if total_span > 0 else None)
        # Calibration: invert the affine fit and ask how wrong `f` is when used
        # as a measurement of `c` — the practical question behind H2.
        c_hat = (f - beta0) / beta1
        calibration_frame = pd.DataFrame({"c": c, "error": c_hat - c})
        calibration = (calibration_frame.groupby("c", sort=True)["error"]
                       .agg(bias="mean", rmse=lambda e: float(np.sqrt(np.mean(e ** 2))))
                       .reset_index())

    # H3's spread ratio, computed over levels with nonzero variance only.
    #
    # ASSUMPTION: a level whose SD is EXACTLY zero is structurally degenerate,
    #             not merely low-variance.
    # ARTEFACT:   this is not a cosmetic guard. Under the Bernoulli pixel model
    #             theta = 0 or 1 makes every pixel deterministic, so SD(f|c)
    #             vanishes at the grid endpoints and the raw max/min ratio is
    #             `inf` for even the cleanest simple scenario — which would label
    #             every Bernoulli run heteroscedastic and make H3 untestable.
    #             Excluding those levels answers the question actually being
    #             asked (does precision vary across the range where there IS
    #             variance?), and `n_zero_variance_levels` reports how many were
    #             dropped so the exclusion is never invisible.
    sd_values = np.nan_to_num(sd_by_c["sd_f"].to_numpy(), nan=0.0)
    sd_floor = ZERO_SD_RELATIVE_TOL * float(sd_values.max()) if sd_values.size else 0.0
    informative = sd_values > sd_floor
    n_zero_variance_levels = int(np.sum(~informative))
    informative_sd = sd_values[informative]
    sd_ratio = (float(informative_sd.max() / informative_sd.min())
                if informative_sd.size else float("nan"))

    # --- pre-registered verdict (plan.md §1.3), evaluated in this order --------
    if degenerate:
        verdict: Literal["affine", "equivocal", "nonlinear", "degenerate",
                         "fit_failed"] = "degenerate"
    elif curvature_index is not None and curvature_index < KAPPA_AFFINE:
        verdict = "affine"
    elif curvature_index is not None and (
        curvature_index > KAPPA_NONLINEAR
        or min(nested_f.get("p_quadratic", 1.0), nested_f.get("p_cubic", 1.0))
        < NESTED_ALPHA
    ):
        verdict = "nonlinear"
    else:
        verdict = "equivocal"

    return LinearityReport(
        n_trials=n,
        mean_by_c=mean_by_c,
        sd_by_c=sd_by_c,
        clip_by_c=clip_by_c,
        residual_by_c=residual_by_c,
        calibration=calibration,
        # A constant `f` (the all-zero oracle readout, or a fully saturated
        # model) makes both correlations undefined rather than zero. Report NaN
        # instead of letting scipy warn and return NaN anyway.
        pearson_r=(float(stats.pearsonr(c, f).statistic) if f.std() > 0
                   else float("nan")),
        spearman_rho=(float(stats.spearmanr(c, f).statistic) if f.std() > 0
                      else float("nan")),
        beta0=float(beta0),
        beta1=float(beta1),
        beta1_ci=beta1_ci,
        slope_t=slope_t,
        r2=r2,
        poly_coefs=poly_coefs,
        curvature_index=curvature_index,
        kappa_quadratic=kappa_quadratic,
        kappa_cubic=kappa_cubic,
        nested_f=nested_f,
        max_local_slope_ratio=max_local_slope_ratio,
        effective_range_use=effective_range_use,
        sd_ratio=sd_ratio,
        n_zero_variance_levels=n_zero_variance_levels,
        homoscedastic=bool(sd_ratio < SD_RATIO_HOMOSCEDASTIC),
        monotonicity_violations=monotonicity_violations,
        verdict=verdict,
    )


def oracle_weights(data: DataConfig, patches: Sequence[Patch],
                   feature_mode: str) -> FloatArray:
    """The best gain-aligned linear readout, built from the ground truth.

    Returns
    -------
    (n_features,) gain-proportional weights, L2-normalised.

    NOT mean-centred. Centring across features looks harmless but destroys the
    simple scenario outright: there every pixel has gain 1, so centred weights
    are identically zero and the "best possible readout" would be a constant.
    The right readout when all pixels track `c` equally is to average them all.
    Only the scale is arbitrary, so the weights are L2-normalised and the affine
    fit absorbs the offset.

    ASSUMPTION: weights are gain-proportional in the STANDARDISED feature space
                (the representation the LDA sees), not in raw pixel units. On raw
                pixels that corresponds to a_i / sigma_i, which is a different —
                but still entirely legitimate — linear readout.
    ARTEFACT:   the scaler is fitted on training data, so the oracle inherits the
                training-set pixel SDs. Under the Bernoulli model those SDs
                depend on c, so at intermediate contrasts the oracle's implied
                raw-pixel weighting is slightly off-optimal. It remains a
                ground-truth-derived reference, which is all H6 needs.

    Computed on the JITTER-FREE gain field: an oracle that saw each trial's
    realised jitter would be omniscient rather than merely well-informed, and
    would no longer bound what a real linear readout could achieve.

    Feature-mode mapping (plan.md §7.3): `pixels` -> gain per sampled pixel;
    `patch_mean` -> mean gain per patch; `patch_mean_std` -> mean gain on the
    mean features and ZERO on the SD features. That last case is a stated
    limitation: variance cues carry information about `c` that a gain-
    proportional oracle cannot express, so under `patch_mean_std` the oracle is
    a lower bound on the achievable linear readout, not a ceiling.
    """
    gf = jitter_free_gain_field(data)
    indices = patch_pixel_indices(patches, data.plane)
    gains = gf.a.ravel()[indices]
    # A masked pixel ignores c, so an informed readout gives it no weight.
    gains = np.where(gf.random_mask.ravel()[indices], 0.0, gains)

    if feature_mode == "pixels":
        weights = gains
    else:
        pixels_per_patch = patches[0].size ** 2
        per_patch = gains.reshape(len(patches), pixels_per_patch).mean(axis=1)
        weights = (per_patch if feature_mode == "patch_mean"
                   else np.concatenate([per_patch, np.zeros_like(per_patch)]))
    norm = float(np.linalg.norm(weights))
    # An all-zero gain field (the all-dead and all-random controls) has no
    # informative readout. Returning zeros is correct: the oracle then reports
    # `degenerate`, exactly as the LDA should.
    return weights / norm if norm > 0 else weights


def oracle_decision_values(model: TrainedModel, X: FloatArray,
                           w_oracle: FloatArray) -> FloatArray:
    """Apply the oracle weights to the same scaled features the LDA sees."""
    return scaled_features(model, X) @ w_oracle


def weight_attribution(model: TrainedModel, data: DataConfig,
                       patches: Sequence[Patch]) -> WeightAttribution:
    """Compare the learned weight map against the true coupling field."""
    gf: GainField = jitter_free_gain_field(data)
    w_pixels = weights_in_pixel_space(model, data.plane)
    sampled = ~np.isnan(w_pixels)
    w_flat = w_pixels[sampled]
    # Effective gain: a random-masked pixel ignores c, so its true coupling is 0
    # whatever the composed `a` says there. Using raw `a` credited masked pixels
    # with gain ~1 and made the all-random control look like it had signal.
    effective_gain = np.where(gf.random_mask, 0.0, gf.a)
    a_flat = effective_gain[sampled]
    masked_flat = gf.random_mask[sampled]

    abs_correlation = (float(stats.pearsonr(np.abs(w_flat), np.abs(a_flat)).statistic)
                       if w_flat.size > 2 and np.std(np.abs(w_flat)) > 0
                       and np.std(np.abs(a_flat)) > 0 else float("nan"))
    signed = np.sign(w_flat) == np.sign(a_flat)
    sign_agreement = float(signed.mean()) if w_flat.size else float("nan")

    rows: list[dict[str, object]] = []
    if gf.zone_weights.size:
        for kind in dict.fromkeys(gf.zone_kinds):
            selector = np.array([k is kind for k in gf.zone_kinds])
            share = gf.zone_weights[selector].sum(axis=0)[sampled]
            inside = share >= 0.5
            rows.append({
                "kind": str(kind),
                "n_pixels": int(inside.sum()),
                "mean_signed_weight": float(w_flat[inside].mean()) if inside.any()
                else float("nan"),
                "mean_true_gain": float(a_flat[inside].mean()) if inside.any()
                else float("nan"),
            })
        coupling = np.array([k.value != "random" for k in gf.zone_kinds])
        background_share = np.maximum(
            0.0, 1.0 - gf.zone_weights[coupling].sum(axis=0)[sampled]
        ) if coupling.any() else np.ones(w_flat.size)
    else:
        background_share = np.ones(w_flat.size)
    # A masked pixel is not background: it does not track c at the baseline rate.
    background_share = np.where(masked_flat, 0.0, background_share)
    inside_background = background_share >= 0.5
    rows.append({
        "kind": "background",
        "n_pixels": int(inside_background.sum()),
        "mean_signed_weight": float(w_flat[inside_background].mean())
        if inside_background.any() else float("nan"),
        "mean_true_gain": float(a_flat[inside_background].mean())
        if inside_background.any() else float("nan"),
    })

    return WeightAttribution(
        abs_correlation=abs_correlation,
        sign_agreement=sign_agreement,
        mean_signed_weight_by_kind=pd.DataFrame(rows),
        w_pixels=w_pixels,
        true_gain=effective_gain,
    )


def evaluate_model(model: TrainedModel, data: DataConfig, evaluation: EvalConfig,
                   rngs: RngBundle, patches: Sequence[Patch]) -> pd.DataFrame:
    """Probe the model on fresh trials across the contrast grid.

    Returns
    -------
    The long-form trial table (plan.md §7.1): one row per trial, with `f`,
    optionally `f_oracle`, `clip_fraction` and the per-zone jitter columns.
    Aggregation happens at analysis time, or the spread information H3 needs is
    gone.
    """
    trials = build_evaluation_set(data, model.sampling_config, evaluation, patches, rngs)
    table = trials.meta.copy()
    table["f"] = decision_values(model, trials.X)
    if evaluation.compute_oracle:
        w_oracle = oracle_weights(data, patches, model.sampling_config.feature_mode)
        table["f_oracle"] = oracle_decision_values(model, trials.X, w_oracle)
    return table


def run_experiment(cfg: RunConfig, provenance: dict[str, str] | None = None,
                   ids: dict[str, str] | None = None) -> ExperimentResult:
    """Train, evaluate, diagnose. Returns everything; SAVES NOTHING.

    Persistence is the caller's job: `core/` stays free of I/O side effects
    (CLAUDE.md §3, layout comment on `core/`) and free of UI imports (I2).
    """
    streams = spawn_streams(cfg.master_seed)
    patches = build_patches(cfg.sampling, cfg.data.plane, streams["patches"])

    identifiers = ids or {}
    train = build_training_set(cfg.data, cfg.sampling, cfg.training, patches,
                              bundle_for(streams, "train"), streams["labels"])
    model = train_lda(
        train, cfg.training, cfg.sampling, patches,
        data_config_hash=identifiers.get("data_config_id", ""),
        sampling_config_hash=identifiers.get("sampling_config_id", ""),
        master_seed=cfg.master_seed,
        provenance=provenance or {},
    )
    trials = evaluate_model(model, cfg.data, cfg.evaluation,
                            bundle_for(streams, "eval"), patches)

    report = linearity_report(trials, "f")
    if model.diagnostics.fit_failed:
        # A model with nothing to learn must not masquerade as a null result:
        # `degenerate` means "c is not linearly readable", this means "the fit
        # never happened". The oracle report below is unaffected.
        report.verdict = "fit_failed"
    fresh_d_prime, fresh_levels = fresh_separation(trials, cfg.training.c_lo,
                                                   cfg.training.c_hi)
    oracle_report = (linearity_report(trials, "f_oracle")
                     if "f_oracle" in trials.columns else None)
    attribution = weight_attribution(model, cfg.data, patches)
    composition = sampled_area_composition(patches, jitter_free_gain_field(cfg.data),
                                           cfg.data.plane)
    composition.attrs["n_features"] = model.diagnostics.n_features
    composition.attrs["unique_pixels"] = unique_pixels_sampled(patches, cfg.data.plane)

    return ExperimentResult(
        config=cfg,
        run_id=identifiers.get("run_id", ""),
        model_id=identifiers.get("model_id", ""),
        model=model,
        trials=trials,
        report=report,
        oracle_report=oracle_report,
        attribution=attribution,
        composition=composition,
        provenance=provenance or {},
        fresh_d_prime=fresh_d_prime,
        fresh_d_prime_levels=fresh_levels,
    )
