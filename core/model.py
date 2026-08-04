"""LDA training and the single path from features to decision values.

Responsibility
--------------
Fit a scaler and an LDA on training trials, record the diagnostics that say
whether the fit is trustworthy, and expose decision values.

Explicitly NOT this module's job
--------------------------------
Generating data, computing linearity diagnostics, or persisting anything.

Pipeline position
-----------------
    TrialSet(train)  ->  [this module]  ->  f  ->  evaluation

Invariant I4
------------
The scaler is fitted on training features ONLY. `decision_values` is the single
path from raw features to `f`, so no caller can accidentally scale by hand or
skip scaling.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from core.config import PlaneSpec, SamplingConfig, TrainingConfig
from core.dataset import TrialSet
from core.sampling import Patch, patch_pixel_indices

FloatArray = npt.NDArray[np.float64]

# Singular values below this multiple of the largest one count as numerical zero
# when computing effective rank. Matches numpy's default matrix_rank tolerance
# scale, made explicit because rank is a reported diagnostic here.
RANK_TOLERANCE = 1e-12


@dataclass
class TrainingDiagnostics:
    """Whether the fit is trustworthy, independent of what it predicts.

    `top1pct_weight_mass` is a fragility measure: a model whose weight mass sits
    on a handful of pixels will not survive a change of seed.
    """

    train_accuracy: float
    cv_accuracy: float
    cv_accuracy_sd: float
    d_prime: float
    cov_condition_number: float
    effective_rank: int
    n_features: int
    n_train: int
    weight_norm: float
    top1pct_weight_mass: float


@dataclass
class TrainedModel:
    """A fitted model plus everything needed to reproduce and interpret it."""

    lda: LinearDiscriminantAnalysis
    scaler: StandardScaler | None
    patches: list[Patch]
    sampling_config: SamplingConfig
    training_config: TrainingConfig
    data_config_hash: str
    sampling_config_hash: str
    master_seed: int
    diagnostics: TrainingDiagnostics
    provenance: dict[str, str]


def _pooled_covariance_spectrum(X_scaled: FloatArray,
                                y: npt.NDArray[np.int64]) -> tuple[float, int]:
    """Condition number and effective rank of the pooled within-class covariance.

    Computed from the singular values of the class-centred data matrix rather
    than by forming the (n_features, n_features) covariance: with 2500 features
    and a few hundred trials the explicit covariance is both slower and less
    numerically informative than the economy SVD.

    Returns
    -------
    (condition_number, effective_rank). The condition number is `inf` when the
    covariance is singular — which is the EXPECTED case whenever
    n_features > n_train, and is exactly the warning sign plan.md §6.3 is about.
    """
    centred = np.empty_like(X_scaled)
    for label in np.unique(y):
        rows = y == label
        centred[rows] = X_scaled[rows] - X_scaled[rows].mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    if singular.size == 0 or singular[0] == 0.0:
        return float("inf"), 0
    eigenvalues = singular ** 2
    rank = int((eigenvalues > eigenvalues[0] * RANK_TOLERANCE).sum())
    n_possible = min(X_scaled.shape)
    if rank < n_possible or X_scaled.shape[1] > rank:
        return float("inf"), rank
    return float(eigenvalues[0] / eigenvalues[-1]), rank


def train_lda(train: TrialSet, training: TrainingConfig, sampling: SamplingConfig,
              patches: Sequence[Patch], data_config_hash: str,
              sampling_config_hash: str, master_seed: int,
              provenance: dict[str, str] | None = None) -> TrainedModel:
    """Fit scaler + LDA on the two extreme contrast levels.

    Raises
    ------
    ValueError
        If `train.y` is missing, or if either class is absent.

    Warnings
    --------
    Emits an explicit warning when a training extreme sits at 0 or 1 under the
    Bernoulli model, where within-class variance is exactly zero and the
    covariance is singular by construction (plan.md §6.2). Letting
    scikit-learn's generic collinearity warning stand in for this would bury a
    condition that invalidates the fit.
    """
    if train.y is None:
        raise ValueError("training set has no labels; use build_training_set")
    if len(np.unique(train.y)) < 2:
        raise ValueError("training set has only one class")

    if training.c_lo <= 0.0 or training.c_hi >= 1.0:
        warnings.warn(
            f"training extremes ({training.c_lo}, {training.c_hi}) touch the ends "
            "of the contrast range. Under the Bernoulli pixel model the "
            "within-class variance there is exactly zero, so the covariance is "
            "singular and the LDA solution is arbitrary in the null directions "
            "(plan.md §6.2).",
            stacklevel=2,
        )

    scaler: StandardScaler | None = None
    X = train.X
    if training.standardize:
        # Fitted on training features ONLY (I4). Fitting on pooled train+eval
        # data would inject evaluation statistics into the model.
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

    lda = LinearDiscriminantAnalysis(solver=training.solver,
                                     shrinkage=training.shrinkage)
    lda.fit(X, train.y)

    f_train = lda.decision_function(X)
    mean_1, mean_0 = f_train[train.y == 1].mean(), f_train[train.y == 0].mean()
    var_1, var_0 = f_train[train.y == 1].var(), f_train[train.y == 0].var()
    pooled_sd = np.sqrt(0.5 * (var_0 + var_1))
    d_prime = float(abs(mean_1 - mean_0) / pooled_sd) if pooled_sd > 0 else float("inf")

    condition_number, effective_rank = _pooled_covariance_spectrum(X, train.y)

    w = np.ravel(lda.coef_)
    squared = w ** 2
    n_top = max(1, int(round(0.01 * w.size)))
    top_mass = float(np.sort(squared)[-n_top:].sum() / squared.sum()) \
        if squared.sum() > 0 else 0.0

    # Cross-validation costs `cv_folds` extra LDA fits, and with 2500 features
    # and Ledoit-Wolf shrinkage a single fit dominates the whole run. cv_folds=0
    # skips it for sweeps where only the c -> f mapping is of interest.
    if training.cv_folds and training.cv_folds >= 2:
        cv_scores = cross_val_score(
            LinearDiscriminantAnalysis(solver=training.solver,
                                       shrinkage=training.shrinkage),
            X, train.y,
            cv=StratifiedKFold(n_splits=training.cv_folds, shuffle=False),
        )
    else:
        cv_scores = np.array([np.nan])

    diagnostics = TrainingDiagnostics(
        train_accuracy=float(lda.score(X, train.y)),
        cv_accuracy=float(cv_scores.mean()),
        cv_accuracy_sd=float(cv_scores.std()),
        d_prime=d_prime,
        cov_condition_number=condition_number,
        effective_rank=effective_rank,
        n_features=int(X.shape[1]),
        n_train=int(X.shape[0]),
        weight_norm=float(np.linalg.norm(w)),
        top1pct_weight_mass=top_mass,
    )
    return TrainedModel(
        lda=lda,
        scaler=scaler,
        patches=list(patches),
        sampling_config=sampling,
        training_config=training,
        data_config_hash=data_config_hash,
        sampling_config_hash=sampling_config_hash,
        master_seed=master_seed,
        diagnostics=diagnostics,
        provenance=provenance or {},
    )


def scaled_features(model: TrainedModel, X: FloatArray) -> FloatArray:
    """Apply the stored scaler (if any).

    Exposed so the oracle readout can act on exactly the representation the LDA
    sees, making `f` and `f_oracle` commensurable (plan.md §7.3).
    """
    return model.scaler.transform(X) if model.scaler is not None else X


def decision_values(model: TrainedModel, X: FloatArray) -> FloatArray:
    """f = w . x + b_lda, from RAW features.

    The single path from features to decision values: no caller scales by hand.
    """
    return np.asarray(model.lda.decision_function(scaled_features(model, X)),
                      dtype=np.float64)


def weights_in_pixel_space(model: TrainedModel, plane: PlaneSpec) -> FloatArray:
    """Map the LDA weight vector back onto the plane.

    Returns
    -------
    (H, W) array; NaN wherever no patch covered the pixel.

    Overlapping patches SUM their contributions, because that is how they enter
    the decision value. Only defined for `feature_mode == "pixels"`; the
    patch-aggregated modes have no per-pixel weight, so a patch's weight is
    spread uniformly over its pixels instead.
    """
    w = np.ravel(model.lda.coef_)
    flat = np.full(plane.height * plane.width, np.nan, dtype=np.float64)
    covered = np.zeros(plane.height * plane.width, dtype=bool)
    indices = patch_pixel_indices(model.patches, plane)

    if model.sampling_config.feature_mode == "pixels":
        per_pixel = w
    else:
        # patch_mean / patch_mean_std: attribute a patch's mean-feature weight
        # evenly across its pixels. The SD features have no pixel-level analogue
        # and are excluded, which is stated as a limitation in plan.md §7.3.
        pixels_per_patch = model.sampling_config.patch_size ** 2
        mean_weights = w[:len(model.patches)]
        per_pixel = np.repeat(mean_weights / pixels_per_patch, pixels_per_patch)

    np.add.at(flat, indices, 0.0)   # no-op that validates index bounds early
    accumulated = np.zeros_like(flat)
    np.add.at(accumulated, indices, per_pixel)
    covered[indices] = True
    flat[covered] = accumulated[covered]
    return flat.reshape(plane.height, plane.width)
