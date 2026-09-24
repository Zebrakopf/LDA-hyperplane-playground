"""Trial generation: turn configs into feature matrices with known ground truth.

Responsibility
--------------
Loop over contrast levels, generate fields, extract features, and record each
trial's true `c` plus its realised nuisance state.

Explicitly NOT this module's job
--------------------------------
Model fitting or diagnostics.

Pipeline position
-----------------
    c grid  ->  [this module]  ->  TrialSet(X, c, y, meta)  ->  model / evaluation

Invariants enforced here
------------------------
I4  Evaluation sets are generated fresh from the `field_eval` / `jitter_eval`
    streams and are never used to fit anything.
I5  Training sets contain ONLY c_lo and c_hi.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.random import Generator

from core.config import DataConfig, EvalConfig, SamplingConfig, TrainingConfig
from core.data import GainField, build_gain_field, generate_field
from core.rng import RngBundle
from core.sampling import Patch, extract_features

FloatArray = npt.NDArray[np.float64]


@dataclass
class TrialSet:
    """Generated trials with their ground truth attached."""

    X: FloatArray                          # (n_trials, n_features)
    c: FloatArray                          # (n_trials,) true latent value
    y: npt.NDArray[np.int64] | None         # (n_trials,) labels; None when evaluating
    meta: pd.DataFrame                     # per-trial c, trial_index, clip_fraction,
                                           # jit_<zone_id>_{dx,dy,dr,dg}

    def __post_init__(self) -> None:
        if self.X.shape[0] != self.c.shape[0]:
            raise ValueError(f"X has {self.X.shape[0]} rows but c has {self.c.shape[0]}")


def generate_trials(data: DataConfig, sampling: SamplingConfig,
                    patches: Sequence[Patch], contrasts: Sequence[float],
                    n_per_level: int, rngs: RngBundle) -> TrialSet:
    """Generate `n_per_level` trials at each contrast in `contrasts`.

    Returns
    -------
    TrialSet with `y=None`; labelling is the caller's decision.

    Performance note
    ----------------
    When `data.jitter.enabled` is False the gain field is deterministic and
    `_draw_jitter` consumes no randomness, so it is built once and reused. This
    changes nothing about the random streams (verified by the determinism tests)
    and removes an O(n_zones * H * W) rebuild per trial.
    """
    reusable_gain_field: GainField | None = (
        None if data.jitter.enabled else build_gain_field(data, rngs.jitter)
    )

    feature_rows: list[FloatArray] = []
    meta_rows: list[dict[str, float]] = []
    trial_index = 0
    for c in contrasts:
        for _ in range(n_per_level):
            _field, gf, diagnostics = generate_field(
                data, float(c), rngs.field, rngs.jitter,
                gain_field=reusable_gain_field,
            )
            feature_rows.append(extract_features(_field, patches, sampling.feature_mode))
            row: dict[str, float] = {
                "c": float(c),
                "trial_index": float(trial_index),
                "clip_fraction": diagnostics.clip_fraction,
            }
            # Per-zone jitter columns: jitter perturbs each zone independently, so
            # two scalar columns could not represent a multi-zone config, and
            # plan.md §1.1 obligation 2 requires the realised state to be
            # regressable onto the distortion.
            for zone_id, offsets in diagnostics.jitter_realised.items():
                for key, value in offsets.items():
                    row[f"jit_{zone_id}_d{key[-1]}"] = value
            meta_rows.append(row)
            trial_index += 1

    X = np.asarray(feature_rows, dtype=np.float64)
    meta = pd.DataFrame(meta_rows)
    meta["trial_index"] = meta["trial_index"].astype(np.int64)
    return TrialSet(X=X, c=meta["c"].to_numpy(), y=None, meta=meta)


def build_training_set(data: DataConfig, sampling: SamplingConfig,
                       training: TrainingConfig, patches: Sequence[Patch],
                       rngs: RngBundle, rng_labels: Generator) -> TrialSet:
    """Trials at the two extremes only, labelled 0 / 1 (invariant I5).

    `training.shuffle_labels` implements the label-shuffle null control: the
    features are untouched and only `y` is permuted, so any residual
    c-dependence in the resulting decision values is a pipeline artefact
    (plan.md §7.5, control 4).
    """
    trials = generate_trials(
        data, sampling, patches,
        contrasts=[training.c_lo, training.c_hi],
        n_per_level=training.n_per_class,
        rngs=rngs,
    )
    y = (trials.c > (training.c_lo + training.c_hi) / 2.0).astype(np.int64)
    if training.shuffle_labels:
        y = _balanced_shuffle(y, rng_labels)
    trials.y = y
    trials.meta = trials.meta.assign(y=y)
    return trials


def _balanced_shuffle(y: npt.NDArray[np.int64],
                      rng: Generator) -> npt.NDArray[np.int64]:
    """Labels that are exactly independent of the true extreme.

    Within each true class, a random half gets label 0 and the other half label
    1. A plain `permutation(y)` does not achieve that: by chance each shuffled
    class holds an unequal mix of c_lo and c_hi trials, which leaks a real slope
    into the "null" control — on the fast preset its |t| reached 9.3 against a
    degeneracy threshold of 10 (docs/decisions.md D14).

    Returns
    -------
    (n_trials,) int array with the same class counts as `y`.
    """
    shuffled = np.empty_like(y)
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        order = rng.permutation(members.size)
        half = members.size // 2
        shuffled[members[order[:half]]] = 0
        shuffled[members[order[half:]]] = 1
    return shuffled


def build_evaluation_set(data: DataConfig, sampling: SamplingConfig,
                         evaluation: EvalConfig, patches: Sequence[Patch],
                         rngs: RngBundle) -> TrialSet:
    """Fresh trials across the full contrast grid, unlabelled (invariant I4)."""
    return generate_trials(
        data, sampling, patches,
        contrasts=evaluation.contrast_grid,
        n_per_level=evaluation.n_per_contrast,
        rngs=rngs,
    )
