"""Determinism and stream-independence (CLAUDE.md §8, test class 2).

A failure here is a P0 bug: it means an unseeded random stream exists somewhere,
and no result in the repository can be reproduced.
"""

from __future__ import annotations

import numpy as np

from core.config import DataConfig, RunConfig, SamplingConfig
from core.dataset import build_evaluation_set, build_training_set, generate_trials
from core.rng import STREAM_NAMES, bundle_for, spawn_streams
from core.sampling import build_patches


def test_stream_names_are_unique() -> None:
    """Names index positions in a spawn, so duplicates would alias two roles."""
    assert len(set(STREAM_NAMES)) == len(STREAM_NAMES)


def test_same_seed_gives_identical_trials(uniform_data: DataConfig,
                                         sampling: SamplingConfig) -> None:
    def run() -> np.ndarray:
        streams = spawn_streams(11)
        patches = build_patches(sampling, uniform_data.plane, streams["patches"])
        return generate_trials(uniform_data, sampling, patches, [0.2, 0.8], 5,
                               bundle_for(streams, "train")).X

    np.testing.assert_array_equal(run(), run())


def test_different_seed_gives_different_trials(uniform_data: DataConfig,
                                               sampling: SamplingConfig) -> None:
    def run(seed: int) -> np.ndarray:
        streams = spawn_streams(seed)
        patches = build_patches(sampling, uniform_data.plane, streams["patches"])
        return generate_trials(uniform_data, sampling, patches, [0.2, 0.8], 5,
                               bundle_for(streams, "train")).X

    assert not np.array_equal(run(11), run(12))


def test_training_and_evaluation_draws_are_independent(
        uniform_data: DataConfig, sampling: SamplingConfig, small_run: RunConfig) -> None:
    """Trials at the SAME contrast must differ between the two roles.

    This is the substantive content of invariant I4: evaluation trials are fresh
    draws, not a replay of training trials. Separate `field_*`/`jitter_*` streams
    are what guarantee it, which is why there is no train/test "split" stream.
    """
    streams = spawn_streams(13)
    patches = build_patches(sampling, uniform_data.plane, streams["patches"])
    train_like = generate_trials(uniform_data, sampling, patches, [0.5], 20,
                                 bundle_for(streams, "train")).X
    eval_like = generate_trials(uniform_data, sampling, patches, [0.5], 20,
                                bundle_for(streams, "eval")).X
    assert not np.array_equal(train_like, eval_like)


def test_gain_field_caching_does_not_change_the_random_stream(
        uniform_data: DataConfig, sampling: SamplingConfig) -> None:
    """The jitter-disabled fast path must be observationally identical.

    `generate_trials` reuses one gain field when jitter is off. That is only safe
    because `_draw_jitter` consumes no randomness in that case. If it ever starts
    to, this test fails rather than the results silently shifting.
    """
    streams = spawn_streams(17)
    patches = build_patches(sampling, uniform_data.plane, streams["patches"])
    cached = generate_trials(uniform_data, sampling, patches, [0.3, 0.7], 6,
                             bundle_for(streams, "train")).X

    # Rebuild the field per trial, the slow way, from a fresh set of streams.
    from core.data import generate_field
    from core.sampling import extract_features
    streams2 = spawn_streams(17)
    _ = build_patches(sampling, uniform_data.plane, streams2["patches"])
    bundle = bundle_for(streams2, "train")
    rows = []
    for c in (0.3, 0.7):
        for _ in range(6):
            field, _gf, _diag = generate_field(uniform_data, c, bundle.field,
                                               bundle.jitter)
            rows.append(extract_features(field, patches, sampling.feature_mode))
    np.testing.assert_array_equal(cached, np.asarray(rows))


def test_patch_layout_is_reproducible(sampling: SamplingConfig,
                                      uniform_data: DataConfig) -> None:
    random_layout = sampling.model_copy(
        update={"layout": "random_uniform", "n_patches": 7})
    first = build_patches(random_layout, uniform_data.plane,
                          spawn_streams(19)["patches"])
    second = build_patches(random_layout, uniform_data.plane,
                           spawn_streams(19)["patches"])
    assert first == second


def test_full_run_is_bit_identical(small_run: RunConfig) -> None:
    """Exit criterion 3 of Phase 1, at test scale.

    Compares the trial TABLE, not the provenance sidecar: provenance carries a
    wall-clock timestamp and is deliberately not byte-stable (storage/io.py).
    """
    from core.evaluation import run_experiment
    first = run_experiment(small_run)
    second = run_experiment(small_run)
    np.testing.assert_array_equal(first.trials["f"].to_numpy(),
                                  second.trials["f"].to_numpy())
    assert first.report.curvature_index == second.report.curvature_index
