"""Named random-number streams, derived from one master seed.

Responsibility
--------------
Turn a single integer seed into a fixed set of independent `Generator`s, one per
role. Nothing else in the codebase constructs a generator.

Explicitly NOT this module's job
--------------------------------
Deciding how many draws each stream makes. That is up to the consumer.

Why named streams (CLAUDE.md §7)
--------------------------------
Sharing one stream across roles couples nuisance sources: change the number of
pixel-noise draws and every jitter value shifts too, which makes single-factor
ablations impossible. Deriving children by position in a fixed tuple keeps each
role's draws stable as long as names are only ever APPENDED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.random import Generator

# Order matters for reproducibility: APPEND new names, never insert or reorder,
# because a stream's identity is its position in this tuple.
STREAM_NAMES: tuple[str, ...] = (
    "field_train",     # pixel noise for training trials
    "jitter_train",    # zone geometry jitter for training trials
    "field_eval",      # pixel noise for evaluation trials
    "jitter_eval",     # zone geometry jitter for evaluation trials
    "patches",         # patch layout placement (drawn once, fixed per model)
    "labels",          # label-shuffle control only; unused otherwise
)


@dataclass(frozen=True)
class RngBundle:
    """The generators one dataset build is allowed to touch.

    Training and evaluation get *different* bundles, which is what guarantees no
    evaluation trial is a replay of a training trial (invariant I4). Streams that
    are not per-trial — `patches`, `labels` — deliberately sit outside the
    bundle so they cannot be consumed inside a trial loop.
    """

    field: Generator
    jitter: Generator


def spawn_streams(master_seed: int) -> dict[str, Generator]:
    """Derive one independent generator per name in `STREAM_NAMES`.

    Parameters
    ----------
    master_seed : int
        The only seed a caller ever supplies.

    Returns
    -------
    dict mapping each name in STREAM_NAMES to its own Generator.
    """
    seed_sequence = np.random.SeedSequence(master_seed)
    children = seed_sequence.spawn(len(STREAM_NAMES))
    return {
        name: np.random.default_rng(child)
        for name, child in zip(STREAM_NAMES, children, strict=True)
    }


def bundle_for(
    streams: dict[str, Generator], role: Literal["train", "eval"]
) -> RngBundle:
    """Pick the field/jitter pair belonging to `role`."""
    if role not in ("train", "eval"):
        raise ValueError(f"role must be 'train' or 'eval', got {role!r}")
    return RngBundle(field=streams[f"field_{role}"], jitter=streams[f"jitter_{role}"])
