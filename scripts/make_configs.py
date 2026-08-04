#!/usr/bin/env python3
"""Emit the canonical Phase-1 config files into `configs/`.

Why a generator rather than ten hand-written JSON files: the sampling-ablation
configs (plan.md §7.5, control 5) need explicit patch lists computed from zone
geometry, and hand-typing coordinates is exactly the kind of silent error this
project cannot detect from its output. Run this once; commit the JSON it writes.

    python -m scripts.make_configs

Every config it writes is a complete `RunConfig`, so `run_experiment.py` needs
nothing but the path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.config import (
    DataConfig,
    EvalConfig,
    Falloff,
    JitterSpec,
    NoiseSpec,
    PlaneSpec,
    RunConfig,
    SamplingConfig,
    TrainingConfig,
    ZoneKind,
    ZoneSpec,
)
from storage.io import save_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
PLANE = PlaneSpec(height=100, width=100)

# Shared geometry, so the scenarios differ only in the factor under study.
HEAT_GEOMETRY = [("heat1", 25.0, 25.0, 15.0), ("heat2", 70.0, 30.0, 10.0)]
DEAD_GEOMETRY = [("dead1", 30.0, 72.0, 18.0), ("dead2", 75.0, 75.0, 12.0)]
ANTI_GEOMETRY = [("anti1", 50.0, 50.0, 12.0)]
RANDOM_GEOMETRY = [("rand1", 15.0, 85.0, 14.0), ("rand2", 88.0, 12.0, 10.0)]

UNIFORM_SAMPLING = SamplingConfig(
    name="uniform_grid_step10",
    layout="uniform_grid",
    patch_size=5,
    grid_step=10,
    feature_mode="pixels",
)


def _zones(kind: ZoneKind, geometry: list[tuple[str, float, float, float]],
           gain: float) -> list[ZoneSpec]:
    return [
        ZoneSpec(id=zone_id, kind=kind, center_x=x, center_y=y, radius=r,
                 gain=gain, falloff=Falloff.SMOOTHSTEP, falloff_width=3.0)
        for zone_id, x, y, r in geometry
    ]


def _patches_covering(zones: list[ZoneSpec], patch_size: int = 5, step: int = 5,
                      inside: bool = True) -> list[tuple[int, int]]:
    """Grid patches whose CENTRE falls inside (or outside) the given zones.

    Used only to build the sampling-ablation configs. Note this reads zone
    geometry to place patches, which is legitimate here — the ablation exists
    precisely to measure how much placement matters — but it is why those configs
    are generated once and committed rather than computed at run time, where it
    would look like the observer peeking at the world (invariant I1).
    """
    half = patch_size / 2.0
    chosen: list[tuple[int, int]] = []
    for row in range(0, PLANE.height - patch_size + 1, step):
        for col in range(0, PLANE.width - patch_size + 1, step):
            centre_y, centre_x = row + half, col + half
            hit = any(np.hypot(centre_x - z.center_x, centre_y - z.center_y) <= z.radius
                      for z in zones)
            if hit == inside:
                chosen.append((row, col))
    return chosen


def build_configs() -> dict[str, RunConfig]:
    """All canonical Phase-1 run configs, keyed by filename stem."""
    heat = _zones(ZoneKind.HEAT, HEAT_GEOMETRY, gain=2.0)
    strong = _zones(ZoneKind.STRONG, HEAT_GEOMETRY, gain=1.0)
    dead = _zones(ZoneKind.DEAD, DEAD_GEOMETRY, gain=0.05)
    anti = _zones(ZoneKind.ANTI, ANTI_GEOMETRY, gain=-0.8)
    random_zones = _zones(ZoneKind.RANDOM, RANDOM_GEOMETRY, gain=1.0)

    low_noise = NoiseSpec(pixel_model="bernoulli")
    heat_dead_data = DataConfig(name="heat_dead", plane=PLANE,
                                zones=heat + dead, noise=low_noise)

    configs: dict[str, RunConfig] = {}

    # ---- controls (plan.md §7.5) --------------------------------------------
    configs["control_simple"] = RunConfig(
        data=DataConfig(name="control_simple", plane=PLANE, noise=low_noise),
        sampling=UNIFORM_SAMPLING,
    )
    configs["control_all_dead"] = RunConfig(
        data=DataConfig(name="control_all_dead", plane=PLANE,
                        background_gain=0.0, noise=low_noise),
        sampling=UNIFORM_SAMPLING,
    )
    configs["control_all_random"] = RunConfig(
        data=DataConfig(
            name="control_all_random", plane=PLANE, noise=low_noise,
            # One circle larger than the plane diagonal masks every pixel.
            zones=[ZoneSpec(id="rand_all", kind=ZoneKind.RANDOM, center_x=50.0,
                            center_y=50.0, radius=200.0, falloff=Falloff.HARD)],
        ),
        sampling=UNIFORM_SAMPLING,
    )
    configs["control_label_shuffle"] = RunConfig(
        data=heat_dead_data.model_copy(update={"name": "control_label_shuffle"}),
        sampling=UNIFORM_SAMPLING,
        training=TrainingConfig(shuffle_labels=True),
    )

    # ---- substantive scenarios (plan.md §7.6) -------------------------------
    configs["scenario_heat_dead"] = RunConfig(data=heat_dead_data,
                                              sampling=UNIFORM_SAMPLING)
    configs["scenario_strong_dead"] = RunConfig(
        # The clip-free counterpart: background_gain 0.4 with strong gain 1.0
        # gives the same 2.5x zone-to-background ratio without ever saturating.
        data=DataConfig(name="strong_dead", plane=PLANE, background_gain=0.4,
                        zones=strong + dead, noise=low_noise),
        sampling=UNIFORM_SAMPLING,
    )
    configs["scenario_heat_dead_anti"] = RunConfig(
        data=DataConfig(name="heat_dead_anti", plane=PLANE,
                        zones=heat + dead + anti, noise=low_noise),
        sampling=UNIFORM_SAMPLING,
    )
    configs["scenario_heat_dead_anti_random"] = RunConfig(
        data=DataConfig(name="heat_dead_anti_random", plane=PLANE,
                        zones=heat + dead + anti + random_zones, noise=low_noise),
        sampling=UNIFORM_SAMPLING,
    )
    configs["scenario_heat_dead_jitter"] = RunConfig(
        data=heat_dead_data.model_copy(update={
            "name": "heat_dead_jitter",
            "jitter": JitterSpec(enabled=True, center_sigma_px=2.0,
                                 radius_sigma_px=1.0, gain_sigma=0.0),
        }),
        sampling=UNIFORM_SAMPLING,
    )

    # ---- sampling ablation (plan.md §7.5, control 5) ------------------------
    for label, patch_list in {
        "heat_only": _patches_covering(heat, inside=True),
        "dead_only": _patches_covering(dead, inside=True),
    }.items():
        configs[f"ablation_{label}"] = RunConfig(
            data=heat_dead_data.model_copy(update={"name": f"heat_dead__{label}"}),
            sampling=SamplingConfig(name=f"manual_{label}", layout="manual",
                                    patch_size=5, patches=patch_list,
                                    feature_mode="pixels"),
        )

    return configs


def main() -> None:
    configs = build_configs()
    for stem, cfg in configs.items():
        path = save_config(cfg, CONFIG_DIR, name=stem)
        n_patches = (len(cfg.sampling.patches) if cfg.sampling.layout == "manual"
                     else "grid")
        print(f"wrote {path.name:44s} zones={len(cfg.data.zones):2d} "
              f"patches={n_patches}")
    print(f"\n{len(configs)} configs in {CONFIG_DIR}")
    print("Default evaluation grid: "
          f"{len(EvalConfig().contrast_grid)} contrast levels")


if __name__ == "__main__":
    main()
