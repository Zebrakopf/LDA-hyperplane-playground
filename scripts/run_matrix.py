#!/usr/bin/env python3
"""Sweep the Phase-1 factor matrix (plan.md §7.6) over one or more base configs.

    # see what would run, and how long it will take, without running it
    python -m scripts.run_matrix --dry-run

    # the pixel-model x noise factor only, one seed, no cross-validation
    python -m scripts.run_matrix --factors pixel_noise --seeds 0 --no-cv

    # the full grid on the substantive scenarios, five seeds
    python -m scripts.run_matrix --seeds 0 1 2 3 4

Factors are applied ONE AT A TIME to the base config, not fully crossed: a full
cross of every factor is thousands of runs and answers no question that the
one-at-a-time sweep leaves open. Each cell's deviation from the base is printed,
so nothing is silently dropped.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core.config import JitterSpec, NoiseSpec, RunConfig, TrainingConfig
from core.evaluation import run_experiment
from storage.io import load_run_config, provenance, run_identifiers, save_result

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = REPO_ROOT / "configs" / "scenario_heat_dead.json"

# Noise levels per pixel model (plan.md §7.6). Bernoulli has NO noise parameter —
# its variance is theta*(1-theta) — so it contributes ONE cell, not three. Writing
# the factor as a flat 3x3 grid would silently produce three identical runs.
PIXEL_NOISE_CELLS: list[tuple[str, NoiseSpec]] = [
    ("bernoulli/intrinsic", NoiseSpec(pixel_model="bernoulli")),
    ("clipped_gaussian/low", NoiseSpec(pixel_model="clipped_gaussian", sigma=0.05)),
    ("clipped_gaussian/medium", NoiseSpec(pixel_model="clipped_gaussian", sigma=0.15)),
    ("clipped_gaussian/high", NoiseSpec(pixel_model="clipped_gaussian", sigma=0.30)),
    ("beta/low", NoiseSpec(pixel_model="beta", beta_concentration=100.0)),
    ("beta/medium", NoiseSpec(pixel_model="beta", beta_concentration=25.0)),
    ("beta/high", NoiseSpec(pixel_model="beta", beta_concentration=8.0)),
]

JITTER_CELLS: list[tuple[str, JitterSpec]] = [
    ("jitter/off", JitterSpec(enabled=False)),
    ("jitter/geometry", JitterSpec(enabled=True, center_sigma_px=2.0,
                                   radius_sigma_px=1.0, gain_sigma=0.0)),
    # gain_sigma > 0 is a separate cell because it is a THIRD path to clipping,
    # so it probes jitter-induced saturation rather than jitter as such.
    ("jitter/gain", JitterSpec(enabled=True, center_sigma_px=2.0,
                               radius_sigma_px=1.0, gain_sigma=0.1)),
]

FEATURE_CELLS = [("features/pixels", "pixels"),
                 ("features/patch_mean", "patch_mean"),
                 ("features/patch_mean_std", "patch_mean_std")]

SOLVER_CELLS: list[tuple[str, str, object]] = [
    ("solver/lsqr+auto", "lsqr", "auto"),
    ("solver/svd+none", "svd", None),
]


@dataclass(frozen=True)
class Cell:
    """One point in the sweep: a label and the config it implies."""

    factor: str
    label: str
    config: RunConfig


def build_cells(base: RunConfig, factors: set[str]) -> list[Cell]:
    """Expand the requested factors into concrete run configs."""
    cells: list[Cell] = []
    if "pixel_noise" in factors:
        for label, noise in PIXEL_NOISE_CELLS:
            cfg = base.model_copy(deep=True)
            cfg.data.noise = noise
            cfg.data.name = f"{base.data.name}__{label.replace('/', '_')}"
            cells.append(Cell("pixel_noise", label, cfg))
    if "jitter" in factors:
        for label, jitter in JITTER_CELLS:
            cfg = base.model_copy(deep=True)
            cfg.data.jitter = jitter
            cfg.data.name = f"{base.data.name}__{label.replace('/', '_')}"
            cells.append(Cell("jitter", label, cfg))
    if "features" in factors:
        for label, mode in FEATURE_CELLS:
            cfg = base.model_copy(deep=True)
            cfg.sampling.feature_mode = mode          # type: ignore[assignment]
            cfg.data.name = f"{base.data.name}__{label.replace('/', '_')}"
            cells.append(Cell("features", label, cfg))
    if "solver" in factors:
        for label, solver, shrinkage in SOLVER_CELLS:
            cfg = base.model_copy(deep=True)
            # Both fields in one validated step: with validate_assignment on,
            # setting solver="svd" first would be rejected while shrinkage is
            # still "auto".
            cfg.training = TrainingConfig.model_validate(
                {**cfg.training.model_dump(), "solver": solver,
                 "shrinkage": shrinkage})
            cfg.data.name = f"{base.data.name}__{label.replace('/', '_')}"
            cells.append(Cell("solver", label, cfg))
    return _drop_duplicate_cells(cells)


def _drop_duplicate_cells(cells: list[Cell]) -> list[Cell]:
    """Drop cells whose config is identical to an earlier one.

    Several factor levels equal the base config (`jitter/off`,
    `features/pixels`, `solver/lsqr+auto` on the default base), so without this
    the same experiment runs three times per seed. Compared on everything except
    the display name, which each cell rewrites.
    """
    seen: set[str] = set()
    unique: list[Cell] = []
    for cell in cells:
        signature = cell.config.model_copy(
            update={"data": cell.config.data.model_copy(update={"name": ""})}
        ).model_dump_json()
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(cell)
    return unique


ALL_FACTORS = ("pixel_noise", "jitter", "features", "solver")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, nargs="+", default=[DEFAULT_BASE])
    parser.add_argument("--factors", nargs="+", choices=ALL_FACTORS,
                        default=list(ALL_FACTORS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--no-cv", action="store_true",
                        help="skip cross-validation (saves cv_folds LDA fits per run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the cells and stop")
    args = parser.parse_args(argv)

    cells: list[Cell] = []
    for base_path in args.base:
        base = load_run_config(base_path)
        cells.extend(build_cells(base, set(args.factors)))

    total = len(cells) * len(args.seeds)
    print(f"{len(cells)} cells x {len(args.seeds)} seeds = {total} runs")
    for cell in cells:
        print(f"  [{cell.factor:12s}] {cell.label:26s} -> {cell.config.data.name}")
    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return 0

    rows: list[dict[str, object]] = []
    for index, cell in enumerate(cells, start=1):
        for seed in args.seeds:
            cfg = cell.config.model_copy(deep=True)
            cfg.master_seed = seed
            if args.no_cv:
                cfg.training.cv_folds = 0
            identifiers = run_identifiers(cfg)
            result = run_experiment(cfg, provenance=provenance(), ids=identifiers)
            save_result(result, args.results_dir)
            report = result.report
            rows.append({
                "factor": cell.factor,
                "cell": cell.label,
                "seed": seed,
                **report.summary_row(),
                "oracle_verdict": (result.oracle_report.verdict
                                   if result.oracle_report else None),
                "oracle_kappa": (result.oracle_report.curvature_index
                                 if result.oracle_report else None),
            })
            print(f"[{index}/{len(cells)}] {cell.label:26s} seed={seed}  "
                  f"verdict={report.verdict:10s} "
                  f"κ={report.curvature_index if report.curvature_index is None else f'{report.curvature_index:.4f}'}  "
                  f"sd_ratio={report.sd_ratio:.2f}  "
                  f"clip={report.clip_by_c['mean_clip_fraction'].mean():.4f}",
                  flush=True)

    table = pd.DataFrame(rows)
    path = args.results_dir / "matrix_table.csv"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)

    print("\n" + "=" * 110)
    columns = ["factor", "cell", "seed", "verdict", "kappa", "sd_ratio",
               "n_zero_variance_levels", "calibration_rmse", "mean_clip_fraction",
               "oracle_verdict", "oracle_kappa"]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(table[columns].to_string(index=False,
                                      float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
