#!/usr/bin/env python3
"""Run one or more experiments from config files. The Phase-1 entrypoint.

    # one config, default seed
    python -m scripts.run_experiment configs/scenario_heat_dead.json

    # every config, five seeds each, with figures
    python -m scripts.run_experiment configs/*.json --seeds 0 1 2 3 4 --figures

    # the mandatory control battery, quickly
    python -m scripts.run_experiment configs/control_*.json --quick

Reads configs, runs the pipeline, prints the diagnostics, and writes artefacts to
`results/` and `models/`. Everything scientific happens in `core/`; this file only
orchestrates and reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core.evaluation import ExperimentResult, run_experiment
from storage.io import (
    load_run_config,
    provenance,
    run_identifiers,
    save_model,
    save_result,
)
from storage.registry import Registry, RegistryEntry

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "results"
DEFAULT_MODELS = REPO_ROOT / "models"

# --quick shrinks the trial counts for a smoke test. Numbers chosen to keep every
# diagnostic computable (>= 2 trials per level for an SD, >= 4 for the cubic fit)
# while running in seconds. NOT valid for reporting: plan.md §1.3 pre-registers
# 200 trials per level and the verdict thresholds assume it.
QUICK_N_PER_CLASS = 60
QUICK_N_PER_CONTRAST = 25


def describe(result: ExperimentResult) -> str:
    """Human-readable summary of one run, in the register CLAUDE.md §10 requires.

    Reports kappa and the residual pattern next to r, never r alone, and always
    names the spread.
    """
    report = result.report
    diagnostics = result.model.diagnostics
    kappa = "n/a (degenerate)" if report.curvature_index is None \
        else f"{report.curvature_index:.4f}"
    calibration = ("n/a" if report.calibration is None
                   else f"{report.calibration['rmse'].mean():.4f}")
    oracle = ("not computed" if result.oracle_report is None else
              f"{result.oracle_report.verdict} "
              f"(κ = {result.oracle_report.curvature_index if result.oracle_report.curvature_index is None else f'{result.oracle_report.curvature_index:.4f}'})")

    lines = [
        f"── {result.config.data.name}  [seed {result.config.master_seed}] "
        f"run {result.run_id}",
        f"   OBSERVATION  verdict={report.verdict}   κ={kappa}   "
        f"ρ={report.spearman_rho:.4f}   r={report.pearson_r:.4f}   R²={report.r2:.4f}",
        f"                slope β₁={report.beta1:.4f} "
        f"CI[{report.beta1_ci[0]:.4f}, {report.beta1_ci[1]:.4f}]"
        f"{'  <- covers 0' if report.beta1_ci[0] <= 0 <= report.beta1_ci[1] else ''}",
        f"                nested F quadratic p={report.nested_f.get('p_quadratic'):.3g}"
        f"   monotonicity violations={report.monotonicity_violations}",
        f"                spread across trials: SD max/min={report.sd_ratio:.3f} "
        f"({'homoscedastic' if report.homoscedastic else 'HETEROSCEDASTIC'})"
        f"   calibration RMSE (in c units)={calibration}",
        f"                clip fraction (mean over grid)="
        f"{report.clip_by_c['mean_clip_fraction'].mean():.4f}"
        f"   clipping reachable by config={result.config.data.clipping_is_reachable()}",
        f"   ORACLE       {oracle}",
        f"   MODEL        n_features={diagnostics.n_features} "
        f"unique_pixels={result.composition.attrs.get('unique_pixels')} "
        f"n_train={diagnostics.n_train}  train_acc={diagnostics.train_accuracy:.3f} "
        f"cv_acc={diagnostics.cv_accuracy:.3f}±{diagnostics.cv_accuracy_sd:.3f} "
        f"d'={diagnostics.d_prime:.2f}",
        f"                cov cond={diagnostics.cov_condition_number:.3g} "
        f"eff_rank={diagnostics.effective_rank} "
        f"top1%_weight_mass={diagnostics.top1pct_weight_mass:.3f}",
        f"   ATTRIBUTION  |w|~|a| r={result.attribution.abs_correlation:.3f}   "
        f"sign agreement={result.attribution.sign_agreement:.3f}",
    ]
    by_kind = result.attribution.mean_signed_weight_by_kind
    for row in by_kind.itertuples():
        lines.append(f"                {row.kind:>11s}: n={row.n_pixels:5d}  "
                     f"mean w={row.mean_signed_weight:+.4f}  "
                     f"true gain={row.mean_true_gain:+.3f}")
    lines.append("   SAMPLED AREA (weight-weighted fraction of sampled pixels)")
    for row in result.composition.itertuples():
        lines.append(f"                {row.kind:>12s}: {row.weight_fraction:.3f}  "
                     f"mean gain={row.mean_gain:+.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", type=Path, help="RunConfig JSON files")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="override master seeds; one run per seed")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--figures", action="store_true",
                        help="also write the four-panel diagnostic PNG")
    parser.add_argument("--no-save", action="store_true",
                        help="print diagnostics only; write nothing")
    parser.add_argument("--quick", action="store_true",
                        help=f"smoke test: {QUICK_N_PER_CLASS} train/class, "
                             f"{QUICK_N_PER_CONTRAST} per contrast. NOT for reporting.")
    args = parser.parse_args(argv)

    if args.quick:
        print("!! --quick: trial counts are below the pre-registered minimum "
              "(plan.md §1.3). Smoke test only.\n")

    summaries: list[dict[str, object]] = []
    registry = Registry(args.results_dir)

    for config_path in args.configs:
        cfg = load_run_config(config_path)
        for seed in (args.seeds if args.seeds is not None else [cfg.master_seed]):
            run_cfg = cfg.model_copy(deep=True)
            run_cfg.master_seed = seed
            if args.quick:
                run_cfg.training.n_per_class = QUICK_N_PER_CLASS
                run_cfg.evaluation.n_per_contrast = QUICK_N_PER_CONTRAST

            identifiers = run_identifiers(run_cfg)
            result = run_experiment(run_cfg, provenance=provenance(), ids=identifiers)
            print(describe(result), "\n", flush=True)

            row = {"config_file": config_path.name, "seed": seed,
                   **{k: v for k, v in identifiers.items()},
                   **result.report.summary_row()}
            if result.oracle_report is not None:
                row["oracle_verdict"] = result.oracle_report.verdict
                row["oracle_kappa"] = result.oracle_report.curvature_index
            summaries.append(row)

            if args.no_save:
                continue
            paths = save_result(result, args.results_dir)
            save_model(result, args.models_dir)
            registry.add(RegistryEntry(
                kind="result", identifier=result.run_id,
                name=f"{run_cfg.data.name}@seed{seed}",
                path=str(paths["trials"].relative_to(args.results_dir)),
                details={"verdict": result.report.verdict,
                         "kappa": result.report.curvature_index,
                         "model_id": result.model_id},
            ))
            if args.figures:
                from scripts.figures import save_figure
                print(f"   figure -> {save_figure(result, args.results_dir).name}\n")

    table = pd.DataFrame(summaries)
    print("=" * 100)
    print("SUMMARY  (κ is the headline; r is never read alone — CLAUDE.md §10)")
    print("=" * 100)
    columns = ["config_file", "seed", "verdict", "kappa", "spearman_rho", "pearson_r",
               "sd_ratio", "homoscedastic", "calibration_rmse", "mean_clip_fraction"]
    if "oracle_verdict" in table.columns:
        columns += ["oracle_verdict", "oracle_kappa"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not args.no_save:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.results_dir / "summary_table.csv"
        table.to_csv(summary_path, index=False)
        print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
