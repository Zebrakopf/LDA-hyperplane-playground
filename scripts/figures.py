"""Static matplotlib figures for Phase-1 headless runs.

Responsibility
--------------
Render the four panels that every result is read through, so a run's conclusion
can be checked by eye as well as by number.

Explicitly NOT this module's job
--------------------------------
Interactive plotting. Plotly figures for the Streamlit app arrive in Phase 2 as
`ui/plots.py`; keeping the headless figures here avoids importing a UI stack into
a batch run (see docs/decisions.md).

The four panels
---------------
1. f(c) with spread band, plus the oracle readout on a second axis
2. residuals from the affine fit — the SHAPE of any failure
3. SD(f|c) and clip activity — the H3 and saturation evidence
4. learned weight map beside the true gain field — the attribution check
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # headless: no display in a batch run or a test
import matplotlib.pyplot as plt
import numpy as np

from core.evaluation import ExperimentResult

# A deliberately small, colour-blind-safe set. Not a brand palette; these figures
# are diagnostic instruments, so the only requirement is that the four series stay
# distinguishable in greyscale print.
COLOUR_LDA = "#1f6fb4"
COLOUR_ORACLE = "#c8621f"
COLOUR_REFERENCE = "#888888"
COLOUR_FILL = "#1f6fb4"


def _panel_mapping(ax: plt.Axes, result: ExperimentResult) -> None:
    """f(c): mean curve, +/-1 SD band, and the affine fit it is judged against."""
    report = result.report
    curve = report.mean_by_c
    c = curve["c"].to_numpy()
    mean_f = curve["mean_f"].to_numpy()
    sd_f = curve["sd_f"].to_numpy()

    ax.fill_between(c, mean_f - sd_f, mean_f + sd_f, color=COLOUR_FILL, alpha=0.18,
                    linewidth=0, label="±1 SD across trials")
    ax.plot(c, mean_f, color=COLOUR_LDA, marker="o", markersize=3, label="LDA  f(c)")
    ax.plot(c, report.beta0 + report.beta1 * c, color=COLOUR_REFERENCE,
            linestyle="--", linewidth=1.2, label="affine fit")

    if result.oracle_report is not None:
        # Oracle is on the same scaled features but its own scale, so it gets a
        # twin axis: comparing raw f across readouts is meaningless (CLAUDE.md
        # §11, trap 8). Only the SHAPE is being compared here.
        twin = ax.twinx()
        oracle_curve = result.oracle_report.mean_by_c
        twin.plot(oracle_curve["c"], oracle_curve["mean_f"], color=COLOUR_ORACLE,
                  linewidth=1.2, linestyle="-.", label="oracle (right axis)")
        twin.set_ylabel("oracle decision value", color=COLOUR_ORACLE, fontsize=8)
        twin.tick_params(axis="y", labelcolor=COLOUR_ORACLE, labelsize=7)
        twin.legend(loc="lower right", fontsize=7, frameon=False)

    kappa = report.curvature_index
    kappa_text = "n/a" if kappa is None else f"{kappa:.3f}"
    ax.set_title(f"verdict: {report.verdict}   κ = {kappa_text}   "
                 f"ρ = {report.spearman_rho:.3f}   r = {report.pearson_r:.3f}",
                 fontsize=9)
    ax.set_xlabel("latent contrast c  (ground truth)")
    ax.set_ylabel("LDA decision value f")
    ax.legend(loc="upper left", fontsize=7, frameon=False)


def _panel_residuals(ax: plt.Axes, result: ExperimentResult) -> None:
    """Mean residual per c: where and how the mapping departs from affine."""
    residuals = result.report.residual_by_c
    c = residuals["c"].to_numpy()
    ax.axhline(0.0, color=COLOUR_REFERENCE, linewidth=1.0)
    ax.fill_between(c, residuals["ci_lo"], residuals["ci_hi"], color=COLOUR_FILL,
                    alpha=0.25, linewidth=0, label="95% CI")
    ax.plot(c, residuals["mean_residual"], color=COLOUR_LDA, marker="o", markersize=3)
    ax.set_title("residual from the affine fit  (systematic curvature)", fontsize=9)
    ax.set_xlabel("latent contrast c")
    ax.set_ylabel("mean residual")
    ax.legend(loc="best", fontsize=7, frameon=False)


def _panel_spread(ax: plt.Axes, result: ExperimentResult) -> None:
    """SD(f|c) for H3, with clip activity on a twin axis to attribute curvature."""
    report = result.report
    ax.plot(report.sd_by_c["c"], report.sd_by_c["sd_f"], color=COLOUR_LDA,
            marker="o", markersize=3, label="SD(f | c)")
    ax.set_xlabel("latent contrast c")
    ax.set_ylabel("SD of f across trials")
    verdict = "homoscedastic" if report.homoscedastic else "HETEROSCEDASTIC"
    ax.set_title(f"spread: {verdict}   max/min = {report.sd_ratio:.2f}", fontsize=9)

    twin = ax.twinx()
    twin.plot(report.clip_by_c["c"], report.clip_by_c["mean_clip_fraction"],
              color=COLOUR_ORACLE, linestyle=":", label="clipped pixel fraction")
    twin.set_ylabel("clip fraction", color=COLOUR_ORACLE, fontsize=8)
    twin.tick_params(axis="y", labelcolor=COLOUR_ORACLE, labelsize=7)
    twin.set_ylim(bottom=0.0)

    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="best", fontsize=7, frameon=False)


def _panel_attribution(ax_left: plt.Axes, ax_right: plt.Axes,
                       result: ExperimentResult) -> None:
    """Learned weight map beside the true coupling field."""
    attribution = result.attribution
    weights = attribution.w_pixels
    limit = np.nanmax(np.abs(weights)) if np.isfinite(weights).any() else 1.0
    ax_left.imshow(weights, cmap="RdBu_r", vmin=-limit, vmax=limit,
                   interpolation="nearest")
    ax_left.set_title(f"learned weights   |w|·|a| r = {attribution.abs_correlation:.2f}"
                      f"   sign agree = {attribution.sign_agreement:.2f}", fontsize=8)
    ax_left.set_xticks([])
    ax_left.set_yticks([])

    gain = attribution.true_gain
    gain_limit = max(1.0, float(np.abs(gain).max()))
    image = ax_right.imshow(gain, cmap="RdBu_r", vmin=-gain_limit, vmax=gain_limit,
                            interpolation="nearest")
    ax_right.set_title("true coupling gain a (ground truth)", fontsize=8)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    plt.colorbar(image, ax=ax_right, fraction=0.046)


def figure_for_result(result: ExperimentResult) -> plt.Figure:
    """Build the four-panel diagnostic figure for one run."""
    fig = plt.figure(figsize=(12.0, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 1.0])

    _panel_mapping(fig.add_subplot(grid[0, :]), result)
    _panel_residuals(fig.add_subplot(grid[1, 0]), result)
    _panel_spread(fig.add_subplot(grid[1, 1]), result)
    _panel_attribution(fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1]), result)

    fig.suptitle(
        f"{result.config.data.name}  |  sampling: {result.config.sampling.name}  |  "
        f"pixel model: {result.config.data.noise.pixel_model}  |  "
        f"seed {result.config.master_seed}  |  run {result.run_id}",
        fontsize=10,
    )
    return fig


def save_figure(result: ExperimentResult, directory: Path) -> Path:
    """Write the diagnostic figure as PNG. Returns the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.config.data.name}__{result.run_id}.png"
    figure = figure_for_result(result)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path
