"""Interactive Plotly figures for the Streamlit app.

Responsibility
--------------
Turn a gain field, a trial table or a `LinearityReport` into a figure. Nothing
here computes anything scientific — if a number is needed it comes from
`core/evaluation.py`.

Explicitly NOT this module's job
--------------------------------
Simulation, fitting or aggregation. Also not the headless figures: those are
static matplotlib in `scripts/figures.py` (docs/decisions.md D7).

Visual rules applied throughout (docs/decisions.md D16)
------------------------------------------------------
* Colours come from one validated categorical palette, assigned by meaning and
  never cycled. Gain maps use a diverging blue <-> red scale with a neutral gray
  at 0 ("does not track c"), never a rainbow.
* ONE y-axis per chart. The first version put the oracle on a second axis and
  clip fraction on another; two scales on one plot invite reading a crossing as
  meaningful. The LDA and oracle curves are now both expressed on a common,
  interpretable scale (see `_affine_normalise`), and clip fraction has its own
  panel.
* Every chart has a hover layer; line charts use a unified crosshair.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.config import DataConfig, ZoneKind
from core.evaluation import LinearityReport, WeightAttribution
from core.sampling import Patch

FloatArray = npt.NDArray[np.float64]

# Validated categorical palette (light-surface steps), fixed order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
    "#e34948")
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
MUTED = "#8a8985"                 # reference lines, recessive ink
GRID = "rgba(128,128,128,0.18)"

# Zone kinds keep one colour each everywhere (canvas, legends, tables). Patches
# get yellow, which no zone uses: the world and the observer never look alike.
ZONE_COLOURS: dict[str, str] = {
    ZoneKind.HEAT.value: ORANGE,
    ZoneKind.STRONG.value: AQUA,
    ZoneKind.DEAD.value: BLUE,
    ZoneKind.ANTI.value: VIOLET,
    ZoneKind.RANDOM.value: RED,
}
PATCH_COLOUR = YELLOW
COLOUR_LDA = BLUE
COLOUR_ORACLE = ORANGE

# Diverging scale for gain: blue (tracks c) <- gray (does not) -> red (inverted).
# Anti-correlation is the "opposite" pole, so it takes the warm arm.
GAIN_SCALE = [
    [0.0, "#b0302f"], [0.25, "#e87d7c"], [0.5, "#f0efec"],
    [0.75, "#6da7ec"], [1.0, "#184f95"],
]
# Ordinal ramp for "which contrast level" (histograms): one hue, light -> dark.
LEVEL_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

FONT = dict(family="Inter, 'Source Sans Pro', system-ui, sans-serif", size=12)


def _style(figure: go.Figure, height: int, *, legend: bool = True) -> go.Figure:
    """Shared layout: transparent surface, recessive grid, compact margins."""
    figure.update_layout(
        height=height, font=FONT, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=FONT),
    )
    figure.update_xaxes(gridcolor=GRID, zeroline=False, showline=False)
    figure.update_yaxes(gridcolor=GRID, zeroline=False, showline=False)
    return figure


def _affine_normalise(report: LinearityReport) -> tuple[FloatArray, FloatArray, bool]:
    """Express a mean curve in units where its own affine fit runs 0 -> 1.

        g(c) = (f(c) - fit(c_min)) / (fit(c_max) - fit(c_min))

    On this scale a perfectly affine readout lies on the diagonal, so curvature
    is visible as distance from it, and two readouts with unrelated raw units
    (LDA and oracle) can share ONE axis honestly. Returns (c, g, ok); ok is False
    for a degenerate report, whose slope is zero and cannot normalise.
    """
    curve = report.mean_by_c
    c = curve["c"].to_numpy()
    if report.verdict == "degenerate" or report.beta1 == 0.0:
        return c, curve["mean_f"].to_numpy(), False
    lo = report.beta0 + report.beta1 * c.min()
    span = report.beta1 * (c.max() - c.min())
    return c, (curve["mean_f"].to_numpy() - lo) / span, True


# --------------------------------------------------------------------------- #
# The canvas
# --------------------------------------------------------------------------- #

def plane_figure(field: FloatArray, gain: FloatArray, data: DataConfig,
                 patches: list[Patch], *, show_zones: bool = True,
                 show_patches: bool = True, show_gain: bool = False) -> go.Figure:
    """One realised pixel field (or the gain map) with zone and patch overlays."""
    if show_gain:
        limit = max(1.0, float(np.abs(gain).max()))
        image = go.Heatmap(
            z=gain, colorscale=GAIN_SCALE, zmin=-limit, zmax=limit,
            colorbar=dict(title=dict(text="gain", side="right"), thickness=10,
                          outlinewidth=0),
            hovertemplate="x %{x} · y %{y}<br>gain %{z:.2f}<extra></extra>")
    else:
        image = go.Heatmap(
            z=field, colorscale="Greys_r", zmin=0.0, zmax=1.0,
            colorbar=dict(title=dict(text="pixel", side="right"), thickness=10,
                          outlinewidth=0),
            hovertemplate="x %{x} · y %{y}<br>pixel %{z:.2f}<extra></extra>")
    figure = go.Figure(data=[image])

    # Invisible scatter lattice: Plotly only reports point selections for
    # scatter-like traces, so without it clicks never register (D12). Opacity 0
    # in all three states, or the lattice smudges the field after a click.
    stride = 1 if max(data.plane.height, data.plane.width) <= 160 else 2
    mesh_x, mesh_y = np.meshgrid(np.arange(0, data.plane.width, stride),
                                 np.arange(0, data.plane.height, stride))
    figure.add_trace(go.Scattergl(
        x=mesh_x.ravel(), y=mesh_y.ravel(), mode="markers",
        marker=dict(size=9, opacity=0.0, color="#000000"),
        selected=dict(marker=dict(opacity=0.0)),
        unselected=dict(marker=dict(opacity=0.0)),
        hovertemplate="click to place · x %{x} · y %{y}<extra></extra>",
        showlegend=False, name="click target"))

    if show_patches:
        for patch in patches:
            figure.add_shape(type="rect", x0=patch.col - 0.5, y0=patch.row - 0.5,
                             x1=patch.col + patch.size - 0.5,
                             y1=patch.row + patch.size - 0.5,
                             line=dict(color=PATCH_COLOUR, width=1.3),
                             fillcolor="rgba(0,0,0,0)", layer="above")
    if show_zones:
        for zone in data.zones:
            colour = ZONE_COLOURS.get(zone.kind.value, MUTED)
            figure.add_shape(type="circle",
                             x0=zone.center_x - zone.radius,
                             y0=zone.center_y - zone.radius,
                             x1=zone.center_x + zone.radius,
                             y1=zone.center_y + zone.radius,
                             line=dict(color=colour, width=2.5),
                             fillcolor="rgba(0,0,0,0)", layer="above")
            figure.add_annotation(
                x=zone.center_x, y=zone.center_y, showarrow=False,
                text=f"<b>{zone.id}</b><br>{zone.kind.value} · {zone.gain:g}",
                font=dict(size=10, color="#0b0b0b"), bgcolor="rgba(255,255,255,0.8)",
                bordercolor=colour, borderwidth=1, borderpad=2)

    _style(figure, 580, legend=False)
    figure.update_layout(
        dragmode=False, clickmode="event+select",
        xaxis=dict(range=[-0.5, data.plane.width - 0.5], constrain="domain",
                   showgrid=False, title=None),
        # Row 0 at the top, matching array indexing, so a click's y is a row.
        yaxis=dict(range=[data.plane.height - 0.5, -0.5], scaleanchor="x",
                   showgrid=False, title=None),
    )
    return figure


# --------------------------------------------------------------------------- #
# Analysis figures
# --------------------------------------------------------------------------- #

def mapping_figure(report: LinearityReport,
                   oracle: LinearityReport | None = None) -> go.Figure:
    """f(c) against the straight line it is judged against, on ONE axis.

    Both curves are affinely normalised (see `_affine_normalise`), so the dashed
    diagonal IS the affine fit, and any gap between a curve and it is curvature.
    The ±1 SD band is normalised the same way.
    """
    figure = go.Figure()
    c, g, normalised = _affine_normalise(report)
    sd = np.nan_to_num(report.mean_by_c["sd_f"].to_numpy())
    if normalised:
        sd = sd / abs(report.beta1 * (c.max() - c.min()))
        figure.add_trace(go.Scatter(x=[c.min(), c.max()], y=[0, 1], mode="lines",
                                    name="perfectly affine",
                                    line=dict(color=MUTED, dash="dash", width=1.5),
                                    hoverinfo="skip"))
    figure.add_trace(go.Scatter(
        x=np.concatenate([c, c[::-1]]), y=np.concatenate([g + sd, (g - sd)[::-1]]),
        fill="toself", fillcolor="rgba(42,120,214,0.14)", line=dict(width=0),
        name="±1 SD across trials", hoverinfo="skip"))
    figure.add_trace(go.Scatter(
        x=c, y=g, mode="lines+markers", name="LDA readout",
        line=dict(color=COLOUR_LDA, width=2.5), marker=dict(size=8),
        customdata=report.mean_by_c["mean_f"],
        hovertemplate="c %{x:.2f}<br>LDA %{y:.3f} (raw %{customdata:.3g})"
                      "<extra></extra>"))
    if oracle is not None and normalised:
        oc, og, ok = _affine_normalise(oracle)
        if ok:
            figure.add_trace(go.Scatter(
                x=oc, y=og, mode="lines", name="oracle (true gains)",
                line=dict(color=COLOUR_ORACLE, width=2, dash="dot"),
                hovertemplate="c %{x:.2f}<br>oracle %{y:.3f}<extra></extra>"))
    _style(figure, 380)
    figure.update_layout(hovermode="x unified")
    figure.update_xaxes(title="latent contrast c (ground truth)")
    figure.update_yaxes(title=("readout, scaled so its affine fit runs 0 → 1"
                               if normalised else "LDA decision value (raw)"))
    return figure


def scatter_figure(trials: pd.DataFrame, max_points: int = 4000) -> go.Figure:
    """Every trial's decision value — the spread behind the mean curve."""
    shown = trials
    if len(trials) > max_points:
        shown = trials.iloc[::int(np.ceil(len(trials) / max_points))]
    figure = go.Figure(go.Scattergl(
        x=shown["c"], y=shown["f"], mode="markers",
        marker=dict(size=5, opacity=0.35, color=COLOUR_LDA),
        hovertemplate="c %{x:.2f}<br>f %{y:.3g}<extra></extra>"))
    _style(figure, 380, legend=False)
    figure.update_xaxes(title=f"c  ({len(shown)} of {len(trials)} trials)")
    figure.update_yaxes(title="decision value f (raw)")
    return figure


def residual_figure(report: LinearityReport) -> go.Figure:
    """Mean departure from the affine fit, in units of c, with 95% CI.

    Dividing the residual by the slope turns it into "how far off, in c, would
    you be if you read this f through the straight line?" — the practical
    meaning of curvature. A flat line at 0 is affine; a smile is a one-sided
    bend; an S is saturation.
    """
    residuals = report.residual_by_c
    c = residuals["c"].to_numpy()
    scale = report.beta1 if report.verdict != "degenerate" and report.beta1 else 1.0
    unit = "c units" if scale != 1.0 else "raw f units"
    mean = residuals["mean_residual"].to_numpy() / scale
    lo, hi = (residuals["ci_lo"].to_numpy() / scale,
              residuals["ci_hi"].to_numpy() / scale)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    figure = go.Figure()
    figure.add_hline(y=0.0, line=dict(color=MUTED, width=1))
    figure.add_trace(go.Scatter(
        x=np.concatenate([c, c[::-1]]), y=np.concatenate([hi, lo[::-1]]),
        fill="toself", fillcolor="rgba(42,120,214,0.18)", line=dict(width=0),
        name="95% CI", hoverinfo="skip"))
    figure.add_trace(go.Scatter(
        x=c, y=mean, mode="lines+markers", name="mean residual",
        line=dict(color=COLOUR_LDA, width=2.5), marker=dict(size=8),
        hovertemplate="c %{x:.2f}<br>off by %{y:+.4f} " + unit + "<extra></extra>"))
    _style(figure, 300)
    figure.update_layout(hovermode="x unified")
    figure.update_xaxes(title="latent contrast c")
    figure.update_yaxes(title=f"departure from affine ({unit})")
    return figure


def spread_figure(report: LinearityReport) -> go.Figure:
    """SD(f|c) and clip activity as two stacked panels sharing the c axis."""
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           row_heights=[0.65, 0.35])
    figure.add_trace(go.Scatter(
        x=report.sd_by_c["c"], y=report.sd_by_c["sd_f"], mode="lines+markers",
        name="SD of f across trials", line=dict(color=COLOUR_LDA, width=2.5),
        marker=dict(size=8), hovertemplate="c %{x:.2f}<br>SD %{y:.3g}<extra></extra>"),
        row=1, col=1)
    figure.add_trace(go.Bar(
        x=report.clip_by_c["c"], y=report.clip_by_c["mean_clip_fraction"],
        name="clipped pixel fraction", marker=dict(color=ORANGE, cornerradius=4),
        hovertemplate="c %{x:.2f}<br>clipped %{y:.1%}<extra></extra>"),
        row=2, col=1)
    _style(figure, 340)
    figure.update_layout(hovermode="x unified", bargap=0.35)
    figure.update_yaxes(title="SD(f | c)", row=1, col=1)
    figure.update_yaxes(title="clipped", tickformat=".0%", rangemode="tozero",
                        row=2, col=1)
    figure.update_xaxes(title="latent contrast c", row=2, col=1)
    return figure


def histogram_figure(trials: pd.DataFrame, n_levels: int = 5) -> go.Figure:
    """Distributions of f at a few contrast levels — overlap the mean hides.

    Levels are ordered, so they take an ordinal one-hue ramp rather than
    categorical colours.
    """
    levels = np.unique(trials["c"])
    picked = levels[np.linspace(0, len(levels) - 1,
                                min(n_levels, len(levels))).astype(int)]
    figure = go.Figure()
    for colour, level in zip(LEVEL_RAMP, picked):
        figure.add_trace(go.Histogram(
            x=trials.loc[trials["c"] == level, "f"], name=f"c = {level:.2f}",
            opacity=0.7, nbinsx=40, marker=dict(color=colour),
            hovertemplate=f"c = {level:.2f}<br>f %{{x:.3g}}<br>%{{y}} trials"
                          "<extra></extra>"))
    _style(figure, 300)
    figure.update_layout(barmode="overlay")
    figure.update_xaxes(title="decision value f (raw)")
    figure.update_yaxes(title="trials")
    return figure


def weight_figure(attribution: WeightAttribution) -> go.Figure:
    """Learned weights beside the true coupling gain (masked pixels shown as 0)."""
    weights = attribution.w_pixels
    finite = np.abs(weights[np.isfinite(weights)])
    # Colour limit at the 98th percentile of |w|, not the maximum: a handful of
    # extreme pixels otherwise set the scale and wash every other patch out to
    # near-white. Values beyond the limit saturate at the end colour.
    limit = float(np.percentile(finite, 98)) if finite.size else 1.0
    limit = limit or 1.0
    gain_limit = max(1.0, float(np.abs(attribution.true_gain).max()))
    figure = make_subplots(rows=1, cols=2, horizontal_spacing=0.06,
                           subplot_titles=("what the LDA weighted",
                                           "what actually tracks c"))
    figure.add_trace(go.Heatmap(
        z=weights, colorscale=GAIN_SCALE, zmin=-limit, zmax=limit, showscale=False,
        hovertemplate="x %{x} · y %{y}<br>weight %{z:.3g}<extra></extra>"), 1, 1)
    figure.add_trace(go.Heatmap(
        z=attribution.true_gain, colorscale=GAIN_SCALE, zmin=-gain_limit,
        zmax=gain_limit, colorbar=dict(title="gain", thickness=10, outlinewidth=0),
        hovertemplate="x %{x} · y %{y}<br>gain %{z:.2f}<extra></extra>"), 1, 2)
    _style(figure, 360, legend=False)
    for column, anchor in ((1, "x"), (2, "x2")):
        figure.update_yaxes(autorange="reversed", scaleanchor=anchor, showgrid=False,
                            row=1, col=column)
        figure.update_xaxes(showgrid=False, row=1, col=column)
    figure.update_layout(margin=dict(l=8, r=8, t=30, b=8))
    return figure


def comparison_figure(curves: dict[str, LinearityReport]) -> go.Figure:
    """Several models' mapping on one axis, each affinely normalised.

    Raw decision values have arbitrary units, so only shape is comparable; on
    the normalised scale every perfectly affine model lies on the diagonal.
    Colours follow list position in fixed palette order.
    """
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                name="perfectly affine", hoverinfo="skip",
                                line=dict(color=MUTED, dash="dash", width=1.5)))
    for index, (label, report) in enumerate(curves.items()):
        c, g, ok = _affine_normalise(report)
        if not ok:
            continue
        colour = CATEGORICAL[index % len(CATEGORICAL)]
        figure.add_trace(go.Scatter(
            x=c, y=g, mode="lines+markers", name=label,
            line=dict(color=colour, width=2), marker=dict(size=7),
            hovertemplate=f"{label}<br>c %{{x:.2f}} · %{{y:.3f}}<extra></extra>"))
    _style(figure, 380)
    figure.update_xaxes(title="latent contrast c")
    figure.update_yaxes(title="readout, scaled so its affine fit runs 0 → 1")
    return figure
