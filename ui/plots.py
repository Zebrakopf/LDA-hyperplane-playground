"""Interactive Plotly figures for the Streamlit app.

Responsibility
--------------
Turn a `GainField`, a trial table or a `LinearityReport` into a figure. Nothing
here computes anything scientific — if a number is needed it comes from
`core/evaluation.py`.

Explicitly NOT this module's job
--------------------------------
Any simulation, fitting or aggregation. Also not the headless figures: those are
static matplotlib in `scripts/figures.py`, kept separate so a batch run never
imports a UI stack (docs/decisions.md D7).

Pipeline position
-----------------
    ExperimentResult / GainField  ->  [this module]  ->  st.plotly_chart
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.config import DataConfig, ZoneKind
from core.data import GainField
from core.evaluation import LinearityReport, WeightAttribution
from core.sampling import Patch

FloatArray = npt.NDArray[np.float64]

# One colour per zone kind, used for outlines on the canvas and nowhere else.
# Chosen to stay distinguishable against a greyscale pixel field, which rules out
# grey and near-white.
ZONE_COLOURS: dict[str, str] = {
    ZoneKind.HEAT.value: "#e2571e",      # orange: amplified, saturating
    ZoneKind.STRONG.value: "#1f9d55",    # green: amplified, clip-free
    ZoneKind.DEAD.value: "#2f6fb5",      # blue: weak
    ZoneKind.ANTI.value: "#8e44ad",      # purple: inverted
    ZoneKind.RANDOM.value: "#c0392b",    # red: masked
}
PATCH_COLOUR = "#f5d000"                 # yellow: the observer, never a zone
COLOUR_LDA = "#1f6fb4"
COLOUR_ORACLE = "#c8621f"
COLOUR_REFERENCE = "#8a8a8a"


def plane_figure(field: FloatArray, gf: GainField, data: DataConfig,
                 patches: list[Patch], *, show_zones: bool = True,
                 show_patches: bool = True, show_gain: bool = False) -> go.Figure:
    """The canvas: one realised pixel field with zone and patch overlays.

    Parameters
    ----------
    show_gain : bool
        Display the coupling gain field instead of a pixel realisation. Useful
        because the gain field is what the zones actually *mean*, while a single
        realisation at mid contrast can look like uniform noise.

    The overlays are deliberately different shapes and colour families: circles in
    kind-specific colours are data zones (the world), yellow squares are sampling
    patches (the observer). Conflating the two is the failure mode invariant I1
    exists to prevent, so the figure never lets them look alike.
    """
    if show_gain:
        limit = max(1.0, float(np.abs(gf.a).max()))
        image = go.Heatmap(z=gf.a, colorscale="RdBu_r", zmin=-limit, zmax=limit,
                           colorbar={"title": "gain a", "thickness": 12},
                           hovertemplate="x=%{x} y=%{y}<br>gain=%{z:.3f}<extra></extra>")
    else:
        image = go.Heatmap(z=field, colorscale="Greys_r", zmin=0.0, zmax=1.0,
                           colorbar={"title": "pixel", "thickness": 12},
                           hovertemplate="x=%{x} y=%{y}<br>value=%{z:.3f}<extra></extra>")

    figure = go.Figure(data=[image])

    # An invisible, clickable scatter lattice over every pixel centre.
    #
    # This is not decoration and it is not optional: Plotly's point-selection API
    # only fires for scatter-like traces. A Heatmap is NOT selectable, so
    # `st.plotly_chart(on_select=...)` over a bare heatmap returns nothing however
    # hard the user clicks — the only thing left responding is drag and zoom, which
    # is exactly what it looks like when the feature is "broken". The lattice gives
    # the selection API something it will actually report, at one-pixel resolution.
    #
    # A headless AppTest cannot deliver a Plotly selection event, so this path is
    # verified by driving a real browser (see docs/decisions.md D12).
    stride = 1 if max(data.plane.height, data.plane.width) <= 160 else 2
    rows = np.arange(0, data.plane.height, stride)
    cols = np.arange(0, data.plane.width, stride)
    mesh_x, mesh_y = np.meshgrid(cols, rows)
    figure.add_trace(go.Scattergl(
        x=mesh_x.ravel(), y=mesh_y.ravel(), mode="markers",
        # Size 9 with opacity 0: invisible, but a forgiving hit area so a click
        # lands on the pixel the user aimed at rather than requiring precision.
        marker={"size": 9, "opacity": 0.0, "color": "#000000"},
        # Both selection states pinned to invisible. Plotly otherwise applies its
        # own selected/unselected opacities once a point is picked, which makes
        # the lattice show up as a grey smudge around wherever you last clicked.
        selected={"marker": {"opacity": 0.0}},
        unselected={"marker": {"opacity": 0.0}},
        hovertemplate="click to place here<br>x=%{x} y=%{y}<extra></extra>",
        showlegend=False, name="click target"))

    if show_patches:
        for patch in patches:
            figure.add_shape(type="rect", x0=patch.col - 0.5, y0=patch.row - 0.5,
                             x1=patch.col + patch.size - 0.5,
                             y1=patch.row + patch.size - 0.5,
                             line={"color": PATCH_COLOUR, "width": 1.2},
                             fillcolor="rgba(0,0,0,0)", layer="above")
    if show_zones:
        for zone in data.zones:
            colour = ZONE_COLOURS.get(zone.kind.value, "#000000")
            figure.add_shape(type="circle",
                             x0=zone.center_x - zone.radius, y0=zone.center_y - zone.radius,
                             x1=zone.center_x + zone.radius, y1=zone.center_y + zone.radius,
                             line={"color": colour, "width": 2},
                             fillcolor="rgba(0,0,0,0)", layer="above")
            figure.add_annotation(x=zone.center_x, y=zone.center_y,
                                  text=f"{zone.id}<br>{zone.kind.value} g={zone.gain:g}",
                                  showarrow=False, font={"color": colour, "size": 9},
                                  bgcolor="rgba(255,255,255,0.55)")

    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 4, "b": 0}, height=460,
        # dragmode False and clickmode event+select together mean a press is
        # interpreted as "select this point", never as the start of a pan or a
        # zoom rectangle. Scroll zoom is disabled on the Streamlit side via the
        # chart `config`, because a stray trackpad scroll over the canvas used to
        # rescale the plane and make the coordinates meaningless.
        dragmode=False, clickmode="event+select",
        xaxis={"range": [-0.5, data.plane.width - 0.5], "constrain": "domain",
               "showgrid": False, "title": "x"},
        # Row 0 at the top, matching array indexing, so a click's y maps to a row.
        yaxis={"range": [data.plane.height - 0.5, -0.5], "scaleanchor": "x",
               "showgrid": False, "title": "y (row)"},
    )
    return figure


def mapping_figure(report: LinearityReport,
                   oracle: LinearityReport | None = None) -> go.Figure:
    """f(c): mean curve with a +/-1 SD band, the affine fit, and the oracle.

    The oracle goes on a secondary axis on purpose. Decision values have arbitrary
    units, so comparing raw `f` between two readouts is meaningless — only the
    SHAPE is comparable (CLAUDE.md §11, trap 8).
    """
    curve = report.mean_by_c
    c = curve["c"].to_numpy()
    mean_f = curve["mean_f"].to_numpy()
    sd_f = np.nan_to_num(curve["sd_f"].to_numpy())

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(go.Scatter(
        x=np.concatenate([c, c[::-1]]),
        y=np.concatenate([mean_f + sd_f, (mean_f - sd_f)[::-1]]),
        fill="toself", fillcolor="rgba(31,111,180,0.18)", line={"width": 0},
        name="±1 SD across trials", hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=c, y=mean_f, mode="lines+markers", name="LDA  f(c)",
                                line={"color": COLOUR_LDA, "width": 2},
                                marker={"size": 5}))
    figure.add_trace(go.Scatter(x=c, y=report.beta0 + report.beta1 * c, mode="lines",
                                name="affine fit", line={"color": COLOUR_REFERENCE,
                                                         "dash": "dash", "width": 1.5}))
    if oracle is not None:
        figure.add_trace(go.Scatter(
            x=oracle.mean_by_c["c"], y=oracle.mean_by_c["mean_f"], mode="lines",
            name="oracle (right axis)",
            line={"color": COLOUR_ORACLE, "dash": "dashdot", "width": 1.8}),
            secondary_y=True)
        figure.update_yaxes(title_text="oracle decision value", secondary_y=True,
                            showgrid=False)

    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 6, "b": 0}, height=340,
        xaxis_title="latent contrast c  (ground truth)",
        legend={"orientation": "h", "y": -0.22},
    )
    figure.update_yaxes(title_text="LDA decision value f", secondary_y=False)
    return figure


def scatter_figure(trials: pd.DataFrame, max_points: int = 4000) -> go.Figure:
    """Raw per-trial decision values, so the spread is visible as points.

    Subsamples above `max_points` purely for browser responsiveness; the number
    plotted is stated in the axis title so the thinning is never invisible.
    """
    shown = trials
    if len(trials) > max_points:
        # Deterministic stride rather than a random sample: the figure must not
        # change when the page reruns.
        stride = int(np.ceil(len(trials) / max_points))
        shown = trials.iloc[::stride]
    figure = go.Figure(go.Scattergl(
        x=shown["c"], y=shown["f"], mode="markers",
        marker={"size": 3, "opacity": 0.35, "color": COLOUR_LDA}, name="trials"))
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 6, "b": 0}, height=300, showlegend=False,
        xaxis_title=f"latent contrast c   ({len(shown)} of {len(trials)} trials shown)",
        yaxis_title="f")
    return figure


def residual_figure(report: LinearityReport) -> go.Figure:
    """Mean residual from the affine fit, with a 95% CI: the SHAPE of any failure.

    This is the panel that catches curvature a correlation coefficient hides.
    """
    residuals = report.residual_by_c
    c = residuals["c"].to_numpy()
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=np.concatenate([c, c[::-1]]),
        y=np.concatenate([residuals["ci_hi"], residuals["ci_lo"][::-1]]),
        fill="toself", fillcolor="rgba(31,111,180,0.22)", line={"width": 0},
        name="95% CI", hoverinfo="skip"))
    figure.add_hline(y=0.0, line={"color": COLOUR_REFERENCE, "width": 1})
    figure.add_trace(go.Scatter(x=c, y=residuals["mean_residual"], mode="lines+markers",
                                line={"color": COLOUR_LDA}, name="mean residual"))
    figure.update_layout(margin={"l": 0, "r": 0, "t": 6, "b": 0}, height=280,
                         showlegend=False, xaxis_title="latent contrast c",
                         yaxis_title="residual from affine fit")
    return figure


def spread_figure(report: LinearityReport) -> go.Figure:
    """SD(f|c) for H3, with clip activity so curvature can be attributed."""
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(go.Scatter(x=report.sd_by_c["c"], y=report.sd_by_c["sd_f"],
                                mode="lines+markers", name="SD(f | c)",
                                line={"color": COLOUR_LDA}))
    figure.add_trace(go.Scatter(x=report.clip_by_c["c"],
                                y=report.clip_by_c["mean_clip_fraction"],
                                mode="lines", name="clipped pixel fraction",
                                line={"color": COLOUR_ORACLE, "dash": "dot"}),
                     secondary_y=True)
    figure.update_layout(margin={"l": 0, "r": 0, "t": 6, "b": 0}, height=280,
                         xaxis_title="latent contrast c",
                         legend={"orientation": "h", "y": -0.28})
    figure.update_yaxes(title_text="SD of f across trials", secondary_y=False)
    figure.update_yaxes(title_text="clip fraction", secondary_y=True, showgrid=False,
                        rangemode="tozero")
    return figure


def histogram_figure(trials: pd.DataFrame, n_levels: int = 5) -> go.Figure:
    """Overlaid distributions of `f` at a few contrast levels.

    Makes overlap between adjacent contrasts visible, which the mean curve hides:
    two levels can have well-separated means and still be indistinguishable on a
    single trial.
    """
    levels = np.unique(trials["c"])
    picked = levels[np.linspace(0, len(levels) - 1, min(n_levels, len(levels))).astype(int)]
    figure = go.Figure()
    for level in picked:
        subset = trials.loc[trials["c"] == level, "f"]
        figure.add_trace(go.Histogram(x=subset, name=f"c = {level:.2f}", opacity=0.55,
                                      nbinsx=40))
    figure.update_layout(barmode="overlay", margin={"l": 0, "r": 0, "t": 6, "b": 0},
                         height=280, xaxis_title="f", yaxis_title="trials",
                         legend={"orientation": "h", "y": -0.28})
    return figure


def weight_figure(attribution: WeightAttribution) -> go.Figure:
    """Learned weights beside the true coupling gain.

    The single most diagnostic panel in the app: it answers "did the model latch
    onto the pixels that actually carry the signal?" Large weight inside a random
    zone means overfitting to nuisance variance, or a broken index mapping.
    """
    weights = attribution.w_pixels
    limit = float(np.nanmax(np.abs(weights))) if np.isfinite(weights).any() else 1.0
    gain_limit = max(1.0, float(np.abs(attribution.true_gain).max()))

    figure = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                           subplot_titles=("learned weights w (NaN = unsampled)",
                                           "true coupling gain a"))
    figure.add_trace(go.Heatmap(z=weights, colorscale="RdBu_r", zmin=-limit, zmax=limit,
                                showscale=False,
                                hovertemplate="x=%{x} y=%{y}<br>w=%{z:.4f}<extra></extra>"),
                     row=1, col=1)
    figure.add_trace(go.Heatmap(z=attribution.true_gain, colorscale="RdBu_r",
                                zmin=-gain_limit, zmax=gain_limit,
                                colorbar={"title": "gain", "thickness": 12},
                                hovertemplate="x=%{x} y=%{y}<br>a=%{z:.3f}<extra></extra>"),
                     row=1, col=2)
    for column in (1, 2):
        figure.update_yaxes(autorange="reversed", scaleanchor=f"x{column if column > 1 else ''}",
                            row=1, col=column)
    figure.update_layout(margin={"l": 0, "r": 0, "t": 28, "b": 0}, height=330)
    return figure


def comparison_figure(curves: dict[str, LinearityReport]) -> go.Figure:
    """Several models' f(c) on one axis, each affinely rescaled to [0, 1].

    Rescaling is mandatory, not cosmetic: decision values have arbitrary units, so
    an unscaled overlay would show differences that are purely units. After
    rescaling, any visible difference is a difference in SHAPE — which is the only
    thing worth comparing across models.
    """
    figure = go.Figure()
    for label, report in curves.items():
        curve = report.mean_by_c
        values = curve["mean_f"].to_numpy()
        span = float(values.max() - values.min())
        normalised = (values - values.min()) / span if span > 0 else values * 0.0
        figure.add_trace(go.Scatter(x=curve["c"], y=normalised, mode="lines+markers",
                                    name=label, marker={"size": 4}))
    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfectly affine",
                                line={"color": COLOUR_REFERENCE, "dash": "dash"}))
    figure.update_layout(margin={"l": 0, "r": 0, "t": 6, "b": 0}, height=360,
                         xaxis_title="latent contrast c",
                         yaxis_title="f, rescaled to [0, 1]",
                         legend={"orientation": "h", "y": -0.22})
    return figure
