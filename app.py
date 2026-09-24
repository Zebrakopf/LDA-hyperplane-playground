"""LDA hyperplane check — interactive explorer.

Run locally:

    streamlit run app.py

Responsibility
--------------
Layout and event handling only. Every number comes from `core/`, every figure from
`ui/plots.py`, and all state, callbacks and caching from `ui/controls.py`.

Layout
------
    sidebar   speed preset · world · observer · training · configs
    main      header + train button
              canvas (tools above it)   |   placement + zone editor
              results: verdict, metrics, tabs

State rules (see the ui/controls.py module docstring)
-----------------------------------------------------
Widgets are bound by `key=` and created without `value=`; anything that changes
settings runs in an `on_click` / `on_change` callback. Both rules exist to fix
real bugs from the first version (docs/decisions.md D15).

Invariant I1 on screen
----------------------
Data zones are coloured circles; sampling patches are yellow squares. They are
never drawn alike.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from core.config import ZoneKind
from core.sampling import Patch
from storage.io import save_model, save_result
from ui import controls as ctl
from ui import plots

st.set_page_config(page_title="LDA hyperplane check", layout="wide",
                   initial_sidebar_state="expanded")

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"
STATE = st.session_state

TOOLS = {
    **{kind.value: kind.value for kind in ZoneKind},
    "patch": "＋ patch",
    "erase": "− patch",
    "look": "look",
}
ZONE_KIND_VALUES = {kind.value for kind in ZoneKind}

# Status colours with an icon and a word, never colour alone.
VERDICTS = {
    "affine": ("#0ca30c", "#ffffff", "✓", "affine",
               "the readout is a straight function of c across the range"),
    "equivocal": ("#fab219", "#0b0b0b", "~", "equivocal",
                  "some curvature, too small to call a clear bend"),
    "nonlinear": ("#d03b3b", "#ffffff", "✗", "nonlinear",
                  "the readout bends — reading f as a measurement of c is biased"),
    "degenerate": ("#8a8985", "#ffffff", "○", "degenerate",
                   "no slope above noise — the correct answer when nothing tracks c"),
    "fit_failed": ("#ec835a", "#0b0b0b", "!", "fit failed",
                   "the LDA had nothing to learn from — see the model tab"),
}

CSS = """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1560px;}
section[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
div[data-testid="stMetric"] {
  border: 1px solid rgba(128,128,128,0.22); border-radius: 12px;
  padding: 10px 14px 8px 14px; background: rgba(128,128,128,0.04);
}
div[data-testid="stMetricValue"] {font-size: 1.55rem;}
.lda-title {font-size: 1.55rem; font-weight: 700; margin: 0; line-height: 1.2;}
.lda-sub {color: rgba(128,128,128,0.95); margin: 2px 0 0 0; font-size: 0.95rem;}
.lda-verdict {display: inline-flex; gap: 8px; align-items: center;
  padding: 6px 14px; border-radius: 999px; font-weight: 650; font-size: 1.02rem;}
.lda-verdict-note {color: rgba(128,128,128,0.95); margin-left: 10px;}
.lda-chip {display: inline-flex; align-items: center; gap: 5px; margin-right: 12px;
  font-size: 0.83rem; color: rgba(128,128,128,0.95);}
.lda-dot {width: 11px; height: 11px; border-radius: 50%; border: 2.5px solid;}
.lda-square {width: 10px; height: 10px; border: 2px solid;}
.lda-empty {border: 1px dashed rgba(128,128,128,0.4); border-radius: 12px;
  padding: 22px 26px; color: rgba(128,128,128,0.95);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


CHART_CONFIG = {"displayModeBar": False}


def show(target, figure, key: str) -> None:
    """Render an analysis chart. The Plotly toolbar is hidden: it floats over the
    legend on hover and its zoom tools are not needed for these small plots."""
    target.plotly_chart(figure, width="stretch", key=key, config=CHART_CONFIG)


def widget_range(key: str, low: float, high: float) -> tuple[float, float]:
    """Widen a widget's range to include its current (e.g. loaded) value.

    A loaded config may carry a value outside the slider's usual range (a
    radius of 250, a zone centre beyond a shrunk plane). Streamlit rejects a
    key-bound value outside [min, max], so the range grows instead.
    """
    value = STATE[key]
    return (min(low, value), max(high, value))


# --------------------------------------------------------------------------- #
# Pre-render: bring derived state up to date before any widget is drawn
# --------------------------------------------------------------------------- #

ctl.init_state()
ctl.sync_zones_from_widgets()
patch_note = ctl.normalise_patches()
if patch_note:
    STATE["status"] = patch_note
STATE["preset"] = ctl.matching_preset()
STATE["place_x"] = int(min(STATE["place_x"], STATE["plane_width"] - 1))
STATE["place_y"] = int(min(STATE["place_y"], STATE["plane_height"] - 1))
# Old canvas keys need no cleanup: Streamlit discards the state of any widget
# that is not rendered in a run (deleting them by hand raised KeyError).
current_canvas_key = f"canvas_{STATE['canvas_generation']}"
if STATE["notice"]:
    st.toast(STATE["notice"])
    STATE["notice"] = ""


# --------------------------------------------------------------------------- #
# Sidebar: every setting
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### Settings")
    st.selectbox(
        "speed preset", list(ctl.PRESETS) + [ctl.CUSTOM_PRESET], key="preset",
        on_change=ctl.on_preset_change,
        help="Trades trials for speed. `fast` returns in well under a second and is "
             "for exploring; `careful` matches the pre-registered settings and is "
             "the only one whose verdict is worth quoting. Shows `custom` once you "
             "change any of the numbers it controls.")

    with st.expander("World — how c drives the pixels", expanded=True):
        size = st.columns(2)
        size[0].number_input(
            "height", 20, 400, key="plane_height", step=10,
            help="Rows of pixels. Patches pushed off the plane by a shrink are moved "
                 "back on, and exact duplicates merged, with a notice.")
        size[1].number_input(
            "width", 20, 400, key="plane_width", step=10,
            help="Columns of pixels. 100×100 is the size the study was specified "
                 "for; whether conclusions depend on it is untested.")
        st.slider(
            "background gain", *widget_range("background_gain", -1.5, 2.0),
            step=0.05, key="background_gain",
            help="How strongly pixels OUTSIDE every zone track c. 0 is the all-dead "
                 "null control. Above 1 the whole background saturates.")
        st.slider(
            "pivot", 0.0, 1.0, step=0.05, key="pivot",
            help="Contrast around which gains rotate; zones are invisible here. "
                 "Anything other than 0.5 makes clipping reachable.")
        st.radio(
            "zone overlap", ["blend", "replace"], horizontal=True, key="overlap_mode",
            help="blend = overlapping zones average (bounded by construction); "
                 "replace = the highest-priority zone wins, like z-order.")
        st.radio(
            "pixel model", ["bernoulli", "clipped_gaussian", "beta"],
            key="pixel_model",
            help="bernoulli: c is the chance a pixel is white; variance vanishes at "
                 "the ends. clipped_gaussian: continuous, closest to constant "
                 "precision, with censoring bias. beta: bounded, never clipped.")
        # The noise and jitter sliders are ALWAYS drawn, and disabled when they
        # do not apply. Drawing them conditionally looked tidier but lost values:
        # Streamlit discards a widget's state whenever it is not rendered, so
        # switching pixel model and back silently reset ν = 50 to the default.
        st.slider("pixel noise σ (clipped_gaussian)", 0.01, 0.5, step=0.01,
                  key="sigma", disabled=STATE["pixel_model"] != "clipped_gaussian",
                  help="Gaussian noise per pixel before values are censored into "
                       "[0, 1]. Only used by the clipped_gaussian model.")
        st.slider("beta precision ν (beta)", *widget_range("beta_concentration", 2.0,
                                                          200.0),
                  step=1.0, key="beta_concentration",
                  disabled=STATE["pixel_model"] != "beta",
                  help="Higher = less noise. Only used by the beta model; its "
                       "variance depends on the mean, so precision still varies.")
        st.radio(
            "random-zone draw", ["bernoulli_half", "uniform"], horizontal=True,
            key="random_zone_dist",
            help="What a masked pixel inside a `random` zone shows instead of "
                 "tracking c: a coin flip or a flat draw. Both are pure nuisance.")
        st.toggle(
            "jitter zones between trials", key="jitter_enabled",
            help="A COHERENT nuisance: whole zones move together within a trial, so "
                 "averaging trials does not remove its effect on the curve's shape.")
        jitter_off = not STATE["jitter_enabled"]
        st.slider("centre σ (px)", 0.0, 10.0, step=0.5, key="jitter_center",
                  disabled=jitter_off,
                  help="How far each zone's centre wanders between trials.")
        st.slider("radius σ (px)", 0.0, 10.0, step=0.5, key="jitter_radius",
                  disabled=jitter_off,
                  help="How much each zone's size wanders between trials.")
        st.slider("gain σ", 0.0, 0.5, step=0.01, key="jitter_gain",
                  disabled=jitter_off,
                  help="> 0 is a path to clipping: it can push |gain| past 1 on "
                       "individual trials even when every configured gain is "
                       "legal.")

    with st.expander("Observer — what the LDA sees", expanded=True):
        st.slider(
            "patch size", 2, 20, key="patch_size",
            help="Side of each square sampling window, in pixels. With `pixels` "
                 "features this squares the feature count per patch.")
        st.slider(
            "grid step", 1, 40, key="grid_step",
            help="Spacing used by the 'grid patches' button. Smaller than the patch "
                 "size makes patches overlap and duplicate pixels.")
        st.radio(
            "features", ["pixels", "patch_mean", "patch_mean_std"], key="feature_mode",
            help="pixels: every sampled pixel (slow to fit). patch_mean: one value "
                 "per patch, fast and higher SNR. patch_mean_std: adds within-patch "
                 "SD so the model can use variance cues.")
        multiplier = {"pixels": STATE["patch_size"] ** 2, "patch_mean": 1,
                      "patch_mean_std": 2}[STATE["feature_mode"]]
        st.metric(
            "features", len(STATE["patches"]) * multiplier,
            help="Length of the vector handed to the LDA. Once it exceeds the number "
                 "of training trials the covariance is rank-deficient — expected, "
                 "and why shrinkage is on by default. Main driver of fit time.")

    with st.expander("Training & evaluation", expanded=False):
        extremes = st.columns(2)
        extremes[0].slider(
            "train c_lo", *widget_range("c_lo", 0.0, 0.45), step=0.01, key="c_lo",
            help="Lower training extreme. At exactly 0 a Bernoulli world is all "
                 "black with zero variance and the fit is arbitrary.")
        extremes[1].slider(
            "train c_hi", *widget_range("c_hi", 0.55, 1.0), step=0.01, key="c_hi",
            help="Upper training extreme. c_lo and c_hi are the ONLY levels the "
                 "model learns from; everything between tests the assumption.")
        st.slider(
            "training trials per extreme", *widget_range("n_per_class", 20, 800),
            step=10, key="n_per_class",
            help="Fields generated at each extreme. Too few relative to the feature "
                 "count and the model overfits nuisance variance.")
        st.slider(
            "evaluation contrast levels", *widget_range("grid_points", 5, 41), step=1,
            key="grid_points", on_change=ctl.on_grid_points_change,
            help="Points between 0 and 1 the trained model is probed at. More "
                 "levels resolve the curve's shape better.")
        if STATE["contrast_grid_override"]:
            st.caption(f"using the loaded config's own {len(STATE['contrast_grid_override'])}"
                       "-level grid; move the slider to replace it")
        st.slider(
            "trials per contrast level", *widget_range("n_per_contrast", 10, 400),
            step=10, key="n_per_contrast",
            help="Fresh trials at each level, never reused from training. Sets how "
                 "precisely the spread is measured; the pre-registered minimum is "
                 "200.")
        solver = st.columns(2)
        solver[0].radio(
            "solver", ["lsqr", "eigen", "svd"], key="solver",
            on_change=ctl.on_solver_change,
            help="lsqr with shrinkage is the stable choice when features outnumber "
                 "trials. svd is unregularised and cannot take shrinkage at all.")
        solver[1].radio(
            "shrinkage", ["auto", "none"], key="shrinkage",
            disabled=STATE["solver"] == "svd",
            help="Ledoit-Wolf covariance shrinkage. Forced off for svd, which "
                 "rejects it.")
        st.number_input(
            "cross-validation folds", 0, 10, key="cv_folds",
            help="Held-out accuracy on the training extremes. Each fold is a full "
                 "extra fit, so it is off (0) in the fast presets.")
        st.number_input(
            "master seed", 0, 10_000, key="master_seed",
            help="The only seed: pixel noise, jitter and patch placement all derive "
                 "from it, so the same seed reproduces a run exactly.")
        st.toggle(
            "shuffle labels (null control)", key="shuffle_labels",
            help="Relabels half of each training extreme at random. Any c-dependence "
                 "left in f is a pipeline artefact; this must come out degenerate.")
        st.toggle(
            "compute the oracle readout", key="compute_oracle",
            help="A readout built from the TRUE gains. It separates 'the LDA failed' "
                 "from 'no linear readout was possible'.")

    with st.expander("Configs", expanded=False):
        available = ctl.available_configs()
        if STATE["load_choice"] not in available:
            STATE["load_choice"] = None
        st.selectbox(
            "load", list(available), key="load_choice", index=None,
            placeholder="choose a config…",
            help="Ready-made worlds from configs/ plus anything saved under "
                 "configs/user/. The control_* ones must come out affine or "
                 "degenerate — they are what makes the tool trustworthy.")
        st.button("load onto canvas", key="load_button", width="stretch",
                  on_click=ctl.on_load_config,
                  help="Replace everything on screen with the chosen config, ready "
                       "to edit. Zone editors are reset, not carried over.")
        st.text_input("save as", key="save_name",
                      help="File name stem (letters, digits, - and _). Saved to "
                           "configs/user/; reusing a name overwrites it.")
        st.button("save current config", key="save_button", width="stretch",
                  on_click=ctl.on_save_config,
                  help="Write the current world, observer, training and evaluation "
                       "settings as one JSON file that the scripts can also run.")


# --------------------------------------------------------------------------- #
# Header and training
# --------------------------------------------------------------------------- #

run_cfg = ctl.run_config()
data_cfg = ctl.data_config()

header, action = st.columns([4, 1], vertical_alignment="center")
header.markdown(
    "<p class='lda-title'>LDA hyperplane check</p>"
    "<p class='lda-sub'>Train an LDA on the two extremes of a latent variable — does "
    "its readout stay a straight line in between?</p>", unsafe_allow_html=True)
train_clicked = action.button(
    "▶  Train & evaluate", key="train", type="primary", width="stretch",
    disabled=run_cfg is None,
    help="Fit an LDA on c_lo vs c_hi only, then probe it with fresh trials across "
         "the whole contrast range. Identical settings reuse the cached result.")

if train_clicked and run_cfg is not None:
    with st.spinner("training on the extremes, then probing the whole range…"):
        try:
            ctl.register_model(ctl.cached_experiment(run_cfg.model_dump_json()))
            STATE["status"] = "trained a new model"
        except Exception as error:                      # surfaced, not swallowed
            st.error(f"Run failed — {type(error).__name__}: {error}")


# --------------------------------------------------------------------------- #
# Canvas and placement
# --------------------------------------------------------------------------- #

canvas_col, side_col = st.columns([2.35, 1], gap="large")

with canvas_col:
    st.segmented_control(
        "click on the canvas to place", list(TOOLS), key="tool",
        format_func=lambda key: TOOLS.get(key, key),
        on_change=ctl.on_tool_change,
        help="Zone kinds (coloured circles) change the WORLD — how c drives pixels. "
             "Patches (yellow squares) change the OBSERVER — what the LDA sees. "
             "`look` places nothing.")

    view, toggle_a, toggle_b, toggle_c = st.columns([2.2, 1, 1, 1],
                                                    vertical_alignment="bottom")
    view.slider(
        "preview at contrast c", 0.0, 1.0, step=0.01, key="preview_contrast",
        help="Which contrast the canvas shows. Zones are invisible at c = pivot by "
             "construction — slide away from 0.5 to see them.")
    toggles = (toggle_a, toggle_b, toggle_c)
    toggles[0].toggle("gain map", key="show_gain",
                      help="Draw each pixel's coupling gain instead of one noisy "
                           "realisation: blue tracks c, gray does not, red runs "
                           "backwards.")
    toggles[1].toggle("zones", key="show_zones",
                      help="Draw zone circles and labels. Off shows the raw field "
                           "the model receives.")
    toggles[2].toggle("patches", key="show_patches",
                      help="Draw the sampling windows. A dense grid can hide the "
                           "pixels underneath.")

    field, gain, preview_clip = ctl.cached_preview(
        data_cfg.model_dump_json(), STATE["preview_contrast"], STATE["master_seed"])
    if STATE["show_gain"]:
        gain = ctl.cached_effective_gain(data_cfg.model_dump_json())
    patches = [Patch(row=r, col=c, size=STATE["patch_size"])
               for r, c in STATE["patches"]]

    figure = plots.plane_figure(field, gain, data_cfg, patches,
                                show_zones=STATE["show_zones"],
                                show_patches=STATE["show_patches"],
                                show_gain=STATE["show_gain"])
    event = st.plotly_chart(
        figure, width="stretch", key=current_canvas_key, on_select="rerun",
        selection_mode="points",
        config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False})

    # A click is handled once, then the chart is re-keyed so the next run starts
    # with an empty selection. Streamlit refuses writes to a widget's own state
    # (the first version tried, failed, and fell back to a guard that was never
    # read); a fresh key is the supported way to reset it, and it lets a second
    # click on the same pixel register.
    points = (event.selection.get("points")
              if event is not None and event.selection else None)
    if points and STATE["tool"] not in (None, "look"):
        ctl.place_with_current_tool(float(points[0].get("x", 0.0)),
                                    float(points[0].get("y", 0.0)))
        STATE["canvas_generation"] += 1
        st.rerun()

    legend = "".join(
        f"<span class='lda-chip'><span class='lda-dot' style='border-color:{c}'>"
        f"</span>{kind}</span>" for kind, c in plots.ZONE_COLOURS.items())
    legend += (f"<span class='lda-chip'><span class='lda-square' "
               f"style='border-color:{plots.PATCH_COLOUR}'></span>patch</span>")
    st.markdown(legend, unsafe_allow_html=True)
    st.caption(f"{len(STATE['patches'])} patches · {len(STATE['zones'])} zones · "
               f"{preview_clip:.1%} of pixels clipped at c = "
               f"{STATE['preview_contrast']:.2f} · config {ctl.config_fingerprint()}")
    if not ctl.has_patches():
        st.warning("No sampling patches — the observer sees nothing and cannot be "
                   "trained. Use **grid patches** or place some with **＋ patch**.")

with side_col:
    st.markdown("**Place at a coordinate**")
    place = st.columns(2)
    place[0].number_input("x (column)", 0, STATE["plane_width"] - 1, key="place_x",
                          help="Column, counted from the left edge of the plane.")
    place[1].number_input("y (row)", 0, STATE["plane_height"] - 1, key="place_y",
                          help="Row, counted from the TOP edge — matching how the "
                               "canvas is drawn and the arrays are indexed.")
    actions = st.columns(2)
    actions[0].button(
        (f"place {TOOLS[STATE['tool']]}" if STATE["tool"] not in (None, "look")
         else "pick a tool first"), key="place_button",
        width="stretch", on_click=ctl.on_place_numeric,
        disabled=STATE["tool"] in (None, "look"),
        help="Does exactly what a canvas click does, at an exact coordinate. Also "
             "the fallback if a click ever fails to register.")
    actions[1].button("undo last", key="undo_button", width="stretch",
                      on_click=ctl.on_undo,
                      help="Remove the most recently placed zone or patch (one step).")
    bulk = st.columns(2)
    bulk[0].button("grid patches", key="grid_button", width="stretch",
                   on_click=ctl.on_grid_patches,
                   help="Replace all patches with a uniform grid at the grid step set "
                        "in the Observer settings.")
    bulk[1].button("clear patches", key="clear_patches_button", width="stretch",
                   on_click=ctl.on_clear_patches,
                   help="Remove every sampling window — a starting point for placing "
                        "your own by hand.")
    st.button("clear zones", key="clear_zones_button", width="stretch",
                   on_click=ctl.on_clear_zones,
                   help="Remove every zone, leaving the uniform world that must come "
                        "out affine — the sanity check.")
    if STATE["status"]:
        st.caption(f"↳ {STATE['status']}")

    st.markdown(f"**Zones** · {len(STATE['zones'])}")
    if not STATE["zones"]:
        st.markdown("<div class='lda-empty'>Pick a zone kind above the canvas and "
                    "click on it to place one.</div>", unsafe_allow_html=True)
    for zone in list(STATE["zones"]):
        ctl.ensure_zone_widget_state(zone)
        zid = zone["id"]
        key = lambda field, zid=zid: ctl.zone_key(zid, field)
        with st.expander(f"{zid} · {zone['kind']} · gain {zone['gain']:g}"):
            st.markdown(f"<div style='height:3px;border-radius:2px;background:"
                        f"{plots.ZONE_COLOURS.get(zone['kind'], '#888')}'></div>",
                        unsafe_allow_html=True)
            xy = st.columns(2)
            xy[0].slider("x", *widget_range(key("center_x"), 0.0,
                                            float(STATE["plane_width"] - 1)),
                         step=1.0, key=key("center_x"),
                         help="Centre column of the zone, from the left edge.")
            xy[1].slider("y", *widget_range(key("center_y"), 0.0,
                                            float(STATE["plane_height"] - 1)),
                         step=1.0, key=key("center_y"),
                         help="Centre row of the zone; row 0 is the TOP edge.")
            st.slider("radius", *widget_range(key("radius"), 1.0, 200.0), step=1.0,
                      key=key("radius"),
                      help="May exceed the plane — that is how a whole-plane control "
                           "is expressed.")
            if zone["kind"] != ZoneKind.RANDOM.value:
                st.slider("gain", *widget_range(key("gain"), -3.0, 3.0), step=0.05,
                          key=key("gain"),
                          help="How steeply the zone tracks c. |gain| > 1 makes pixels "
                               "clip — the saturation condition, not a mistake.")
                st.slider("offset", *widget_range(key("offset"), -0.5, 0.5),
                          step=0.01, key=key("offset"),
                          help="A contrast-INDEPENDENT brightness shift. Nonzero also "
                               "makes clipping reachable; leave at 0 unless probing "
                               "that.")
            edge = st.columns([3, 2])
            edge[0].selectbox("edge", ["hard", "smoothstep", "gaussian"],
                              key=key("falloff"),
                              help="How membership fades at the rim. smoothstep ends "
                                   "at the radius; gaussian extends past it.")
            edge[1].slider("edge width", *widget_range(key("falloff_width"), 0.0,
                                                       15.0),
                           step=0.5, key=key("falloff_width"),
                           help="Pixels over which membership decays; a zone only "
                                "reaches its full gain in its core.")
            st.number_input("priority", *widget_range(key("priority"), -10, 10),
                            key=key("priority"),
                            help="Only used in `replace` overlap mode: the highest "
                                 "priority wins shared pixels.")
            st.button("remove zone", key=f"remove:{zid}", on_click=ctl.on_remove_zone,
                      args=(zid,),
                      help="Delete this zone from the world. Already-trained models "
                           "are unaffected.")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

st.divider()
models = STATE["models"]
if not models:
    st.markdown(
        "<div class='lda-empty'><b>No model yet.</b> The default world is uniform — "
        "every pixel tracks c equally — so pressing <b>Train & evaluate</b> now "
        "should give <b>affine</b>. That is the sanity check. Then add zones and see "
        "what bends it.</div>", unsafe_allow_html=True)
    st.stop()

if STATE["active_model"] not in models:
    STATE["active_model"] = next(reversed(models))
order = list(reversed(models))                          # newest first
select, load_btn, delete_btn, save_btn = st.columns([5, 1.2, 1, 1.2],
                                                    vertical_alignment="bottom")
select.selectbox(
    "model", order, key="active_model",
    format_func=lambda rid: (f"{VERDICTS[models[rid].report.verdict][2]}  "
                             f"{ctl.summarise(models[rid]).label}"),
    help="Every model trained this session, newest first. Train several worlds and "
         "flip between them here, or overlay them in the compare tab.")
active_id = STATE["active_model"]
result = models[active_id]
report = result.report
load_btn.button("load its config", key="load_model_button", width="stretch",
                on_click=ctl.on_load_model_config, args=(active_id,),
                help="Put this model's world and observer back on the canvas, to edit "
                     "and re-run.")
delete_btn.button("delete", key="delete_model_button", width="stretch",
                  on_click=ctl.on_delete_model, args=(active_id,),
                  help="Drop this model from the session. Files already saved to "
                       "disk are untouched.")
if save_btn.button("save to disk", key="save_model_button", width="stretch",
                   help="Write the model to models/ and the per-trial table to "
                        "results/, with a sidecar recording config, seed and "
                        "package versions."):
    save_model(result, MODELS_DIR)
    st.toast(f"saved {save_result(result, RESULTS_DIR)['trials'].name}")

if run_cfg is None or run_cfg.model_dump_json() != result.config.model_dump_json():
    st.warning("The canvas or settings have changed since this model was trained — "
               "the figures below describe the **model**, not what is on screen now.")

colour, ink, icon, word, meaning = VERDICTS[report.verdict]
st.markdown(
    f"<span class='lda-verdict' style='background:{colour};color:{ink}'>{icon} "
    f"{word}</span><span class='lda-verdict-note'>{meaning}</span>",
    unsafe_allow_html=True)
st.write("")

def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


metric_cols = st.columns(6)
metric_cols[0].metric(
    "κ curvature", _fmt(report.curvature_index),
    help=f"√(κ₂² + κ₃²): quadratic κ₂ = {_fmt(report.kappa_quadratic)} (one-sided "
         f"bend), cubic κ₃ = {_fmt(report.kappa_cubic)} (S-shape, i.e. saturation). "
         "Below 0.05 affine, above 0.15 a clear bend. The headline number.")
metric_cols[1].metric(
    "range use", _fmt(report.effective_range_use, 2),
    help="Share of the output range spent on the middle half of c. A straight line "
         "is exactly 0.50; an S-curve spends more, an end-expanded curve less.")
metric_cols[2].metric(
    "ρ Spearman", _fmt(report.spearman_rho),
    help="Is f at least monotone in c? High ρ with high κ means usable as an "
         "ordering but not as a measurement.")
metric_cols[3].metric(
    "r Pearson", _fmt(report.pearson_r),
    help="Never read alone: a clearly S-shaped curve can sit at r = 0.99.")
metric_cols[4].metric(
    "SD max / min", _fmt(report.sd_ratio, 2),
    "constant" if report.homoscedastic else "varies",
    delta_color="normal" if report.homoscedastic else "inverse",
    help="How much the PRECISION of f varies across c (H3). Levels with no variance "
         f"at all are excluded ({report.n_zero_variance_levels} here).")
metric_cols[5].metric(
    "calibration RMSE", _fmt(None if report.calibration is None
                             else float(report.calibration["rmse"].mean())),
    help="Read c back off f through the affine fit: the typical error, in units "
         "of c.")

summary = ctl.summarise(result)
if summary.below_preregistered_minimum:
    st.caption(f"⚠︎ {summary.n_per_contrast} trials per level is below the "
               f"pre-registered minimum of {ctl.PRE_REGISTERED_MIN_PER_LEVEL} — fine "
               "for exploring; use the **careful** preset before quoting a verdict.")
if report.verdict == "fit_failed":
    st.error(f"**The fit failed:** {result.model.diagnostics.fit_problem}")

tabs = st.tabs(["Mapping", "Precision & residuals", "What the model used",
                "Model detail", "Compare"])

with tabs[0]:
    left, right = st.columns([3, 2])
    show(left, plots.mapping_figure(report, result.oracle_report), "fig_mapping")
    show(right, plots.scatter_figure(result.trials), "fig_scatter")
    if result.oracle_report is not None:
        oracle = result.oracle_report
        st.caption(
            f"Oracle (a readout built from the true gains): "
            f"{VERDICTS[oracle.verdict][2]} **{VERDICTS[oracle.verdict][3]}**, "
            f"κ = {_fmt(oracle.curvature_index)}. If the oracle is affine where the "
            "LDA is not, the method is at fault; if both bend, the world does it.")
    show(st, plots.histogram_figure(result.trials), "fig_hist")

with tabs[1]:
    left, right = st.columns(2)
    show(left, plots.residual_figure(report), "fig_residual")
    left.caption("How far off you would be, in units of c, reading f through a "
                 "straight line. Flat at zero is affine; a smile is a one-sided "
                 "bend; an S is saturation.")
    show(right, plots.spread_figure(report), "fig_spread")
    right.caption("Trial-to-trial spread of f at each level, and how many pixels "
                  "were clipped there. Clipping that appears only at the ends is the "
                  "signature of saturation.")
    st.dataframe(report.mean_by_c.round(4), width="stretch", hide_index=True)

with tabs[2]:
    show(st, plots.weight_figure(result.attribution), "fig_weights")
    left, right = st.columns(2)
    with left:
        st.markdown("**Weight by zone kind**")
        st.dataframe(result.attribution.mean_signed_weight_by_kind.round(4),
                     width="stretch", hide_index=True)
        st.caption(f"|w| vs |gain| correlation "
                   f"{_fmt(result.attribution.abs_correlation, 2)} · sign agreement "
                   f"{_fmt(result.attribution.sign_agreement, 2)}. Expect negative "
                   "weight in anti zones and ~0 in dead and random ones.")
    with right:
        st.markdown("**What the patches actually cover**")
        st.dataframe(result.composition.round(4), width="stretch", hide_index=True)
        st.caption("Placement alone can decide the answer. Random-zone pixels carry "
                   "gain 0 here: they do not track c.")

with tabs[3]:
    d = result.model.diagnostics
    row = st.columns(4)
    row[0].metric("features", d.n_features,
                  help="Length of the feature vector the LDA was fitted on.")
    row[1].metric("training trials", d.n_train,
                  help="Both extremes combined, so twice the per-extreme count.")
    row[2].metric("fresh d′", _fmt(result.fresh_d_prime, 2),
                  help="Separation of the two extremes on FRESH trials at c = "
                       f"{result.fresh_d_prime_levels[0]:.2f} and "
                       f"{result.fresh_d_prime_levels[1]:.2f}. The honest number.")
    row[3].metric("in-sample d′", _fmt(d.d_prime, 2),
                  help="The same separation on the training data itself. Much larger "
                       "than fresh d′ means the model memorised nuisance variance.")
    row = st.columns(4)
    row[0].metric("in-sample accuracy", _fmt(d.train_accuracy, 3),
                  help="Near 1.0 is normal and means little — the extremes are easy "
                       "to separate. Not evidence the readout is linear.")
    row[1].metric("cross-val accuracy",
                  "off" if np.isnan(d.cv_accuracy) else _fmt(d.cv_accuracy, 3),
                  help="Held-out accuracy on the training extremes; set folds in the "
                       "Training settings.")
    row[2].metric("covariance",
                  "singular" if not np.isfinite(d.cov_condition_number)
                  else f"{d.cov_condition_number:.3g}",
                  help="Singular is EXPECTED whenever features outnumber trials — why "
                       "shrinkage is the default.")
    row[3].metric("top 1% weight mass", _fmt(d.top1pct_weight_mass, 3),
                  help="Share of the weight vector carried by its largest 1% of "
                       "entries. High = the model leans on a few pixels and will not "
                       "survive a change of seed.")
    with st.expander("slope and curvature detail"):
        st.json({"beta0": report.beta0, "beta1": report.beta1,
                 "beta1_ci": list(report.beta1_ci), "slope_t": report.slope_t,
                 "r2": report.r2, "kappa_quadratic": report.kappa_quadratic,
                 "kappa_cubic": report.kappa_cubic,
                 "orthogonal_poly_coefs": report.poly_coefs,
                 "nested_f": report.nested_f,
                 "max_local_slope_ratio": report.max_local_slope_ratio,
                 "monotonicity_violations": report.monotonicity_violations})
    with st.expander("config that produced this"):
        st.json(json.loads(result.config.model_dump_json()))

with tabs[4]:
    if len(models) < 2:
        st.markdown("<div class='lda-empty'>Train at least two models to compare "
                    "them — e.g. the same world under <b>bernoulli</b> and "
                    "<b>clipped_gaussian</b>, or with patches on heat zones versus "
                    "dead zones.</div>", unsafe_allow_html=True)
    else:
        chosen = st.multiselect(
            "models to overlay", order, default=order[:4], key="compare_choice",
            format_func=lambda rid: ctl.summarise(models[rid]).label,
            help="Each curve is rescaled so its own straight-line fit runs 0 → 1, so "
                 "only SHAPE is compared — raw decision values have arbitrary units.")
        if chosen:
            show(st, plots.comparison_figure(
                {f"{i + 1}. {ctl.summarise(models[r]).label}": models[r].report
                 for i, r in enumerate(chosen)}), "fig_compare")
            st.dataframe(pd.DataFrame([
                {"model": ctl.summarise(models[r]).label,
                 **models[r].report.summary_row()} for r in chosen]),
                width="stretch", hide_index=True)
