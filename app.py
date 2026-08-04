"""LDA hyperplane check — interactive explorer.

Run locally:

    streamlit run app.py

Responsibility
--------------
Layout and event handling only. Every number comes from `core/`, every figure from
`ui/plots.py`, and all state and caching from `ui/controls.py`. Nothing here
computes anything scientific — if you find yourself doing arithmetic in this file,
it belongs in `core/`.

Layout (following the original sketch in docs/original_spec.md §4)
-----------------------------------------------------------------
    toolbar          zone / patch tools, speed preset, run button
    left    centre   right      model registry | canvas | data configs
    below            global data controls, then the analysis panel

Invariant I1 on screen
----------------------
Data zones are circles in kind-specific colours; sampling patches are yellow
squares. They are never drawn alike, because conflating the world with the
observer is the mistake this whole project is built to avoid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from core.config import RunConfig, ZoneKind
from core.data import GainField
from core.sampling import Patch
from storage.io import load_run_config, save_config, save_model, save_result
from ui import controls as ctl
from ui import plots

st.set_page_config(page_title="LDA hyperplane check", layout="wide",
                   initial_sidebar_state="collapsed")

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = REPO_ROOT / "models"

ZONE_KIND_VALUES = {kind.value for kind in ZoneKind}

TOOL_LABELS = {
    **{kind.value: f"zone: {kind.value}" for kind in ZoneKind},
    "patch": "add sampling patch",
    "erase_patch": "remove sampling patch",
    "none": "look only (no placing)",
}

ctl.init_state()


# --------------------------------------------------------------------------- #
# Toolbar
# --------------------------------------------------------------------------- #

st.markdown("#### LDA hyperplane check &nbsp;·&nbsp; "
            "<span style='font-weight:400;font-size:0.85em;color:#666'>"
            "does an LDA trained on the two extremes of a latent variable read out "
            "linearly in between?</span>", unsafe_allow_html=True)

toolbar = st.columns([2.4, 1.1, 1.1, 1.2, 1.0, 1.0])
with toolbar[0]:
    st.session_state["tool"] = st.radio(
        "click on the canvas to…", list(TOOL_LABELS), horizontal=False,
        format_func=lambda key: TOOL_LABELS[key],
        index=list(TOOL_LABELS).index(st.session_state["tool"]),
        help="What a click places. Zones (circles) change the WORLD — how the "
             "latent contrast drives the pixels. Patches (yellow squares) change "
             "the OBSERVER — which pixels the LDA gets to see. The two are kept "
             "strictly separate; that separation is what stops the model peeking "
             "at how the data was made.")
with toolbar[1]:
    preset = st.selectbox(
        "speed preset", list(ctl.PRESETS),
        index=list(ctl.PRESETS).index(st.session_state["preset"]),
        help="Trades trials for speed. `fast` returns in well under a second and "
             "is for exploring; `careful` matches the pre-registered settings and "
             "is the only one whose verdict is worth quoting. Changing this "
             "rewrites the trial counts and feature mode below.")
    if preset != st.session_state["preset"]:
        ctl.apply_preset(preset)
        st.rerun()
with toolbar[2]:
    st.session_state["preview_contrast"] = st.slider(
        "preview contrast c", 0.0, 1.0, st.session_state["preview_contrast"], 0.01,
        help="Which contrast level the canvas shows. Zones are invisible at "
             "c = pivot by construction, so slide away from 0.5 to see them.")
with toolbar[3]:
    st.session_state["show_gain"] = st.toggle(
        "show gain field", st.session_state["show_gain"],
        help="Display the coupling gain a instead of a pixel realisation. The gain "
             "field is what the zones actually mean.")
    st.session_state["show_zones"] = st.toggle(
        "outline zones", st.session_state["show_zones"],
        help="Draw the zone circles and their labels. Turn off to see the raw "
             "pixel field the model actually receives.")
    st.session_state["show_patches"] = st.toggle(
        "outline patches", st.session_state["show_patches"],
        help="Draw the sampling windows. With a dense grid these can hide the "
             "pixels underneath, so turning them off is often clearer.")
with toolbar[4]:
    st.write("")
    if st.button("grid patches", width="stretch",
                 help="Replace all patches with a uniform grid at the current step."):
        ctl.auto_grid_patches()
        st.rerun()
    if st.button("clear zones", width="stretch",
                 help="Remove every zone, leaving a uniform world in which every "
                      "pixel tracks the contrast equally. That uniform case is the "
                      "sanity check: it must come out affine."):
        st.session_state["zones"] = []
        st.rerun()
with toolbar[5]:
    st.write("")
    run_clicked = st.button(
        "▶ Train & evaluate", type="primary", width="stretch",
        help="Fit an LDA on the two training extremes only, then probe it across "
             "the whole contrast range with fresh trials. Identical settings reuse "
             "the cached result rather than retraining.")
    if st.button("clear patches", width="stretch",
                 help="Remove every sampling window. The observer then sees "
                      "nothing and cannot be trained — useful mainly as a "
                      "starting point for placing your own by hand."):
        st.session_state["patches"] = []
        st.rerun()

# Build the configs once per rerun, so a validation error surfaces here rather
# than halfway through a run.
#
# An empty patch list is handled separately and deliberately. `SamplingConfig`
# rejects a manual layout with no patches, which is correct for the library but
# wrong for the app: pressing "clear patches" is a legitimate thing to do on the
# way to placing your own, and it must produce a readable warning and a live
# canvas, not a pydantic traceback where the picture used to be.
has_patches = bool(st.session_state["patches"])
try:
    data_cfg = ctl.data_config()
    sampling_cfg = ctl.sampling_config() if has_patches else None
    run_cfg = ctl.run_config() if has_patches else None
except Exception as error:                        # pydantic ValidationError et al.
    st.error(f"Invalid configuration: {error}")
    st.stop()


# --------------------------------------------------------------------------- #
# Left panel · canvas · right panel
# --------------------------------------------------------------------------- #

left, centre, right = st.columns([1.05, 2.5, 1.05], gap="medium")

# ---- canvas --------------------------------------------------------------- #
with centre:
    if not st.session_state["patches"]:
        st.warning("No sampling patches: the observer sees nothing. "
                   "Click **grid patches**, or place some by clicking the canvas.")

    field, gain, preview_clip = ctl.cached_preview(
        data_cfg.model_dump_json(), st.session_state["preview_contrast"],
        st.session_state["master_seed"])
    patches = [Patch(row=r, col=c, size=st.session_state["patch_size"])
               for r, c in st.session_state["patches"]]

    # A lightweight stand-in for the GainField the figure needs: only `a` is used
    # for display, and rebuilding the full field per rerun would be wasteful.
    display_gf = GainField(
        a=ctl.cached_gain_field(data_cfg.model_dump_json()) if st.session_state["show_gain"]
        else gain,
        b=np.zeros_like(gain), random_mask=np.zeros(gain.shape, dtype=bool),
        zone_weights=np.zeros((0, *gain.shape)), zone_ids=(), zone_kinds=(),
        jitter_realised={})

    figure = plots.plane_figure(
        field, display_gf, data_cfg, patches,
        show_zones=st.session_state["show_zones"],
        show_patches=st.session_state["show_patches"],
        show_gain=st.session_state["show_gain"])
    event = st.plotly_chart(
        figure, width="stretch", key="canvas",
        on_select="rerun", selection_mode="points",
        # scrollZoom off: a stray trackpad scroll over the canvas used to rescale
        # the plane, after which clicked coordinates no longer matched the pixels.
        # The modebar is hidden because every tool on it (pan, zoom, lasso) either
        # does nothing here or fights the click-to-place interaction.
        config={"scrollZoom": False, "displayModeBar": False, "doubleClick": False},
    )

    # Handle a canvas click.
    #
    # Streamlit replays the current selection on EVERY rerun, so a naive handler
    # re-places the same zone on every later interaction. The fix is to clear the
    # widget's own selection state after acting on it, which also makes a genuine
    # second click on the same pixel work — the coordinate-comparison approach this
    # replaced silently swallowed those, so you could never put two patches in the
    # same place.
    selected = (event.selection.get("points") if event and event.selection else None)
    if selected and st.session_state["tool"] != "none":
        point = selected[0]
        x, y = float(point.get("x", 0.0)), float(point.get("y", 0.0))
        ctl.place_with_current_tool(x, y)
        try:
            st.session_state["canvas"] = {"selection": {"points": [], "point_indices": [],
                                                        "box": [], "lasso": []}}
        except Exception:
            # Streamlit refuses to overwrite some widget states; the coordinate
            # guard below is the fallback so a replay cannot place a second zone.
            st.session_state["last_click"] = (round(x, 3), round(y, 3))
        st.rerun()

    caption = (f"{len(st.session_state['patches'])} patches · "
               f"{len(st.session_state['zones'])} zones · "
               f"clip fraction at c={st.session_state['preview_contrast']:.2f}: "
               f"{preview_clip:.3f} · config {ctl.config_fingerprint()}")
    st.caption(caption)
    if st.session_state["status"]:
        st.caption(f"↳ {st.session_state['status']}")

    # Numeric placement, as a guaranteed path that does not depend on Plotly
    # events firing. Clicking is the nice way; this is the way that cannot break,
    # and it is also how you place something at an exact coordinate.
    place = st.columns([1, 1, 1.4, 1.2])
    place_x = place[0].number_input(
        "x", 0, data_cfg.plane.width - 1, data_cfg.plane.width // 2, key="place_x",
        help="Column, counted from the left edge of the plane.")
    place_y = place[1].number_input(
        "y", 0, data_cfg.plane.height - 1, data_cfg.plane.height // 2, key="place_y",
        help="Row, counted from the TOP edge — matching how the canvas is drawn "
             "and how the arrays are indexed.")
    place[2].write("")
    if place[2].button(f"place {TOOL_LABELS[st.session_state['tool']]} here",
                       width="stretch", disabled=st.session_state["tool"] == "none",
                       help="Does exactly what a canvas click does, at an exact "
                            "coordinate. Both routes run the same code, so this is "
                            "also the fallback if a click ever fails to register."):
        ctl.place_with_current_tool(float(place_x), float(place_y))
        st.rerun()
    place[3].write("")
    if place[3].button("undo last", width="stretch",
                       help="Removes the most recently placed zone or patch."):
        st.session_state["status"] = ctl.undo_last_placement()
        st.rerun()

# ---- left: model registry ------------------------------------------------- #
with left:
    st.markdown("**Trained models**")
    models: dict[str, object] = st.session_state["models"]
    if not models:
        st.caption("None yet. Set up the world and the observer, then press "
                   "**Train & evaluate**.")
    else:
        summaries = {mid: ctl.summarise(result) for mid, result in models.items()}
        active = st.radio(
            "active model", list(summaries),
            format_func=lambda mid: (
                f"{summaries[mid].verdict}"
                f"{'' if summaries[mid].kappa is None else f' κ={summaries[mid].kappa:.4f}'}"
                f" — {summaries[mid].label}"),
            index=(list(summaries).index(st.session_state["active_model"])
                   if st.session_state["active_model"] in summaries else 0),
            label_visibility="collapsed",
            help="Which trained model the analysis panel below describes. Models "
                 "stay in this list for the session, so you can train several "
                 "worlds and flip between them.")
        st.session_state["active_model"] = active

        buttons = st.columns(2)
        if buttons[0].button("load its config", width="stretch",
                             help="Put this model's world and observer back on the "
                                  "canvas, so it can be edited and re-run."):
            ctl.load_run_config_into_state(models[active].config)
            st.rerun()
        if buttons[1].button("delete", width="stretch",
                             help="Drop this model from the session list. Anything "
                                  "already saved to disk is untouched."):
            del models[active]
            st.session_state["active_model"] = next(iter(models), None)
            st.rerun()
        if st.button("save model + results to disk", width="stretch",
                     help="Write the fitted model to models/ and the full per-trial "
                          "table to results/, with a sidecar recording the config, "
                          "the seed and the package versions needed to reproduce it."):
            result = models[active]
            save_model(result, MODELS_DIR)
            paths = save_result(result, RESULTS_DIR)
            st.success(f"wrote {paths['trials'].name}")

# ---- right: data configs -------------------------------------------------- #
with right:
    st.markdown("**Configs on disk**")
    available = ctl.available_configs()
    chosen = st.selectbox(
        "load a config", ["—"] + list(available), label_visibility="collapsed",
        help="Ready-made worlds from configs/. The control_* entries are the ones "
             "that make the tool trustworthy: all-dead, all-random and "
             "label-shuffle must all come out `degenerate`.")
    if chosen != "—" and st.button(
            "load onto canvas", width="stretch",
            help="Replace everything on screen with this config, ready to edit."):
        ctl.load_run_config_into_state(load_run_config(available[chosen]))
        st.session_state["status"] = f"loaded {chosen}"
        st.rerun()

    st.divider()
    save_name = st.text_input(
        "save current as", value="my_scenario", label_visibility="visible",
        help="Filename stem, without .json. Reusing a name overwrites it.")
    if st.button("＋ save config", width="stretch",
                 help="Write the current world, observer, training and evaluation "
                      "settings to configs/ as one JSON file. Saved configs also "
                      "run headlessly via scripts/run_experiment.py."):
        path = save_config(ctl.run_config(save_name), ctl.CONFIG_DIR, name=save_name)
        st.success(f"wrote configs/{path.name}")

    st.divider()
    st.markdown("**Zone list**")
    if not st.session_state["zones"]:
        st.caption("Click the canvas with a zone tool selected.")
    for zone in list(st.session_state["zones"]):
        colour = plots.ZONE_COLOURS.get(zone["kind"], "#000")
        with st.expander(f"{zone['id']} · {zone['kind']} · g={zone['gain']:g}"):
            st.markdown(f"<div style='height:3px;background:{colour}'></div>",
                        unsafe_allow_html=True)
            zone["center_x"] = st.slider(
                "x", 0.0, float(data_cfg.plane.width - 1), zone["center_x"], 1.0,
                key=f"{zone['id']}x",
                help="Centre column of the zone, from the left edge.")
            zone["center_y"] = st.slider(
                "y", 0.0, float(data_cfg.plane.height - 1), zone["center_y"], 1.0,
                key=f"{zone['id']}y",
                help="Centre row of the zone. Row 0 is the TOP edge, matching how "
                     "the canvas is drawn and how the arrays are indexed.")
            zone["radius"] = st.slider("radius", 1.0, 200.0, zone["radius"], 1.0,
                                      key=f"{zone['id']}r",
                                      help="May exceed the plane — that is how a "
                                           "whole-plane control is expressed.")
            if zone["kind"] != ZoneKind.RANDOM.value:
                zone["gain"] = st.slider(
                    "gain a", -3.0, 3.0, zone["gain"], 0.05, key=f"{zone['id']}g",
                    help="|a| > 1 makes theta leave [0,1] and clip. That is the "
                         "saturation condition, not a mistake.")
                zone["offset"] = st.slider(
                    "offset b", -0.5, 0.5, zone["offset"], 0.01, key=f"{zone['id']}o",
                    help="A contrast-INDEPENDENT brightness shift: it moves the "
                         "zone up or down without changing how it responds to the "
                         "latent variable. Anything nonzero also makes clipping "
                         "reachable. Leave at 0 unless you are probing that.")
            zone["falloff"] = st.selectbox(
                "edge", ["hard", "smoothstep", "gaussian"],
                index=["hard", "smoothstep", "gaussian"].index(zone["falloff"]),
                key=f"{zone['id']}f",
                help="How membership fades at the rim. `hard` is a step, which a "
                     "5-wide patch straddling the edge averages across — a real "
                     "effect worth studying, but a choice rather than a default.")
            zone["falloff_width"] = st.slider(
                "edge width", 0.0, 15.0, zone["falloff_width"], 0.5,
                key=f"{zone['id']}fw",
                help="Pixels over which membership decays. Because overlap is "
                     "averaged, a zone only reaches its nominal gain in its core — "
                     "a wide edge means a large weak skirt.")
            zone["priority"] = st.number_input(
                "priority (replace mode)", -10, 10, int(zone["priority"]),
                key=f"{zone['id']}p",
                help="Only used when zone overlap is set to `replace`: the highest "
                     "priority wins the shared pixels, like z-order in a drawing "
                     "program. Ignored in the default `blend` mode.")
            if st.button("remove", key=f"{zone['id']}del",
                         help="Delete this zone. The world changes immediately; any "
                              "already-trained model does not."):
                ctl.remove_zone(zone["id"])
                st.rerun()


# --------------------------------------------------------------------------- #
# Global controls
# --------------------------------------------------------------------------- #

st.divider()
with st.expander("World, observer and training settings", expanded=True):
    world, observer, training = st.columns(3, gap="large")

    with world:
        st.markdown("**World** — how `c` drives the pixels")
        size = st.columns(2)
        st.session_state["plane_height"] = size[0].number_input(
            "plane height", 20, 400, st.session_state["plane_height"], 10,
            help="Rows of pixels. Larger planes are slower to generate and give "
                 "the grid button more patches, so features grow quickly.")
        st.session_state["plane_width"] = size[1].number_input(
            "plane width", 20, 400, st.session_state["plane_width"], 10,
            help="Columns of pixels. 100×100 is the size the study was specified "
                 "for; whether conclusions depend on it is untested.")
        st.session_state["background_gain"] = st.slider(
            "background gain", -1.5, 2.0, st.session_state["background_gain"], 0.05,
            help="How strongly pixels OUTSIDE every zone track c. Set 0 for the "
                 "all-dead null control; below 1 to give `strong` zones room to "
                 "beat the background without clipping.")
        st.session_state["pivot"] = st.slider(
            "pivot", 0.0, 1.0, st.session_state["pivot"], 0.05,
            help="Contrast around which gains rotate. Zones are invisible here. "
                 "Anything other than 0.5 makes clipping reachable.")
        st.session_state["overlap_mode"] = st.radio(
            "zone overlap", ["blend", "replace"], horizontal=True,
            index=["blend", "replace"].index(st.session_state["overlap_mode"]),
            help="blend = partition-of-unity average (bounded by construction); "
                 "replace = highest-priority zone wins, like z-order paint.")
        st.session_state["pixel_model"] = st.radio(
            "pixel model", ["bernoulli", "clipped_gaussian", "beta"], horizontal=True,
            index=["bernoulli", "clipped_gaussian", "beta"].index(
                st.session_state["pixel_model"]),
            help="bernoulli: contrast IS the chance a pixel is white; no clipping "
                 "bias, but variance vanishes at the ends. clipped_gaussian: "
                 "continuous, with censoring bias. beta: bounded, no clipping — "
                 "the no-artefact reference.")
        if st.session_state["pixel_model"] == "clipped_gaussian":
            st.session_state["sigma"] = st.slider(
                "pixel noise σ", 0.01, 0.5, st.session_state["sigma"], 0.01,
                help="Gaussian noise added per pixel before values are censored "
                     "into [0,1]. Roughly constant across the range, which is why "
                     "this pixel model is the closest to homoscedastic.")
        elif st.session_state["pixel_model"] == "beta":
            st.session_state["beta_concentration"] = st.slider(
                "beta precision ν (higher = less noise)", 2.0, 200.0,
                st.session_state["beta_concentration"], 1.0,
                help="Concentration of the Beta draw. Its variance depends on the "
                     "mean, so precision still varies across the contrast range "
                     "even though nothing is ever clipped.")
        st.session_state["random_zone_dist"] = st.radio(
            "random-zone draw", ["bernoulli_half", "uniform"], horizontal=True,
            index=["bernoulli_half", "uniform"].index(
                st.session_state["random_zone_dist"]),
            help="What a masked pixel inside a `random` zone shows instead of "
                 "tracking the contrast. `bernoulli_half` is a coin flip per "
                 "pixel (high variance); `uniform` is a flat draw in [0,1]. Both "
                 "are contrast-independent, so both are pure nuisance.")
        st.session_state["jitter_enabled"] = st.toggle(
            "geometric jitter (zones move between trials)",
            st.session_state["jitter_enabled"],
            help="A COHERENT nuisance: it moves whole zones together within a "
                 "trial, so averaging over trials does not remove its effect on "
                 "the shape of f(c).")
        if st.session_state["jitter_enabled"]:
            jitter = st.columns(3)
            st.session_state["jitter_center"] = jitter[0].slider(
                "centre σ px", 0.0, 10.0, st.session_state["jitter_center"], 0.5,
                help="How far each zone's centre wanders between trials. Zone "
                     "edges then cross patches differently on every trial, which "
                     "can make precision depend on the contrast level.")
            st.session_state["jitter_radius"] = jitter[1].slider(
                "radius σ px", 0.0, 10.0, st.session_state["jitter_radius"], 0.5,
                help="How much each zone's size wanders between trials.")
            st.session_state["jitter_gain"] = jitter[2].slider(
                "gain σ", 0.0, 0.5, st.session_state["jitter_gain"], 0.01,
                help="> 0 is a THIRD path to clipping: it can push |a| past 1 on "
                     "individual trials even when every configured gain is legal.")

    with observer:
        st.markdown("**Observer** — what the LDA gets to see")
        st.session_state["patch_size"] = st.slider(
            "patch size", 2, 20, st.session_state["patch_size"], 1,
            help="Side length of each square sampling window, in pixels. With "
                 "`pixels` features this squares the feature count, so 5 means 25 "
                 "features per patch.")
        st.session_state["grid_step"] = st.slider(
            "grid step (for the grid button)", 1, 40, st.session_state["grid_step"], 1,
            help="Step smaller than patch size makes patches overlap, which "
                 "duplicates pixels in the feature vector without adding "
                 "information.")
        st.session_state["feature_mode"] = st.radio(
            "features", ["pixels", "patch_mean", "patch_mean_std"],
            index=["pixels", "patch_mean", "patch_mean_std"].index(
                st.session_state["feature_mode"]),
            help="pixels: every sampled pixel (fast to generate, slow to fit — "
                 "2500 features on a step-10 grid). patch_mean: one value per "
                 "patch, much faster and higher SNR. patch_mean_std: adds the "
                 "within-patch SD, letting the model use variance cues.")
        n_patches = len(st.session_state["patches"])
        multiplier = {"pixels": st.session_state["patch_size"] ** 2,
                      "patch_mean": 1, "patch_mean_std": 2}[
            st.session_state["feature_mode"]]
        st.metric("features the model will see", n_patches * multiplier,
                  help="Length of the vector handed to the LDA. Once this exceeds "
                       "the number of training trials the covariance estimate is "
                       "rank-deficient — expected here, and why shrinkage is on by "
                       "default. It is also the main driver of fitting time.")
        st.caption(f"{n_patches} patches × {multiplier}")

    with training:
        st.markdown("**Training & evaluation**")
        extremes = st.columns(2)
        st.session_state["c_lo"] = extremes[0].slider(
            "train c_lo", 0.0, 0.45, st.session_state["c_lo"], 0.01,
            help="0.0 is legal but under the Bernoulli model gives an all-black "
                 "field with zero variance, so the covariance is singular.")
        st.session_state["c_hi"] = extremes[1].slider(
            "train c_hi", 0.55, 1.0, st.session_state["c_hi"], 0.01,
            help="The upper training extreme. Together with c_lo these are the "
                 "ONLY contrast levels the model ever sees while learning — "
                 "everything in between is a test of the assumption.")
        st.session_state["n_per_class"] = st.slider(
            "training trials per extreme", 20, 800,
            st.session_state["n_per_class"], 10,
            help="How many fields are generated at each extreme. Too few relative "
                 "to the feature count and the model overfits nuisance variance — "
                 "watch `top 1% weight mass` in the model tab for that.")
        st.session_state["grid_points"] = st.slider(
            "evaluation contrast levels", 5, 41,
            st.session_state["grid_points"], 2,
            help="How many points between 0 and 1 the trained model is probed at. "
                 "More levels resolve the shape of the curve better; the curvature "
                 "index needs enough of them to separate a bend from noise.")
        st.session_state["n_per_contrast"] = st.slider(
            "trials per contrast level", 10, 400,
            st.session_state["n_per_contrast"], 10,
            help="Fresh trials generated at each level — never reused from "
                 "training. This sets how precisely the spread of f is measured, "
                 "so it drives the H3 verdict. The pre-registered minimum is 200.")
        solver_columns = st.columns(2)
        st.session_state["solver"] = solver_columns[0].radio(
            "solver", ["lsqr", "eigen", "svd"],
            index=["lsqr", "eigen", "svd"].index(st.session_state["solver"]),
            help="How scikit-learn solves the discriminant. `lsqr` with shrinkage "
                 "is the stable choice when features outnumber trials. `svd` is "
                 "unregularised and cannot take shrinkage at all. A result that "
                 "only holds for one solver is a result about the solver.")
        shrinkage_choice = solver_columns[1].radio(
            "shrinkage", ["auto", "none"],
            index=0 if st.session_state["shrinkage"] == "auto" else 1,
            help="svd cannot take shrinkage; the config validator rejects that "
                 "pairing, so choosing svd forces none.")
        st.session_state["shrinkage"] = (
            None if shrinkage_choice == "none" or st.session_state["solver"] == "svd"
            else "auto")
        st.session_state["shuffle_labels"] = st.toggle(
            "shuffle labels (null control)", st.session_state["shuffle_labels"],
            help="Permutes the training labels only. Any remaining c-dependence in "
                 "f is a pipeline artefact, not a finding.")
        st.session_state["compute_oracle"] = st.toggle(
            "compute the oracle readout", st.session_state["compute_oracle"],
            help="A readout built from the TRUE gain field. It separates 'the LDA "
                 "failed' from 'no linear readout was possible'.")
        st.session_state["master_seed"] = st.number_input(
            "master seed", 0, 10_000, st.session_state["master_seed"], 1,
            help="The only seed. Everything — pixel noise, jitter, patch placement "
                 "— derives from it, so the same seed reproduces a run exactly. "
                 "Change it to check a finding is not one lucky draw.")


# --------------------------------------------------------------------------- #
# Analysis panel
# --------------------------------------------------------------------------- #

st.divider()

if run_clicked and run_cfg is None:
    st.error("Nothing to train on: place at least one sampling patch first.")
elif run_clicked:
    with st.spinner("training on the extremes, then probing the grid…"):
        try:
            result = ctl.cached_experiment(run_cfg.model_dump_json())
        except Exception as error:
            st.error(f"Run failed: {type(error).__name__}: {error}")
            st.stop()
    ctl.register_model(result)
    st.session_state["status"] = f"trained {result.model_id}"

active_id = st.session_state["active_model"]
if active_id is None or active_id not in st.session_state["models"]:
    st.info("Press **▶ Train & evaluate** to fit an LDA on `c_lo` vs `c_hi` and "
            "probe it across the whole contrast range.")
    st.stop()

result = st.session_state["models"][active_id]
report = result.report
summary = ctl.summarise(result)

# Warn loudly when the displayed model no longer matches the canvas. This is the
# single most confusing state a cached Streamlit app can be in: the figures keep
# showing a model that the settings on screen would no longer produce.
if ctl.run_config().model_dump_json() != result.config.model_dump_json():
    st.warning("The canvas has changed since this model was trained — the figures "
               "below describe the **model**, not what is now on screen. Press "
               "**▶ Train & evaluate** to catch up.")

VERDICT_COLOUR = {"affine": "normal", "equivocal": "off", "nonlinear": "inverse",
                  "degenerate": "off"}
metrics = st.columns(6)
metrics[0].metric("verdict (H2: affine?)", report.verdict,
                  delta=None, delta_color=VERDICT_COLOUR.get(report.verdict, "off"),
                  help="The pre-registered answer, decided by thresholds fixed "
                       "before any result was seen. `affine` = the mapping is "
                       "straight; `nonlinear` = it bends; `equivocal` = in between; "
                       "`degenerate` = the slope is not resolved above noise, which "
                       "is the CORRECT answer when nothing tracks the contrast.")
metrics[1].metric("κ curvature index",
                  "n/a" if report.curvature_index is None
                  else f"{report.curvature_index:.4f}",
                  help="|β₂|/|β₁| from an orthogonal-polynomial fit. Below 0.05 is "
                       "affine, above 0.15 a clear violation. THIS is the headline "
                       "number, not r.")
metrics[2].metric("ρ Spearman", f"{report.spearman_rho:.4f}",
                  help="H1: monotone? A high ρ with a high κ means f is usable as "
                       "an ordering but not as a measurement.")
metrics[3].metric("r Pearson", f"{report.pearson_r:.4f}",
                  help="Never read this alone: a visibly S-shaped curve can sit at "
                       "r = 0.99. The residual panel below is the check.")
metrics[4].metric("SD max/min (H3)", f"{report.sd_ratio:.2f}",
                  "homoscedastic" if report.homoscedastic else "heteroscedastic",
                  delta_color="normal" if report.homoscedastic else "inverse",
                  help="How much the PRECISION of f varies across the range. A "
                       "straight mean curve with a 20× swing in spread still "
                       "breaks f as a constant-precision measurement.")
metrics[5].metric("calibration RMSE",
                  "n/a" if report.calibration is None
                  else f"{report.calibration['rmse'].mean():.4f}",
                  help="Invert the affine fit and use f to estimate c. This is the "
                       "error, in units of c.")

if summary.below_preregistered_minimum:
    st.caption(f"⚠︎ {summary.n_per_contrast} trials per contrast level is below the "
               f"pre-registered minimum of {ctl.PRE_REGISTERED_MIN_PER_LEVEL} "
               "(plan.md §1.3). Fine for exploring; switch to the **careful** "
               "preset before quoting a verdict.")
if report.verdict == "degenerate":
    st.info("**degenerate** is not an error: the slope is not resolved above noise, "
            "so every slope-normalised number (κ, calibration) is 0/0 and is "
            "reported as n/a. This is the expected answer when nothing tracks `c` — "
            "the all-dead, all-random and shuffled-label controls.")

tabs = st.tabs(["mapping", "spread & residuals", "what the model used",
                "model & sampling detail", "compare models"])

with tabs[0]:
    left_pane, right_pane = st.columns([3, 2])
    with left_pane:
        st.plotly_chart(plots.mapping_figure(report, result.oracle_report),
                        width="stretch")
    with right_pane:
        st.plotly_chart(plots.scatter_figure(result.trials), width="stretch")
    st.plotly_chart(plots.histogram_figure(result.trials), width="stretch")
    if result.oracle_report is not None:
        oracle_kappa = result.oracle_report.curvature_index
        st.caption(
            f"Oracle readout (built from the true gain field): "
            f"verdict **{result.oracle_report.verdict}**, "
            f"κ = {'n/a' if oracle_kappa is None else f'{oracle_kappa:.4f}'}. "
            "If the oracle is affine where the LDA is not, the distortion belongs "
            "to the method; if both bend, it is baked into the generative chain.")

with tabs[1]:
    columns = st.columns(2)
    with columns[0]:
        st.plotly_chart(plots.residual_figure(report), width="stretch")
        st.caption("Systematic departure from the affine fit. Flat and centred on "
                   "zero means genuinely affine; a smile or an S is curvature that "
                   "a correlation coefficient will not show you.")
    with columns[1]:
        st.plotly_chart(plots.spread_figure(report), width="stretch")
        st.caption(f"{report.n_zero_variance_levels} contrast level(s) had exactly "
                   "zero variance and are excluded from the SD ratio — under the "
                   "Bernoulli model θ ∈ {0,1} makes every pixel deterministic.")
    st.dataframe(report.mean_by_c, width="stretch", hide_index=True)

with tabs[2]:
    st.plotly_chart(plots.weight_figure(result.attribution), width="stretch")
    columns = st.columns([1, 1])
    with columns[0]:
        st.markdown("**Weight by zone kind**")
        st.dataframe(result.attribution.mean_signed_weight_by_kind,
                     width="stretch", hide_index=True)
        st.caption(f"|w| vs |a| correlation {result.attribution.abs_correlation:.3f} · "
                   f"sign agreement {result.attribution.sign_agreement:.3f}. Expect "
                   "negative weight in `anti` zones and near-zero in `dead` and "
                   "`random`. Large weight inside a random zone means the model is "
                   "fitting nuisance variance.")
    with columns[1]:
        st.markdown("**What the patches actually cover**")
        st.dataframe(result.composition, width="stretch", hide_index=True)
        st.caption("Patch placement can decide the answer on its own: patches on "
                   "heat zones give a near-perfect mapping, patches on dead zones "
                   "give noise. Coupling kinds plus background sum to 1; `random` "
                   "is reported separately because a masked pixel leaves the "
                   "coupling composition entirely.")

with tabs[3]:
    diagnostics = result.model.diagnostics
    columns = st.columns(4)
    columns[0].metric("features", diagnostics.n_features,
                      help="Length of the feature vector the LDA was fitted on.")
    columns[1].metric("training trials", diagnostics.n_train,
                      help="Both extremes combined, so twice the per-extreme count.")
    columns[2].metric("train accuracy", f"{diagnostics.train_accuracy:.3f}",
                      help="Accuracy on the data it was fitted to. Near 1.0 is "
                           "normal and means little — the two extremes are easy to "
                           "separate. It is not evidence the readout is linear.")
    columns[3].metric("d′ between extremes", f"{diagnostics.d_prime:.1f}",
                      help="Separation of the two training classes in units of "
                           "their own spread. Large values mean the extremes are "
                           "far apart, which says nothing about the middle.")
    columns = st.columns(4)
    columns[0].metric("covariance condition",
                      "singular" if not np.isfinite(diagnostics.cov_condition_number)
                      else f"{diagnostics.cov_condition_number:.3g}",
                      help="Conditioning of the within-class covariance. "
                           "`singular` is EXPECTED whenever features outnumber "
                           "training trials — it is not a failure, and it is why "
                           "shrinkage is the default.")
    columns[1].metric("effective rank", diagnostics.effective_rank,
                      help="How many independent directions the training data "
                           "actually spans. Capped by the number of trials, so it "
                           "sits below the feature count whenever features win.")
    columns[2].metric("top 1% weight mass", f"{diagnostics.top1pct_weight_mass:.3f}",
                      help="Share of the weight vector's squared magnitude carried "
                           "by its largest 1% of entries. High values mean the "
                           "model leans on a handful of pixels and will not "
                           "survive a change of seed.")
    columns[3].metric("cross-val accuracy",
                      "not computed" if np.isnan(diagnostics.cv_accuracy)
                      else f"{diagnostics.cv_accuracy:.3f}",
                      help="Held-out accuracy on the training extremes only. Off "
                           "in the fast presets because it costs several extra "
                           "fits and says nothing about the intermediate levels.")
    st.caption("A singular covariance is EXPECTED whenever features outnumber "
               "training trials — that is why shrinkage is on by default. A high "
               "top-1% weight mass means the model leans on a few pixels and will "
               "not survive a change of seed.")
    st.markdown("**Slope and curvature detail**")
    st.json({
        "beta0": report.beta0, "beta1": report.beta1,
        "beta1_ci": list(report.beta1_ci), "slope_t": report.slope_t,
        "r2": report.r2, "orthogonal_poly_coefs": report.poly_coefs,
        "nested_f": report.nested_f,
        "max_local_slope_ratio": report.max_local_slope_ratio,
        "effective_range_use": report.effective_range_use,
        "monotonicity_violations": report.monotonicity_violations,
    }, expanded=False)
    st.markdown("**Config that produced this**")
    st.json(json.loads(result.config.model_dump_json()), expanded=False)

with tabs[4]:
    models = st.session_state["models"]
    if len(models) < 2:
        st.caption("Train at least two models to compare them. A useful pair: the "
                   "same world under `bernoulli` and under `clipped_gaussian`, or "
                   "the same world sampled on heat zones versus dead zones.")
    else:
        chosen = st.multiselect(
            "models to overlay", list(models), default=list(models)[:4],
            format_func=lambda mid: ctl.summarise(models[mid]).label,
            help="Pick two or more to compare their SHAPES. Each curve is "
                 "rescaled to [0,1] first, because decision values have arbitrary "
                 "units and an unscaled overlay would show differences that are "
                 "purely units.")
        if chosen:
            st.plotly_chart(
                plots.comparison_figure({ctl.summarise(models[mid]).label:
                                         models[mid].report for mid in chosen}),
                width="stretch")
            st.caption("Each curve is affinely rescaled to [0, 1]: decision values "
                       "have arbitrary units, so only SHAPE is comparable. The "
                       "dashed diagonal is perfectly affine.")
            table = pd.DataFrame([
                {"model": ctl.summarise(models[mid]).label,
                 **models[mid].report.summary_row()} for mid in chosen])
            st.dataframe(table, width="stretch", hide_index=True)
