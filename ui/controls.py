"""Session state, config assembly and cached computation for the Streamlit app.

Responsibility
--------------
Own everything that survives a rerun, translate widget state into validated
config objects, and wrap the expensive `core/` calls in caches keyed on config
content.

Explicitly NOT this module's job
--------------------------------
Any science, and any layout. Layout is `app.py`; figures are `ui/plots.py`.

Why this module exists at all
----------------------------
Streamlit reruns the whole script on every interaction (CLAUDE.md §11, trap 7).
Without a single place that owns state and cache keys, a widget click can silently
retrain a model or — worse — half-retrain it against a config that has since
changed. Every cache here is keyed on the JSON of a validated config, so a cache
hit is provably a hit on the same experiment.

Invariant I1
------------
Zones live in `state.zones` and become a `DataConfig`; patches live in
`state.patches` and become a `SamplingConfig`. The two lists never merge, and the
config builders below are the only place either is read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

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
from core.data import build_gain_field, generate_field, jitter_free_gain_field
from core.evaluation import ExperimentResult, run_experiment
from core.rng import spawn_streams
from core.sampling import Patch, build_patches
from storage.io import config_hash, provenance, run_identifiers

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

# Speed presets. The point of the app is exploration, so the default must return
# in well under a second: a two-second wait per slider nudge makes the tool
# unusable and pushes people back to guessing. Measured on a 100x100 plane with a
# step-10 grid: fast 0.35 s, standard 3.5 s, careful 6 s.
#
# ARTEFACT: the fast preset is BELOW the pre-registered minimum of 200 trials per
#           contrast level (plan.md §1.3). Its verdicts are for exploration only,
#           and the app labels them as such wherever a verdict is shown.
PRESETS: dict[str, dict[str, int | str]] = {
    "fast (explore)": {"n_per_class": 150, "n_per_contrast": 60, "grid_points": 11,
                       "feature_mode": "patch_mean", "cv_folds": 0},
    "standard": {"n_per_class": 300, "n_per_contrast": 120, "grid_points": 21,
                 "feature_mode": "pixels", "cv_folds": 0},
    "careful (pre-registered)": {"n_per_class": 400, "n_per_contrast": 200,
                                 "grid_points": 21, "feature_mode": "pixels",
                                 "cv_folds": 5},
}
PRE_REGISTERED_MIN_PER_LEVEL = 200

# Default geometry for a newly placed zone, per kind. Gains are the values that
# make each kind mean what its name says (plan.md §4.3).
ZONE_DEFAULTS: dict[str, dict[str, float]] = {
    ZoneKind.HEAT.value: {"radius": 15.0, "gain": 2.0},
    ZoneKind.STRONG.value: {"radius": 15.0, "gain": 1.0},
    ZoneKind.DEAD.value: {"radius": 15.0, "gain": 0.05},
    ZoneKind.ANTI.value: {"radius": 12.0, "gain": -0.8},
    ZoneKind.RANDOM.value: {"radius": 12.0, "gain": 1.0},
}


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

def init_state() -> None:
    """Create every session-state key exactly once.

    Uses `setdefault` semantics so a rerun never clobbers user edits. Zones and
    patches are stored as plain dicts/tuples rather than pydantic objects because
    widgets write into them in place; validation happens in the config builders.
    """
    defaults: dict[str, Any] = {
        "zones": [],                      # list[dict] -> ZoneSpec
        "patches": [],                    # list[tuple[int, int]] -> Patch origins
        "zone_counter": 0,
        "tool": ZoneKind.HEAT.value,      # what a canvas click places
        "plane_height": 100,
        "plane_width": 100,
        "pivot": 0.5,
        "background_gain": 1.0,
        "overlap_mode": "blend",
        "pixel_model": "bernoulli",
        "sigma": 0.1,
        "beta_concentration": 20.0,
        "random_zone_dist": "bernoulli_half",
        "jitter_enabled": False,
        "jitter_center": 2.0,
        "jitter_radius": 1.0,
        "jitter_gain": 0.0,
        "patch_size": 5,
        "grid_step": 10,
        "feature_mode": "patch_mean",
        "c_lo": 0.05,
        "c_hi": 0.95,
        "n_per_class": 150,
        "solver": "lsqr",
        "shrinkage": "auto",
        "shuffle_labels": False,
        "cv_folds": 0,
        "grid_points": 11,
        "n_per_contrast": 60,
        "compute_oracle": True,
        "master_seed": 0,
        "preset": "fast (explore)",
        "preview_contrast": 0.75,
        "show_gain": False,
        "show_zones": True,
        "show_patches": True,
        "models": {},                     # model_id -> ExperimentResult
        "active_model": None,
        "saved_data_configs": {},         # name -> DataConfig JSON
        "last_click": None,
        "last_placed": (None, None),      # ("zone", id) | ("patch", None) — for undo
        "status": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Seed a starting patch grid ONCE, on genuinely first load, and never again.
    #
    # The obvious version of this -- `if not state["patches"]: auto_grid_patches()`
    # -- silently breaks the "clear patches" button: Streamlit reruns the script
    # after the click, `init_state` sees an empty list and refills it, so the
    # observer can never be emptied and the user cannot see what a model with no
    # patches does. A separate flag is the whole fix.
    if "initialised" not in st.session_state:
        st.session_state["initialised"] = True
        auto_grid_patches()


def apply_preset(name: str) -> None:
    """Overwrite the trial-count and feature settings from a named preset."""
    preset = PRESETS[name]
    st.session_state["n_per_class"] = preset["n_per_class"]
    st.session_state["n_per_contrast"] = preset["n_per_contrast"]
    st.session_state["grid_points"] = preset["grid_points"]
    st.session_state["feature_mode"] = preset["feature_mode"]
    st.session_state["cv_folds"] = preset["cv_folds"]
    st.session_state["preset"] = name


# --------------------------------------------------------------------------- #
# Zone and patch editing
# --------------------------------------------------------------------------- #

def add_zone(kind: str, x: float, y: float) -> str:
    """Place a new zone of `kind` centred on (x, y). Returns its id."""
    st.session_state["zone_counter"] += 1
    zone_id = f"{kind}{st.session_state['zone_counter']}"
    defaults = ZONE_DEFAULTS[kind]
    st.session_state["zones"].append({
        "id": zone_id, "kind": kind,
        "center_x": float(np.clip(x, 0, st.session_state["plane_width"] - 1)),
        "center_y": float(np.clip(y, 0, st.session_state["plane_height"] - 1)),
        "radius": defaults["radius"], "gain": defaults["gain"], "offset": 0.0,
        "falloff": Falloff.SMOOTHSTEP.value, "falloff_width": 3.0, "priority": 0,
    })
    return zone_id


def remove_zone(zone_id: str) -> None:
    st.session_state["zones"] = [z for z in st.session_state["zones"]
                                if z["id"] != zone_id]


def add_patch(row: float, col: float) -> None:
    """Place one sampling patch with its top-left corner near (row, col).

    The click marks the patch CENTRE, which is what a user means when they click,
    so the corner is offset by half the patch size and clamped to the plane.

    The offset is `round(coord) - size // 2`, NOT `round(coord - size / 2)`. The
    latter goes through Python's banker's rounding on the .5 cases, so a click at
    row 21 with a 5-wide patch lands at 18 while row 22 lands at 20 — an
    off-by-one that appears for some clicks and not others. Integer division first
    makes the click pixel the exact centre for every odd patch size. Even sizes
    have no exact centre, so they sit half a pixel up and left.
    """
    size = st.session_state["patch_size"]
    max_row = st.session_state["plane_height"] - size
    max_col = st.session_state["plane_width"] - size
    st.session_state["patches"].append((
        int(np.clip(int(round(row)) - size // 2, 0, max_row)),
        int(np.clip(int(round(col)) - size // 2, 0, max_col)),
    ))


def auto_grid_patches() -> None:
    """Replace the patch list with a uniform grid at the current step.

    This is the "somewhat uniform" baseline layout, and the starting point for any
    sampling ablation: place the grid, then delete or add patches by clicking.
    """
    size, step = st.session_state["patch_size"], st.session_state["grid_step"]
    rows = range(0, st.session_state["plane_height"] - size + 1, step)
    cols = range(0, st.session_state["plane_width"] - size + 1, step)
    st.session_state["patches"] = [(r, c) for r in rows for c in cols]


def place_with_current_tool(x: float, y: float) -> str:
    """Apply the selected toolbar tool at plane coordinate (x, y).

    One function for both input routes — a canvas click and the numeric "place
    here" button — so the two can never drift apart in what they do. Returns the
    status line the app shows under the canvas.

    Note the axis convention: `x` is the column and `y` is the row, matching the
    figure's reversed y-axis so that row 0 is at the top. Getting this backwards
    puts zones in the mirrored position and is very easy to do, which is why the
    swap happens here and nowhere else.
    """
    tool = st.session_state["tool"]
    if tool in {kind.value for kind in ZoneKind}:
        zone_id = add_zone(tool, x, y)
        st.session_state["last_placed"] = ("zone", zone_id)
        status = f"placed zone {zone_id} at ({x:.0f}, {y:.0f})"
    elif tool == "patch":
        add_patch(row=y, col=x)
        st.session_state["last_placed"] = ("patch", None)
        status = f"placed a sampling patch at ({x:.0f}, {y:.0f})"
    elif tool == "erase_patch":
        removed = remove_patch_near(row=y, col=x)
        st.session_state["last_placed"] = (None, None)
        status = ("removed a patch" if removed
                  else f"no patch within one patch width of ({x:.0f}, {y:.0f})")
    else:
        status = "tool is 'look only' — nothing placed"
    st.session_state["status"] = status
    return status


def undo_last_placement() -> str:
    """Remove whatever was placed most recently.

    Only one step deep. A full undo stack would mean snapshotting the whole world
    on every edit; for a tool where the alternative is "drag the slider back" the
    single step covers the actual mistake, which is a misplaced click.
    """
    # Subscript, not `.get`: Streamlit's session-state proxy resolves attribute
    # access as a KEY lookup, so `st.session_state.get(...)` raises KeyError for a
    # missing key named "get" instead of returning a default.
    kind, identifier = (st.session_state["last_placed"]
                        if "last_placed" in st.session_state else (None, None))
    if kind == "zone" and identifier:
        remove_zone(identifier)
        st.session_state["last_placed"] = (None, None)
        return f"removed zone {identifier}"
    if kind == "patch" and st.session_state["patches"]:
        st.session_state["patches"].pop()
        st.session_state["last_placed"] = (None, None)
        return "removed the last patch"
    return "nothing to undo"


def remove_patch_near(row: float, col: float) -> bool:
    """Delete the patch whose centre is nearest the click, if one is close enough.

    "Close enough" is one patch width, so a click in empty space does not silently
    delete a distant patch.
    """
    patches = st.session_state["patches"]
    if not patches:
        return False
    size = st.session_state["patch_size"]
    centres = np.array([(r + size / 2, c + size / 2) for r, c in patches])
    distances = np.hypot(centres[:, 0] - row, centres[:, 1] - col)
    nearest = int(np.argmin(distances))
    if distances[nearest] > size:
        return False
    patches.pop(nearest)
    return True


# --------------------------------------------------------------------------- #
# State -> validated configs
# --------------------------------------------------------------------------- #

def data_config(name: str = "interactive") -> DataConfig:
    """Assemble the WORLD config from session state. Reads zones, never patches."""
    return DataConfig(
        name=name,
        plane=PlaneSpec(height=st.session_state["plane_height"],
                        width=st.session_state["plane_width"]),
        pivot=st.session_state["pivot"],
        background_gain=st.session_state["background_gain"],
        overlap_mode=st.session_state["overlap_mode"],
        zones=[ZoneSpec(**z) for z in st.session_state["zones"]],
        jitter=JitterSpec(enabled=st.session_state["jitter_enabled"],
                          center_sigma_px=st.session_state["jitter_center"],
                          radius_sigma_px=st.session_state["jitter_radius"],
                          gain_sigma=st.session_state["jitter_gain"]),
        noise=NoiseSpec(pixel_model=st.session_state["pixel_model"],
                        sigma=st.session_state["sigma"],
                        beta_concentration=st.session_state["beta_concentration"],
                        random_zone_dist=st.session_state["random_zone_dist"]),
    )


def sampling_config(name: str = "interactive") -> SamplingConfig:
    """Assemble the OBSERVER config from session state. Reads patches, never zones.

    Always `layout="manual"`: the app's grid button writes an explicit patch list,
    so what is saved is exactly what was on screen. A `uniform_grid` layout would
    re-derive the patches at run time and silently diverge from the canvas after a
    change of plane size or patch size.
    """
    return SamplingConfig(
        name=name, layout="manual", patch_size=st.session_state["patch_size"],
        grid_step=st.session_state["grid_step"],
        feature_mode=st.session_state["feature_mode"],
        patches=[(int(r), int(c)) for r, c in st.session_state["patches"]],
    )


def training_config() -> TrainingConfig:
    return TrainingConfig(
        c_lo=st.session_state["c_lo"], c_hi=st.session_state["c_hi"],
        n_per_class=st.session_state["n_per_class"],
        solver=st.session_state["solver"], shrinkage=st.session_state["shrinkage"],
        shuffle_labels=st.session_state["shuffle_labels"],
        cv_folds=st.session_state["cv_folds"],
    )


def eval_config() -> EvalConfig:
    return EvalConfig(
        contrast_grid=np.linspace(0.0, 1.0, st.session_state["grid_points"]).tolist(),
        n_per_contrast=st.session_state["n_per_contrast"],
        compute_oracle=st.session_state["compute_oracle"],
    )


def run_config(name: str = "interactive") -> RunConfig:
    return RunConfig(data=data_config(name), sampling=sampling_config(name),
                     training=training_config(), evaluation=eval_config(),
                     master_seed=st.session_state["master_seed"])


def load_run_config_into_state(cfg: RunConfig) -> None:
    """Populate session state from a saved config, so it can be edited on screen."""
    state = st.session_state
    state["zones"] = [z.model_dump(mode="json") for z in cfg.data.zones]
    state["zone_counter"] = len(cfg.data.zones)
    state["plane_height"] = cfg.data.plane.height
    state["plane_width"] = cfg.data.plane.width
    state["pivot"] = cfg.data.pivot
    state["background_gain"] = cfg.data.background_gain
    state["overlap_mode"] = cfg.data.overlap_mode
    state["pixel_model"] = cfg.data.noise.pixel_model
    state["sigma"] = cfg.data.noise.sigma
    state["beta_concentration"] = cfg.data.noise.beta_concentration
    state["random_zone_dist"] = cfg.data.noise.random_zone_dist
    state["jitter_enabled"] = cfg.data.jitter.enabled
    state["jitter_center"] = cfg.data.jitter.center_sigma_px
    state["jitter_radius"] = cfg.data.jitter.radius_sigma_px
    state["jitter_gain"] = cfg.data.jitter.gain_sigma
    state["patch_size"] = cfg.sampling.patch_size
    state["grid_step"] = cfg.sampling.grid_step
    state["feature_mode"] = cfg.sampling.feature_mode
    state["c_lo"] = cfg.training.c_lo
    state["c_hi"] = cfg.training.c_hi
    state["n_per_class"] = cfg.training.n_per_class
    state["solver"] = cfg.training.solver
    state["shrinkage"] = cfg.training.shrinkage
    state["shuffle_labels"] = cfg.training.shuffle_labels
    state["cv_folds"] = cfg.training.cv_folds
    state["grid_points"] = len(cfg.evaluation.contrast_grid)
    state["n_per_contrast"] = cfg.evaluation.n_per_contrast
    state["master_seed"] = cfg.master_seed

    if cfg.sampling.layout == "manual" and cfg.sampling.patches:
        state["patches"] = [(int(r), int(c)) for r, c in cfg.sampling.patches]
    else:
        # A committed config may use a generated layout; realise it once so the
        # canvas shows the patches the model will actually see.
        streams = spawn_streams(cfg.master_seed)
        state["patches"] = [(p.row, p.col) for p in
                            build_patches(cfg.sampling, cfg.data.plane,
                                          streams["patches"])]


def available_configs() -> dict[str, Path]:
    """Committed run configs on disk, newest name order, for the load dropdown."""
    return {path.stem: path for path in sorted(CONFIG_DIR.glob("*.json"))}


# --------------------------------------------------------------------------- #
# Cached computation
# --------------------------------------------------------------------------- #
# Every cache is keyed on the JSON of a validated config, never on a widget value.
# That makes a cache hit provably a hit on the same experiment, and it means an
# edit anywhere in the config invalidates the cache exactly once.

@st.cache_data(show_spinner=False, max_entries=64)
def cached_preview(data_json: str, contrast: float, seed: int
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    """One realised field plus its gain field, for the canvas.

    Returns
    -------
    (field (H, W), gain (H, W), clip_fraction)

    Uses its own generator seeded from `seed` so the preview never consumes from
    an experiment's streams — the canvas must not perturb reproducibility.
    """
    data = DataConfig.model_validate_json(data_json)
    rng_field = np.random.default_rng(seed)
    rng_jitter = np.random.default_rng(seed + 1)
    field, gf, diagnostics = generate_field(data, contrast, rng_field, rng_jitter)
    return field, gf.a, diagnostics.clip_fraction


@st.cache_data(show_spinner=False, max_entries=16)
def cached_gain_field(data_json: str) -> np.ndarray:
    """The jitter-free coupling gain field, for the 'show gain' canvas mode."""
    return jitter_free_gain_field(DataConfig.model_validate_json(data_json)).a


@st.cache_resource(show_spinner=False, max_entries=32)
def cached_experiment(run_json: str) -> ExperimentResult:
    """Train and evaluate. Cached on the full RunConfig, so identical settings
    never retrain.

    `cache_resource` rather than `cache_data`: the result holds a fitted sklearn
    estimator, and we want the object itself back rather than a pickled copy per
    caller. Nothing mutates an `ExperimentResult` after it is built.
    """
    cfg = RunConfig.model_validate_json(run_json)
    return run_experiment(cfg, provenance=provenance(), ids=run_identifiers(cfg))


@dataclass(frozen=True)
class RunSummary:
    """Just enough about a trained model for the registry list."""

    model_id: str
    label: str
    verdict: str
    kappa: float | None
    sd_ratio: float
    n_per_contrast: int
    below_preregistered_minimum: bool


def summarise(result: ExperimentResult) -> RunSummary:
    report = result.report
    per_level = result.config.evaluation.n_per_contrast
    return RunSummary(
        model_id=result.model_id,
        label=f"{result.config.data.name} · seed {result.config.master_seed} · "
              f"{len(result.config.data.zones)} zones · "
              f"{result.model.diagnostics.n_features} feat",
        verdict=report.verdict,
        kappa=report.curvature_index,
        sd_ratio=report.sd_ratio,
        n_per_contrast=per_level,
        below_preregistered_minimum=per_level < PRE_REGISTERED_MIN_PER_LEVEL,
    )


def register_model(result: ExperimentResult) -> None:
    """Keep the result in session state and make it the active model."""
    st.session_state["models"][result.model_id] = result
    st.session_state["active_model"] = result.model_id


def config_fingerprint() -> str:
    """Short hash of the current data+sampling state, shown next to the canvas.

    Lets a user tell at a glance whether the displayed model matches what is now
    on the canvas — the single most confusing failure mode in a Streamlit app that
    caches expensive results.

    Tolerates an empty patch list, which `SamplingConfig` rejects: this string is
    decoration, and it must never be the thing that takes the page down while the
    user is midway through replacing the patch layout by hand.
    """
    world = config_hash(data_config())
    try:
        observer = config_hash(sampling_config())
    except Exception:
        observer = "no-patches"
    return f"{world}·{observer}"
