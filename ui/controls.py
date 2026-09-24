"""Session state, config assembly and cached computation for the Streamlit app.

Responsibility
--------------
Own everything that survives a rerun, translate widget state into validated
config objects, and wrap the expensive `core/` calls in caches keyed on config
content.

Explicitly NOT this module's job
--------------------------------
Any science, and any layout. Layout is `app.py`; figures are `ui/plots.py`.

The state model (read this before touching a widget)
----------------------------------------------------
Every widget is bound to a session-state KEY and is created WITHOUT a `value=`
argument; `init_state` puts the default in session state first. Streamlit then
writes a user's change into that key *before* the script reruns, so everything
computed from state — the canvas, the configs, the fingerprint — already sees
the new value on the same interaction.

The first version did `state[k] = st.slider(..., state[k])` with no key. That
looked equivalent but was not: the canvas was built from state before the
widgets further down wrote their new values, so every setting took effect one
interaction late, and a keyless widget's identity included its value so arrow-key
nudges only worked once (docs/decisions.md D15).

The corollary: a key may only be changed programmatically BEFORE its widget is
drawn in the current run. So every button that changes settings — load a
config, apply a preset, clear, undo — does its work in an `on_click` callback,
which Streamlit runs before the script body.

Zone editors follow the same rule with keys `zone:<id>:<field>`.
`sync_zones_from_widgets` copies those keys into the zone dicts at the top of
each run, so a slider edit is visible on the canvas immediately, and
`forget_zone_widgets` drops the keys whenever the zone list is replaced so a
loaded zone never inherits an old slider's value.

Invariant I1
------------
Zones live in `state.zones` and become a `DataConfig`; patches live in
`state.patches` and become a `SamplingConfig`. The two lists never merge, and the
config builders below are the only place either is read.
"""

from __future__ import annotations

import re
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
from core.data import generate_field, jitter_free_gain_field
from core.evaluation import ExperimentResult, run_experiment
from core.rng import spawn_streams
from core.sampling import build_patches
from storage.io import config_hash, provenance, run_identifiers, save_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
# Configs saved from the app go here, not into configs/ itself, so the canonical
# experiment definitions (checked against make_configs.py by the test suite) are
# never mixed with scratch scenarios.
USER_CONFIG_DIR = CONFIG_DIR / "user"

# Speed presets. The point of the app is exploration, so the default must return
# in well under a second: a two-second wait per slider nudge makes the tool
# unusable and pushes people back to guessing. Measured on a 100x100 plane with a
# step-10 grid: fast ~0.35 s, standard ~3.5 s, careful ~25 s (5-fold CV).
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
CUSTOM_PRESET = "custom"
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
ZONE_FIELDS = ("center_x", "center_y", "radius", "gain", "offset", "falloff",
               "falloff_width", "priority")

# Widget-bound settings and their defaults. Every one of these keys is also the
# `key=` of exactly one widget in app.py.
SETTING_DEFAULTS: dict[str, Any] = {
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
    "tool": ZoneKind.HEAT.value,
    "tool_last": ZoneKind.HEAT.value,
    "preview_contrast": 0.75,
    "show_gain": False,
    "show_zones": True,
    "show_patches": True,
    "place_x": 50,
    "place_y": 50,
    "save_name": "my_scenario",
    "load_choice": None,
}


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

def init_state() -> None:
    """Create every session-state key exactly once; never clobber user edits."""
    internal: dict[str, Any] = {
        "zones": [],                      # list[dict] -> ZoneSpec
        "patches": [],                    # list[tuple[int, int]] -> Patch origins
        "zone_counter": 0,
        "models": {},                     # run_id -> ExperimentResult
        "active_model": None,
        "last_placed": (None, None),      # ("zone", id) | ("patch", None) — for undo
        "canvas_generation": 0,           # bumped after each handled click
        "contrast_grid_override": None,   # a loaded non-uniform grid, kept verbatim
        "status": "",
        "notice": "",                     # one-shot message from a callback
    }
    for key, value in {**SETTING_DEFAULTS, **internal}.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Seed a starting patch grid ONCE per session. Refilling "whenever the list
    # is empty" silently broke the clear-patches button (docs/decisions.md D11).
    if "initialised" not in st.session_state:
        st.session_state["initialised"] = True
        auto_grid_patches()


def matching_preset() -> str:
    """The preset whose settings the current state matches, or `custom`."""
    for name, preset in PRESETS.items():
        if all(st.session_state[k] == v for k, v in preset.items()):
            return name
    return CUSTOM_PRESET


def on_preset_change() -> None:
    """Callback for the preset selectbox: overwrite the preset's fields."""
    name = st.session_state["preset"]
    if name in PRESETS:
        for key, value in PRESETS[name].items():
            st.session_state[key] = value
        st.session_state["contrast_grid_override"] = None


def on_grid_points_change() -> None:
    """Moving the level slider replaces any loaded non-uniform grid."""
    st.session_state["contrast_grid_override"] = None


def on_tool_change() -> None:
    """Keep a tool selected when the active one is clicked again.

    A segmented control DESELECTS its option when that option is clicked a second
    time. Clicking "heat" while heat was already active therefore silently set
    the tool to nothing, and the next canvas click placed nothing — found by
    clicking through the app in a browser.
    """
    if st.session_state["tool"] is None:
        st.session_state["tool"] = st.session_state["tool_last"]
    else:
        st.session_state["tool_last"] = st.session_state["tool"]


def on_solver_change() -> None:
    """svd cannot take shrinkage; switch it off rather than fail validation."""
    if st.session_state["solver"] == "svd":
        st.session_state["shrinkage"] = "none"


# --------------------------------------------------------------------------- #
# Zones
# --------------------------------------------------------------------------- #

def _zone_key(zone_id: str, field: str) -> str:
    return f"zone:{zone_id}:{field}"


def forget_zone_widgets() -> None:
    """Drop every zone-editor widget key.

    Called whenever the zone list is replaced or a zone removed. Without it a
    loaded zone that shares an id with an old one (both `heat1`) keeps the old
    zone's slider values, because an existing key wins over the new data.
    """
    for key in [k for k in st.session_state if str(k).startswith("zone:")]:
        del st.session_state[key]


def ensure_zone_widget_state(zone: dict[str, Any]) -> None:
    """Seed a zone's editor keys from its dict, if they do not exist yet."""
    for field in ZONE_FIELDS:
        key = _zone_key(zone["id"], field)
        if key not in st.session_state:
            st.session_state[key] = zone[field]


def sync_zones_from_widgets() -> None:
    """Copy zone-editor values into the zone dicts. Run at the top of each run."""
    for zone in st.session_state["zones"]:
        for field in ZONE_FIELDS:
            key = _zone_key(zone["id"], field)
            if key in st.session_state:
                zone[field] = st.session_state[key]


def zone_key(zone_id: str, field: str) -> str:
    """Public accessor so app.py never hand-builds a zone widget key."""
    return _zone_key(zone_id, field)


def _next_zone_id(kind: str) -> str:
    """A fresh id that cannot collide with any existing zone.

    The first version used `f"{kind}{counter}"` with `counter = len(zones)` after
    a load, so a config holding only `heat2` produced a second `heat2` and a
    DuplicateElementKey crash on the next placement.
    """
    taken = {z["id"] for z in st.session_state["zones"]}
    while True:
        st.session_state["zone_counter"] += 1
        candidate = f"{kind}{st.session_state['zone_counter']}"
        if candidate not in taken:
            return candidate


def add_zone(kind: str, x: float, y: float) -> str:
    """Place a new zone of `kind` centred on (x, y). Returns its id."""
    zone_id = _next_zone_id(kind)
    defaults = ZONE_DEFAULTS[kind]
    zone = {
        "id": zone_id, "kind": kind,
        "center_x": float(np.clip(x, 0, st.session_state["plane_width"] - 1)),
        "center_y": float(np.clip(y, 0, st.session_state["plane_height"] - 1)),
        "radius": defaults["radius"], "gain": defaults["gain"], "offset": 0.0,
        "falloff": Falloff.SMOOTHSTEP.value, "falloff_width": 3.0, "priority": 0,
    }
    st.session_state["zones"].append(zone)
    return zone_id


def remove_zone(zone_id: str) -> None:
    st.session_state["zones"] = [z for z in st.session_state["zones"]
                                 if z["id"] != zone_id]
    for field in ZONE_FIELDS:
        st.session_state.pop(_zone_key(zone_id, field), None)


def on_remove_zone(zone_id: str) -> None:
    remove_zone(zone_id)
    st.session_state["status"] = f"removed zone {zone_id}"


def on_clear_zones() -> None:
    st.session_state["zones"] = []
    forget_zone_widgets()
    st.session_state["status"] = "cleared all zones"


# --------------------------------------------------------------------------- #
# Patches
# --------------------------------------------------------------------------- #

def add_patch(row: float, col: float) -> bool:
    """Place one sampling patch whose CENTRE is at (row, col).

    Offset is `round(coord) - size // 2`, not `round(coord - size / 2)`: the
    latter hits banker's rounding on .5 and gave an off-by-one for some clicks
    only. Clamped so the whole patch stays on the plane.
    """
    size = st.session_state["patch_size"]
    max_row = st.session_state["plane_height"] - size
    max_col = st.session_state["plane_width"] - size
    origin = (int(np.clip(int(round(row)) - size // 2, 0, max_row)),
              int(np.clip(int(round(col)) - size // 2, 0, max_col)))
    # An identical patch duplicates 25 features without adding information, and
    # `normalise_patches` would merge it on the next run anyway — so refuse it
    # here and say so, rather than letting it vanish silently.
    if origin in st.session_state["patches"]:
        return False
    st.session_state["patches"].append(origin)
    return True


def auto_grid_patches() -> None:
    """Replace the patch list with a uniform grid at the current step."""
    size, step = st.session_state["patch_size"], st.session_state["grid_step"]
    rows = range(0, st.session_state["plane_height"] - size + 1, step)
    cols = range(0, st.session_state["plane_width"] - size + 1, step)
    st.session_state["patches"] = [(r, c) for r in rows for c in cols]


def on_grid_patches() -> None:
    auto_grid_patches()
    st.session_state["status"] = (f"placed a {len(st.session_state['patches'])}-patch "
                                  "grid")


def on_clear_patches() -> None:
    st.session_state["patches"] = []
    st.session_state["status"] = "cleared all patches"


def normalise_patches() -> str:
    """Keep every patch fully on the plane, merging any that collide.

    Needed because plane size and patch size are editable after patches exist.
    Shrinking the plane used to leave patches drawn off-plane; the pipeline then
    clamped them silently onto the edge, where 100 patches collapsed to 36
    positions and 64 duplicated feature blocks went into the LDA unannounced.

    Returns a status message when anything changed, else "".
    """
    size = st.session_state["patch_size"]
    max_row = max(st.session_state["plane_height"] - size, 0)
    max_col = max(st.session_state["plane_width"] - size, 0)
    before = list(st.session_state["patches"])
    seen: set[tuple[int, int]] = set()
    after: list[tuple[int, int]] = []
    moved = 0
    for row, col in before:
        clamped = (int(min(max(row, 0), max_row)), int(min(max(col, 0), max_col)))
        moved += clamped != (row, col)
        if clamped not in seen:
            seen.add(clamped)
            after.append(clamped)
    if after == before:
        return ""
    st.session_state["patches"] = after
    merged = len(before) - len(after)
    parts = []
    if moved:
        parts.append(f"{moved} patch{'es' if moved != 1 else ''} moved back onto the "
                     "plane")
    if merged:
        parts.append(f"{merged} identical patch{'es' if merged != 1 else ''} merged")
    return " · ".join(parts)


def remove_patch_near(row: float, col: float) -> bool:
    """Delete the patch whose centre is nearest the click, within one patch width."""
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
# Placement (canvas click and numeric button share this)
# --------------------------------------------------------------------------- #

def place_with_current_tool(x: float, y: float) -> str:
    """Apply the selected tool at plane coordinate (x, y).

    `x` is the column and `y` the row (row 0 at the top), matching the canvas.
    """
    tool = st.session_state["tool"]
    if tool in {kind.value for kind in ZoneKind}:
        zone_id = add_zone(tool, x, y)
        st.session_state["last_placed"] = ("zone", zone_id)
        status = f"placed zone {zone_id} at ({x:.0f}, {y:.0f})"
    elif tool == "patch":
        if add_patch(row=y, col=x):
            st.session_state["last_placed"] = ("patch", None)
            status = f"placed a sampling patch at ({x:.0f}, {y:.0f})"
        else:
            status = f"a patch already sits exactly at ({x:.0f}, {y:.0f})"
    elif tool == "erase":
        removed = remove_patch_near(row=y, col=x)
        st.session_state["last_placed"] = (None, None)
        status = ("removed a patch" if removed
                  else f"no patch within one patch width of ({x:.0f}, {y:.0f})")
    else:
        status = "tool is 'look' — nothing placed"
    st.session_state["status"] = status
    return status


def on_place_numeric() -> None:
    place_with_current_tool(float(st.session_state["place_x"]),
                            float(st.session_state["place_y"]))


def undo_last_placement() -> str:
    """Remove whatever was placed most recently (one step deep)."""
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


def on_undo() -> None:
    st.session_state["status"] = undo_last_placement()


# --------------------------------------------------------------------------- #
# State -> validated configs
# --------------------------------------------------------------------------- #

def data_config(name: str = "interactive") -> DataConfig:
    """Assemble the WORLD config from session state. Reads zones, never patches."""
    state = st.session_state
    return DataConfig(
        name=name,
        plane=PlaneSpec(height=state["plane_height"], width=state["plane_width"]),
        pivot=state["pivot"],
        background_gain=state["background_gain"],
        overlap_mode=state["overlap_mode"],
        zones=[ZoneSpec(**z) for z in state["zones"]],
        jitter=JitterSpec(enabled=state["jitter_enabled"],
                          center_sigma_px=state["jitter_center"],
                          radius_sigma_px=state["jitter_radius"],
                          gain_sigma=state["jitter_gain"]),
        noise=NoiseSpec(pixel_model=state["pixel_model"], sigma=state["sigma"],
                        beta_concentration=state["beta_concentration"],
                        random_zone_dist=state["random_zone_dist"]),
    )


def sampling_config(name: str = "interactive") -> SamplingConfig:
    """Assemble the OBSERVER config. Reads patches, never zones.

    Always `layout="manual"`, so what is saved is exactly what was on screen.
    Raises (pydantic) when there are no patches; callers check `has_patches`.
    """
    state = st.session_state
    return SamplingConfig(
        name=name, layout="manual", patch_size=state["patch_size"],
        grid_step=state["grid_step"], feature_mode=state["feature_mode"],
        patches=[(int(r), int(c)) for r, c in state["patches"]],
    )


def training_config() -> TrainingConfig:
    state = st.session_state
    shrinkage = None if (state["shrinkage"] == "none" or state["solver"] == "svd") \
        else "auto"
    return TrainingConfig(c_lo=state["c_lo"], c_hi=state["c_hi"],
                          n_per_class=state["n_per_class"], solver=state["solver"],
                          shrinkage=shrinkage, shuffle_labels=state["shuffle_labels"],
                          cv_folds=state["cv_folds"])


def eval_config() -> EvalConfig:
    state = st.session_state
    grid = (state["contrast_grid_override"]
            or np.linspace(0.0, 1.0, state["grid_points"]).tolist())
    return EvalConfig(contrast_grid=grid, n_per_contrast=state["n_per_contrast"],
                      compute_oracle=state["compute_oracle"])


def has_patches() -> bool:
    return bool(st.session_state["patches"])


def run_config(name: str = "interactive") -> RunConfig | None:
    """The full run config, or None when there are no patches to observe with."""
    if not has_patches():
        return None
    return RunConfig(data=data_config(name), sampling=sampling_config(name),
                     training=training_config(), evaluation=eval_config(),
                     master_seed=st.session_state["master_seed"])


# --------------------------------------------------------------------------- #
# Loading and saving
# --------------------------------------------------------------------------- #

def load_run_config_into_state(cfg: RunConfig) -> None:
    """Populate session state from a config. MUST run inside a callback.

    Every setting is written, including ones without a visible default (cv_folds
    now has a widget; a non-uniform contrast grid is kept verbatim as an
    override rather than silently replaced by linspace).
    """
    state = st.session_state
    forget_zone_widgets()
    state["zones"] = [z.model_dump(mode="json") for z in cfg.data.zones]
    # Counter past the highest numeric suffix, so new ids never collide.
    suffixes = [int(m.group(1)) for z in state["zones"]
                if (m := re.search(r"(\d+)$", z["id"]))]
    state["zone_counter"] = max(suffixes, default=0)
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
    state["shrinkage"] = "none" if cfg.training.shrinkage is None else "auto"
    state["shuffle_labels"] = cfg.training.shuffle_labels
    state["cv_folds"] = cfg.training.cv_folds
    grid = list(cfg.evaluation.contrast_grid)
    state["grid_points"] = len(grid)
    uniform = np.allclose(grid, np.linspace(0.0, 1.0, len(grid)))
    state["contrast_grid_override"] = None if uniform else grid
    state["n_per_contrast"] = cfg.evaluation.n_per_contrast
    state["compute_oracle"] = cfg.evaluation.compute_oracle
    state["master_seed"] = cfg.master_seed
    state["last_placed"] = (None, None)

    if cfg.sampling.layout == "manual" and cfg.sampling.patches:
        state["patches"] = [(int(r), int(c)) for r, c in cfg.sampling.patches]
    else:
        # A generated layout is realised once, so the canvas shows the patches
        # the model will actually see.
        streams = spawn_streams(cfg.master_seed)
        state["patches"] = [(p.row, p.col) for p in
                            build_patches(cfg.sampling, cfg.data.plane,
                                          streams["patches"])]


def available_configs() -> dict[str, Path]:
    """Committed configs, then the user's saved ones (prefixed `user/`)."""
    found = {path.stem: path for path in sorted(CONFIG_DIR.glob("*.json"))}
    found.update({f"user/{path.stem}": path
                  for path in sorted(USER_CONFIG_DIR.glob("*.json"))})
    return found


def on_load_config() -> None:
    from storage.io import load_run_config
    choice = st.session_state["load_choice"]
    if not choice:
        st.session_state["notice"] = "pick a config to load first"
        return
    try:
        load_run_config_into_state(load_run_config(available_configs()[choice]))
    except Exception as error:                     # corrupt or outdated file
        st.session_state["notice"] = f"could not load {choice}: {error}"
        return
    st.session_state["status"] = f"loaded {choice}"


def on_load_model_config(run_id: str) -> None:
    result = st.session_state["models"].get(run_id)
    if result is not None:
        load_run_config_into_state(result.config)
        st.session_state["status"] = "loaded the model's config onto the canvas"


SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def on_save_config() -> None:
    """Save the current run config to configs/user/<name>.json."""
    name = str(st.session_state["save_name"]).strip()
    if not SAFE_NAME.match(name):
        st.session_state["notice"] = ("use 1-64 letters, digits, '-' or '_' for the "
                                      "config name")
        return
    cfg = run_config(name)
    if cfg is None:
        st.session_state["notice"] = "place at least one sampling patch before saving"
        return
    path = save_config(cfg, USER_CONFIG_DIR, name=name)
    st.session_state["status"] = f"saved configs/user/{path.name}"


# --------------------------------------------------------------------------- #
# Cached computation — keyed on validated config JSON, never on widget values
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False, max_entries=64)
def cached_preview(data_json: str, contrast: float, seed: int
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    """One realised field plus its gain field, for the canvas.

    Returns (field (H, W), gain (H, W), clip_fraction). Uses its own generator,
    so the preview never consumes an experiment's streams.
    """
    data = DataConfig.model_validate_json(data_json)
    field, gf, diagnostics = generate_field(data, contrast,
                                            np.random.default_rng(seed),
                                            np.random.default_rng(seed + 1))
    return field, gf.a, diagnostics.clip_fraction


@st.cache_data(show_spinner=False, max_entries=16)
def cached_effective_gain(data_json: str) -> np.ndarray:
    """Jitter-free coupling gain, with masked (random-zone) pixels shown as 0."""
    gf = jitter_free_gain_field(DataConfig.model_validate_json(data_json))
    return np.where(gf.random_mask, 0.0, gf.a)


@st.cache_resource(show_spinner=False, max_entries=32)
def cached_experiment(run_json: str) -> ExperimentResult:
    """Train and evaluate; identical settings never retrain."""
    cfg = RunConfig.model_validate_json(run_json)
    return run_experiment(cfg, provenance=provenance(), ids=run_identifiers(cfg))


@dataclass(frozen=True)
class RunSummary:
    """Just enough about a trained model for the registry list."""

    run_id: str
    label: str
    verdict: str
    kappa: float | None
    sd_ratio: float
    n_per_contrast: int
    below_preregistered_minimum: bool


def summarise(result: ExperimentResult) -> RunSummary:
    report, cfg = result.report, result.config
    return RunSummary(
        run_id=result.run_id,
        label=(f"{len(cfg.data.zones)} zones · {cfg.data.noise.pixel_model} · "
               f"{result.model.diagnostics.n_features} feat · "
               f"{cfg.evaluation.n_per_contrast}/level · seed {cfg.master_seed}"),
        verdict=report.verdict,
        kappa=report.curvature_index,
        sd_ratio=report.sd_ratio,
        n_per_contrast=cfg.evaluation.n_per_contrast,
        below_preregistered_minimum=(cfg.evaluation.n_per_contrast
                                     < PRE_REGISTERED_MIN_PER_LEVEL),
    )


def register_model(result: ExperimentResult) -> None:
    """Keep the result and make it active.

    Keyed on `run_id`, which includes the evaluation settings. Keying on
    `model_id` (data + sampling + training + seed only) made a re-run with a
    different trial count silently overwrite the earlier result.
    """
    st.session_state["models"][result.run_id] = result
    st.session_state["active_model"] = result.run_id


def on_delete_model(run_id: str) -> None:
    models = st.session_state["models"]
    models.pop(run_id, None)
    st.session_state["active_model"] = next(reversed(models), None) if models else None


def config_fingerprint() -> str:
    """Short hash of the current world+observer, shown next to the canvas."""
    world = config_hash(data_config())
    observer = config_hash(sampling_config()) if has_patches() else "no-patches"
    return f"{world}·{observer}"
