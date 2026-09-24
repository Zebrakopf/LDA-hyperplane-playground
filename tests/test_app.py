"""Headless tests for the Streamlit app, via `streamlit.testing.v1.AppTest`.

Every test drives the REAL widgets and buttons in app.py (by key), not helper
functions. The first version of this file called `ctl.add_zone(...)` directly,
so swapping x/y in the click handler, deleting the preset callback or making
"clear patches" do nothing all still passed (docs/decisions.md D15). Each bug
the September review found in the app has a test here that reproduces it.

What AppTest cannot do is deliver a Plotly click. The canvas's structural
preconditions are checked below; the click itself is verified in a browser.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parent.parent
APP_PATH = str(REPO / "app.py")
STARTUP_TIMEOUT = 120


def _app(**state) -> AppTest:
    """A small, fast app: 30x30 plane, patch_mean features, tiny trial counts."""
    app = AppTest.from_file(APP_PATH, default_timeout=STARTUP_TIMEOUT)
    defaults = dict(plane_height=30, plane_width=30, grid_step=10, patch_size=5,
                    feature_mode="patch_mean", n_per_class=30, n_per_contrast=20,
                    grid_points=5, cv_folds=0)
    for key, value in {**defaults, **state}.items():
        app.session_state[key] = value
    return app.run()


def _ok(app: AppTest) -> AppTest:
    assert not app.exception, [e.value for e in app.exception]
    return app


def _train(app: AppTest) -> AppTest:
    return _ok(app.button(key="train").click().run())


def _load(app: AppTest, name: str) -> AppTest:
    app.selectbox(key="load_choice").set_value(name).run()
    return _ok(app.button(key="load_button").click().run())


def _fingerprint(app: AppTest) -> str:
    caption = next(c.value for c in app.caption if "config " in c.value)
    return caption.rsplit("config ", 1)[1]


# --------------------------------------------------------------------------- #
# Basic paths
# --------------------------------------------------------------------------- #

def test_app_starts_with_a_patch_grid_and_an_empty_results_prompt() -> None:
    app = _ok(_app())
    assert app.session_state["patches"]
    assert any("No model yet" in m.value for m in app.markdown)


def test_train_produces_a_labelled_verdict() -> None:
    app = _train(_app())
    assert len(app.session_state["models"]) == 1
    assert any("lda-verdict" in m.value for m in app.markdown)
    assert any("pre-registered minimum" in c.value for c in app.caption)


def test_all_dead_world_is_degenerate_through_the_widgets() -> None:
    app = _app()
    app.slider(key="background_gain").set_value(0.0).run()
    app = _train(app)
    active = app.session_state["models"][app.session_state["active_model"]]
    assert active.report.verdict == "degenerate"


def test_fully_saturated_training_shows_fit_failed() -> None:
    app = _app()
    app.slider(key="background_gain").set_value(2.0).run()
    app = _train(app)
    active = app.session_state["models"][app.session_state["active_model"]]
    assert active.report.verdict == "fit_failed"
    assert any("fit failed" in e.value for e in app.error)


# --------------------------------------------------------------------------- #
# Bugs from the review, each reproduced through the UI
# --------------------------------------------------------------------------- #

def test_settings_take_effect_on_the_same_interaction() -> None:
    """Keyless widgets used to apply one interaction late (D15)."""
    app = _app()
    before = _fingerprint(app)
    app.slider(key="background_gain").set_value(0.45).run()
    assert _fingerprint(_ok(app)) != before
    assert app.session_state["background_gain"] == 0.45


def test_numeric_place_button_places_at_x_column_y_row() -> None:
    app = _app()
    app.session_state["tool"] = "dead"
    app.number_input(key="place_x").set_value(7).run()
    app.number_input(key="place_y").set_value(21).run()
    app = _ok(app.button(key="place_button").click().run())
    zone = app.session_state["zones"][-1]
    assert (zone["kind"], zone["center_x"], zone["center_y"]) == ("dead", 7.0, 21.0)


def test_zone_editor_edits_reach_the_world_immediately() -> None:
    app = _app()
    app.session_state["tool"] = "heat"
    app = _ok(app.button(key="place_button").click().run())
    zid = app.session_state["zones"][-1]["id"]
    before = _fingerprint(app)
    app.slider(key=f"zone:{zid}:gain").set_value(1.5).run()
    assert app.session_state["zones"][-1]["gain"] == 1.5
    assert _fingerprint(_ok(app)) != before


def test_loading_a_config_replaces_zone_values_with_the_same_id() -> None:
    """A loaded heat1 used to keep the OLD heat1's slider values (D15)."""
    app = _app(plane_height=100, plane_width=100)
    app.session_state["tool"] = "heat"
    app.number_input(key="place_x").set_value(10).run()
    app.number_input(key="place_y").set_value(10).run()
    app = _ok(app.button(key="place_button").click().run())
    assert app.session_state["zones"][-1]["id"] == "heat1"
    app.slider(key="zone:heat1:gain").set_value(1.5).run()
    app = _load(app, "scenario_heat_dead")
    heat1 = next(z for z in app.session_state["zones"] if z["id"] == "heat1")
    assert (heat1["center_x"], heat1["center_y"], heat1["gain"]) == (25.0, 25.0, 2.0)
    assert app.slider(key="zone:heat1:gain").value == 2.0


def test_loading_a_config_exposes_every_setting_it_changes() -> None:
    """cv_folds was loaded with no widget, and the preset still said 'fast'.

    control_simple uses exactly the careful preset's numbers, so the preset box
    must now say so; nudging any of them must turn it into `custom`.
    """
    app = _load(_app(), "control_simple")
    assert app.number_input(key="cv_folds").value == 5
    assert app.selectbox(key="preset").value == "careful (pre-registered)"
    app.slider(key="n_per_contrast").set_value(190).run()
    assert _ok(app).selectbox(key="preset").value == "custom"


def test_new_zone_ids_never_collide_with_loaded_ones(tmp_path: Path) -> None:
    """A config holding only heat2 used to yield a second heat2 and a crash."""
    from core.config import RunConfig
    from ui.controls import USER_CONFIG_DIR

    cfg = RunConfig.model_validate_json(
        (REPO / "configs" / "scenario_heat_dead.json").read_text("utf-8"))
    cfg.data.zones = [z for z in cfg.data.zones if z.id == "heat2"]
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = USER_CONFIG_DIR / "_test_only_heat2.json"
    path.write_text(cfg.model_dump_json(), encoding="utf-8")
    try:
        app = _load(_app(), "user/_test_only_heat2")
        app.session_state["tool"] = "heat"
        app = _ok(app.button(key="place_button").click().run())
        ids = [z["id"] for z in app.session_state["zones"]]
        assert len(ids) == len(set(ids)) == 2
    finally:
        path.unlink()


def test_clear_patches_after_training_does_not_crash() -> None:
    app = _train(_app())
    app = _ok(app.button(key="clear_patches_button").click().run())
    assert app.session_state["patches"] == []
    assert any("observer sees nothing" in w.value for w in app.warning)
    assert app.button(key="train").disabled


def test_saving_without_patches_explains_instead_of_crashing() -> None:
    app = _ok(_app().button(key="clear_patches_button").click().run())
    app = _ok(app.button(key="save_button").click().run())
    assert any("sampling patch" in t.value for t in app.toast)


def test_models_differing_only_in_evaluation_are_both_kept() -> None:
    """Keyed on model_id, the second run used to overwrite the first."""
    app = _train(_app())
    app.slider(key="n_per_contrast").set_value(40).run()
    app = _train(app)
    assert len(app.session_state["models"]) == 2


def test_shrinking_the_plane_moves_patches_back_on_and_merges_duplicates() -> None:
    app = _app(plane_height=100, plane_width=100)
    assert len(app.session_state["patches"]) == 100
    app.number_input(key="plane_width").set_value(50).run()
    app = _ok(app)
    patches = app.session_state["patches"]
    assert all(c + 5 <= 50 for _, c in patches)
    assert len(patches) == len(set(patches))
    assert any("moved back onto the plane" in c.value for c in app.caption)


def test_stale_model_warning_after_editing_the_world() -> None:
    app = _train(_app())
    assert not any("changed since this model" in w.value for w in app.warning)
    app.slider(key="background_gain").set_value(0.4).run()
    assert any("changed since this model" in w.value for w in _ok(app).warning)


def test_preset_selectbox_rewrites_the_trial_counts() -> None:
    from ui.controls import PRESETS

    app = _app()
    app.selectbox(key="preset").set_value("careful (pre-registered)").run()
    app = _ok(app)
    for key, value in PRESETS["careful (pre-registered)"].items():
        assert app.session_state[key] == value


def test_clear_zones_and_grid_buttons_work() -> None:
    app = _app()
    app.session_state["tool"] = "heat"
    app = _ok(app.button(key="place_button").click().run())
    app = _ok(app.button(key="clear_zones_button").click().run())
    assert app.session_state["zones"] == []
    app = _ok(app.button(key="clear_patches_button").click().run())
    app = _ok(app.button(key="grid_button").click().run())
    assert len(app.session_state["patches"]) == 9          # 30x30, step 10


def test_undo_removes_the_last_placement() -> None:
    app = _app()
    app.session_state["tool"] = "patch"
    before = len(app.session_state["patches"])
    app = _ok(app.button(key="place_button").click().run())
    assert len(app.session_state["patches"]) == before + 1
    app = _ok(app.button(key="undo_button").click().run())
    assert len(app.session_state["patches"]) == before


# --------------------------------------------------------------------------- #
# Canvas preconditions (a click itself needs a browser; D12)
# --------------------------------------------------------------------------- #

SELECTABLE_PLOTLY_TYPES = {"scatter", "scattergl", "bar", "histogram", "box",
                           "violin"}


def _canvas():
    from core.config import DataConfig, PlaneSpec, ZoneKind
    from core.data import jitter_free_gain_field
    from core.sampling import Patch
    from tests.conftest import zone
    from ui.plots import plane_figure

    plane = PlaneSpec(height=30, width=30)
    data = DataConfig(name="canvas", plane=plane,
                      zones=[zone(ZoneKind.HEAT, gain=2.0, radius=6.0)])
    gain = jitter_free_gain_field(data).a
    return plane_figure(np.zeros((30, 30)), gain, data,
                        [Patch(row=0, col=0, size=5)]), plane


def test_canvas_has_an_invisible_selectable_lattice_over_every_pixel() -> None:
    figure, plane = _canvas()
    lattice = next(t for t in figure.data if t.type in SELECTABLE_PLOTLY_TYPES)
    covered = set(zip(np.asarray(lattice.x).tolist(), np.asarray(lattice.y).tolist()))
    assert covered == {(x, y) for x in range(plane.width) for y in range(plane.height)}
    assert lattice.marker.opacity == 0.0
    assert lattice.selected.marker.opacity == 0.0
    assert lattice.unselected.marker.opacity == 0.0
    assert figure.layout.dragmode is False
    assert "select" in (figure.layout.clickmode or "")


def test_canvas_chart_disables_scroll_zoom_and_toolbar() -> None:
    source = (REPO / "app.py").read_text(encoding="utf-8")
    assert '"scrollZoom": False' in source and '"displayModeBar": False' in source


def test_no_figure_uses_a_second_y_axis() -> None:
    """One axis per chart (D16): dual axes invite reading crossings as meaning."""
    source = (REPO / "ui" / "plots.py").read_text(encoding="utf-8")
    assert "secondary_y" not in source


# --------------------------------------------------------------------------- #
# Tooltip coverage (D13)
# --------------------------------------------------------------------------- #

HELPABLE_WIDGETS = frozenset({
    "button", "download_button", "checkbox", "toggle", "radio", "selectbox",
    "multiselect", "slider", "select_slider", "text_input", "number_input",
    "text_area", "file_uploader", "color_picker", "metric", "link_button",
    "segmented_control", "pills",
})
MIN_TOOLTIP_CHARS = 30


def _widget_calls() -> list[ast.Call]:
    tree = ast.parse((REPO / "app.py").read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr in HELPABLE_WIDGETS]


def test_every_widget_has_a_tooltip() -> None:
    missing = [(n.lineno, n.func.attr) for n in _widget_calls()
               if not any(k.arg == "help" for k in n.keywords)]
    assert not missing, missing


def test_tooltips_are_substantive() -> None:
    """Static tooltips must say more than the label (dynamic f-strings skipped)."""
    weak = []
    for node in _widget_calls():
        for keyword in node.keywords:
            if (keyword.arg == "help" and isinstance(keyword.value, ast.Constant)
                    and len(keyword.value.value) < MIN_TOOLTIP_CHARS):
                weak.append((node.lineno, keyword.value.value))
    assert not weak, weak


def test_clicking_the_active_tool_again_keeps_it_selected() -> None:
    """A segmented control deselects on a second click; that must not drop the tool."""
    app = _app()
    app.session_state["tool"] = None            # what the widget reports
    app.session_state["tool_last"] = "dead"
    from ui import controls as ctl
    import streamlit as st
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(st, "session_state", app.session_state)
        ctl.on_tool_change()
    assert app.session_state["tool"] == "dead"


def test_hidden_or_disabled_settings_keep_their_values() -> None:
    """Switching pixel model away and back must not reset its noise parameter."""
    app = _app()
    app.radio(key="pixel_model").set_value("beta").run()
    app.slider(key="beta_concentration").set_value(50.0).run()
    app.radio(key="pixel_model").set_value("bernoulli").run()
    app.radio(key="pixel_model").set_value("beta").run()
    assert _ok(app).slider(key="beta_concentration").value == 50.0
    assert app.session_state["beta_concentration"] == 50.0
