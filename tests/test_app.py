"""Headless tests for the Streamlit app, via `streamlit.testing.v1.AppTest`.

Why these exist: an app is the one part of this repo a reader cannot check by
eye in CI, and a Streamlit script fails at RUN time rather than import time — a
typo in a widget call only surfaces when someone clicks. AppTest runs the real
script in-process, so a broken layout, a bad session-state key or an exception in
a cached function fails the suite instead of the user's afternoon.

These tests deliberately use the fastest possible settings: they check that the
app WORKS, not what it finds. The scientific behaviour is tested in
`test_pipeline.py`, where it belongs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
# Generous because the first run imports sklearn and builds a plane; the app's own
# interaction budget is far tighter (see ui/controls.py PRESETS).
STARTUP_TIMEOUT = 120


def _tiny_app() -> AppTest:
    """Boot the app on a small plane so a full train+evaluate is near-instant."""
    app = AppTest.from_file(APP_PATH, default_timeout=STARTUP_TIMEOUT)
    app.session_state["plane_height"] = 30
    app.session_state["plane_width"] = 30
    app.session_state["grid_step"] = 10
    app.session_state["patch_size"] = 5
    app.session_state["feature_mode"] = "patch_mean"
    app.session_state["n_per_class"] = 30
    app.session_state["n_per_contrast"] = 20
    app.session_state["grid_points"] = 5
    app.session_state["cv_folds"] = 0
    return app


def test_app_starts_without_exception() -> None:
    app = _tiny_app().run()
    assert not app.exception, app.exception
    # The landing state must tell the user what to do, not show an empty page.
    assert any("Train & evaluate" in info.value for info in app.info)


def test_default_state_places_a_patch_grid() -> None:
    """A brand-new session must not open with an observer that sees nothing."""
    app = AppTest.from_file(APP_PATH, default_timeout=STARTUP_TIMEOUT).run()
    assert not app.exception, app.exception
    assert app.session_state["patches"], "init_state should seed a patch grid"


def test_clearing_patches_warns() -> None:
    app = _tiny_app().run()
    app.session_state["patches"] = []
    app.run()
    assert not app.exception, app.exception
    assert any("observer sees nothing" in warning.value for warning in app.warning)


def test_adding_a_zone_updates_the_canvas_state() -> None:
    app = _tiny_app().run()
    before = len(app.session_state["zones"])
    app.session_state["tool"] = "heat"
    # Simulate what a canvas click does. AppTest cannot deliver a Plotly selection
    # event, so the handler's effect is exercised through the same helper the
    # handler calls -- keeping the test honest about what it covers.
    from ui import controls as ctl
    import streamlit as st
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(st, "session_state", app.session_state)
        ctl.add_zone("heat", 15.0, 15.0)
    app.run()
    assert not app.exception, app.exception
    assert len(app.session_state["zones"]) == before + 1
    assert app.session_state["zones"][-1]["kind"] == "heat"


def test_train_and_evaluate_produces_a_verdict() -> None:
    """The end-to-end path a user actually takes: press the button, read a verdict."""
    app = _tiny_app().run()
    app.button(key=None) if False else None          # documented: buttons by label
    run_button = [b for b in app.button if "Train & evaluate" in b.label]
    assert run_button, "the primary run button is missing"
    run_button[0].click().run()
    assert not app.exception, app.exception
    assert app.session_state["models"], "a trained model should be registered"
    active = app.session_state["models"][app.session_state["active_model"]]
    assert active.report.verdict in {"affine", "equivocal", "nonlinear", "degenerate"}
    # The exploration presets sit below the pre-registered minimum, and the app must
    # say so rather than presenting the verdict as publishable.
    assert any("pre-registered minimum" in caption.value for caption in app.caption)


def test_all_dead_world_is_reported_degenerate_in_the_app() -> None:
    """The null control, driven through the UI rather than the API."""
    app = _tiny_app()
    app.session_state["background_gain"] = 0.0
    app.run()
    [b for b in app.button if "Train & evaluate" in b.label][0].click().run()
    assert not app.exception, app.exception
    active = app.session_state["models"][app.session_state["active_model"]]
    assert active.report.verdict == "degenerate"
    assert any("degenerate" in info.value for info in app.info)


def test_stale_model_warning_appears_after_editing_the_world() -> None:
    """Editing the canvas after training must not silently mislabel the figures."""
    app = _tiny_app().run()
    [b for b in app.button if "Train & evaluate" in b.label][0].click().run()
    assert not app.exception, app.exception
    app.session_state["background_gain"] = 0.42        # world no longer matches
    app.run()
    assert any("canvas has changed" in warning.value for warning in app.warning)


def test_speed_preset_rewrites_the_trial_counts() -> None:
    app = _tiny_app().run()
    from ui.controls import PRESETS
    import streamlit as st
    from ui import controls as ctl
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(st, "session_state", app.session_state)
        ctl.apply_preset("careful (pre-registered)")
    app.run()
    assert not app.exception, app.exception
    assert (app.session_state["n_per_contrast"]
            == PRESETS["careful (pre-registered)"]["n_per_contrast"])


# --------------------------------------------------------------------------- #
# Canvas click regression guards
# --------------------------------------------------------------------------- #
# The click-to-place feature shipped broken once: the canvas was a bare Plotly
# Heatmap, and Plotly's point-selection API only fires for scatter-like traces.
# Nothing raised, nothing failed a test — clicks simply did nothing while drag
# and zoom kept working, which reads exactly like "the feature is broken".
#
# AppTest cannot deliver a Plotly selection event, so it cannot test clicking
# end-to-end (that is done against a real browser, docs/decisions.md D12). What it
# CAN do is assert the structural preconditions without which a click can never
# work. Those are what regressed, so those are what is guarded here.

SELECTABLE_PLOTLY_TYPES = {"scatter", "scattergl", "bar", "histogram", "box",
                           "violin"}


def _canvas_figure():
    """Build the canvas figure directly, with a small plane and one zone."""
    import numpy as np

    from core.config import DataConfig, PlaneSpec, ZoneKind
    from core.data import jitter_free_gain_field
    from core.sampling import Patch
    from tests.conftest import zone
    from ui.plots import plane_figure

    plane = PlaneSpec(height=30, width=30)
    data = DataConfig(name="canvas", plane=plane,
                      zones=[zone(ZoneKind.HEAT, gain=2.0, radius=6.0)])
    gf = jitter_free_gain_field(data)
    field = np.zeros((plane.height, plane.width))
    return plane_figure(field, gf, data, [Patch(row=0, col=0, size=5)]), plane


def test_canvas_has_a_selectable_trace() -> None:
    """Without a scatter-like trace, no click can ever be reported. THE guard."""
    figure, _plane = _canvas_figure()
    types = {trace.type for trace in figure.data}
    assert types & SELECTABLE_PLOTLY_TYPES, (
        f"canvas traces are {types}; Plotly reports point selections only for "
        f"{SELECTABLE_PLOTLY_TYPES}, so clicking is dead")


def test_click_lattice_covers_every_pixel_and_stays_invisible() -> None:
    """The lattice must span the plane, and must never be visible."""
    figure, plane = _canvas_figure()
    lattice = next(t for t in figure.data if t.type in SELECTABLE_PLOTLY_TYPES)
    assert lattice.x.min() == 0 and lattice.y.min() == 0
    assert lattice.x.max() == plane.width - 1
    assert lattice.y.max() == plane.height - 1
    # Invisible in all three states, or it smudges the pixel field it sits on.
    assert lattice.marker.opacity == 0.0
    assert lattice.selected.marker.opacity == 0.0
    assert lattice.unselected.marker.opacity == 0.0


def test_canvas_disables_drag_so_a_press_is_a_click() -> None:
    """dragmode must be off, or a press starts a pan instead of selecting."""
    figure, _plane = _canvas_figure()
    assert figure.layout.dragmode is False
    assert "select" in (figure.layout.clickmode or "")


def test_placing_tools_route_through_one_function() -> None:
    """Canvas clicks and the numeric button must do the same thing.

    Both call `place_with_current_tool`, so this exercises the shared path once
    per tool — including that x is the column and y the row, which is very easy to
    get backwards and produces mirrored zones when it is.
    """
    import streamlit as st

    from ui import controls as ctl

    app = _tiny_app().run()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(st, "session_state", app.session_state)

        app.session_state["tool"] = "dead"
        ctl.place_with_current_tool(7.0, 21.0)
        placed = app.session_state["zones"][-1]
        assert (placed["center_x"], placed["center_y"]) == (7.0, 21.0)

        app.session_state["tool"] = "patch"
        before = len(app.session_state["patches"])
        ctl.place_with_current_tool(7.0, 21.0)
        assert len(app.session_state["patches"]) == before + 1
        # Click marks the patch CENTRE; the stored origin is the top-left corner.
        assert app.session_state["patches"][-1] == (21 - 2, 7 - 2)

        # Two clicks on the same spot must place two patches, not one.
        ctl.place_with_current_tool(7.0, 21.0)
        assert len(app.session_state["patches"]) == before + 2

        assert ctl.undo_last_placement() == "removed the last patch"
        assert len(app.session_state["patches"]) == before + 1

        app.session_state["tool"] = "erase_patch"
        ctl.place_with_current_tool(7.0, 21.0)
        assert len(app.session_state["patches"]) == before

        app.session_state["tool"] = "none"
        assert "nothing placed" in ctl.place_with_current_tool(5.0, 5.0)


# --------------------------------------------------------------------------- #
# Tooltip coverage
# --------------------------------------------------------------------------- #

# Streamlit widgets that accept a `help=` tooltip. Anything in this set that
# appears in app.py without one is a control the user has to guess at.
HELPABLE_WIDGETS = frozenset({
    "button", "download_button", "checkbox", "toggle", "radio", "selectbox",
    "multiselect", "slider", "select_slider", "text_input", "number_input",
    "text_area", "file_uploader", "color_picker", "metric", "link_button",
})


def _widget_calls_without_help(source_path: Path) -> list[tuple[int, str, str]]:
    """Every helpable widget call in a source file that has no `help=`.

    Static analysis rather than AppTest introspection, deliberately: this needs to
    hold for widgets nested inside branches a given test run never reaches — the
    per-zone editors, the pixel-model-specific noise sliders, the comparison tab
    that only appears with two models trained. A runtime check would silently pass
    on the branches it did not visit.
    """
    import ast

    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    missing: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in HELPABLE_WIDGETS:
            continue
        if any(keyword.arg == "help" for keyword in node.keywords):
            continue
        label = "<dynamic>"
        if node.args and isinstance(node.args[0], ast.Constant):
            label = str(node.args[0].value)
        missing.append((node.lineno, node.func.attr, label))
    return missing


def test_every_widget_has_a_tooltip() -> None:
    """No control in the app is left unexplained.

    The app exposes a lot of knobs whose meaning is not guessable from the label —
    "pivot", "background gain", "shrinkage", "d′" — and it is aimed at colleagues
    who did not write it. A missing tooltip is a real defect here, not polish.
    """
    app_file = Path(__file__).resolve().parent.parent / "app.py"
    missing = _widget_calls_without_help(app_file)
    assert not missing, "widgets without a help tooltip:\n" + "\n".join(
        f"  app.py:{line}  st.{widget}({label!r})" for line, widget, label in missing)


# Floor on tooltip length. 30 rather than something larger because a few controls
# genuinely need only a short sentence -- an x coordinate really is just "column,
# counted from the left edge of the plane" -- while anything under 30 characters
# cannot be saying more than the label already does.
MIN_TOOLTIP_CHARS = 30


def test_tooltips_are_substantive() -> None:
    """A tooltip that restates the label teaches nothing.

    Two ways to fail: being too short to carry information, or echoing the label
    (`help="the plane height"` beside a slider labelled "plane height"). The second
    check is the one that matters — length is easy to game, word overlap is not.
    """
    import ast

    app_file = Path(__file__).resolve().parent.parent / "app.py"
    tree = ast.parse(app_file.read_text(encoding="utf-8"))
    too_short: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in HELPABLE_WIDGETS:
            continue
        for keyword in node.keywords:
            if keyword.arg != "help":
                continue
            # Help strings are written as implicitly concatenated literals, which
            # parse to a single Constant; anything else is dynamic and skipped.
            if isinstance(keyword.value, ast.Constant) and \
                    isinstance(keyword.value.value, str):
                text = keyword.value.value
                if len(text) < MIN_TOOLTIP_CHARS:
                    too_short.append((node.lineno, f"too short: {text}"))
                    continue
                label = (node.args[0].value
                         if node.args and isinstance(node.args[0], ast.Constant)
                         and isinstance(node.args[0].value, str) else "")
                if label:
                    label_words = {w.strip("():,.").lower()
                                   for w in label.split() if len(w) > 3}
                    tip_words = {w.strip("():,.").lower() for w in text.split()}
                    # Every substantial word of the label appearing in a tooltip
                    # barely longer than the label means it is an echo.
                    if (label_words and label_words <= tip_words
                            and len(text) < len(label) * 3):
                        too_short.append((node.lineno, f"echoes its label: {text}"))
    assert not too_short, "uninformative tooltips:\n" + "\n".join(
        f"  app.py:{line}  {text}" for line, text in too_short)
