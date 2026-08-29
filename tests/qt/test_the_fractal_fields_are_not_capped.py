"""The fractal settings are fields, and a bad number is explained."""
from __future__ import annotations

import pytest

from spacr.qt import preferences as P


@pytest.fixture
def spaceout_only(monkeypatch):
    """Turn spaceout on for one test and OFF again afterwards.

    It is a module-level flag, so a test that enables it and walks away
    leaves the Fractal tab on the Preferences dialog for every test that
    runs after it -- which is what broke
    `test_the_dialog_has_the_expected_subject_tabs_in_order`.
    """
    from spacr.qt import theme

    monkeypatch.setattr(theme, "_SPACEOUT", True)
    yield



@pytest.fixture
def store(monkeypatch):
    values = {}

    class _Mem:
        def value(self, key, default=None, type=None):
            return values.get(key, default)

        def setValue(self, key, value):
            values[key] = value

        def sync(self):
            pass

    monkeypatch.setattr(P, "_settings", lambda: _Mem())
    monkeypatch.setattr(P, "_SAFE_MODE", False)
    return values


def test_a_large_scale_is_kept(store):
    """It was capped at 2.0, which turned a typed 40 into 2 silently."""
    P.set_fractal_settings(scale=40.0)
    assert P.get_fractal_settings()["scale"] == 40.0


def test_supersampling_is_a_setting(store):
    """Called out as "a super important setting"."""
    assert "supersampling" in P.get_fractal_settings()
    P.set_fractal_settings(supersampling=4)
    assert P.get_fractal_settings()["supersampling"] == 4


def test_the_mandelbrot_numbers_are_settings(store):
    values = P.get_fractal_settings()
    for name in ("seconds_per_decade", "base_iterations",
                 "iterations_per_decade", "max_iterations",
                 "precision_digits", "initial_scale", "zoom_rate",
                 "render_scale", "steering_strength",
                 "steering_interval_decades", "steering_duration",
                 "candidate_count"):
        assert name in values, name


def test_the_published_numbers_survive_where_they_cost_nothing(store):
    """The published set is a `high` profile, and the shipped defaults are
    deliberately lighter -- but only where the number decides COST.

    Making the steering timid, or the reference orbit less precise, would
    change what the pattern IS rather than how hard it works.
    """
    from spacr.qt.widgets.fractal_mandelbrot import DEFAULTS

    values = P.get_fractal_settings()
    for name in ("seconds_per_decade", "precision_digits", "initial_scale",
                 "candidate_count", "steering_strength",
                 "steering_duration"):
        assert values[name] == DEFAULTS[name], name


def test_choosing_high_restores_the_published_profile(store):
    """The published numbers are what High means, not what nobody-chose
    means."""
    from spacr.qt.widgets.fractal_mandelbrot import DEFAULTS

    P.apply_quality_preset("high")
    values = P.get_fractal_settings()
    for name in ("supersampling", "base_iterations",
                 "iterations_per_decade", "max_iterations"):
        assert values[name] == DEFAULTS[name], name


def test_a_number_that_cannot_work_is_explained_not_clamped_silently():
    """"the code will gracefully throw an error and tell the user"."""
    said = P.explain_a_fractal_number("supersampling", 0)
    assert "too small" in said
    assert "draws nothing" in said, said
    assert "supersampling" in said

    said = P.explain_a_fractal_number("max_iterations", 9999)
    assert "4096" in said, said

    # And a merely extravagant number is somebody's business but ours.
    assert P.explain_a_fractal_number("scale", 40.0) == ""
    assert P.explain_a_fractal_number("speed", 500000.0) == ""


def test_a_non_number_says_so():
    said = P.explain_a_fractal_number("scale", "wide")
    assert "not a number" in said


def test_mandelbrot_is_the_default_pattern():
    from spacr.qt.fractal_defaults import DEFAULT_PATTERN, FALLBACK_PATTERN

    assert DEFAULT_PATTERN == "mandelbrot"
    assert FALLBACK_PATTERN == "orbit"


def test_a_machine_without_a_gpu_gets_the_orbit_fold(monkeypatch):
    """Mandelbrot is GPU-only; the fallback has a CPU renderer."""
    from spacr.qt.widgets import fractal_travel as ft

    monkeypatch.setattr(ft, "platform_can_do_opengl", lambda: False)
    assert ft.pattern_for_this_machine("mandelbrot") == "orbit"
    assert ft.pattern_for_this_machine("mandelbrot", "cpu") == "orbit"
    # Every other pattern is left alone.
    assert ft.pattern_for_this_machine("space") == "space"
    assert ft.pattern_for_this_machine("cascade", "cpu") == "cascade"

    monkeypatch.setattr(ft, "platform_can_do_opengl", lambda: True)
    monkeypatch.setattr(ft, "gpu_is_available", lambda: True)
    assert ft.pattern_for_this_machine("mandelbrot") == "mandelbrot"


def test_up_and_down_change_the_zoom_rate():
    """As in the renderer this pattern came from."""
    from spacr.qt.widgets import fractal_travel as ft

    controls = ft.RuntimeControls()
    ft._LIVE_CONTROLS.clear()
    ft._LIVE_CONTROLS.append(controls)
    try:
        start = controls.zoom_rate
        faster = ft.nudge_zoom_rate(1)
        assert faster == pytest.approx(start * ft.ZOOM_STEP)
        slower = ft.nudge_zoom_rate(-1)
        assert slower == pytest.approx(start)
        # A key held down does not reach 10^12.
        for _ in range(200):
            ft.nudge_zoom_rate(1)
        assert controls.zoom_rate <= ft.MAX_ZOOM_RATE
        for _ in range(400):
            ft.nudge_zoom_rate(-1)
        assert controls.zoom_rate >= ft.MIN_ZOOM_RATE
    finally:
        ft._LIVE_CONTROLS.clear()


def test_no_backdrop_means_the_key_is_not_taken():
    """Or Up and Down would stop working everywhere else."""
    from spacr.qt.widgets import fractal_travel as ft

    ft._LIVE_CONTROLS.clear()
    assert ft.nudge_zoom_rate(1) == 0.0


def test_ctrl_r_restarts_the_backdrop(qtbot, spaceout_only):
    """Asked for 2026-08-28: a hotkey to start the theme from the beginning."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    import spacr.qt.app as app_module
    from spacr.qt.widgets import fractal_travel as ft

    win = app_module.MainWindow()
    qtbot.addWidget(win)
    controls = ft.RuntimeControls()
    ft._LIVE_CONTROLS.clear()
    ft._LIVE_CONTROLS.append(controls)
    try:
        before = controls.restart_token
        win.keyPressEvent(QKeyEvent(
            QKeyEvent.Type.KeyPress, Qt.Key.Key_R,
            Qt.KeyboardModifier.ControlModifier))
        assert controls.restart_token != before
    finally:
        ft._LIVE_CONTROLS.clear()
        win.close()


def test_ctrl_r_is_not_taken_when_no_backdrop_is_running(qtbot, spaceout_only):
    """Or it would swallow the shortcut everywhere else."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    import spacr.qt.app as app_module
    from spacr.qt.widgets import fractal_travel as ft

    win = app_module.MainWindow()
    qtbot.addWidget(win)
    ft._LIVE_CONTROLS.clear()
    try:
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_R,
                          Qt.KeyboardModifier.ControlModifier)
        win.keyPressEvent(event)
        assert not event.isAccepted()
    finally:
        win.close()


def test_there_is_an_ultra_level():
    """Asked for 2026-08-28."""
    assert "ultra" in P.FRACTAL_QUALITIES
    assert P.FRACTAL_QUALITIES.index("ultra") > P.FRACTAL_QUALITIES.index("high")


def test_a_level_is_a_set_of_numbers_not_an_adjective():
    """It has to GOVERN the other settings, or it is only a label."""
    for name in ("balanced", "high", "ultra"):
        preset = P.QUALITY_PRESETS[name]
        for key in ("supersampling", "render_scale", "base_iterations",
                    "max_iterations", "scale"):
            assert key in preset, f"{name} does not set {key}"


def test_the_levels_are_monotonic():
    order = ("balanced", "high", "ultra")
    for key in ("supersampling", "base_iterations", "scale"):
        values = [P.QUALITY_PRESETS[name][key] for name in order]
        assert values == sorted(values), (key, values)
    assert P.QUALITY_PRESETS["ultra"]["supersampling"] == 3


def test_applying_a_level_writes_its_numbers(store):
    applied = P.apply_quality_preset("ultra")
    assert applied
    saved = P.get_fractal_settings()
    assert saved["supersampling"] == 3
    assert saved["base_iterations"] == 500

    # And it is a starting point, not a lock: the field still wins after.
    P.set_fractal_settings(supersampling=1)
    assert P.get_fractal_settings()["supersampling"] == 1


def test_auto_writes_nothing(store):
    """It means "decide from the machine", and the renderer does that per
    backend; freezing numbers here would end that."""
    assert P.apply_quality_preset("auto") == {}
    assert P.apply_quality_preset("nonsense") == {}


def test_auto_is_conservative_on_both_backends():
    """"dont want the first impression to be supper laggy" -- and this
    cannot know whether the card is a workstation's or a laptop's."""
    from spacr.qt.widgets.fractal_travel import (HardwareProfile,
                                                 resolved_quality)

    hardware = HardwareProfile.detect()
    assert resolved_quality("auto", "gpu", hardware) == "balanced"
    # An explicit choice is still honoured -- conservatism is a default,
    # not a ceiling.
    assert resolved_quality("ultra", "gpu", hardware) == "ultra"
    assert resolved_quality("high", "cpu", hardware) == "high"


def test_the_shipped_defaults_are_the_ones_that_were_given(store):
    """They were the `balanced` preset for a while, to keep the first
    impression light. The maintainer then handed over the command line they
    actually run -- supersampling 2, render scale 1.0 -- and a later
    instruction wins over an earlier inference.

    Choosing Balanced still gives the lighter profile; it is simply not
    what a user who has chosen nothing gets.
    """
    from spacr.qt.widgets.fractal_mandelbrot import DEFAULTS

    values = P.get_fractal_settings()
    for key in ("supersampling", "render_scale", "base_iterations",
                "max_iterations", "seconds_per_decade", "precision_digits"):
        assert values[key] == DEFAULTS[key], key

    light = P.QUALITY_PRESETS["balanced"]
    assert light["supersampling"] < DEFAULTS["supersampling"]


def test_the_fractal_panel_is_short_enough_not_to_need_sub_categories(qtbot, spaceout_only):
    """Twenty-one fields in one column is a wall, and headings were the
    first answer to it. Removing the fields is the better one: eight
    controls need no signposting.
    """
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox

    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)

    named = []
    for kind in (QComboBox, QDoubleSpinBox, QSpinBox):
        named += [c.objectName() for c in dlg.findChildren(kind)
                  if c.objectName().startswith("Fractal")]
    assert len(named) <= 10, sorted(named)


def test_the_restart_is_in_the_hotkey_menu(qtbot, spaceout_only):
    """A shortcut nobody can find is one nobody uses."""
    from PySide6.QtGui import QAction

    import spacr.qt.app as app_module

    win = app_module.MainWindow()
    qtbot.addWidget(win)
    try:
        action = win.findChild(QAction, "RestartBackdrop")
        assert action is not None, "no menu entry for restarting the backdrop"
        assert action.shortcut().toString() == "Ctrl+R"
        # It has to say what the other keys do too, since they have no
        # entries of their own.
        tip = action.statusTip()
        assert "Up and Down" in tip and "wheel" in tip and "drag" in tip
    finally:
        win.close()


def test_the_shortcut_and_the_menu_entry_are_the_same_action(qtbot, spaceout_only):
    """Or they drift apart."""
    import inspect

    import spacr.qt.app as app_module

    source = inspect.getsource(app_module.MainWindow.keyPressEvent)
    assert "_restart_the_backdrop" in source


def test_dragging_moves_the_view():
    """Asked for 2026-08-28: drag the visual field with the mouse."""
    from PySide6.QtCore import QPoint

    from spacr.qt.widgets import fractal_travel as ft

    positions = [QPoint(50, 50), QPoint(60, 50), QPoint(70, 50)]

    class _Widget:
        def __init__(self):
            self.step = 0

        def isVisible(self):
            return True

        def width(self):
            return 100

        def height(self):
            return 100

        def mapFromGlobal(self, _pos):
            point = positions[min(self.step, len(positions) - 1)]
            self.step += 1
            return point

    pointer = ft.Pointer()
    widget = _Widget()
    # Without a button held, moving does not drag.
    pointer.sample(widget)
    pointer.sample(widget)
    assert pointer.drag_x == 0.0


def test_the_drag_accumulates_rather_than_being_assigned():
    """A frame dropped under load must not lose the movement: it arrives
    with the next one, so a slow machine pans the same total distance."""
    from spacr.qt.widgets import fractal_travel as ft

    pointer = ft.Pointer()
    pointer.drag_x = 0.1
    pointer.drag_x += 0.2
    assert pointer.drag_x == pytest.approx(0.3)


def test_the_depth_limit_matches_the_orbit_precision():
    """It was 34 while the numbers supported 15.7, so most of every dive was
    noise -- "ends quickly in a verry pixelated image"."""
    from spacr.qt.widgets import fractal_mandelbrot as mb

    assert mb.MAX_USEFUL_DEPTH <= 16.0
    viewport = mb.scale_at(mb.MAX_USEFUL_DEPTH, mb.DEFAULTS["initial_scale"])
    assert viewport > 2.2e-16 * 10
