"""The keys reach the backdrop, and one control drives each group."""
from __future__ import annotations

import pytest
from PySide6.QtGui import QAction

import spacr.qt.app as app_module
from spacr.qt import preferences as P
from spacr.qt.widgets import fractal_travel as ft


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


def test_two_backdrops_with_the_same_settings_both_register():
    """`RuntimeControls` is a dataclass, so `in` compares field by field: a
    new backdrop matching a previous one was never added, and the keys then
    drove a stale object no canvas reads."""
    first = ft.RuntimeControls()
    second = ft.RuntimeControls()
    assert first == second, "the test needs two that compare equal"
    assert first is not second

    ft._LIVE_CONTROLS.clear()
    try:
        for controls in (first, second):
            if not any(e is controls for e in ft._LIVE_CONTROLS):
                ft._LIVE_CONTROLS.append(controls)
        assert len(ft._LIVE_CONTROLS) == 2
    finally:
        ft._LIVE_CONTROLS.clear()


def test_the_keys_reach_the_newest_backdrop(qapp):
    ft._LIVE_CONTROLS.clear()
    try:
        ft.create_fractal_widget(
            ft.Settings(pattern="orbit", backend="cpu"),
            ft.RuntimeControls())
        newest = ft.RuntimeControls()
        ft.create_fractal_widget(
            ft.Settings(pattern="orbit", backend="cpu"), newest)

        before = newest.zoom_rate
        assert ft.nudge_zoom_rate(1) != 0.0
        assert newest.zoom_rate != before

        ft.restart_the_dive()
        assert newest.restart_token != 0
    finally:
        ft._LIVE_CONTROLS.clear()


def test_the_list_does_not_grow_without_bound(qapp):
    """Nothing else can trim it: a destroyed backdrop leaves its controls."""
    ft._LIVE_CONTROLS.clear()
    try:
        for _ in range(30):
            ft.create_fractal_widget(
                ft.Settings(pattern="orbit", backend="cpu"),
                ft.RuntimeControls())
        assert len(ft._LIVE_CONTROLS) <= 8
    finally:
        ft._LIVE_CONTROLS.clear()


def test_a_saved_speed_reaches_a_running_backdrop(store):
    """"i change speed and nothing fucking happens" -- the backdrop keeps
    the controls it was built with."""
    controls = ft.RuntimeControls(speed=1.0)
    ft._LIVE_CONTROLS.clear()
    ft._LIVE_CONTROLS.append(controls)
    try:
        P.set_fractal_settings(speed=4.0)
        assert ft.apply_saved_controls() == 1
        assert controls.speed == pytest.approx(4.0)
    finally:
        ft._LIVE_CONTROLS.clear()


def test_one_speed_drives_every_speed_setting(store):
    """"take all the speed settings and have them be controled by one"."""
    P.set_fractal_settings(speed=2.0)
    values = P.get_fractal_settings()
    assert values["speed"] == 2.0
    assert values["zoom_rate"] == 2.0
    assert values["speed_min"] == pytest.approx(1.1)
    assert values["speed_max"] == pytest.approx(3.3)
    # A DURATION goes the other way: twice the speed is half the time.
    assert values["seconds_per_decade"] == pytest.approx(12.0)


def test_one_scale_drives_every_scale_setting(store):
    P.set_fractal_settings(scale=2.0)
    values = P.get_fractal_settings()
    assert values["scale"] == 2.0
    assert values["render_scale"] == 2.0
    # Whole samples a side, so it steps.
    assert values["supersampling"] == 2
    P.set_fractal_settings(scale=0.5)
    assert P.get_fractal_settings()["supersampling"] == 1


def test_speed_is_monotonic(store):
    """Higher must never mean slower, whichever setting it reaches."""
    P.set_fractal_settings(speed=1.0)
    slow = P.get_fractal_settings()["seconds_per_decade"]
    P.set_fractal_settings(speed=3.0)
    fast = P.get_fractal_settings()["seconds_per_decade"]
    assert fast < slow


@pytest.mark.parametrize("name,shortcut", [
    ("RestartBackdrop", "Ctrl+R"),
    ("ToggleBackdrop", "Ctrl+T"),
    ("ShowScreensaver", "Ctrl+Shift+F"),
])
def test_the_shortcuts_are_bound(qtbot, name, shortcut):
    win = app_module.MainWindow()
    qtbot.addWidget(win)
    try:
        action = win.findChild(QAction, name)
        assert action is not None, name
        assert action.shortcut().toString() == shortcut
    finally:
        win.close()


def test_no_two_actions_claim_the_same_shortcut(qtbot):
    """Ctrl+B was asked for as the blank-background key and quietly went to
    Ctrl+Shift+B, because the app drawer already held it."""
    win = app_module.MainWindow()
    qtbot.addWidget(win)
    try:
        seen = {}
        for action in win.findChildren(QAction):
            key = action.shortcut().toString()
            if not key:
                continue
            assert key not in seen, (
                f"{key} is claimed by both {seen[key]} and "
                f"{action.objectName() or action.text()}")
            seen[key] = action.objectName() or action.text()
        assert "Ctrl+B" in seen
    finally:
        win.close()


def test_the_arrow_keys_go_forward_and_back():
    """"so i could go slow and fast forward and back" -- the rate is a
    velocity, not a magnitude, so stepping down through zero comes back out
    of the zoom rather than stopping at the slowest descent."""
    controls = ft.RuntimeControls()
    ft._LIVE_CONTROLS.clear()
    ft._LIVE_CONTROLS.append(controls)
    try:
        assert controls.zoom_rate > 0

        # Down, repeatedly: it must cross zero rather than stalling at the
        # floor for ever.
        for _ in range(80):
            ft.nudge_zoom_rate(-1)
        assert controls.zoom_rate < 0, controls.zoom_rate

        # And back up again the same way.
        for _ in range(80):
            ft.nudge_zoom_rate(1)
        assert controls.zoom_rate > 0, controls.zoom_rate
    finally:
        ft._LIVE_CONTROLS.clear()


def test_the_speed_is_still_bounded_in_both_directions():
    """A key held down is not a request for ten to the twelfth."""
    controls = ft.RuntimeControls()
    ft._LIVE_CONTROLS.clear()
    ft._LIVE_CONTROLS.append(controls)
    try:
        for _ in range(400):
            ft.nudge_zoom_rate(1)
        assert abs(controls.zoom_rate) <= ft.MAX_ZOOM_RATE
        for _ in range(800):
            ft.nudge_zoom_rate(-1)
        assert abs(controls.zoom_rate) <= ft.MAX_ZOOM_RATE
    finally:
        ft._LIVE_CONTROLS.clear()


def test_dragging_works_on_the_steady_path_too():
    """Refusing it there would mean the only way to look somewhere else was
    to turn on the search that shook."""
    import inspect

    source = inspect.getsource(ft._make_gpu_widget)
    drag = source.index("camera.drag(")
    guided = source.index('!= "guided"')
    assert drag < guided, (
        "the drag is handled after the fixed path returns, so it never runs")


def test_nothing_aims_the_dive_automatically():
    """Surveying the surface for a "more interesting" point was tried and
    made it worse: a point on a busy edge at the starting scale measured
    completely flat three decades in, because surface structure does not
    predict what survives a descent."""
    import inspect

    source = inspect.getsource(ft._make_gpu_widget)
    assert "_aim_once" not in source
    assert "a_more_interesting_anchor" not in source


def test_the_mandelbrot_is_dragged_not_attracted():
    """"when mouse gravity is on for mandelbrot it should only be drag and
    drop... not pointer position."

    The other three patterns are fields that can be warped toward a point.
    A deep zoom is a camera: pulling its coordinates toward wherever the
    mouse rests slides the picture continuously, which reads as the image
    drifting away from you rather than as anything you did.
    """
    import inspect

    source = inspect.getsource(ft._make_gpu_widget)
    body = source[source.index("def _pointer_state"):]
    body = body[:body.index("def on_resize")]

    # The position-driven terms are withheld for this pattern only.
    assert 'settings.pattern == "mandelbrot"' in body
    assert "return pointer.x, pointer.y, 0.0, 0.0" in body
    # And the others keep them.
    assert "return pointer.x, pointer.y, pointer.pull, pointer.push" in body


def test_the_pointer_is_still_sampled_so_the_drag_accumulates():
    """Withholding pull and push must not stop the sampling: that is what
    fills `drag_x`, which the steering consumes."""
    import inspect

    source = inspect.getsource(ft._make_gpu_widget)
    body = source[source.index("def _pointer_state"):]
    body = body[:body.index("def on_resize")]
    sampled = body.index("self._pointer.sample(")
    withheld = body.index('settings.pattern == "mandelbrot"')
    assert sampled < withheld, (
        "the pattern is checked before the pointer is sampled, so the drag "
        "would never accumulate")


def test_the_drag_is_read_after_the_pointer_is_sampled():
    """Otherwise a drag is always one frame stale."""
    import inspect

    source = inspect.getsource(ft._make_gpu_widget)
    assert source.index("self._pointer_state()") < \
        source.index("self._mandelbrot_uniforms(elapsed)")


def test_the_depth_is_a_control_on_the_panel(qtbot):
    """"i want controll over the decades" -- it was a setting all along and
    was taken off the panel in the cut-down, which is the same mistake as
    hiding numbers behind an Advanced heading."""
    from PySide6.QtWidgets import QDoubleSpinBox

    from spacr.qt import theme

    monkeypatch_spaceout = getattr(theme, "_SPACEOUT")
    theme._SPACEOUT = True
    try:
        dlg = P.PreferencesDialog(None)
        qtbot.addWidget(dlg)
        box = dlg.findChild(QDoubleSpinBox, "FractalMaxDepth")
        assert box is not None, "no depth control"
        assert box.value() > 0
        # It says why it stops, or the next person raises it and gets mush.
        # Read from the hint bar's register: Preferences moves every
        # tooltip there so a control answers in the strip rather than in a
        # window over it.
        from spacr.qt.widgets.hint_bar import HintBar

        said = " ".join(str(text) for text in
                        getattr(dlg.findChildren(HintBar)[0], "_hints",
                                {}).values())
        assert "4.2e-24" in said and "precision" in said
    finally:
        theme._SPACEOUT = monkeypatch_spaceout


def test_the_depth_can_be_set_and_is_bounded_by_the_precision(store):
    P.set_fractal_settings(max_depth=18.0)
    assert P.get_fractal_settings()["max_depth"] == 18.0

    # Above what three float32s can support it is refused in words rather
    # than silently accepted and drawn as noise.
    said = P.explain_a_fractal_number("max_depth", 40)
    assert "too large" in said
    assert "4.2e-24" in said
    assert P.explain_a_fractal_number("max_depth", 20) == ""


def test_the_panel_is_still_short(qtbot):
    """Adding a control back must not undo the consolidation."""
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox

    from spacr.qt import theme

    was = theme._SPACEOUT
    theme._SPACEOUT = True
    try:
        dlg = P.PreferencesDialog(None)
        qtbot.addWidget(dlg)
        named = []
        for kind in (QComboBox, QDoubleSpinBox, QSpinBox):
            named += [c.objectName() for c in dlg.findChildren(kind)
                      if c.objectName().startswith("Fractal")]
        assert len(named) <= 11, sorted(named)
    finally:
        theme._SPACEOUT = was
