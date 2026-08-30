"""The maintainer's fractal_travel.py, as spaceout's backdrop.

Two renderers: a GLSL shader when vispy is importable, and a Numba
orbit-fold otherwise. The settings are spaceout-only, and the whole thing
winds down when a run starts.
"""

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import fractal_travel as F  # noqa: E402


# --- which renderer, and the honest answer about it -----------------------


def test_auto_resolves_to_something_real():
    """Never 'auto': a caller showing the user their backend cannot show
    them a word that means "ask again"."""
    assert F.resolve_backend("auto") in ("gpu", "cpu")


def test_auto_follows_whether_vispy_is_installed():
    expected = "gpu" if F.gpu_is_available() else "cpu"
    assert F.resolve_backend("auto") == expected


def test_an_unknown_backend_falls_back_rather_than_raising():
    assert F.resolve_backend("sideways") in ("gpu", "cpu")


def test_asking_for_the_gpu_never_stops_the_application(qtbot):
    """A backdrop is not worth refusing to start over."""
    widget = F.create_fractal_widget(F.Settings(backend="gpu"))
    qtbot.addWidget(widget)
    assert widget.backend_name in ("gpu", "cpu")
    widget.shutdown()


# --- the maintainer's defaults --------------------------------------------


def test_the_defaults_are_the_two_command_lines():
    assert F.DEFAULT_SPEED == 4.0
    assert F.DEFAULT_DREAM == 1.5
    assert F.DEFAULT_SCALE == 1.0
    assert F.DEFAULT_BACKEND == "auto"
    assert F.DEFAULT_QUALITY == "auto"


def test_a_silly_stored_value_is_clamped_not_refused():
    settings = F.Settings(scale=99.0, fps=9999, quality="sideways").validated()
    assert settings.scale == 2.0
    assert settings.fps == 240
    assert settings.quality == "auto"


# --- variable speed --------------------------------------------------------


def test_speed_is_constant_when_variable_speed_is_off():
    controls = F.RuntimeControls(speed=4.0, variable_speed=False)
    assert controls.speed_at(0.0) == 4.0
    assert controls.speed_at(97.3) == 4.0


def test_variable_speed_breathes_around_the_set_value():
    """It modulates the preference rather than replacing it, so the number
    the user typed still means something."""
    controls = F.RuntimeControls(speed=4.0, variable_speed=True)
    seen = [controls.speed_at(t) for t in range(0, 120, 3)]
    assert min(seen) < 4.0 < max(seen)
    assert min(seen) > 0.0
    assert max(seen) < 4.0 * 1.6


def test_variable_speed_never_reaches_zero():
    controls = F.RuntimeControls(speed=0.15, variable_speed=True)
    assert min(controls.speed_at(t / 4) for t in range(800)) > 0.0


# --- the renderer itself ---------------------------------------------------


def test_the_cpu_engine_draws_something_that_is_not_flat():
    numba = pytest.importorskip("numba")  # noqa: F841
    engine = F.OrbitEngine(2)
    frame = engine.render(96, 64, 0.0, 4.0, 1.5, 5)
    assert frame.shape == (64, 96, 3)
    assert frame.std() > 0


def test_the_picture_moves_with_time():
    pytest.importorskip("numba")
    engine = F.OrbitEngine(2)
    first = engine.render(96, 64, 0.0, 4.0, 1.5, 5).copy()
    for step in range(1, 5):          # walk the whole jitter ring
        later = engine.render(96, 64, step * 2.0, 4.0, 1.5, 5)
    assert (later != first).any()


def test_the_first_frame_is_the_picture_not_a_fade_from_black():
    """The ring is filled with the first real sample, so nothing fades up."""
    pytest.importorskip("numba")
    engine = F.OrbitEngine(2)
    frame = engine.render(96, 64, 0.0, 4.0, 1.5, 5)
    assert frame.mean() > 20


def test_the_camera_moves():
    early = F.state_at_seconds(0.0, 4.0, 1.5)
    later = F.state_at_seconds(30.0, 4.0, 1.5)
    assert early.depth != later.depth
    assert (early.tx, early.ty) != (later.tx, later.ty)


# --- threads ---------------------------------------------------------------


def test_it_leaves_the_machine_some_cores():
    """A backdrop that takes every core starves the run it sits behind."""
    hardware = F.HardwareProfile(logical_cpus=32)
    assert F.resolved_cpu_threads(F.Settings(), hardware) < 32


def test_it_never_asks_for_more_than_24():
    hardware = F.HardwareProfile(logical_cpus=128)
    assert F.resolved_cpu_threads(F.Settings(), hardware) <= 24


def test_a_small_machine_still_gets_one():
    hardware = F.HardwareProfile(logical_cpus=1)
    assert F.resolved_cpu_threads(F.Settings(), hardware) == 1


def test_an_explicit_thread_count_is_honoured():
    hardware = F.HardwareProfile(logical_cpus=32)
    assert F.resolved_cpu_threads(F.Settings(cpu_threads=3), hardware) == 3


# --- no second Qt binding --------------------------------------------------


def test_nothing_here_imports_pyqt6():
    """Two Qt bindings in one process do not raise -- they segfault.

    Checked on the IMPORTS rather than on the text: this module explains in
    prose that the script it came from was PyQt6, and a substring search
    fails on its own docstring.
    """
    import ast

    tree = ast.parse(open(F.__file__).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "PyQt6" not in imported
    assert "PySide6" in imported


def test_the_gpu_path_asks_vispy_for_pyside6():
    """vispy will import whichever binding it is told to."""
    source = open(F.__file__).read()
    assert 'use_app("pyside6")' in source


def test_asking_whether_there_is_a_gpu_does_not_import_vispy():
    """`find_spec` looks a module up without executing it, so the check
    costs nothing on a machine that has no vispy."""
    import inspect

    assert "find_spec" in inspect.getsource(F.gpu_is_available)


# --- the settings, and that they are spaceout-only -------------------------


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    return tmp_path


def test_the_stored_defaults_are_the_command_lines(sandbox):
    from spacr.qt.preferences import get_fractal_settings

    values = get_fractal_settings()
    assert values["speed"] == 4.0
    assert values["dream"] == 1.5
    assert values["scale"] == 1.0
    assert values["backend"] == "auto"


def test_every_setting_round_trips(sandbox):
    from spacr.qt.preferences import get_fractal_settings, set_fractal_settings

    set_fractal_settings(pattern="cascade", backend="cpu", quality="high",
                         scale=0.5, speed=2.0, dream=0.25,
                         variable_speed=True)
    stored = get_fractal_settings()
    # The four the test set, checked by name rather than by whole-dict
    # equality: this file has now grown the sweep bounds and the sweep time,
    # and an exact dict makes every future setting a failing test rather
    # than a new row.
    assert stored["pattern"] == "cascade"
    assert stored["backend"] == "cpu"
    assert stored["quality"] == "high"
    assert stored["scale"] == 0.5
    assert stored["speed"] == 2.0
    assert stored["dream"] == 0.25
    assert stored["variable_speed"] is True


def test_only_a_number_that_cannot_work_is_clamped(sandbox):
    from spacr.qt.preferences import get_fractal_settings, set_fractal_settings

    set_fractal_settings(dream=99.0, speed=-5.0)
    values = get_fractal_settings()
    assert values["dream"] == 99.0, "large usable values must survive storage"
    assert values["speed"] == 0.0, "travel cannot run backwards"


def test_an_unknown_backend_is_refused(sandbox):
    from spacr.qt.preferences import set_fractal_settings

    with pytest.raises(ValueError, match="unknown fractal backend"):
        set_fractal_settings(backend="sideways")


def test_an_unknown_setting_is_refused(sandbox):
    from spacr.qt.preferences import set_fractal_settings

    with pytest.raises(ValueError, match="unknown fractal setting"):
        set_fractal_settings(colour="green")


def test_the_rows_are_absent_in_an_ordinary_launch(qtbot, sandbox):
    """A settings page advertising the hidden mode is the giveaway."""
    from PySide6.QtWidgets import QComboBox

    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    assert dialog.findChild(QComboBox, "FractalBackend") is None


def test_all_six_rows_are_there_under_spaceout(qtbot, sandbox, monkeypatch):
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QWidget

    import spacr.qt.theme as theme
    from spacr.qt.preferences import PreferencesDialog

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    assert dialog.findChild(QComboBox, "FractalBackend") is not None
    assert dialog.findChild(QComboBox, "FractalQuality") is not None
    for name in ("FractalScale", "FractalSpeed", "FractalDream"):
        assert dialog.findChild(QDoubleSpinBox, name) is not None
    assert dialog.findChild(QWidget, "FractalVariableSpeed") is not None


def test_the_backend_row_says_which_renderer_this_machine_gets(
        qtbot, sandbox, monkeypatch):
    """'auto' cannot be labelled: the answer depends on the machine."""
    from PySide6.QtWidgets import QLabel

    import spacr.qt.theme as theme
    from spacr.qt.preferences import PreferencesDialog

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    said = dialog.findChild(QLabel, "FractalBackendNote").text()
    assert ("GPU" in said) or ("CPU" in said)


def test_the_defaults_module_needs_neither_numpy_nor_qt():
    """`preferences` reads these before any window exists, so pulling numba
    onto the launch path for an ordinary session would be a real cost."""
    import ast

    from spacr.qt import fractal_defaults

    tree = ast.parse(open(fractal_defaults.__file__).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"numpy", "numba", "PySide6", "vispy"}


# --- winding down for a run ------------------------------------------------


class _Backdrop:
    """A stand-in with the same surface the real widgets expose."""

    backend_name = "cpu"

    def __init__(self):
        self.paused = False

    def pause(self):
        if self.paused:
            return False
        self.paused = True
        return True

    def resume(self):
        if not self.paused:
            return False
        self.paused = False
        return True


class _Angry(_Backdrop):
    def pause(self):
        raise RuntimeError("no")

    def resume(self):
        raise RuntimeError("no")


class _Window:
    def __init__(self, children):
        self._children = list(children)

    def findChildren(self, _kind):
        return self._children


class _Screen:
    def __init__(self, children):
        self._window = _Window(children)

    def window(self):
        return self._window


def test_a_run_pauses_the_backdrop():
    from spacr.qt.screens.app_screen import _pause_the_fractal

    backdrop = _Backdrop()
    assert _pause_the_fractal(_Screen([backdrop])) == 1
    assert backdrop.paused


def test_finishing_gives_the_cores_back():
    from spacr.qt.screens.app_screen import (
        _pause_the_fractal, _resume_the_fractal)

    backdrop = _Backdrop()
    screen = _Screen([backdrop])
    _pause_the_fractal(screen)
    assert _resume_the_fractal(screen) == 1
    assert not backdrop.paused


def test_pausing_twice_reports_the_second_as_a_no_op():
    from spacr.qt.screens.app_screen import _pause_the_fractal

    screen = _Screen([_Backdrop()])
    assert _pause_the_fractal(screen) == 1
    assert _pause_the_fractal(screen) == 0


def test_a_screen_with_no_backdrop_is_fine():
    """Every ordinary launch is this case."""
    from spacr.qt.screens.app_screen import _pause_the_fractal

    assert _pause_the_fractal(_Screen([])) == 0


def test_a_backdrop_that_will_not_stop_does_not_fail_the_run():
    """A run must not die because a decoration misbehaved."""
    from spacr.qt.screens.app_screen import _pause_the_fractal

    assert _pause_the_fractal(_Screen([_Angry()])) == 0


def test_one_angry_backdrop_does_not_stop_the_others():
    from spacr.qt.screens.app_screen import _pause_the_fractal

    willing = _Backdrop()
    assert _pause_the_fractal(_Screen([_Angry(), willing])) == 1
    assert willing.paused


def test_the_run_button_actually_calls_it():
    """Wired, not merely defined."""
    import inspect

    from spacr.qt.screens import app_screen

    assert "_pause_the_fractal(self)" in inspect.getsource(
        app_screen.AppScreen._on_run)
    assert "_resume_the_fractal(self)" in inspect.getsource(
        app_screen.AppScreen._on_finished)


def test_the_real_widget_stops_computing_when_paused(qtbot):
    """The whole point: paused means no frames, not fewer frames."""
    pytest.importorskip("numba")
    from PySide6.QtCore import QEventLoop, QTimer

    widget = F.create_fractal_widget(F.Settings(backend="cpu"))
    qtbot.addWidget(widget)
    widget.resize(360, 220)
    widget.show()

    loop = QEventLoop()
    QTimer.singleShot(3000, loop.quit)
    loop.exec()
    assert widget._frames > 0, "it never drew anything to begin with"

    assert widget.pause() is True
    # ONE FRAME MAY STILL LAND: a render already dispatched to the worker
    # cannot be un-dispatched, and waiting for it is what makes the stop
    # graceful rather than a kill. What must not happen is a SECOND one.
    loop = QEventLoop()
    QTimer.singleShot(900, loop.quit)
    loop.exec()
    before = widget._frames
    loop = QEventLoop()
    QTimer.singleShot(1500, loop.quit)
    loop.exec()
    assert widget._frames == before, "it kept rendering while paused"

    assert widget.resume() is True
    loop = QEventLoop()
    QTimer.singleShot(1200, loop.quit)
    loop.exec()
    assert widget._frames > before, "it did not start again"
    widget.shutdown()


def test_the_paused_widget_says_so(qtbot):
    pytest.importorskip("numba")
    widget = F.create_fractal_widget(F.Settings(backend="cpu"))
    qtbot.addWidget(widget)
    widget.pause()
    assert "paused for a run" in widget.stats_text()
    widget.shutdown()


# --- actually drawn ---------------------------------------------------------


def test_an_ordinary_launch_gets_no_fractal(qtbot, sandbox):
    """The mode stays hidden: a normal session keeps the ambient engine."""
    from PySide6.QtWidgets import QWidget

    from spacr.qt.app import install_the_spaceout_fractal

    screen = QWidget()
    qtbot.addWidget(screen)
    assert install_the_spaceout_fractal(screen) is False
    assert not [c for c in screen.findChildren(QWidget)
                if hasattr(c, "backend_name")]


def test_spaceout_installs_it(qtbot, sandbox, monkeypatch):
    """The assertion that was missing: everything else tested the widget
    directly, so nothing noticed that nothing built it."""
    from PySide6.QtWidgets import QWidget

    import spacr.qt.theme as theme
    from spacr.qt.app import install_the_spaceout_fractal

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    screen = QWidget()
    qtbot.addWidget(screen)
    screen.resize(640, 400)
    assert install_the_spaceout_fractal(screen) is True
    drawn = [c for c in screen.findChildren(QWidget)
             if hasattr(c, "backend_name")]
    assert len(drawn) == 1
    drawn[0].shutdown()


def test_it_sits_behind_the_screen_not_over_it(qtbot, sandbox, monkeypatch):
    from PySide6.QtWidgets import QWidget

    import spacr.qt.theme as theme
    from spacr.qt.app import install_the_spaceout_fractal

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    screen = QWidget()
    qtbot.addWidget(screen)
    screen.resize(640, 400)
    install_the_spaceout_fractal(screen)
    drawn = [c for c in screen.findChildren(QWidget)
             if hasattr(c, "backend_name")][0]
    assert drawn.size() == screen.rect().size()
    drawn.shutdown()


def test_it_follows_the_window_being_resized(qtbot, sandbox, monkeypatch):
    """Without this it keeps its first size and a resized window shows bare
    ground beside it."""
    from PySide6.QtWidgets import QApplication, QWidget

    import spacr.qt.theme as theme
    from spacr.qt.app import install_the_spaceout_fractal

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    screen = QWidget()
    qtbot.addWidget(screen)
    screen.resize(640, 400)
    screen.show()
    install_the_spaceout_fractal(screen)
    drawn = [c for c in screen.findChildren(QWidget)
             if hasattr(c, "backend_name")][0]
    screen.resize(900, 560)
    QApplication.processEvents()
    assert drawn.width() == screen.width()
    drawn.shutdown()


def test_a_fractal_that_cannot_be_built_leaves_the_old_backdrop(
        qtbot, sandbox, monkeypatch):
    """False rather than raising, so the machine still gets its animation."""
    from PySide6.QtWidgets import QWidget

    import spacr.qt.theme as theme
    from spacr.qt.widgets import fractal_travel

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    monkeypatch.setattr(fractal_travel, "create_fractal_widget",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError))
    from spacr.qt.app import install_the_spaceout_fractal

    screen = QWidget()
    qtbot.addWidget(screen)
    assert install_the_spaceout_fractal(screen) is False


def test_the_backdrop_uses_the_stored_settings(qtbot, sandbox, monkeypatch):
    """One place decides what is offered and what is drawn."""
    import spacr.qt.theme as theme
    from spacr.qt import app as qt_app
    from spacr.qt.preferences import set_fractal_settings

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    set_fractal_settings(speed=1.25, dream=0.5, backend="cpu")
    seen = {}

    def _spy(settings, controls, *args, **kwargs):
        seen["speed"] = controls.speed
        seen["dream"] = controls.dream
        seen["backend"] = settings.backend
        raise RuntimeError("stop here, the arguments are what matter")

    monkeypatch.setattr(
        "spacr.qt.widgets.fractal_travel.create_fractal_widget", _spy)
    from PySide6.QtWidgets import QWidget

    qt_app.install_the_spaceout_fractal(QWidget())
    assert seen == {"speed": 1.25, "dream": 0.5, "backend": "cpu"}


# --- the platform guard -----------------------------------------------------


def test_the_offscreen_platform_is_not_offered_a_gl_canvas(monkeypatch):
    """Getting this wrong does not raise -- Qt dumps core, which no `except`
    around the constructor can catch."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert F.platform_can_do_opengl() is False
    assert F.gpu_is_available() is False
    assert F.resolve_backend("auto") == "cpu"


@pytest.mark.parametrize("platform", ["offscreen", "minimal", "vnc"])
def test_every_headless_platform_is_refused(monkeypatch, platform):
    monkeypatch.setenv("QT_QPA_PLATFORM", platform)
    assert F.platform_can_do_opengl() is False


def test_a_real_display_is_allowed(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    monkeypatch.setenv("DISPLAY", ":0")
    assert F.platform_can_do_opengl() is True


def test_no_display_at_all_is_refused(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert F.platform_can_do_opengl() is False


def test_an_explicit_gpu_request_still_respects_the_guard(qtbot, monkeypatch):
    """Asking for a renderer this platform would crash on is still a crash."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widget = F.create_fractal_widget(F.Settings(backend="gpu"))
    qtbot.addWidget(widget)
    assert widget.backend_name == "cpu"
    widget.shutdown()


# --- every install path, not just one --------------------------------------


def test_install_ambient_returns_the_fractal_under_spaceout(
        qtbot, sandbox, monkeypatch):
    """Hooked at `install_ambient` rather than at its callers.

    There are three call sites -- the module screens, the Home screen and
    the setup slides -- and hooking one left Home showing the old Julia set,
    which is what the maintainer saw and reported.
    """
    from PySide6.QtWidgets import QWidget

    import spacr.qt.theme as theme
    from spacr.qt.widgets.ambient import AmbientWidget, install_ambient

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(480, 320)
    widget = install_ambient(host, None)
    assert not isinstance(widget, AmbientWidget)
    assert hasattr(widget, "backend_name")
    widget.shutdown()


def test_an_ordinary_launch_still_gets_the_ambient_engine(qtbot, sandbox):
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.ambient import AmbientWidget, install_ambient

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(480, 320)
    assert isinstance(install_ambient(host, None), AmbientWidget)


def test_the_old_spaceout_theme_is_no_longer_reached(qtbot, sandbox,
                                                     monkeypatch):
    """`dressed()` swaps the theme to SPACEOUT_THEME, which IS the old
    artwork. Returning before it is what retires that engine."""
    import spacr.qt.theme as theme
    from spacr.qt.widgets import ambient

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    used = []
    monkeypatch.setattr(ambient, "dressed",
                        lambda t, p: used.append((t, p)) or (t, p))
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    widget = ambient.install_ambient(host, None)
    assert used == [], "the ambient engine was still dressed for spaceout"
    widget.shutdown()


def test_it_answers_to_the_ambient_widget_s_own_verb(qtbot, sandbox):
    """`_discard_ambient` and the Home teardown call `set_animating`."""
    widget = F.create_fractal_widget(F.Settings(backend="cpu"))
    qtbot.addWidget(widget)
    assert widget.set_animating(False) is True
    assert widget.is_paused() is True
    assert widget.set_animating(True) is True
    assert widget.is_paused() is False
    widget.shutdown()


def test_the_home_screen_path_gets_it_too(qtbot, sandbox, monkeypatch):
    """Home installs its own backdrop, which is the one that was still old."""
    import inspect

    from spacr.qt.widgets import home

    # It reaches the fractal the same way: through `install_ambient`.
    assert "install_ambient" in inspect.getsource(home.HomePage._install_ambient)


# --- the second pattern -----------------------------------------------------


def test_every_pattern_is_offered():
    assert F.PATTERNS == ("orbit", "cascade", "space", "mandelbrot")


def test_the_cascade_draws_something_that_is_not_flat():
    pytest.importorskip("numba")
    from spacr.qt.widgets.fractal_cascade import CascadeEngine

    frame = CascadeEngine(2).render(64, 48, 0.0, 4.0, 1.5, 4)
    assert frame.shape == (48, 64, 3)
    assert frame.std() > 0


def test_the_cascade_keeps_no_history_between_frames():
    """Its four samples are of one instant, so there is nothing to remember
    -- which is also why pausing it cannot show a seam."""
    from spacr.qt.widgets.fractal_cascade import CascadeEngine

    engine = CascadeEngine(2)
    assert not hasattr(engine, "ring")
    assert not hasattr(engine, "slot")


def test_the_same_instant_gives_the_same_picture():
    """A true supersample is deterministic in t; the orbit pattern's
    temporal jitter is not."""
    pytest.importorskip("numba")
    from spacr.qt.widgets.fractal_cascade import CascadeEngine

    engine = CascadeEngine(2)
    first = engine.render(48, 32, 3.0, 4.0, 1.5, 4).copy()
    second = engine.render(48, 32, 3.0, 4.0, 1.5, 4)
    assert (first == second).all()


def test_the_cascade_travels_with_time():
    pytest.importorskip("numba")
    from spacr.qt.widgets.fractal_cascade import CascadeEngine

    engine = CascadeEngine(2)
    early = engine.render(48, 32, 0.0, 4.0, 1.5, 4).copy()
    later = engine.render(48, 32, 20.0, 4.0, 1.5, 4)
    assert (early != later).any()


def test_the_two_scale_windows_hand_over_without_a_reset():
    """At phase 1 the second window has reached exactly where the first
    started, so the cascade never jumps."""
    import math

    from spacr.qt.widgets.fractal_cascade import LOG_SCALE_BASE, SCALE_BASE

    assert math.isclose(math.exp(-LOG_SCALE_BASE) * SCALE_BASE, 1.0,
                        rel_tol=1e-9)


@pytest.mark.parametrize("pattern", ["orbit", "cascade"])
def test_each_pattern_builds_a_widget(qtbot, pattern):
    pytest.importorskip("numba")
    widget = F.create_fractal_widget(
        F.Settings(pattern=pattern, backend="cpu"))
    qtbot.addWidget(widget)
    assert pattern in widget.stats_text()
    widget.shutdown()


def test_they_do_not_share_a_budget():
    """The cascade is four samples a pixel inside one frame; sharing the
    orbit's pixel count would make it unusable."""
    import inspect

    body = inspect.getsource(F._make_cpu_widget)
    assert "115_000" in body and "460_000" in body


def test_an_unknown_pattern_falls_back(qtbot):
    assert F.Settings(pattern="spiral").validated().pattern == F.DEFAULT_PATTERN


def test_the_pattern_is_a_preference(sandbox):
    from spacr.qt.preferences import get_fractal_settings, set_fractal_settings

    set_fractal_settings(pattern="cascade")
    assert get_fractal_settings()["pattern"] == "cascade"


def test_an_unknown_pattern_is_refused(sandbox):
    from spacr.qt.preferences import set_fractal_settings

    with pytest.raises(ValueError, match="unknown fractal pattern"):
        set_fractal_settings(pattern="spiral")


def test_the_row_is_there_under_spaceout(qtbot, sandbox, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    import spacr.qt.theme as theme
    from spacr.qt.preferences import PreferencesDialog

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    combo = dialog.findChild(QComboBox, "FractalPattern")
    assert combo is not None
    assert [combo.itemData(i) for i in range(combo.count())] == \
        list(F.PATTERNS)


def test_the_chosen_pattern_reaches_the_backdrop(sandbox, monkeypatch):
    """The wiring, not the widget. A setting the backdrop never reads is the
    shape of bug this file has already hit twice."""
    import spacr.qt.theme as theme
    from spacr.qt.preferences import set_fractal_settings
    from spacr.qt.widgets import ambient

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    set_fractal_settings(pattern="cascade")
    seen = {}

    def _spy(settings, controls, *args, **kwargs):
        seen["pattern"] = settings.pattern
        raise RuntimeError("the argument is what matters")

    monkeypatch.setattr(
        "spacr.qt.widgets.fractal_travel.create_fractal_widget", _spy)
    from PySide6.QtWidgets import QWidget

    ambient._the_spaceout_fractal(QWidget())
    assert seen["pattern"] == "cascade"


def test_the_cascade_module_imports_no_pyqt6():
    import ast

    from spacr.qt.widgets import fractal_cascade

    tree = ast.parse(open(fractal_cascade.__file__).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "PyQt6" not in imported


# --- every row explains itself ---------------------------------------------


FRACTAL_ROWS = ("Pattern", "Backend", "Quality", "Scale", "Speed", "Dream",
                "Variable speed")


@pytest.mark.parametrize("row", FRACTAL_ROWS)
def test_every_fractal_row_has_a_caption(row):
    """A caption ships with its row, or the row explains nothing."""
    from spacr.qt.preferences import PREFERENCE_TIPS

    assert row in PREFERENCE_TIPS
    assert len(PREFERENCE_TIPS[row]) > 40


def test_the_dream_caption_does_not_invent_a_storage_ceiling():
    """The caption agrees with the deliberately uncapped stored value."""
    from spacr.qt.preferences import PREFERENCE_TIPS

    caption = PREFERENCE_TIPS["Dream"].lower()
    assert "1.5 is the default" in caption
    assert "without a fixed ceiling" in caption
    assert "1.5 is the maximum" not in caption


def test_the_captions_reach_the_hint_bar_through_the_labels(
        qtbot, sandbox, monkeypatch):
    """A caption in the catalogue that never renders is the same as none.

    Preferences moves label help into its non-modal hint bar, so the label is
    the registered hover target and no native tooltip remains to cover it.
    """
    from PySide6.QtWidgets import QFormLayout, QLabel

    import spacr.qt.theme as theme
    from spacr.qt.preferences import PreferencesDialog
    from spacr.qt.widgets.hint_bar import HintBar

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    bar = dialog.findChild(HintBar)
    assert bar is not None

    explained = {}
    duplicates = []
    for form in dialog.findChildren(QFormLayout):
        for index in range(form.rowCount()):
            item = form.itemAt(index, QFormLayout.LabelRole)
            if item is None:
                continue
            label = item.widget()
            if isinstance(label, QLabel):
                text = (label.text() or "").replace("&", "").strip()
                if text in FRACTAL_ROWS:
                    explained[text] = bool(bar.explains(label))
                    if (label.toolTip() or "").strip():
                        duplicates.append(text)

    assert set(explained) == set(FRACTAL_ROWS), "a row is missing its label"
    assert all(explained.values()), f"unexplained: {explained}"
    assert not duplicates, f"native tooltip duplicates the hint bar: {duplicates}"


def test_the_caption_says_what_the_cascade_costs():
    """The one number a user needs before choosing it."""
    from spacr.qt.preferences import PREFERENCE_TIPS

    assert "four times" in PREFERENCE_TIPS["Pattern"]


def test_the_backend_caption_says_the_gpu_can_be_absent():
    from spacr.qt.preferences import PREFERENCE_TIPS

    assert "vispy" in PREFERENCE_TIPS["Backend"]


def test_the_scale_caption_says_it_is_not_the_look():
    """Scale and quality are easy to confuse; the caption separates them."""
    from spacr.qt.preferences import PREFERENCE_TIPS

    assert "does not change what the fractal looks like" in \
        PREFERENCE_TIPS["Scale"]


# --- lifetime: the three ways it crashed on a real launch ------------------


def test_only_one_backdrop_survives_repeated_installs(qtbot, sandbox,
                                                      monkeypatch):
    """`install_ambient` runs again whenever a screen is rebuilt. The old
    fractal used to stay parented AND RUNNING: four live canvases, four
    vispy timers and four render threads at once."""
    from PySide6.QtWidgets import QWidget

    import spacr.qt.theme as theme
    from spacr.qt.preferences import set_fractal_settings
    from spacr.qt.widgets.ambient import install_ambient

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    set_fractal_settings(backend="cpu")
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(400, 260)

    made = [install_ambient(host, None) for _ in range(3)]
    live = [c for c in host.findChildren(QWidget)
            if hasattr(c, "backend_name")]
    assert len(live) == 1
    assert all(w._stopped for w in made[:-1]), "an old backdrop kept running"
    made[-1].shutdown()


def test_retiring_reports_how_many_it_stopped(qtbot, sandbox, monkeypatch):
    from PySide6.QtWidgets import QWidget

    import spacr.qt.theme as theme
    from spacr.qt.widgets.ambient import _retire_fractals_on, install_ambient

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    host = QWidget()
    qtbot.addWidget(host)
    assert _retire_fractals_on(host) == 0
    install_ambient(host, None)
    assert _retire_fractals_on(host) == 1


def test_retiring_a_host_with_no_backdrop_is_fine(qtbot):
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.ambient import _retire_fractals_on

    host = QWidget()
    qtbot.addWidget(host)
    assert _retire_fractals_on(host) == 0


def test_the_render_thread_is_not_parented_to_the_widget():
    """A QThread whose parent dies while it runs prints "Destroyed while
    thread is still running" and takes the process with it. The backdrop is
    reparented and deleted with its screen, so that is the ordinary path."""
    import inspect

    body = inspect.getsource(F._make_cpu_widget)
    assert "QThread()" in body
    assert "QThread(self)" not in body


def test_the_thread_is_joined_when_qt_frees_the_widget():
    import inspect

    assert "_join_on_destroy(self, self._thread)" in inspect.getsource(
        F._make_cpu_widget)


def test_the_joiner_never_touches_the_widget():
    """`destroyed` fires mid-teardown; reaching for the widget from there is
    a second crash on top of the one this prevents."""
    import inspect

    body = inspect.getsource(F._join_on_destroy)
    inner = body.split("def _join(")[1].split("try:")[0]
    assert "widget" not in inner


def test_a_deleted_backdrop_does_not_take_the_process_down(qtbot, sandbox):
    """The crash as reported: a backdrop deleted with its parent and no
    shutdown() call anywhere."""
    import gc

    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    host.resize(360, 240)
    widget = F.create_fractal_widget(F.Settings(backend="cpu"))
    widget.setParent(host)
    host.show()
    loop = QEventLoop()
    QTimer.singleShot(1500, loop.quit)
    loop.exec()

    host.deleteLater()
    del widget, host
    gc.collect()
    loop = QEventLoop()
    QTimer.singleShot(800, loop.quit)
    loop.exec()
    gc.collect()          # reaching here at all is the assertion


def test_the_gpu_timer_stops_at_the_first_late_tick():
    """vispy's Timer is not a QTimer and outlives the canvas. Its own
    handler catches, logs and RETRIES, which is the 2,4,8...4096 storm."""
    import inspect

    body = inspect.getsource(F._make_gpu_widget)
    assert "_dead" in body
    assert "except RuntimeError:" in body
    assert "self.stop_timer()" in body


def test_the_gpu_canvas_stops_when_qt_frees_it():
    import inspect

    body = inspect.getsource(F._make_gpu_widget)
    assert "self.native.destroyed.connect(self._on_native_destroyed)" in body


def test_shutdown_is_safe_to_call_twice(qtbot, sandbox):
    widget = F.create_fractal_widget(F.Settings(backend="cpu"))
    qtbot.addWidget(widget)
    widget.shutdown()
    widget.shutdown()
