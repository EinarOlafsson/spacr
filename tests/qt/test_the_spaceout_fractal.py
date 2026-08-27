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

    set_fractal_settings(backend="cpu", quality="high", scale=0.5,
                         speed=2.0, dream=0.25, variable_speed=True)
    assert get_fractal_settings() == {
        "backend": "cpu", "quality": "high", "scale": 0.5,
        "speed": 2.0, "dream": 0.25, "variable_speed": True}


def test_a_number_out_of_range_is_clamped(sandbox):
    from spacr.qt.preferences import get_fractal_settings, set_fractal_settings

    set_fractal_settings(dream=99.0, speed=-5.0)
    values = get_fractal_settings()
    assert values["dream"] == 1.5
    assert values["speed"] == 0.15


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
