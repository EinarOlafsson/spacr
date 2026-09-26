"""Item 530: a saved fractal setting reaches the running backdrop at once.

"in spaceout model, i noticed that i need to restart the application for
fractal theme changes to take effect."

THE CAUSE. The window builds one spaceout backdrop and keeps it for the
session. Saving Preferences pushed the runtime numbers into it, but the
pattern, backend, quality, scale and the Mandelbrot reference orbit are
fixed when it is constructed, and nothing constructed it again. The fix
rebuilds just the backdrop -- in place, keeping every reference to it --
when one of those changes, and leaves the running one alone otherwise.

Offscreen, so the CPU renderer is the one built; the GPU path is covered by
a stand-in that records what it was asked for, and was measured for real
under xvfb (Mesa llvmpipe) -- see the item file.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QWidget

import spacr.qt.theme as theme
from spacr.qt import preferences as P
from spacr.qt.widgets import ambient as A
from spacr.qt.widgets import fractal_travel as ft


@pytest.fixture
def spaceout(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setattr(P, "_SAFE_MODE", False, raising=False)
    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    monkeypatch.setattr(ft, "_LIVE_CONTROLS", [])
    P.set_fractal_settings(pattern="orbit", backend="cpu",
                           quality="balanced", scale=1.0)
    made = []
    yield made
    for host in made:
        A._retire_fractals_on(host)
        host.deleteLater()
    QApplication.processEvents()


def _window_with_a_backdrop(made):
    """A top-level holding a host, the way the main window holds the dock.

    The backdrop is kept as ``_dock_backdrop`` on the window, which is the
    reference a rebuild has to move.
    """
    window = QWidget()
    host = QWidget(window)
    window.resize(800, 600)
    host.resize(800, 600)
    made.append(window)
    window._dock_backdrop = A.install_ambient(host, None)
    assert window._dock_backdrop is not None
    assert hasattr(window._dock_backdrop, "_spaceout_built_from")
    return window, host


def _save(**values):
    """What the Preferences dialog's Save does with the Fractal tab."""
    P.set_fractal_settings(**values)
    ft.apply_saved_controls()
    ft.restart_the_dive()
    return A.rebuild_the_spaceout_backdrops()


@pytest.mark.parametrize("pattern", ["cascade", "space", "orbit"])
def test_a_new_pattern_is_drawn_without_a_restart(spaceout, pattern):
    P.set_fractal_settings(pattern="space" if pattern != "space" else "orbit")
    window, host = _window_with_a_backdrop(spaceout)
    old = window._dock_backdrop

    assert _save(pattern=pattern) == 1
    new = window._dock_backdrop
    assert new is not old, "the window still holds the retired backdrop"
    assert f"· {pattern} ·" in new.stats_text()
    assert new.parentWidget() is host
    assert old.parentWidget() is None, "the old one is still on screen"
    assert A._live_spaceout_fractals() == [new]


def test_a_new_quality_is_drawn_without_a_restart(spaceout):
    window, _host = _window_with_a_backdrop(spaceout)
    assert "CPU/balanced" in window._dock_backdrop.stats_text()

    assert _save(quality="high") == 1
    assert "CPU/high" in window._dock_backdrop.stats_text()


def test_a_new_scale_changes_the_size_the_renderer_draws(spaceout):
    window, host = _window_with_a_backdrop(spaceout)
    before = window._dock_backdrop._target_size()

    assert _save(scale=0.5) == 1
    after = window._dock_backdrop._target_size()
    assert after[0] * after[1] < before[0] * before[1], (before, after)


@pytest.mark.parametrize("name, value", [("max_iterations", 900),
                                         ("precision_digits", 80)])
def test_the_reference_orbit_numbers_rebuild_it(spaceout, name, value):
    """The Mandelbrot reference orbit is iterated once, at construction."""
    window, _host = _window_with_a_backdrop(spaceout)
    old = window._dock_backdrop

    assert _save(**{name: value}) == 1
    new = window._dock_backdrop
    assert new is not old
    index = A._BUILT_FROM_SETTINGS.index(name)
    assert new._spaceout_built_from[index] == value


def test_the_backend_switches_without_a_restart(spaceout, monkeypatch):
    asked = []

    class _Gpu(QWidget):
        backend_name = "gpu"

        def __init__(self):
            super().__init__()
            self._paused = False

        def stats_text(self):
            return "GPU"

        def pause(self):
            self._paused = True
            return True

        def resume(self):
            self._paused = False
            return True

        def is_paused(self):
            return self._paused

        def shutdown(self):
            pass

    def _make(settings, controls, hardware):
        asked.append(settings.pattern)
        return _Gpu()

    monkeypatch.setattr(ft, "gpu_is_available", lambda: True)
    monkeypatch.setattr(ft, "_make_gpu_widget", _make)
    window, _host = _window_with_a_backdrop(spaceout)
    assert window._dock_backdrop.backend_name == "cpu"

    assert _save(backend="gpu") == 1
    assert window._dock_backdrop.backend_name == "gpu"
    assert asked == ["orbit"]

    assert _save(backend="cpu") == 1
    assert window._dock_backdrop.backend_name == "cpu"


@pytest.mark.parametrize("name, value, attribute", [
    ("speed", 2.5, "speed"),
    ("dream", 0.4, "dream"),
    ("variable_speed", True, "variable_speed"),
    ("speed_min", 0.3, "speed_min"),
    ("speed_max", 3.0, "speed_max"),
    ("speed_period", 40.0, "speed_period"),
    ("pointer_gravity", False, "follow_pointer"),
    ("pointer_size", 1.7, "pointer_size"),
    ("pointer_strength", 0.6, "pointer_strength"),
    ("zoom_rate", 3.0, "zoom_rate"),
])
def test_a_runtime_setting_reaches_the_running_backdrop(
        spaceout, name, value, attribute):
    """No rebuild: the canvas reads these from its controls every frame."""
    window, _host = _window_with_a_backdrop(spaceout)
    running = window._dock_backdrop
    controls = running._spaceout_controls
    assert getattr(controls, attribute) != value

    assert _save(**{name: value}) == 0
    assert window._dock_backdrop is running, "rebuilt for no reason"
    assert getattr(controls, attribute) == value


@pytest.mark.parametrize("name, value", [
    ("path", "guided"),
    ("steering", 0.8),
    ("max_depth", 12.0),
    ("seconds_per_decade", 10.0),
    ("base_iterations", 250),
    ("iterations_per_decade", 70.0),
    ("initial_scale", 2.0),
    ("steering_strength", 0.7),
    ("steering_interval_decades", 3.0),
    ("steering_duration", 5.0),
    ("candidate_count", 12),
])
def test_a_setting_read_each_frame_is_not_a_rebuild(spaceout, name, value):
    """The GPU canvas reads these from the store per frame; see item 530."""
    window, _host = _window_with_a_backdrop(spaceout)
    running = window._dock_backdrop

    assert _save(**{name: value}) == 0
    assert window._dock_backdrop is running
    assert P.get_fractal_settings()[name] == pytest.approx(value)


def test_render_scale_changes_the_running_renderer(spaceout):
    window, _host = _window_with_a_backdrop(spaceout)
    running = window._dock_backdrop
    before = running._target_size()

    assert _save(render_scale=0.5) == 0
    assert window._dock_backdrop is running
    after = running._target_size()
    assert after[0] < before[0] and after[1] < before[1]


def test_a_rebuild_keeps_the_controls_the_pause_and_the_visibility(spaceout):
    window, host = _window_with_a_backdrop(spaceout)
    old = window._dock_backdrop
    controls = old._spaceout_controls
    old.pause()
    old.hide()

    assert _save(pattern="cascade") == 1
    new = window._dock_backdrop
    assert new._spaceout_controls is controls
    assert new.is_paused(), "a backdrop paused for a run started drawing"
    assert new.isHidden()


def test_the_old_resize_filter_is_dropped(spaceout):
    window, host = _window_with_a_backdrop(spaceout)
    for pattern in ("cascade", "space", "orbit"):
        _save(pattern=pattern)
    following = [f for f in host.findChildren(A._FractalTracksItsHost)
                 if f._widget is not None]
    assert len(following) == 1
    assert following[0]._widget is window._dock_backdrop


def test_a_screen_holding_its_own_backdrop_gets_the_new_one(spaceout):
    screen = QWidget()
    screen.resize(640, 480)
    spaceout.append(screen)
    screen._ambient = A.install_ambient(screen, None)
    old = screen._ambient

    assert _save(pattern="cascade") == 1
    assert screen._ambient is not old
    assert screen._ambient.parentWidget() is screen


def test_a_backdrop_that_cannot_be_rebuilt_keeps_the_running_one(
        spaceout, monkeypatch):
    window, host = _window_with_a_backdrop(spaceout)
    running = window._dock_backdrop

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no renderer")

    monkeypatch.setattr(A, "_build_the_spaceout_fractal", _boom)
    assert _save(pattern="cascade") == 0
    assert window._dock_backdrop is running
    assert running.parentWidget() is host


def test_a_heavy_import_defers_the_rebuild(spaceout, monkeypatch):
    window, _host = _window_with_a_backdrop(spaceout)
    running = window._dock_backdrop
    later = []

    def _busy(*_args, **_kwargs):
        raise ft._HeavyImportInProgress("busy")

    class _Timer:
        @staticmethod
        def singleShot(ms, callback):
            later.append((ms, callback))

    monkeypatch.setattr(A, "_build_the_spaceout_fractal", _busy)
    monkeypatch.setattr(A, "QTimer", _Timer)
    assert _save(pattern="cascade") == 0
    assert window._dock_backdrop is running
    assert later and later[0][1] is A.rebuild_the_spaceout_backdrops


def test_nothing_is_rebuilt_outside_spaceout(spaceout, monkeypatch):
    window, _host = _window_with_a_backdrop(spaceout)
    running = window._dock_backdrop
    monkeypatch.setattr(theme, "spaceout_enabled", lambda: False)
    assert _save(pattern="cascade") == 0
    assert window._dock_backdrop is running


def test_saving_preferences_rebuilds_the_running_backdrop(spaceout, qtbot):
    """The wiring: the dialog's Save is what the user presses."""
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox

    window, _host = _window_with_a_backdrop(spaceout)
    old = window._dock_backdrop

    dialog = P.PreferencesDialog()
    qtbot.addWidget(dialog)
    combo = dialog.findChild(QComboBox, "FractalPattern")
    combo.setCurrentIndex(combo.findData("cascade"))
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()

    assert P.get_fractal_settings()["pattern"] == "cascade"
    assert window._dock_backdrop is not old
    assert "· cascade ·" in window._dock_backdrop.stats_text()
