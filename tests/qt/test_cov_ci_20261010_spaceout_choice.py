"""Every branch of the live Spaceout animation switch on real Qt widgets."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt import preferences as prefs, theme
from spacr.qt.widgets import ambient


@pytest.fixture
def spaceout(tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    store.setValue("prefs/brand_palette_revision", True)
    was_spaceout = theme.spaceout_enabled()
    theme.enable_spaceout()
    yield store
    theme.enable_spaceout() if was_spaceout else theme.disable_spaceout()


class Fractal(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._spaceout_built_from = ("test",)
        self._paused = False
        self.retired = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def is_paused(self):
        return self._paused

    def shutdown(self):
        self.retired = True


def _host(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 200)
    host.show()
    return host


def _switch():
    ambient._apply_spaceout_animation_choice(QApplication.instance())


def test_turning_animation_off_and_on_restores_each_fractals_own_state(qtbot, spaceout):
    host = _host(qtbot)
    running = Fractal(host)
    running.show()
    hidden_paused = Fractal(host)
    hidden_paused.pause()
    hidden_paused.hide()
    host._ambient = running

    prefs.set_ambient_animation("none")
    assert not prefs.get_ambient_enabled()
    _switch()
    assert running.is_paused() and running.isHidden()
    assert running.property("spacrPreferenceDisabled") is True
    assert running.property("spacrPreferenceWasHidden") is False
    assert hidden_paused.property("spacrPreferenceWasHidden") is True
    assert hidden_paused.property("spacrPreferenceWasPaused") is True
    _switch()
    assert running.property("spacrPreferenceWasHidden") is False

    prefs.set_ambient_animation(ambient.SPACEOUT_THEME)
    assert prefs.get_ambient_theme() == ambient.SPACEOUT_THEME
    _switch()
    assert not running.isHidden() and not running.is_paused()
    assert running.property("spacrPreferenceDisabled") is False
    assert hidden_paused.isHidden() and hidden_paused.is_paused()
    assert hidden_paused.property("spacrPreferenceDisabled") is False
    assert host._ambient is running
    assert not running.retired
    _switch()
    assert host._ambient is running and not running.is_paused()

    theme.disable_spaceout()
    prefs.set_ambient_animation("none")
    _switch()
    assert not running.isHidden() and not running.is_paused()


def test_a_disabled_fractal_replaced_by_a_regular_theme_keeps_its_saved_state(
        qtbot, spaceout):
    host = _host(qtbot)
    old = Fractal(host)
    old.show()
    host._ambient = old
    prefs.set_ambient_animation("none")
    _switch()
    assert old.isHidden() and old.is_paused()

    prefs.set_ambient_animation("blobs")
    _switch()
    new = host._ambient
    assert isinstance(new, ambient.AmbientWidget)
    assert new.theme() == "blobs"
    assert old.retired
    assert not new.isHidden()
    assert new.is_animating()


def test_regular_backdrops_stay_until_spaceout_is_chosen_then_retire_hidden(
        qtbot, monkeypatch, spaceout):
    host = _host(qtbot)
    old = ambient.install_ambient(host, theme="blobs", palette="spacr")
    host._ambient = old
    old.hide()
    prefs.set_ambient_animation("drift")
    _switch()
    assert host._ambient is old

    def install(target, *, theme, palette):
        widget = Fractal(target)
        widget.show()
        return widget

    monkeypatch.setattr(ambient, "install_ambient", install)
    prefs.set_ambient_animation(ambient.SPACEOUT_THEME)
    _switch()
    new = host._ambient
    assert isinstance(new, Fractal)
    assert new.isHidden()
    assert old.property("spacrRetiringBackdrop") is True
    assert old.parentWidget() is None


def test_a_dead_widget_is_skipped_and_a_busy_backdrop_is_retried(qtbot, monkeypatch, spaceout):
    host = _host(qtbot)
    Fractal(host).show()
    prefs.set_ambient_animation("blobs")
    retried = []
    monkeypatch.setattr(ambient.QTimer, "singleShot",
                        lambda delay, callback: retried.append(delay))

    def busy(*_args, **_kwargs):
        raise RuntimeError("Internal C++ object already deleted.")

    monkeypatch.setattr(ambient, "install_ambient", busy)
    _switch()
    assert retried == []

    def not_yet(*_args, **_kwargs):
        raise ValueError("heavy import")

    monkeypatch.setattr(ambient, "install_ambient", not_yet)
    monkeypatch.setattr(ambient, "_the_backdrop_wants_a_retry", lambda error: True)
    _switch()
    assert retried == [400]

    monkeypatch.setattr(ambient, "_the_backdrop_wants_a_retry", lambda error: False)
    _switch()
    assert retried == [400]
