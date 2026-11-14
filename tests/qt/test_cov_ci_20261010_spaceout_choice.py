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
