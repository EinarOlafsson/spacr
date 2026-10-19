"""A field's faded chrome is drawn by a native label under it, not a hook.

Over the window's animated field every widget is painted on every frame, and
the fade's Python paint hook re-entered Python once per field per frame: 27
times on the Regression screen. A field with a parent now has its cached
chrome shown by a plain label one step under it in the stacking order. The
window must show the same pixels as the hook drew, dark and light, and a
repaint must not enter the fade's Python at all. Counted and compared, not
timed.
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QLineEdit,  # noqa: E402
                               QSpinBox, QVBoxLayout, QWidget)

from spacr.qt import preferences as prefs  # noqa: E402
from spacr.qt import theme as theme_mod  # noqa: E402
from spacr.qt.widgets import field_fade as ff  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def sandbox(qapp, tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / "spacr-qt.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    assert str(tmp_path) in prefs._settings().fileName()
    saved_qss = dict(theme_mod._WIDGET_QSS)
    saved_sheet = qapp.styleSheet()
    ff.invalidate_field_fade()
    yield store
    ff.uninstall_field_fade(qapp)
    ff.invalidate_field_fade()
    theme_mod._WIDGET_QSS.clear()
    theme_mod._WIDGET_QSS.update(saved_qss)
    qapp.setStyleSheet(saved_sheet)


def _form(qtbot):
    host = QWidget()
    host.resize(420, 260)
    layout = QVBoxLayout(host)
    fields = [QLineEdit("value"), QComboBox(), QSpinBox(), QDoubleSpinBox()]
    fields[1].addItem("chosen option")
    for field in fields:
        layout.addWidget(field)
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    for field in fields:
        field.clearFocus()
    return host, fields


def _grab(widget):
    return widget.grab().toImage().convertToFormat(QImage.Format_RGB32)


def _hooked(fields):
    """Put the fields back on the paint hook, as before the underlay."""
    for field in fields:
        layer = ff._UNDERLAYS.get(field)
        if layer is not None:
            layer.hide()
        field.installEventFilter(ff._filter)
        field.update()


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_underlay_shows_the_pixels_the_hook_drew(qtbot, qapp, sandbox,
                                                     theme):
    prefs.set_theme_choice(theme)
    prefs.apply_preferences_to_app(qapp)
    host, fields = _form(qtbot)
    qtbot.wait(50)
    assert all(ff._UNDERLAYS.get(field) is not None
               and ff._UNDERLAYS[field].isVisible() for field in fields)
    underlaid = _grab(host)

    _hooked(fields)
    qtbot.wait(50)
    hooked = _grab(host)
    assert underlaid == hooked


def test_a_repaint_does_not_enter_the_fade(qtbot, qapp, sandbox):
    prefs.set_theme_choice("dark")
    prefs.apply_preferences_to_app(qapp)
    host, fields = _form(qtbot)
    entered = []

    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_filename == ff.__file__:
            entered.append(frame.f_code.co_name)

    sys.setprofile(profile)
    try:
        host.repaint()
    finally:
        sys.setprofile(None)
    assert entered == []


def test_the_underlay_follows_focus_size_and_hiding(qtbot, qapp, sandbox):
    prefs.set_theme_choice("dark")
    prefs.apply_preferences_to_app(qapp)
    host, fields = _form(qtbot)
    edit = fields[0]
    layer = ff._UNDERLAYS[edit]
    resting = layer.pixmap().cacheKey()

    edit.setFocus()
    qtbot.waitUntil(lambda: layer.pixmap().cacheKey() != resting, timeout=2000)
    edit.clearFocus()
    qtbot.waitUntil(lambda: layer.pixmap().cacheKey() == resting, timeout=2000)

    edit.setFixedWidth(200)
    qtbot.waitUntil(lambda: layer.geometry() == edit.geometry(), timeout=2000)

    edit.hide()
    qtbot.waitUntil(lambda: not layer.isVisible(), timeout=2000)
    edit.show()
    qtbot.waitUntil(lambda: layer.isVisible(), timeout=2000)


def test_a_parentless_field_keeps_the_hook(qtbot, qapp, sandbox):
    prefs.apply_preferences_to_app(qapp)
    edit = QLineEdit()
    qtbot.addWidget(edit)
    edit.show()
    qtbot.waitExposed(edit)
    assert ff._UNDERLAYS.get(edit) is None
