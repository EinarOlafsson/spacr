"""Recomposing the sheet for a late block leaves screens that wear it alone.

Each module screen installs the blocks registered after the window sheet
was composed on itself (``theme.ensure_widget_qss_applied``). Recomposing
the sheet only to include those blocks used to give every hidden screen a
stale serial, so each restyled itself in full at its first show: 0.55-0.8 s
per screen on an Intel Mac. Counted in ``setStyleSheet`` calls, not timed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget  # noqa: E402

from spacr.qt import theme  # noqa: E402


@pytest.fixture
def isolated_sheet(qapp):
    saved_attr = getattr(qapp, theme._WINDOW_SHEET_ATTRIBUTE, None)
    saved_serial = getattr(qapp, theme._WINDOW_SHEET_SERIAL, 0)
    saved_history = getattr(qapp, theme._WINDOW_SHEET_HISTORY, None)
    yield qapp
    _HOSTS.clear()
    theme.unregister_widget_qss("LateProbeBlock")
    setattr(qapp, theme._WINDOW_SHEET_ATTRIBUTE, saved_attr)
    setattr(qapp, theme._WINDOW_SHEET_SERIAL, saved_serial)
    setattr(qapp, theme._WINDOW_SHEET_HISTORY, saved_history)


_HOSTS = []


def _screen(qtbot):
    """A hidden module-screen-like page inside a window, marked as a sheet target."""
    host = QWidget()
    qtbot.addWidget(host)
    _HOSTS.append(host)
    screen = QWidget(host)
    QVBoxLayout(screen).addWidget(QLabel("probe", objectName="LateProbeBlock"))
    screen.hide()
    theme.mark_as_a_sheet_target(screen)
    return screen


def _counted(widget, monkeypatch):
    calls = []
    real = type(widget).setStyleSheet

    def counting(self, text):
        if self is widget:
            calls.append(len(text))
        return real(self, text)

    monkeypatch.setattr(type(widget), "setStyleSheet", counting)
    return calls


def test_a_screen_that_wears_the_new_block_keeps_its_sheet(
        isolated_sheet, qtbot, monkeypatch):
    app = isolated_sheet
    theme.apply_stylesheet_per_window(app, theme.stylesheet(theme="dark"))
    wearing, missing = _screen(qtbot), _screen(qtbot)
    for screen in (wearing, missing):
        assert theme._sheet_one_window(screen)

    theme.register_widget_qss(
        "LateProbeBlock", lambda palette, opacity: "QLabel#LateProbeBlock { color: red; }",
        replace=True)
    theme.ensure_widget_qss_applied("LateProbeBlock", root=wearing)
    assert "registered widget QSS: LateProbeBlock" in wearing.styleSheet()

    theme.apply_stylesheet_per_window(app, theme.stylesheet(theme="dark"))
    calls = _counted(wearing, monkeypatch)
    assert not theme._sheet_one_window(wearing)
    assert calls == []
    assert wearing.property(theme._WINDOW_SHEET_SERIAL) == getattr(
        app, theme._WINDOW_SHEET_SERIAL)

    assert theme._sheet_one_window(missing), (
        "a screen without the new block must take the new sheet")


def test_a_real_theme_change_still_restyles_a_screen_that_wears_every_block(
        isolated_sheet, qtbot):
    app = isolated_sheet
    theme.apply_stylesheet_per_window(app, theme.stylesheet(theme="dark"))
    screen = _screen(qtbot)
    assert theme._sheet_one_window(screen)
    theme.apply_stylesheet_per_window(app, theme.stylesheet(theme="light"))
    assert theme._sheet_one_window(screen)
