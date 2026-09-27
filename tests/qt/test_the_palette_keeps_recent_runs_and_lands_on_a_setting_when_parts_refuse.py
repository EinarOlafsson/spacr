"""Ctrl-K keeps its recent runs when the app list cannot be read, and still
lands on a setting when the screen cannot open its heading.

With no app registry the palette cannot say which apps are hidden, so it
offers every recent run rather than none. Revealing a setting on a screen
whose heading opener raises still gives the setting's control the focus.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit, QMainWindow, QStackedWidget, QWidget

from spacr.qt import command_palette as CP


class _Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self._stack = QStackedWidget(self)
        self.setCentralWidget(self._stack)
        self._screens = {}


class _Model:
    def __init__(self, widgets):
        self._widgets = widgets

    def _label_for(self, key):
        return key.replace("_", " ").title()

    def plain_tooltip_for(self, key):
        return f"what {key} does"

    def collect(self):
        return {k: 0 for k in self._widgets}


@pytest.fixture
def window(qtbot):
    win = _Window()
    qtbot.addWidget(win)
    return win


def test_without_the_app_list_every_recent_run_is_offered(
        window, monkeypatch, qtbot, tmp_path):
    from spacr import run_journal
    from spacr.qt import app as qt_app

    def _no_registry():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(qt_app, "visible_apps", _no_registry)
    monkeypatch.setattr(run_journal, "recent_runs", lambda limit=8: [
        {"app_key": "mask", "dir": tmp_path, "status": "ok",
         "elapsed_s": 1.0}])

    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)

    recent = [c.label for c in pal._commands if c.section == "Recent runs"]
    assert recent == ["Recent · mask  (ok, 1.0s)"]
    assert not any(c.section.startswith("Apps") for c in pal._commands)


def test_the_first_row_selected_is_a_command_not_a_heading(window, qtbot):
    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)
    row = pal._list.currentRow()
    assert pal._list.item(row - 1).flags() == Qt.NoItemFlags
    assert pal._list.item(row).flags() != Qt.NoItemFlags
    assert pal._list.item(row).data(Qt.UserRole) is pal._commands[0]


def test_a_heading_that_will_not_open_still_focuses_the_setting(
        window, qtbot):
    screen = QWidget()
    qtbot.addWidget(screen)
    field = QLineEdit(screen)
    screen.app_key = "mask"
    screen._settings_model = _Model({"cell_channel": field})

    def _refuses(key):
        raise RuntimeError(f"heading of {key} is gone")

    screen._open_the_heading_of = _refuses
    window._stack.addWidget(screen)
    window._stack.setCurrentWidget(screen)
    window.show()
    qtbot.waitExposed(window)
    pal = CP.CommandPalette(window)
    qtbot.addWidget(pal)

    pal._reveal_setting("cell_channel")

    qtbot.waitUntil(lambda: field.hasFocus(), timeout=2000)
