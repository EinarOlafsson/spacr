"""Application shortcuts: a key a menu already owns is left to the menu, and
each shortcut's handler survives the part it drives being unavailable.

Binding a second holder to a key makes Qt treat it as ambiguous and fire
neither, so ``install`` must not bind Ctrl+End over a menu action that
already carries it. The handlers behind Ctrl+Alt+0, Ctrl+F and
Ctrl+Shift+H each call into an optional module; when that module raises,
the key does nothing rather than raising out of Qt's event loop -- and
Ctrl+F still puts the caret in the settings search box when the settings
column cannot be opened first.
"""
from __future__ import annotations

import os
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QAction, QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import QLineEdit, QMainWindow, QWidget  # noqa: E402

from spacr.qt import shortcuts  # noqa: E402

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("this part is not available")


def _own_shortcuts(window):
    return {sc.key().toString() for sc in window.findChildren(
        QShortcut, options=Qt.FindDirectChildrenOnly)}


def test_a_key_a_menu_action_owns_is_not_bound_a_second_time(
        qapp, qtbot, monkeypatch):
    import spacr.qt.help_search as help_search

    monkeypatch.setattr(help_search, "install_window_hooks", _boom)
    window = QMainWindow()
    qtbot.addWidget(window)
    owner = QAction("Jump to the end", window)
    owner.setShortcut(QKeySequence("Ctrl+End"))
    window.addAction(owner)

    shortcuts.install(window)

    bound = _own_shortcuts(window)
    assert "Ctrl+End" not in bound
    assert {"Ctrl+K", "Ctrl+F", "F1"} <= bound


def test_reset_every_scale_is_asked_of_the_window_and_may_fail(
        qapp, monkeypatch):
    from spacr.qt import gui_scale

    asked = []
    window = QMainWindow()
    monkeypatch.setattr(gui_scale, "reset_every_scale", asked.append)
    shortcuts._reset_every_scale(window)
    assert asked == [window]

    monkeypatch.setattr(gui_scale, "reset_every_scale", _boom)
    assert shortcuts._reset_every_scale(window) is None
    window.deleteLater()


def test_ctrl_f_focuses_the_search_box_when_the_column_cannot_open(
        qapp, qtbot):
    screen = QWidget()
    qtbot.addWidget(screen)
    box = QLineEdit("cell", screen)
    screen._settings_search = types.SimpleNamespace(_input=box)
    screen.reveal_settings = _boom
    screen.show()
    qtbot.waitExposed(screen)
    screen.activateWindow()
    window = types.SimpleNamespace(
        _stack=types.SimpleNamespace(currentWidget=lambda: screen))

    shortcuts._focus_settings_search(window)

    qtbot.waitUntil(box.hasFocus)
    assert box.selectedText() == "cell"


def test_the_help_search_key_reaches_the_field_or_does_nothing(
        qapp, monkeypatch):
    import spacr.qt.help_search as help_search

    focused = []
    window = object()
    monkeypatch.setattr(help_search, "focus_field", focused.append)
    shortcuts._focus_help_search(window)
    assert focused == [window]

    monkeypatch.setattr(help_search, "focus_field", _boom)
    assert shortcuts._focus_help_search(window) is None
