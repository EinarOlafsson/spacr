"""The Ctrl+End sweep survives every console that is not quite a console.

The sweep exists because two live bindings for one key make Qt fire
NEITHER handler, so a console panel's own copy is stood down inside a
window that binds the key too. It walks whatever is on screen, and what is
on screen includes panels built before the binding existed, panels whose
C++ side has already gone, and panels that are not showing at all.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import QMainWindow, QWidget  # noqa: E402
from shiboken6 import delete as _delete_cpp_side  # noqa: E402

from spacr.qt import shortcuts as sc  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def window(qapp):
    win = QMainWindow()
    yield win
    win.close()
    win.deleteLater()


class _PanelWithoutABinding:
    """A console built before the window bound the key for everyone."""

    _end_shortcut = None


class _HiddenPanel:
    def __init__(self):
        self._end_shortcut = None
        self.jumped = False

    def isVisible(self):
        return False

    def jump_to_the_end(self):                     # pragma: no cover - guard
        self.jumped = True


class _BrokenPanel:
    def __init__(self):
        self._end_shortcut = None

    def isVisible(self):
        return True

    def jump_to_the_end(self):
        raise RuntimeError("the console's document has gone")


class _LivePanel:
    def __init__(self):
        self._end_shortcut = None
        self.jumped = False

    def isVisible(self):
        return True

    def jump_to_the_end(self):
        self.jumped = True


def test_a_window_that_cannot_be_searched_yields_no_consoles(caplog):
    class _Unsearchable:
        def findChildren(self, _kind):
            raise RuntimeError("the window's C++ side has gone")

    with caplog.at_level(logging.DEBUG, logger=sc.LOG.name):
        assert sc._consoles(_Unsearchable()) == []

    assert any("console panels" in record.message
               for record in caplog.records)


def test_a_panel_without_its_own_binding_is_left_alone(window):
    panel = _PanelWithoutABinding()

    sc._hand_ctrl_end_to_the_window(window, [panel])

    assert panel._end_shortcut is None, (
        "there was nothing to stand down and nothing was invented")


def test_a_binding_whose_widget_has_gone_does_not_stop_the_next_panel(window):
    dead_host = QWidget()
    dead = QShortcut(QKeySequence("Ctrl+End"), dead_host)
    _delete_cpp_side(dead_host)

    live_host = QWidget()
    live = QShortcut(QKeySequence("Ctrl+End"), live_host)
    live.setEnabled(True)

    gone = _PanelWithoutABinding()
    gone._end_shortcut = dead
    still_here = _PanelWithoutABinding()
    still_here._end_shortcut = live

    sc._hand_ctrl_end_to_the_window(window, [gone, still_here])

    assert live.isEnabled() is False, (
        "the panel after the dead one still had to be stood down")


def test_only_the_console_on_screen_is_jumped(window, monkeypatch):
    hidden, broken, live = _HiddenPanel(), _BrokenPanel(), _LivePanel()
    monkeypatch.setattr(sc, "_consoles",
                        lambda _window: [hidden, broken, live])

    sc._jump_to_the_newest_line(window)

    assert hidden.jumped is False, "a console nobody can see is not the one"
    assert live.jumped is True, (
        "a console that raised must not swallow the key for the next one")
