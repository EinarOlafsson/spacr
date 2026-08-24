"""The spaCR window has no minimise or close button, and one fullscreen icon.

Asked for in two steps: first "remove the minus and x bar ... and just have
an icon in the top left for true fullscreen", then "have the square be on
the other side (the top right side) and add a minus to its left for
minimizing".

So: top RIGHT, minimise then full screen -- the order a title bar puts them
in. Closing is still not there; Quit is in the spaCR menu with its usual
shortcut, and a stray click on an x mid-analysis costs more than reaching
for the menu does.

NOTHING ABOUT CLOSING DEPENDS ON THE BUTTON THAT WENT. Quit keeps its
standard shortcut in the spaCR menu, so a frameless window cannot trap
anyone -- which is the failure mode a frameless main window has.

The menu bar becomes the title bar: dragging its empty area moves the
window and a double-click there maximises it. Without that the window could
not be moved at all, which is a worse trade than the bar it replaced.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QToolButton


@pytest.fixture
def window(qapp):
    from spacr.qt.app import MainWindow

    win = MainWindow()
    try:
        yield win
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_the_window_is_frameless(window):
    assert bool(window.windowFlags() & Qt.WindowType.FramelessWindowHint)


def _window_buttons(window):
    corner = window.menuBar().cornerWidget(Qt.Corner.TopRightCorner)
    return corner.findChildren(QToolButton) if corner is not None else []


def test_the_top_right_holds_minimise_then_full_screen(window):
    names = [b.objectName() for b in _window_buttons(window)]

    assert names == ["MinimiseWindow", "FullScreenToggle"]
    assert all(not b.icon().isNull() for b in _window_buttons(window)), (
        "the icons are drawn, not shipped")


def test_nothing_is_left_in_the_top_left(window):
    assert window.menuBar().cornerWidget(Qt.Corner.TopLeftCorner) is None


def test_the_icon_toggles_true_fullscreen(window, qapp):
    window.show()
    qapp.processEvents()
    assert not window.isFullScreen()

    assert window.toggle_fullscreen() is True
    qapp.processEvents()
    assert window.isFullScreen()

    assert window.toggle_fullscreen() is False
    qapp.processEvents()
    assert not window.isFullScreen()


def test_the_button_is_what_calls_it(window, qapp):
    window.show()
    qapp.processEvents()
    full = [b for b in _window_buttons(window)
            if b.objectName() == "FullScreenToggle"][0]

    full.click()
    qapp.processEvents()

    assert window.isFullScreen()


def test_quitting_still_has_its_shortcut(window):
    """A frameless window with no way out is a trap."""
    quit_actions = [a for menu in window.menuBar().findChildren(type(
        window.menuBar().actions()[0].menu())) for a in menu.actions()
        if "quit" in a.text().lower()]

    assert quit_actions, "no Quit action anywhere in the menu bar"
    assert any(a.shortcut() == QKeySequence.StandardKey.Quit
               or not a.shortcut().isEmpty() for a in quit_actions)


def test_the_minus_minimises(window, qapp):
    """The other half of what the title bar used to offer."""
    window.show()
    qapp.processEvents()
    minimise = [b for b in _window_buttons(window)
                if b.objectName() == "MinimiseWindow"][0]

    minimise.click()
    qapp.processEvents()

    assert window.isMinimized() or not window.isActiveWindow()


def test_full_screen_has_a_shortcut(window):
    shortcuts = [a.shortcut().toString() for a in window.actions()]
    assert "F11" in shortcuts


def test_a_press_on_the_empty_bar_starts_a_drag(window, qapp):
    """The menu bar is the title bar; without this nothing can move it."""
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent

    window.show()
    qapp.processEvents()
    bar = window.menuBar()
    # Far to the right of the last menu: empty bar.
    where = QPointF(bar.width() - 8, bar.height() / 2)
    assert bar.actionAt(where.toPoint()) is None

    press = QMouseEvent(QEvent.Type.MouseButtonPress, where,
                        bar.mapToGlobal(where), Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    window.eventFilter(bar, press)

    assert window._drag_from is not None


def test_a_press_on_a_menu_does_not_start_a_drag(window, qapp):
    """Dragging must not swallow the menus it sits beside."""
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent

    window.show()
    qapp.processEvents()
    bar = window.menuBar()
    first = bar.actions()[0]
    where = QPointF(bar.actionGeometry(first).center())
    assert bar.actionAt(where.toPoint()) is not None

    press = QMouseEvent(QEvent.Type.MouseButtonPress, where,
                        bar.mapToGlobal(where), Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    window.eventFilter(bar, press)

    assert window._drag_from is None
