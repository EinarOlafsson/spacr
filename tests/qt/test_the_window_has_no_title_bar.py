"""The spaCR window has no minimise or close button, and one fullscreen icon.

Asked for: "remove the minus and x bar from the spacr window and just have
an icon in the top left for true fullscreen".

The two things the title bar offered were a button that hid the application
and a button that quit it, and neither is what a user reaches for
mid-analysis. Fullscreen is.

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


def test_the_top_left_holds_one_icon(window):
    corner = window.menuBar().cornerWidget(Qt.Corner.TopLeftCorner)

    assert isinstance(corner, QToolButton)
    assert corner.objectName() == "FullScreenToggle"
    assert not corner.icon().isNull(), "the icon is drawn, not shipped"


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
    corner = window.menuBar().cornerWidget(Qt.Corner.TopLeftCorner)

    corner.click()
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
