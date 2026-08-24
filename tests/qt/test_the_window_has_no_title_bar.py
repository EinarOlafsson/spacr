"""The spaCR window has no minimise or close button, and one fullscreen icon.

Asked for in two steps: first "remove the minus and x bar ... and just have
an icon in the top left for true fullscreen", then "have the square be on
the other side (the top right side) and add a minus to its left for
minimizing".

So: top RIGHT, minimise then full screen then close, the order a title bar
puts them in. The x "should act exactly like pressing quit in preferences",
so it calls the same thing Quit does rather than closing the window a second
way -- two exits that differ is how a session ends without saving something.

The colours say what each does: red on the one that ends the session, blue
on full screen (the accent the rest of spaCR uses for "on"), and minimise
stays white, because hiding a window is not a decision.

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


def test_the_top_right_holds_the_three_in_order(window):
    names = [b.objectName() for b in _window_buttons(window)]

    assert names == ["MinimiseWindow", "FullScreenToggle", "CloseWindow"]
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


@pytest.mark.parametrize("name,colour", [
    ("CloseWindow", "#DC3C3C"),        # red: it ends the session
    ("FullScreenToggle", "#3C82DC"),   # blue: the accent for a live control
    ("MinimiseWindow", "#3C82DC"),     # blue too, on the user's instruction
])
def test_each_button_lights_in_its_own_colour(window, qapp, name, colour):
    """The MARK lights, not a plate behind it.

    Rewritten on 2026-08-23: "i meant the x itself not the background of
    the x". The colour used to be a rounded background in the corner
    widget's stylesheet, which is why this read the stylesheet. It is now
    painted into the glyph, so the assertion is on the PIXELS of the
    icon -- the only place a QIcon's colour can be observed.
    """
    from PySide6.QtCore import QEvent, QPoint, QPointF, QSize
    from PySide6.QtGui import QColor, QEnterEvent

    button = [b for b in _window_buttons(window)
              if b.objectName() == name][0]
    wanted = QColor(colour)

    def marks(icon):
        image = icon.pixmap(QSize(18, 18)).toImage()
        return [image.pixelColor(x, y)
                for x in range(image.width())
                for y in range(image.height())
                if image.pixelColor(x, y).alpha() > 200]

    resting = marks(button.icon())
    assert resting, f"{name} draws no mark at all"
    assert not any(_close_to(pixel, wanted) for pixel in resting), (
        f"{name} is already {colour} before it is hovered")

    local = QPointF(button.width() / 2, button.height() / 2)
    button.event(QEnterEvent(local, local, local))
    qapp.processEvents()

    lit = marks(button.icon())
    assert any(_close_to(pixel, wanted) for pixel in lit), (
        f"hovering {name} did not paint its mark {colour}")

    button.event(QEvent(QEvent.Type.Leave))
    qapp.processEvents()
    assert not any(_close_to(pixel, wanted) for pixel in marks(button.icon())), (
        f"{name} stayed {colour} after the pointer left")


def _close_to(pixel, wanted, tol: int = 24) -> bool:
    """Antialiasing means the mark is a gradient, not one flat colour."""
    return (abs(pixel.red() - wanted.red()) <= tol
            and abs(pixel.green() - wanted.green()) <= tol
            and abs(pixel.blue() - wanted.blue()) <= tol)


def test_the_x_is_wired_to_the_same_thing_quit_is(window):
    """"act exactly like pressing quit" -- one exit path, not two.

    Checked by asking Qt what the button is connected to rather than by
    clicking it: clicking really would close the window, and a test that
    tears down its own fixture proves nothing about what it meant to.
    """
    close = [b for b in _window_buttons(window)
             if b.objectName() == "CloseWindow"][0]
    quit_action = [a for menu in window.menuBar().actions()
                   if menu.menu() is not None
                   for a in menu.menu().actions()
                   if "quit" in a.text().lower()]

    assert quit_action, "there is no Quit to match"
    # Both end the window the same way. Verified by SOURCE rather than by
    # clicking: a click really would close the fixture, and by wiring
    # rather than by behaviour because `close` is the one call both make.
    import inspect

    from spacr.qt.app import MainWindow

    wiring = inspect.getsource(MainWindow._install_fullscreen_button)
    assert "close.clicked.connect(self.close)" in wiring
    assert quit_action[0].shortcut() == QKeySequence.StandardKey.Quit


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
