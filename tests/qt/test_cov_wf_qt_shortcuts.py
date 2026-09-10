"""Two keyboard promises whose failure is silent: the AI key and the click
that closes the cheat sheet.

``Ctrl+/`` is advertised as "toggle the AI switch on the screen you are
looking at". The handler finds that screen by walking the window's
:class:`AppScreen` children and stopping at the first VISIBLE one -- so the
interesting cases are the ones where the walk finds nothing to flip: a
window whose screens are all hidden, and a screen that has no AI switch at
all. Both must leave the app exactly as it was; a handler that reached for
a switch that is not there would raise inside a key press, and Qt turns an
exception in a slot into a message on stderr the user never sees while the
key silently does nothing forever.

The shortcut overlay is dismissed by "any key or any click". The click half
is delivered by an event filter on the scroll area's VIEWPORT -- the card
scrolls when the map is taller than the window, so the viewport, not the
overlay, is what the pointer lands on there. If that filter stops firing,
clicking the middle of the cheat sheet leaves the user staring at a dimmed
window that will not go away, and the click must also be SWALLOWED rather
than passed on to whatever sits underneath.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QMainWindow, QWidget

from spacr.qt import shortcuts as sc


class _Switch:
    """The AI toggle reduced to the two calls ``_toggle_ai`` makes on it.

    Recording every ``setChecked`` is the point: "the switch was never
    touched" and "the switch was set back to the value it already had" look
    identical from ``isChecked()`` alone.
    """

    def __init__(self, checked: bool = False):
        self.checked = bool(checked)
        self.history: list[bool] = []

    def isChecked(self) -> bool:
        return self.checked

    def setChecked(self, value) -> None:
        self.checked = bool(value)
        self.history.append(self.checked)


@pytest.fixture
def window(qapp):
    """A real shown main window -- visibility is what the handler tests."""
    win = QMainWindow()
    win.resize(900, 600)
    win.show()
    yield win
    win.hide()
    win.close()
    win.deleteLater()


@pytest.fixture
def screen_factory(qapp):
    """Build real ``AppScreen`` instances without running its constructor.

    ``_toggle_ai`` selects its target with ``findChildren(AppScreen)``, so
    the candidate has to BE an ``AppScreen`` for the test to exercise the
    real selection. The full constructor builds a settings model, a console
    and a figures card, none of which this key touches; ``QWidget.__init__``
    gives a genuine ``AppScreen`` instance whose Qt-side lifecycle hooks are
    the plain widget ones.
    """
    from spacr.qt.screens.app_screen import AppScreen

    class _StubScreen(AppScreen):
        def __init__(self, parent, switch=None):
            QWidget.__init__(self, parent)
            if switch is not None:
                self._ai_switch = switch

        # AppScreen's own show/hide/paint hooks reach into state the real
        # constructor would have made; the widget defaults are correct here.
        def showEvent(self, event):
            QWidget.showEvent(self, event)

        def hideEvent(self, event):
            QWidget.hideEvent(self, event)

        def paintEvent(self, event):
            QWidget.paintEvent(self, event)

        def resizeEvent(self, event):
            QWidget.resizeEvent(self, event)

        def eventFilter(self, obj, event):
            return QWidget.eventFilter(self, obj, event)

    return _StubScreen


# --------------------------------------------------------------------------
# Ctrl+/ -- the AI switch on the screen in front
# --------------------------------------------------------------------------

def test_the_ai_key_flips_the_switch_and_flips_it_back(window, screen_factory):
    """The binding is a TOGGLE, so pressing it twice has to return the user
    to where they started. A handler that only ever turned the AI on would
    leave no way to turn it off from the keyboard, and the switch would
    disagree with the console the second time.
    """
    switch = _Switch(checked=False)
    screen = screen_factory(window, switch)
    screen.show()
    assert screen.isVisible() is True

    sc._toggle_ai(window)
    sc._toggle_ai(window)

    assert switch.history == [True, False]
    assert switch.isChecked() is False


def test_a_screen_nobody_can_see_keeps_its_ai_switch_alone(
        window, screen_factory):
    """Screens live on a stack and only one is on screen; the others are
    still children of the window. If the key flipped the first child rather
    than the visible one, pressing Ctrl+/ would silently arm the AI on a
    module the user is not even looking at -- and the visible screen's own
    switch would stay off, so nothing on screen would show it happened.

    Both halves are driven here: the same call is made once while the screen
    is hidden (nothing must move) and once while it is shown.
    """
    switch = _Switch(checked=False)
    screen = screen_factory(window, switch)
    screen.hide()
    assert screen.isVisible() is False

    sc._toggle_ai(window)
    assert switch.history == []          # the walk found no visible screen
    assert switch.isChecked() is False

    screen.show()
    sc._toggle_ai(window)
    assert switch.history == [True]      # ... and the same call does flip it


def test_a_visible_screen_with_no_ai_switch_is_left_alone(
        window, screen_factory):
    """Not every screen carries an AI toggle -- the interactive ones build a
    different action row. Ctrl+/ is a window-wide binding, so it fires there
    too, and reaching for ``_ai_switch`` unconditionally would raise
    ``AttributeError`` inside the shortcut's slot: the key would look broken
    on that screen and print a traceback to a console the user never reads.

    The absence and its opposite are both driven: the same visible screen is
    given a switch afterwards and the same key then flips it.
    """
    screen = screen_factory(window)
    screen.show()
    assert screen.isVisible() is True
    assert not hasattr(screen, "_ai_switch")

    sc._toggle_ai(window)
    assert not hasattr(screen, "_ai_switch")   # nothing was invented for it

    switch = _Switch(checked=True)
    screen._ai_switch = switch
    sc._toggle_ai(window)
    assert switch.history == [False]
    assert switch.isChecked() is False


def test_a_window_with_no_app_screens_at_all_survives_the_key(window):
    """The key is bound on the main window from ``MainWindow.__init__``, so
    it is live before any module screen has been built -- on the home screen,
    or during startup. Pressing it there must be a no-op rather than an
    exception, and it must still work once a screen exists.
    """
    from spacr.qt.screens.app_screen import AppScreen

    assert window.findChildren(AppScreen) == []
    sc._toggle_ai(window)
    assert window.findChildren(AppScreen) == []


# --------------------------------------------------------------------------
# the cheat-sheet overlay -- a click on the card closes it
# --------------------------------------------------------------------------

@pytest.fixture
def overlay(window):
    """A live cheat-sheet overlay over the shown window."""
    ov = sc.ShortcutOverlay(window)
    ov.show()
    yield ov


def test_a_click_on_the_cheat_sheet_closes_it_and_stops_there(overlay, window):
    """The overlay's whole contract is "it goes away on the next thing you
    do". The card sits inside a scroll area, so a click in the middle of the
    map is delivered to the SCROLL VIEWPORT, not to the overlay widget --
    without the viewport filter the most natural click of all, the one on the
    thing the user is reading, would leave the dimmed sheet stuck on screen.

    The click must also be consumed: returning ``False`` would hand it on to
    the scroll area, which would start a text selection or a drag on a panel
    that is about to be destroyed.
    """
    viewport = overlay._scroll.viewport()
    assert overlay.isVisible() is True

    # A non-click event on the same viewport is NOT a dismissal ...
    passed_on = overlay.eventFilter(viewport, QEvent(QEvent.MouseButtonRelease))
    assert passed_on is False
    assert overlay.isVisible() is True

    # ... and the press is, and is swallowed.
    swallowed = overlay.eventFilter(viewport, QEvent(QEvent.MouseButtonPress))
    assert swallowed is True
    assert overlay.isVisible() is False
    assert [w for w in window.findChildren(QWidget)
            if w.objectName() == sc.OVERLAY_NAME and w.isVisible()] == []


def test_a_press_somewhere_else_is_not_the_cheat_sheets_business(
        overlay, window):
    """The filter is installed on two objects -- the window (to track its
    size) and the scroll viewport (to catch clicks). A press arriving from
    any OTHER widget must fall through to ``QWidget.eventFilter`` unhandled;
    swallowing it would eat clicks belonging to the window underneath while
    the sheet is up.
    """
    stranger = QWidget(window)
    handled = overlay.eventFilter(stranger, QEvent(QEvent.MouseButtonPress))
    assert handled is False
    assert overlay.isVisible() is True

    # The viewport's identical event is the one that is handled.
    assert overlay.eventFilter(overlay._scroll.viewport(),
                               QEvent(QEvent.MouseButtonPress)) is True
    assert overlay.isVisible() is False


def test_the_overlay_follows_the_window_it_dims(overlay, window):
    """It dims "everything behind it", which is only true while it covers the
    whole window. A window resized while the sheet is up -- dragging a corner,
    or the tiling WM that resized it as it mapped -- would otherwise leave a
    bright undimmed band down one side with the card off-centre in it.
    """
    window.resize(1100, 720)
    overlay.eventFilter(window, QEvent(QEvent.Resize))

    assert overlay.geometry() == window.rect()
    card = overlay._card.geometry()
    assert card.width() <= overlay.width()
    assert card.height() <= overlay.height()
    # centred, to within the odd-pixel rounding of an integer divide
    assert abs((overlay.width() - card.width()) // 2 - card.x()) <= 1
