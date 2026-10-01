"""Item 510: the FEATURES window looks like every other spaCR window.

    "in make masks, the features button spawns a window that should have an
     ocupacy, and have rounded edges remove the maximize minimize and close
     at the top and add a close red button at the bottom to the right of
     measure which should be blue. and the window should have the travelling
     blue rim. make this window like other spacr windows."

The fix is not a stylesheet written into this one window. spaCR already
dresses every dialog through `spacr.qt.widgets.glass`, an application event
filter that recognises a dialog by its TYPE -- and this window was a bare
``QWidget`` shown with the ``Qt.Window`` flag, so the filter never saw it and
it kept the operating system's title bar, its square corners and its opaque
background.

So what is asserted here is the two halves of that: the window is a
``QDialog``, which is what the sweep reaches, and the sweep then actually
reaches it -- card, rim, rounded corners, no native buttons -- plus the two
actions at the bottom in the colours the request names.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt                        # noqa: E402
from PySide6.QtGui import QKeyEvent                          # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog          # noqa: E402

from spacr.qt.screens.measure_inputs import MeasureInputsScreen  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def glassed(app):
    """The dressing filter installed as it is at startup, then taken off.

    A FILTER LEFT ON DECIDES THE LOOK OF EVERY DIALOG EXAMINED AFTER IT, so
    the teardown is not tidiness: a later file's dialog would otherwise be
    looking at a card its author never put there.
    """
    from spacr.qt.preferences import apply_preferences_to_app
    from spacr.qt.widgets.glass import (install_glass_everywhere,
                                        uninstall_glass_everywhere)

    apply_preferences_to_app(app)
    install_glass_everywhere(app)
    yield app
    uninstall_glass_everywhere(app)
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.fixture
def window(glassed):
    """The FEATURES window, shown and settled, taken down afterwards."""
    screen = MeasureInputsScreen(threaded=False)
    screen.resize(720, 520)
    screen.show()
    for _ in range(8):
        glassed.processEvents()
    yield screen
    screen.close()
    screen.deleteLater()
    glassed.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    glassed.processEvents()


def test_it_is_a_dialog_which_is_what_the_sweep_recognises():
    """The whole fix in one assertion. `glass.wants_glass` tests the type."""
    from spacr.qt.widgets.glass import wants_glass

    screen = MeasureInputsScreen(threaded=False)
    try:
        assert isinstance(screen, QDialog)
        assert wants_glass(screen) is True
    finally:
        screen.deleteLater()


def test_it_gets_the_card_and_the_travelling_rim(window):
    """The translucent body and the rim that goes round it."""
    from spacr.qt.widgets.glass import GLASSED
    from spacr.qt.widgets.setup_card import SetupCard

    assert window.property(GLASSED) is True
    cards = window.findChildren(SetupCard)
    assert cards, "no card behind the window's contents"
    assert window.layout().indexOf(cards[0]) == -1, (
        "the card must sit behind the contents, not inside the layout")


def test_the_native_title_bar_and_its_buttons_are_gone(window):
    """"remove the maximize minimize and close at the top" -- the request."""
    flags = window.windowFlags()
    assert flags & Qt.WindowType.FramelessWindowHint
    assert window.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


def test_the_corners_are_rounded(window):
    """Cut from the window's shape, not merely composited away."""
    assert not window.mask().isEmpty(), "the window was never cut to shape"
    assert not window.mask().contains(window.rect().topLeft()), (
        "the top-left corner pixel is still part of the window")


def test_measure_is_blue_and_close_is_red_and_to_its_right(window):
    """The two actions, in the roles and the order the request names."""
    assert window.run_button.objectName() == "PrimaryButton"
    assert window.close_button.objectName() == "DangerButton"
    assert window.close_button.text() == "Close"

    row = window.close_button.parentWidget().layout()
    assert row is not None
    assert window.run_button.x() < window.close_button.x(), (
        "Close must be to the right of Measure")
    assert window.close_button.y() > window.inputs.y(), (
        "the actions belong at the bottom, under the table")


def test_neither_action_steals_the_return_key(window):
    """A dialog makes its buttons auto-default; Measure is minutes of work."""
    assert window.run_button.autoDefault() is False
    assert window.close_button.autoDefault() is False


def test_close_closes_it_and_stops_the_run(window):
    """The red button is the way out the missing title bar took away."""
    stopped = []
    window.stop_the_runners = lambda: stopped.append(True)
    window.close_button.click()
    assert not window.isVisible()
    assert stopped, "closing left the run going"


def test_escape_stops_the_run_rather_than_hiding_it_behind_the_window(
        window, glassed):
    """`QDialog.reject` hides without a close event.

    That would leave a measure run -- minutes of work, on a thread -- going
    behind a window nobody can see, and a running QThread destroyed at exit
    aborts the process. So the hook is `done`, which every way out goes
    through.
    """
    stopped = []
    window.stop_the_runners = lambda: stopped.append(True)
    window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Escape,
                                   Qt.KeyboardModifier.NoModifier))
    glassed.processEvents()
    assert stopped, "Escape left the run going"
    assert not window.isVisible()
