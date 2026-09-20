"""A module's description lands in the hint strip, not over the grid.

Asked for on 2026-09-01: "the tooltips for modules should only be shown
at the bottom of the screen".

IT WENT TO THE STATUS BAR UNTIL 2026-09-03, the line in the bottom LEFT,
with a four-second linger -- and the maintainer reported what that looked
like: "in the bottom of the screen to the left is text that also flickers
sometimes like its going to what is hovered and something else back and
forthe." Two writers alternating, this filter and Qt restoring the
permanent message. It routes to the page's own hint strip now, which holds
the last module for thirty seconds and carries its API and Tutorial
links.

Home already worked this way, and the reason is written on ``AppTile``:
these blurbs run to several hundred characters, which is fine in a fixed
line the eye can skip and wrong in a box drawn on top of the grid the
user is reading to choose between. The sidebar and the fold strip kept
popping them.

The hook is ``QEvent.ToolTip``, not hover -- it fires when Qt has already
decided to show a tooltip, so the description appears after the same
delay the user is used to, and returning True is what suppresses the
popup.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint
from PySide6.QtGui import QHelpEvent
from PySide6.QtWidgets import QLabel, QMainWindow, QPushButton, QStatusBar

from spacr.qt import module_hints as M


class _Window(QMainWindow):
    """A main window with the ONE method the filter routes through.

    `_ModuleHints` no longer writes anywhere itself: it hands the app key to
    the window, which knows which page is in front and therefore which strip
    to write to. Recording the calls here is what lets these tests stay
    about the FILTER rather than about a whole MainWindow.
    """

    def __init__(self):
        super().__init__()
        self.hinted: list = []

    def _show_module_hint(self, key):
        self.hinted.append(key)


@pytest.fixture
def window(qtbot):
    win = _Window()
    win.setStatusBar(QStatusBar())
    qtbot.addWidget(win)
    return win


def _module_button(parent, name="Mask", summary="Segment cells and nuclei.",
                   labelled=True):
    """A module button.

    `labelled` matters: a widget that shows its own name has its popup
    diverted to the status bar, and an ICON-ONLY one keeps the popup as
    well, because the popup is the only thing identifying it.
    """
    button = QPushButton(name if labelled else "", parent)
    button.setProperty(M.NAME_PROPERTY, name)
    button.setProperty(M.SUMMARY_PROPERTY, summary)
    # The KEY is what the filter routes on now -- the strip resolves the
    # sentence and both links from it, so nothing composes a line any more.
    button.setProperty(M.KEY_PROPERTY, name.lower())
    return button


def _tooltip_event(widget):
    return QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1),
                      widget.mapToGlobal(QPoint(1, 1)))


# ---------------------------------------------------------------------------
# The text
# ---------------------------------------------------------------------------

def test_the_line_joins_the_name_and_the_summary(window):
    button = _module_button(window)
    assert M.module_hint_text(button) == "Mask — Segment cells and nuclei."


def test_a_module_with_no_summary_still_gives_its_name(window):
    button = _module_button(window, summary="")
    assert M.module_hint_text(button) == "Mask"


def test_a_widget_that_is_not_a_module_contributes_nothing(window):
    assert M.module_hint_text(QLabel("plain", window)) == ""


def test_the_text_comes_from_the_properties_not_the_tooltip(window):
    """So a language switch retranslates the source rather than
    translating an already-translated tooltip."""
    button = _module_button(window)
    button.setToolTip("something else entirely")
    assert "something else entirely" not in M.module_hint_text(button)


# ---------------------------------------------------------------------------
# The diversion
# ---------------------------------------------------------------------------

def test_a_module_tooltip_is_routed_to_the_hint_strip(window):
    hints = M._ModuleHints(window)
    button = _module_button(window)

    handled = hints.eventFilter(button, _tooltip_event(button))

    assert handled is True, "the popup was not suppressed"
    assert window.hinted == ["mask"], (
        f"the filter routed {window.hinted!r}")
    assert window.statusBar().currentMessage() == "", (
        "the status bar is being written on hover again -- that is the "
        "flicker reported on 2026-09-03")


def test_a_plain_widget_keeps_its_ordinary_tooltip(window):
    """The filter is application-wide, so it must not eat every tooltip
    in the program."""
    hints = M._ModuleHints(window)
    label = QLabel("plain", window)
    label.setToolTip("an ordinary tooltip")

    assert hints.eventFilter(label, _tooltip_event(label)) is False
    assert window.hinted == []


def test_events_that_are_not_tooltips_pass_straight_through(window):
    hints = M._ModuleHints(window)
    button = _module_button(window)
    assert hints.eventFilter(button, QEvent(QEvent.Type.Enter)) is False


def test_a_window_that_cannot_show_a_hint_keeps_its_popup(qtbot):
    """SUPPRESSED ONLY IF IT LANDED SOMEWHERE.

    Losing the description entirely would be worse than the popup this
    replaces, so the filter declines rather than swallowing it.

    A plain QWidget rather than a QMainWindow: what is missing now is
    `_show_module_hint`, which no plain widget has.
    """
    from PySide6.QtWidgets import QWidget

    bare = QWidget()
    qtbot.addWidget(bare)
    assert not hasattr(bare, "statusBar")
    hints = M._ModuleHints(bare)
    button = _module_button(bare)

    assert hints.eventFilter(button, _tooltip_event(button)) is False


def test_the_real_window_has_the_method_this_file_stubs(qtbot):
    """The premise these tests rest on, stated rather than assumed.

    `_Window` above stands in for `MainWindow`, and a stub that no longer
    matches the thing it stands for is a suite that passes while the
    feature is broken.
    """
    from spacr.qt.app import MainWindow

    assert callable(getattr(MainWindow, "_show_module_hint", None)), (
        "MainWindow has no _show_module_hint for the filter to route to")


def test_a_deleted_widget_does_not_raise(window):
    """changeEvent and tooltips both arrive during teardown."""
    class _Gone:
        def property(self, _name):
            raise RuntimeError("Signal source has been deleted")

    assert M.module_hint_text(_Gone()) == ""


def test_the_strip_and_not_the_status_bar_is_the_target(window):
    """The four-second linger is gone with the status-bar write.

    A module nobody is pointing at should not be described for the rest of
    the session -- which the linger existed to prevent. The strip solves it
    the same way and for longer: `ModuleHintBar.HOLD_MS` puts the prompt
    back after thirty seconds, and the reason it is longer is that the
    strip carries two links that have to be reachable.
    """
    from spacr.qt.widgets.module_hint_bar import ModuleHintBar

    assert not hasattr(M, "LINGER_MS"), (
        "the status-bar linger is back, and so is the flicker")
    assert ModuleHintBar.HOLD_MS > 0


# ---------------------------------------------------------------------------
# The widgets that carry it
# ---------------------------------------------------------------------------

def test_the_fold_strip_button_carries_the_properties():
    """Without them the fold strip's own tooltips stay popups."""
    import inspect

    from spacr.qt.widgets import fold_strip

    source = inspect.getsource(fold_strip)
    assert f'setProperty("{M.NAME_PROPERTY}"' in source
    assert f'setProperty("{M.SUMMARY_PROPERTY}"' in source


def test_the_sidebar_button_carries_them_too():
    import inspect

    from spacr.qt import app as app_mod

    # THE ROW, NOT THE DOCK. `Sidebar` became a thin binding to
    # `spacr.qt.widgets.dock` on 2026-09-03; the properties the bottom strip
    # reads are stamped where they belong, on the row itself.
    from spacr.qt.widgets.dock import DockRow

    source = inspect.getsource(DockRow)
    assert M.SUMMARY_PROPERTY in source
    assert M.NAME_PROPERTY in source
    assert app_mod.Sidebar is not None


def test_the_main_window_installs_the_filter():
    import inspect

    from spacr.qt import app as app_mod

    source = inspect.getsource(app_mod.MainWindow)
    assert "install_module_hints" in source


# ---------------------------------------------------------------------------
# An icon-only button keeps its popup
# ---------------------------------------------------------------------------

def test_a_button_with_no_label_keeps_its_popup(window):
    """Reported: the Mask masthead button "tooltip isnt showing up".

    A fold-strip button is an icon and nothing else, so removing its popup
    would take away the only thing identifying it that follows the pointer.
    The description still reaches the strip -- that is what was asked for --
    but the popup is not suppressed.
    """
    hints = M._ModuleHints(window)
    button = _module_button(window, name="Timelapse", labelled=False)

    handled = hints.eventFilter(button, _tooltip_event(button))

    assert handled is False, "an icon-only button lost its popup"
    assert window.hinted == ["timelapse"], (
        "the description did not also reach the strip")


def test_a_labelled_button_still_gives_its_popup_up(window):
    """The other half. A sidebar row has its name written on it, so the
    popup would be a second copy of something already on screen."""
    hints = M._ModuleHints(window)
    button = _module_button(window, name="Mask", labelled=True)

    assert hints.eventFilter(button, _tooltip_event(button)) is True
    assert window.hinted == ["mask"]


# ---------------------------------------------------------------------------
# The ways the hint does not land
#
# Every one of these runs INSIDE an application-wide event filter, on the
# hover path, during teardown as often as not. A hint that cannot be shown
# costs a line at the bottom of the window; an exception raised here reaches
# Qt's event loop from a mouse move.
# ---------------------------------------------------------------------------

def test_nothing_at_all_contributes_nothing(window):
    """The filter is installed on the APPLICATION, so it sees every event."""
    assert M.module_hint_text(None) == ""


def test_a_description_longer_than_the_strip_is_cut_at_a_word(window):
    """Module descriptions run to 154 characters and the cap is 96.

    Cut mid-word it reads as a typo; cut at a space with an ellipsis it
    reads as a description that continues. The trailing punctuation goes
    too, so the line never ends in " ,…".
    """
    summary = ("Segment cells and nuclei with Cellpose, then measure every "
               "object in every channel, and write one row per object, "
               "per field, per well")
    button = _module_button(window, name="Mask", summary=summary)

    line = M.module_hint_text(button)

    assert len(line) <= M.MAX_HINT_CHARS + 1, line
    assert line.endswith("…")
    assert not line[:-1].rstrip().endswith((",", ";", ":", "-", "—"))
    assert line[:-1].rstrip() in f"Mask — {summary}", (
        "the cut has to be a prefix of the real sentence")


def test_a_summary_with_no_name_is_still_the_line(window):
    """A widget can carry a summary and no name; the line is the summary."""
    button = _module_button(window, name="", summary="Segment cells.")
    button.setProperty(M.NAME_PROPERTY, "")

    assert M.module_hint_text(button) == "Segment cells."


def test_a_widget_deleted_between_the_text_and_the_route_keeps_its_popup(
    window,
):
    """`module_hint_text` succeeded; the C++ object went away before `_show`.

    Teardown order makes this ordinary, and the filter has to answer False
    -- which leaves Qt to draw its own tooltip for a widget that is going
    away, and costs nothing.
    """
    hints = M._ModuleHints(window)

    class _GoneSinceTheText:
        def property(self, _name):
            raise RuntimeError("Internal C++ object already deleted")

    assert hints._show(_GoneSinceTheText()) is False
    assert window.hinted == []


def test_the_home_tile_does_not_describe_itself_in_its_own_strip(window):
    """Home IS the grid of modules; a hint naming Home says nothing."""
    hints = M._ModuleHints(window)
    button = _module_button(window, name="Home")
    button.setProperty(M.KEY_PROPERTY, "__home__")

    assert hints._show(button) is False

    button.setProperty(M.KEY_PROPERTY, "")
    assert hints._show(button) is False
    assert window.hinted == []


def test_a_router_that_raises_is_read_as_not_landed(window, monkeypatch):
    """The page can be torn down while its tooltip is still in flight."""
    hints = M._ModuleHints(window)
    button = _module_button(window, name="Mask")

    def gone(_key):
        raise RuntimeError("Internal C++ object already deleted")

    monkeypatch.setattr(window, "_show_module_hint", gone)

    assert hints._show(button) is False


def test_a_widget_whose_label_cannot_be_read_keeps_its_popup(window):
    """`_shows_its_own_name` decides whether the popup is suppressed.

    A widget with no `text()` at all, or one already deleted, is treated as
    icon-only -- so the popup stays, which is the only thing identifying it.
    """
    hints = M._ModuleHints(window)

    class _NoText:
        pass

    class _Gone:
        def text(self):
            raise RuntimeError("Internal C++ object already deleted")

    assert hints._shows_its_own_name(_NoText()) is False
    assert hints._shows_its_own_name(_Gone()) is False


def test_the_filter_ignores_events_for_things_that_are_not_widgets(window):
    """The filter is on the QApplication, so it sees QObjects too."""
    from PySide6.QtCore import QObject

    hints = M._ModuleHints(window)
    plain = QObject()

    assert hints.eventFilter(plain, _tooltip_event(window)) is False


def test_installing_without_an_application_returns_none(monkeypatch):
    """The headless CLI imports the Qt package without ever making an app."""
    from PySide6.QtWidgets import QApplication

    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))

    assert M.install_module_hints(object()) is None


def test_installing_puts_the_filter_on_the_application_not_the_widgets(
    window, qtbot,
):
    """The fold strip builds its buttons lazily, per host masthead.

    A filter installed per widget would miss every button made after it ran,
    which is most of them -- so it goes on the application, and a button
    created AFTERWARDS is still diverted.
    """
    from PySide6.QtWidgets import QApplication

    hints = M.install_module_hints(window)
    assert isinstance(hints, M._ModuleHints)
    try:
        later = _module_button(window, name="Measure")
        qtbot.addWidget(later)

        assert hints.eventFilter(later, _tooltip_event(later)) is True
        assert window.hinted == ["measure"]
    finally:
        QApplication.instance().removeEventFilter(hints)
