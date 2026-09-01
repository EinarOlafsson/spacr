"""A module's description lands in the status bar, not over the grid.

Asked for on 2026-09-01: "the tooltips for modules should only be shown
at the bottom of the screen".

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


@pytest.fixture
def window(qtbot):
    win = QMainWindow()
    win.setStatusBar(QStatusBar())
    qtbot.addWidget(win)
    return win


def _module_button(parent, name="Mask", summary="Segment cells and nuclei."):
    button = QPushButton(parent)
    button.setProperty(M.NAME_PROPERTY, name)
    button.setProperty(M.SUMMARY_PROPERTY, summary)
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

def test_a_module_tooltip_is_shown_in_the_status_bar(window):
    hints = M._ModuleHints(window)
    button = _module_button(window)

    handled = hints.eventFilter(button, _tooltip_event(button))

    assert handled is True, "the popup was not suppressed"
    assert window.statusBar().currentMessage() == (
        "Mask — Segment cells and nuclei.")


def test_a_plain_widget_keeps_its_ordinary_tooltip(window):
    """The filter is application-wide, so it must not eat every tooltip
    in the program."""
    hints = M._ModuleHints(window)
    label = QLabel("plain", window)
    label.setToolTip("an ordinary tooltip")

    assert hints.eventFilter(label, _tooltip_event(label)) is False
    assert window.statusBar().currentMessage() == ""


def test_events_that_are_not_tooltips_pass_straight_through(window):
    hints = M._ModuleHints(window)
    button = _module_button(window)
    assert hints.eventFilter(button, QEvent(QEvent.Type.Enter)) is False


def test_a_window_with_no_status_bar_keeps_its_popup(qtbot):
    """SUPPRESSED ONLY IF IT LANDED SOMEWHERE.

    Losing the description entirely would be worse than the popup this
    replaces, so the filter declines rather than swallowing it.

    A plain QWidget, not a QMainWindow with its bar removed: `statusBar()`
    LAZILY CREATES one, so `setStatusBar(None)` does not produce a window
    without a bar and cannot exercise this path.
    """
    from PySide6.QtWidgets import QWidget

    bare = QWidget()
    qtbot.addWidget(bare)
    assert not hasattr(bare, "statusBar")
    hints = M._ModuleHints(bare)
    button = _module_button(bare)

    assert hints.eventFilter(button, _tooltip_event(button)) is False


def test_a_main_window_always_has_a_bar_to_show_it_in(qtbot):
    """The premise the test above rests on, stated rather than assumed."""
    win = QMainWindow()
    qtbot.addWidget(win)
    win.setStatusBar(None)
    assert win.statusBar() is not None, (
        "QMainWindow no longer creates a status bar on demand")


def test_a_deleted_widget_does_not_raise(window):
    """changeEvent and tooltips both arrive during teardown."""
    class _Gone:
        def property(self, _name):
            raise RuntimeError("Signal source has been deleted")

    assert M.module_hint_text(_Gone()) == ""


def test_the_message_expires_rather_than_describing_nothing(window):
    """A permanent message would describe a module nobody is pointing
    at, for the rest of the session."""
    assert M.LINGER_MS > 0


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

    source = inspect.getsource(app_mod.Sidebar)
    assert M.SUMMARY_PROPERTY in source
    assert M.NAME_PROPERTY in source


def test_the_main_window_installs_the_filter():
    import inspect

    from spacr.qt import app as app_mod

    source = inspect.getsource(app_mod.MainWindow)
    assert "install_module_hints" in source
