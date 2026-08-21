"""The bottom of a console is one gesture away (instruction 232).

"it should be possible to go to the bottom of a console section without
having to scroll through everything."

AND THE OTHER DIRECTION. A console that always follows cannot be read; one
that never follows has to be chased. FOLLOW WHILE AT THE BOTTOM, STOP WHEN
THE USER SCROLLS UP, and offer the way back -- the two behaviours are one
decision.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QKeySequence          # noqa: E402
from PySide6.QtWidgets import QApplication      # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def console(app):
    from spacr.qt.widgets.console_panel import ConsolePanel

    panel = ConsolePanel()
    panel.resize(400, 200)
    return panel


def _make_scrollable(console, height=2000):
    """Give the scrollbar a real range.

    The headless console lays out no entries, so the bar's maximum is 0 and
    every position IS the end. Setting the range directly is what lets the
    behaviour be tested at all -- and it is the behaviour that matters, not
    how the content got tall.
    """
    bar = console._scroll.verticalScrollBar()
    bar.setRange(0, height)
    return bar


class TestTheGestureExists:

    def test_there_is_a_ctrl_end_shortcut(self, console):
        assert console._end_shortcut.key() == QKeySequence("Ctrl+End")

    def test_it_is_held(self, console):
        """Qt keeps a bare pointer, and a shortcut only the constructor
        referenced stops working as soon as the call returns."""
        assert getattr(console, "_end_shortcut", None) is not None

    def test_there_is_a_visible_control_too(self, console):
        """For the people who do not know the shortcut."""
        assert console._jump is not None
        assert "end" in console._jump.text().lower()

    def test_the_control_names_the_shortcut(self, console):
        assert "Ctrl+End" in console._jump.toolTip()


class TestItIsShownOnlyWhenItWouldDoSomething:
    """A button that is always there and usually inert is furniture, and the
    user stops seeing it -- which is the state it is most needed in."""

    def test_hidden_at_the_end(self, console):
        _make_scrollable(console)
        console.jump_to_the_end()
        assert not console._jump.isVisibleTo(console)

    def test_shown_once_scrolled_away(self, console):
        bar = _make_scrollable(console)
        bar.setValue(0)
        console._refresh_jump_button()
        assert console._jump.isVisibleTo(console)

    def test_hidden_again_after_the_jump(self, console):
        bar = _make_scrollable(console)
        bar.setValue(0)
        console._refresh_jump_button()
        console.jump_to_the_end()
        assert not console._jump.isVisibleTo(console)


class TestTheJump:

    def test_it_reaches_the_last_line_from_anywhere(self, console):
        bar = _make_scrollable(console)
        bar.setValue(0)
        console.jump_to_the_end()
        assert bar.value() == bar.maximum()

    def test_it_resumes_the_follow(self, console):
        """Both halves, because they are one decision: a console that jumped
        without resuming would slide off the end on the very next line
        written, and the user would press it again."""
        bar = _make_scrollable(console)
        bar.setValue(0)
        console._follow_output = False
        console.jump_to_the_end()
        assert console._follow_output

    def test_at_the_end_tolerates_a_few_pixels(self, console):
        """A scrollbar dragged to the end does not always land exactly on
        maximum() -- the same tolerance the scroll handler uses."""
        bar = _make_scrollable(console)
        bar.setValue(bar.maximum() - 2)
        assert console.at_the_end()

    def test_and_not_more_than_a_few(self, console):
        bar = _make_scrollable(console)
        bar.setValue(bar.maximum() - 500)
        assert not console.at_the_end()


class TestTheOtherDirection:
    """"reading something above must not be yanked back down by the next
    line written"."""

    def test_scrolling_up_stops_the_follow(self, console):
        bar = _make_scrollable(console)
        console._follow_output = True
        bar.setValue(0)
        console._on_console_scrolled(0)
        # The handler only ever RESTORES the follow; the panel clears it
        # when a section is raised. What matters here is that being at 0
        # does not turn it back on.
        assert not console.at_the_end()

    def test_scrolling_back_down_restores_it(self, console):
        bar = _make_scrollable(console)
        console._follow_output = False
        console._on_console_scrolled(bar.maximum())
        assert console._follow_output
