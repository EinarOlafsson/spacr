"""The gate editor's console transcript and chat box have room to work.

The chat box was a QLineEdit, which cannot be made taller -- one line is
what it is. Making it taller meant making it multi-line, and that put the
Enter key in question: it sends today, and a multi-line box normally makes
Enter a newline. These pin the resolution, which is to keep the habit and
put the newline on Shift+Enter.
"""

import pytest

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QLineEdit

from spacr.qt.widgets.gate_console import (
    CHAT_VISIBLE_LINES, CONSOLE_MIN_HEIGHT, CONSOLE_MIN_WIDTH, GateConsole,
)


def _press(widget, key, modifier=Qt.NoModifier):
    widget.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, key, modifier))


@pytest.fixture
def console(qt_theme_applied, qtbot):
    widget = GateConsole()
    qtbot.addWidget(widget)
    return widget


class TestRoom:

    def test_the_chat_box_is_several_lines_tall(self, console):
        """A one-line box cannot hold a question phrased in words."""
        assert console.chat.height() > console.fontMetrics().lineSpacing() * 2

    def test_the_chat_box_is_no_longer_a_single_line_widget(self, console):
        assert not isinstance(console.chat, QLineEdit)

    def test_the_transcript_has_a_floor(self, console):
        """It always stretched; nothing stopped the rows below squeezing it."""
        assert console.log.minimumHeight() == CONSOLE_MIN_HEIGHT
        assert CONSOLE_MIN_HEIGHT >= 120

    def test_the_height_is_measured_in_lines_not_pixels(self, console,
                                                        qt_theme_applied):
        """A pixel height chosen at one zoom is wrong at every other one.

        The qt suite pins font scale 1.0 (INVARIANTS 9), so this asserts the
        relationship rather than a number: the box is about as tall as the
        lines it promises to show.
        """
        line = console.fontMetrics().lineSpacing()
        assert console.chat.height() >= line * CHAT_VISIBLE_LINES
        assert console.chat.height() < line * (CHAT_VISIBLE_LINES + 3)


class TestTheEnterKey:

    def test_enter_still_sends(self, console):
        asked = []
        console.set_responder(lambda q: asked.append(q) or "answer")
        console.chat.setPlainText("why did this gate drop everything")
        _press(console.chat, Qt.Key_Return)
        assert asked == ["why did this gate drop everything"]

    def test_sending_clears_the_box(self, console):
        console.set_responder(lambda q: "answer")
        console.chat.setPlainText("a question")
        _press(console.chat, Qt.Key_Return)
        assert console.chat.toPlainText() == ""

    def test_shift_enter_makes_a_newline_and_does_not_send(self, console):
        asked = []
        console.set_responder(lambda q: asked.append(q) or "answer")
        console.chat.setPlainText("line one")
        _press(console.chat, Qt.Key_Return, Qt.ShiftModifier)
        assert asked == []
        assert "\n" in console.chat.toPlainText()

    def test_the_placeholder_says_where_the_newline_went(self, console):
        """An invisible keybinding is one nobody uses."""
        assert "Shift+Enter" in console.chat.placeholderText()

    def test_the_expression_input_is_untouched(self, console):
        """Only the chat box changed. The command entry is still one line,
        because an expression is one line."""
        assert isinstance(console.input, QLineEdit)


class TestItIsNeverTooNarrowToRead:
    """Height was only half of "the console is still one line".

    It sits in a HORIZONTAL splitter, so it can be dragged thin as easily
    as short. Measured in the gate editor at 126px -- about fifteen
    characters -- which wraps every line into a ribbon. The transcript was
    549px tall at the time, so the vertical fix had worked and the console
    still read as one line.
    """

    def test_the_panel_has_a_width_floor(self, qt_theme_applied, qtbot):
        widget = GateConsole()
        qtbot.addWidget(widget)
        assert widget.minimumWidth() == CONSOLE_MIN_WIDTH
        assert CONSOLE_MIN_WIDTH >= 280, "narrower than this wraps badly"

    def test_in_the_gate_editor_it_is_hidden_or_readable(self, qt_theme_applied,
                                                          qtbot):
        """Two honest states, not three. The third -- open but unreadable --
        is the one that was reported."""
        from spacr.qt.screens.gate_editor import GateEditorScreen

        screen = GateEditorScreen()
        qtbot.addWidget(screen)
        screen.resize(1400, 900)
        screen.show()
        qt_theme_applied.processEvents()
        width = screen.console.width()
        assert width == 0 or width >= CONSOLE_MIN_WIDTH, (
            f"console is {width}px -- open but too narrow to read")
