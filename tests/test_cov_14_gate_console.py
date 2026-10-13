"""The gating console answers, or stays quiet, but never guesses.

An empty box is not a question. Pressing Enter on nothing must leave the
transcript alone and leave what was typed alone, because a console that echoes
a blank line every time the field is focused turns the record of what was asked
into noise.

A numeric answer is summarised rather than printed row by row: a Series of
60,000 values pasted into a transcript is a hang, and the question behind
``area`` is always about its distribution.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _frame():
    return pd.DataFrame({"area": [1.0, 2.0, 3.0, 4.0],
                         "keep": [True, True, False, False]})


def test_a_numeric_column_is_answered_with_its_distribution():
    """A non-boolean Series is described, not dumped into the transcript."""
    from spacr.qt.widgets.gate_console import evaluate

    answer = evaluate("area", _frame())

    assert "mean" in answer
    assert "count" in answer


def test_a_boolean_column_is_answered_as_a_count():
    """A mask answers the question it was really asked: how many objects."""
    from spacr.qt.widgets.gate_console import evaluate

    assert evaluate("keep", _frame()) == "2 of 4 objects"


def test_an_empty_expression_writes_nothing(qtbot):
    """Enter on an empty box leaves the transcript untouched."""
    from spacr.qt.widgets.gate_console import GateConsole

    console = GateConsole()
    qtbot.addWidget(console)
    console.set_frame(_frame())

    assert console.run("   ") == ""
    assert console.transcript() == ""


def test_an_empty_question_writes_nothing(qtbot):
    """The chat half refuses a blank question the same way."""
    from spacr.qt.widgets.gate_console import GateConsole

    console = GateConsole()
    qtbot.addWidget(console)

    assert console.ask("") == ""
    assert console.transcript() == ""


def test_running_the_input_box_clears_it_once_answered(qtbot):
    """A question that was answered leaves the field ready for the next one."""
    from spacr.qt.widgets.gate_console import GateConsole

    console = GateConsole()
    qtbot.addWidget(console)
    console.set_frame(_frame())
    console.input.setText("keep")

    console.run_input()

    assert console.input.text() == ""
    assert "2 of 4 objects" in console.transcript()
    assert "› keep" in console.transcript()


def test_an_empty_input_box_is_not_cleared_and_not_recorded(qtbot):
    """Nothing typed means nothing echoed and nothing to clear."""
    from spacr.qt.widgets.gate_console import GateConsole

    console = GateConsole()
    qtbot.addWidget(console)
    console.set_frame(_frame())
    console.input.setText("  ")

    console.run_input()

    assert console.input.text() == "  "
    assert console.transcript() == ""
