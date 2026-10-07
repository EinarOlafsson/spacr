"""The strip that speaks for whatever the pointer is on.

Registration moves a control's sentence out of its tooltip and into the bar,
and everything downstream of that move is a case the bar has to get right:
a control with nothing to say must not be watched at all, a control that
already carries an accessible description must keep the one the author wrote,
and a Leave from a control the bar is not currently speaking for must not
blank the line the reader is in the middle of.

The catalog is allowed to be missing. The bar is built while a form is being
assembled, which can be before the language machinery is reachable, and a
strip that raised there would take the form with it.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from spacr.qt.widgets.hint_bar import (DEFAULT_HINT, HintBar,
                                       explain_through_the_bar, hint_bar_of)

pytestmark = pytest.mark.qt


def _enter(bar, widget):
    bar.eventFilter(widget, QEvent(QEvent.Enter))


def _leave(bar, widget):
    bar.eventFilter(widget, QEvent(QEvent.Leave))


# -- the catalog may be unreachable ------------------------------------------

def test_a_bar_whose_catalog_cannot_be_imported_says_the_english(qapp,
                                                                 monkeypatch):
    """No catalog is not an error; it is the source string, shown as written.

    The bar is built while a form is being assembled, so an import that
    fails here would take the form down before it reached the screen.
    """
    stub = types.ModuleType("spacr.qt.i18n")
    monkeypatch.setitem(sys.modules, "spacr.qt.i18n", stub)

    bar = HintBar()
    button = QPushButton("Run")
    bar.explain(button, "Start the pipeline.")
    _enter(bar, button)

    assert bar.text() == "Start the pipeline."

    bar.reset()

    assert bar.text() == DEFAULT_HINT


def test_a_working_catalog_is_what_the_bar_shows(qapp, monkeypatch):
    """The same line goes through the catalog when there is one."""
    import spacr.qt.i18n as i18n

    monkeypatch.setattr(i18n, "tr", lambda text: f"<{text}>")

    bar = HintBar()
    button = QPushButton("Run")
    bar.explain(button, "Start the pipeline.")
    _enter(bar, button)

    assert bar.text() == "<Start the pipeline.>"


# -- registration -------------------------------------------------------------

def test_a_control_with_nothing_to_say_is_not_watched(qapp):
    """An unregistered control must not blank the bar when crossed.

    A registration that succeeded with an empty sentence would make every
    pointer journey across a silent control wipe the line being read.
    """
    bar = HintBar()
    silent = QPushButton("Run")

    assert bar.explain(silent) == ""
    assert bar.count() == 0
    assert bar.explains(silent) == ""

    bar.setText("something the reader is reading")
    _enter(bar, silent)

    assert bar.text() == "something the reader is reading"


def test_a_whitespace_only_tooltip_is_nothing_to_say(qapp):
    bar = HintBar()
    button = QPushButton("Run")
    button.setToolTip("   \t ")

    assert bar.explain(button) == ""
    assert button.toolTip() == "   \t ", (
        "an unregistered control keeps whatever tooltip it had")


def test_the_sentence_moves_out_of_the_tooltip(qapp):
    """One sentence, one place: registering takes the tooltip away."""
    bar = HintBar()
    button = QPushButton("Run")
    button.setToolTip("  Start the pipeline.  ")

    assert bar.explain(button) == "Start the pipeline."
    assert button.toolTip() == ""
    assert button.accessibleDescription() == "Start the pipeline."
    assert bar.count() == 1
    assert bar.explains(button) == "Start the pipeline."


def test_an_accessible_description_the_author_wrote_is_kept(qapp):
    """The bar fills a gap in the accessibility tree; it does not overwrite."""
    bar = HintBar()
    button = QPushButton("Run")
    button.setAccessibleDescription("Runs the selected module.")

    bar.explain(button, "Start the pipeline.")

    assert button.accessibleDescription() == "Runs the selected module."


def test_explicit_text_wins_over_the_tooltip(qapp):
    bar = HintBar()
    button = QPushButton("Run")
    button.setToolTip("the old sentence")

    assert bar.explain(button, "the new sentence") == "the new sentence"
    assert bar.explains(button) == "the new sentence"


# -- hovering -----------------------------------------------------------------

def test_the_bar_says_what_the_pointer_is_on(qapp):
    bar = HintBar()
    button = QPushButton("Run")
    bar.explain(button, "Start the pipeline.")

    _enter(bar, button)

    assert bar.text() == "Start the pipeline."


def test_leaving_the_control_being_shown_restores_the_default(qapp):
    bar = HintBar()
    button = QPushButton("Run")
    bar.explain(button, "Start the pipeline.")
    _enter(bar, button)

    _leave(bar, button)

    assert bar.text() == DEFAULT_HINT


def test_a_hover_leave_restores_the_default_too(qapp):
    """``HoverLeave`` reaches the bar from styled controls; same rule."""
    bar = HintBar()
    button = QPushButton("Run")
    bar.explain(button, "Start the pipeline.")
    _enter(bar, button)

    bar.eventFilter(button, QEvent(QEvent.HoverLeave))

    assert bar.text() == DEFAULT_HINT


def test_leaving_a_neighbour_does_not_blank_the_line_on_screen(qapp):
    """Two adjacent controls send Leave-then-Enter, and the bar must not flicker.

    The pointer moves from the first control to the second; Qt delivers the
    second's Enter before the first's Leave often enough that an
    unconditional reset shows the default between every pair of neighbours.
    """
    bar = HintBar()
    first = QPushButton("Run")
    second = QPushButton("Stop")
    bar.explain(first, "Start the pipeline.")
    bar.explain(second, "Stop the pipeline.")

    _enter(bar, second)
    _leave(bar, first)

    assert bar.text() == "Stop the pipeline."


def test_leaving_a_control_nobody_registered_changes_nothing(qapp):
    bar = HintBar()
    stranger = QPushButton("Run")
    bar.setText("a sentence in progress")

    _leave(bar, stranger)

    assert bar.text() == "a sentence in progress"


def test_events_that_are_not_hovers_are_passed_through(qapp):
    """The filter watches for Enter and Leave and does not eat anything.

    A filter that swallowed the events it inspects would stop the control
    from ever repainting its own hover state.
    """
    bar = HintBar()
    button = QPushButton("Run")
    bar.explain(button, "Start the pipeline.")

    assert bar.eventFilter(button, QEvent(QEvent.Show)) is False
    assert bar.eventFilter(button, QEvent(QEvent.Enter)) is False
    assert bar.eventFilter(button, QEvent(QEvent.Leave)) is False


# -- finding the bar from deep in a form --------------------------------------

def test_a_nested_control_finds_the_bar_of_its_window(qapp):
    """The helper searches the top-level window, not the immediate parent."""
    window = QWidget()
    bar = HintBar(parent=window)
    inner = QWidget(window)
    field = QLabel("threshold", inner)

    assert hint_bar_of(field) is bar
    assert explain_through_the_bar(field, "How bright a pixel must be.")
    assert bar.explains(field) == "How bright a pixel must be."


def test_a_control_with_nothing_to_say_reports_no_registration(qapp):
    """``explain_through_the_bar`` is False when the bar refused it too."""
    window = QWidget()
    HintBar(parent=window)
    field = QLabel("threshold", window)

    assert explain_through_the_bar(field) is False


# -- the strip's own shape ----------------------------------------------------

def test_the_bar_is_bounded_at_three_lines(qapp):
    """A long sentence elides instead of resizing the whole dialog."""
    bar = HintBar()
    line = max(1, bar.fontMetrics().lineSpacing())

    assert bar.maximumHeight() == line * 3 + 12
    assert bar.minimumHeight() >= 28
    assert bar.wordWrap()
