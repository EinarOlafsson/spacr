"""Text fits the box it is drawn in (instruction 350).

THE MEASUREMENT, NOT ONE FIXED DIALOG. The maintainer reported that
Annotate's "Load test data" tooltips do not fit their popup, and asked for a
sweep of the whole package. A sweep needs a check that is known to catch a
real failure before it is trusted anywhere else, so it is written here against
the reported case first.

WHAT WENT WRONG THERE is worth keeping, because it is one of the four causes
350 lists and it looks like good practice: the pane was given
``setMinimumHeight(110)`` with a comment explaining -- correctly -- that a
pane which resizes as the text changes would move the buttons out from under
the pointer hovering them. The intent was right. 110 was a guess, both route
descriptions grew to two paragraphs, and the pane could not grow because not
growing was the point. Measured before the fix: 170 px of text in a 110 px
pane.
"""
from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel

import pytest


def painted_text(widget) -> str:
    """What a widget is actually PAINTING, which is not always its text.

    A control that elides on purpose reports its full logical caption from
    `text()` and paints less. `AiToggleLabel` is the one in spaCR today: it
    caps `minimumSizeHint` at ELIDE_ABOVE_PX so a long secondary caption
    cannot force the whole row wider, and offers `displayed_text()` for
    exactly this question -- its docstring says tests need both "to tell 'the
    toggle says X' from 'the toggle currently fits this much of X'".

    COMPARING `text()` AGAINST THE WIDTH OF SUCH A WIDGET ALWAYS REPORTS
    CLIPPING, by construction. On 2026-09-02 that produced five confident
    false positives across four screens and three locales -- the entire
    result of a sweep -- before anyone noticed the control was doing what it
    was built to do. Ask for the painted text first.
    """
    getter = getattr(widget, "displayed_text", None)
    if callable(getter):
        return getter()
    return widget.text()


def wrapped_height(widget: QLabel, text: str, width: int | None = None) -> int:
    """How tall ``text`` needs to be in ``widget``'s font at ``width``.

    The reusable half of this file. Anything asking "is this string clipped"
    compares this against the widget's own height.
    """
    metrics = QFontMetrics(widget.font())
    box = metrics.boundingRect(
        QRect(0, 0, width if width is not None else widget.width(), 0),
        int(Qt.TextWordWrap | Qt.AlignTop | Qt.AlignLeft),
        text)
    return box.height()


@pytest.fixture
def chooser(qtbot, qt_theme_applied):
    from spacr.qt.widgets.test_data_chooser import TestDataChooser

    dialog = TestDataChooser()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    return dialog


def test_every_route_description_fits_its_pane(chooser):
    """Not just the longest one today -- every string the pane can show."""
    pane = chooser._description
    for text in chooser.every_description():
        needed = wrapped_height(pane, text)
        assert pane.height() >= needed, (
            f"clipped by {needed - pane.height()} px: {text[:60]!r}…")


def test_the_pane_does_not_move_when_the_description_changes(chooser):
    """The reason the original fixed height existed, kept.

    A pane that grew to fit each description as it arrived would move the
    buttons out from under the pointer that is hovering them. Every
    description must therefore produce the SAME height, not merely fit.
    """
    pane = chooser._description
    heights = set()
    for text in chooser.every_description():
        pane.setText(text)
        heights.add(pane.height())
    assert len(heights) == 1, f"the pane resized as the text changed: {heights}"


def test_a_longer_description_would_be_caught(chooser, monkeypatch):
    """The check fails when the text really does not fit.

    Without this the two tests above would pass just as happily against a
    pane sized by luck, and instruction 288 records four tests that passed
    while exercising nothing. Here the failure is manufactured and the
    measurement is asserted to notice it.
    """
    pane = chooser._description
    enormous = ("A sentence that keeps going. " * 200)
    assert wrapped_height(pane, enormous) > pane.height()
