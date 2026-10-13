"""The toggle survives an empty caption and ignores buttons that are not left.

Both branches guard the same thing: a secondary control must never become the
reason something else misbehaves. Eliding a caption that is not there would
measure an empty string against a real width on every resize, and flipping the
switch on a right-click would toggle AI out from under a user who was reaching
for a context menu.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402

from spacr.qt.widgets.ai_toggle_label import AiToggleLabel  # noqa: E402

pytestmark = pytest.mark.qt


def _press(button):
    return QMouseEvent(QEvent.MouseButtonPress, QPointF(4.0, 4.0),
                       QPointF(4.0, 4.0), button, button, Qt.NoModifier)


def test_a_toggle_with_no_caption_still_resizes_without_eliding_nothing(qapp):
    """An empty caption must leave the label empty, not painted with an ellipsis.

    Callers blank the caption to hide a toggle's text while keeping its slot in
    the action row. Running the elision on ``""`` would measure a width for a
    string that is not there, and the "keep at least one character" rule below
    it would then put something back.
    """
    toggle = AiToggleLabel(text="Hyperparameter search")
    toggle.show()
    qapp.processEvents()

    toggle.setText("")

    assert toggle.text() == ""
    assert toggle.displayed_text() == ""

    toggle.resize(24, 20)
    qapp.processEvents()
    assert toggle.displayed_text() == ""


def test_a_long_caption_is_still_elided_when_it_does_not_fit(qapp):
    """The empty-caption guard must not disarm eliding for real text.

    ``Hyperparameter search`` is the caption that held the action row wide
    enough to starve the settings column, so it has to keep shrinking.
    """
    caption = "Hyperparameter search"
    toggle = AiToggleLabel(text=caption)
    # Half the caption's own painted width, so the assertion does not depend
    # on which font the application theme happens to have installed.
    chrome = toggle.width() - toggle.contentsRect().width()
    toggle.resize(toggle.fontMetrics().horizontalAdvance(caption) // 2 + chrome,
                  20)
    toggle.setText(caption)

    shown = toggle.displayed_text()
    assert shown != ""
    assert shown != toggle.text()
    assert toggle.text() == caption


def test_a_right_click_does_not_flip_the_switch(qapp):
    """Only the left button toggles; anything else goes back to Qt.

    A right-press is how a user reaches a context menu. Treating it as a
    toggle would silently route their next console message through a paid
    provider, or silently stop routing it.
    """
    toggle = AiToggleLabel()
    toggle.setChecked(True)
    emitted = []
    toggle.toggled.connect(emitted.append)

    toggle.mousePressEvent(_press(Qt.RightButton))

    assert toggle.isChecked() is True
    assert emitted == []


def test_a_left_click_flips_the_switch_and_announces_it(qapp):
    """The other half of the same branch, so the guard cannot swallow both."""
    toggle = AiToggleLabel()
    emitted = []
    toggle.toggled.connect(emitted.append)

    toggle.mousePressEvent(_press(Qt.LeftButton))

    assert toggle.isChecked() is True
    assert emitted == [True]
