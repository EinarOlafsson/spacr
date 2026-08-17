"""The figure queue's clear control: plain text that flickers accent.

Instruction 111. Three things are asserted here rather than eyeballed,
because all three have been got wrong before in this repository:

* the colours come from the palette, not from a literal, so a theme switch
  carries them (a hex typed into a widget is a hex that stays dark on light);
* the flash ENDS -- a flicker that latches is a highlight, which is a
  different control;
* the timing is the shared one, so this and the console's copy glyph cannot
  drift into two different durations.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from PySide6.QtCore import QEvent, QEventLoop, QPoint, Qt, QTimer
from PySide6.QtGui import QMouseEvent

pytestmark = pytest.mark.qt


def _colour(label) -> str:
    """The colour the stylesheet is currently painting."""
    return label.styleSheet().split("color: ")[1].split(";")[0].strip()


def _settle(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


@pytest.fixture
def queue(qtbot):
    from spacr.qt.widgets.figure_queue import FigureQueue
    widget = FigureQueue()
    qtbot.addWidget(widget)
    return widget


def test_the_control_is_text_not_a_button(queue):
    """A QPushButton would give a rare destructive action the same weight as
    the controls beside it that are used constantly."""
    from PySide6.QtWidgets import QLabel, QPushButton

    label = queue._clear_label
    assert isinstance(label, QLabel)
    assert not isinstance(label, QPushButton)
    # Discoverable as clickable without a border saying so.
    assert label.cursor().shape() == Qt.PointingHandCursor
    # Reachable without a mouse.
    assert label.focusPolicy() == Qt.StrongFocus


def test_clicking_it_empties_the_queue(queue):
    for _ in range(3):
        figure = plt.figure()
        figure.add_subplot(111).plot([0, 1], [1, 0])
        queue.add_figure(figure)
    assert queue._list.count() == 3

    queue._clear_label.clicked.emit()
    assert queue._list.count() == 0


def test_it_flickers_the_accent_and_returns(queue):
    """Flicker, not highlight: it must come back down on its own."""
    from spacr.qt.theme import active_palette
    from spacr.qt.widgets.flash import FLASH_MS

    label = queue._clear_label
    palette = active_palette()
    resting = _colour(label)
    # RESTS AT `error`, NOT `fg_dim`. Changed deliberately on 2026-08-17 at
    # the maintainer's request -- "just make it red like other negative
    # butons" -- because clearing the figures cannot be undone. The flash
    # below is unchanged: the accent is the app-wide "your click landed"
    # mark, shared with the console's copy glyph.
    assert resting.lower() == palette["error"].lower()

    label.flash()
    assert label._flash.active
    assert _colour(label).lower() == palette["accent"].lower()

    _settle(FLASH_MS + 150)
    assert not label._flash.active
    assert _colour(label).lower() == resting.lower()


def test_a_click_on_an_empty_queue_still_flickers(queue):
    """The mark says the click was received, not that there was work --
    clearing an empty queue is otherwise indistinguishable from a dead
    control."""
    assert queue._list.count() == 0
    queue._clear_label.flash()
    assert queue._clear_label._flash.active


def test_keyboard_activation_works(queue):
    """Enter and Space activate it, or it is a control some users cannot
    reach."""
    from PySide6.QtGui import QKeyEvent

    fired = []
    queue._clear_label.clicked.connect(lambda: fired.append(1))
    for key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
        queue._clear_label.keyPressEvent(
            QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))
    assert len(fired) == 3


def test_dragging_off_the_label_cancels(queue):
    """Release outside the rect is a cancelled click, as everywhere else."""
    fired = []
    queue._clear_label.clicked.connect(lambda: fired.append(1))
    outside = QPoint(-50, -50)
    queue._clear_label.mouseReleaseEvent(
        QMouseEvent(QEvent.MouseButtonRelease, outside, Qt.LeftButton,
                    Qt.LeftButton, Qt.NoModifier))
    assert fired == []


def test_the_flash_timing_is_shared_with_the_copy_glyph(queue):
    """One duration, or the two controls read as a bug in whichever you
    notice second."""
    from spacr.qt.widgets.console_panel import _CopyGlyphButton
    from spacr.qt.widgets.flash import Flash

    glyph = _CopyGlyphButton()
    assert isinstance(glyph._flash, Flash)
    assert isinstance(queue._clear_label._flash, Flash)
    assert glyph._flash._duration == queue._clear_label._flash._duration
