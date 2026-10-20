"""The help strip's resize grip, its height bounds and its lookups.

Pinned here, each as what the user sees or gets:

* only a left press on the grip starts a drag; a right press, a move with no
  drag under way, and a release of another button leave the strip's height
  alone and store nothing; the press and release reach the frame, which
  leaves them unaccepted for the parent;
* a strip with no page above it, or one that is not on screen, cannot grow
  past its four-line minimum, while a shown strip under a tall page can;
* a strip measured before its text view exists still reserves four rows of
  its own font, and re-syncing its font then is harmless;
* a widget with no window has no bar, and a widget whose window has no bar
  keeps its tooltip and reports that it was not registered.
"""
from __future__ import annotations

import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QTextDocument
from PySide6.QtWidgets import (QApplication, QDialog, QPushButton, QTextEdit,
                               QVBoxLayout, QWidget)

from spacr.qt.widgets.hint_bar import (HintBar, explain_through_the_bar,
                                       hint_bar_of)

pytestmark = pytest.mark.qt


def _mouse(kind, y, button, buttons=None):
    """A real mouse event at global height ``y``."""
    held = button if buttons is None else buttons
    return QMouseEvent(kind, QPointF(2, 2), QPointF(2, y), button, held,
                       Qt.NoModifier)


@pytest.fixture
def bar(qtbot):
    strip = HintBar("Hover something.")
    qtbot.addWidget(strip)
    return strip


def test_a_right_press_on_the_grip_does_not_start_a_drag(bar):
    handle = bar._resize_handle
    height = bar.height()
    event = _mouse(QEvent.Type.MouseButtonPress, 300, Qt.RightButton)
    QApplication.sendEvent(handle, event)
    assert handle._drag is None
    assert not event.isAccepted()
    assert bar.height() == height


def test_a_move_with_no_drag_under_way_leaves_the_height(bar):
    handle = bar._resize_handle
    height = bar.height()
    committed = []
    bar.helpHeightCommitted.connect(committed.append)
    event = _mouse(QEvent.Type.MouseMove, 10, Qt.NoButton, Qt.LeftButton)
    QApplication.sendEvent(handle, event)
    assert handle._drag is None
    assert bar.height() == height
    assert bar._manual_height is None
    assert committed == []


def test_releasing_another_button_mid_drag_does_not_commit(bar):
    handle = bar._resize_handle
    committed = []
    bar.helpHeightCommitted.connect(committed.append)
    press = _mouse(QEvent.Type.MouseButtonPress, 300, Qt.LeftButton)
    QApplication.sendEvent(handle, press)
    assert press.isAccepted()
    assert handle._drag == (300.0, bar.height())

    other = _mouse(QEvent.Type.MouseButtonRelease, 300, Qt.RightButton,
                   Qt.LeftButton)
    QApplication.sendEvent(handle, other)
    assert not other.isAccepted()
    assert handle._drag is not None
    assert committed == []

    left = _mouse(QEvent.Type.MouseButtonRelease, 300, Qt.LeftButton,
                  Qt.NoButton)
    QApplication.sendEvent(handle, left)
    assert left.isAccepted()
    assert handle._drag is None
    assert committed == [bar.height()]


def test_a_release_with_no_drag_commits_nothing(bar):
    handle = bar._resize_handle
    committed = []
    bar.helpHeightCommitted.connect(committed.append)
    event = _mouse(QEvent.Type.MouseButtonRelease, 300, Qt.LeftButton,
                   Qt.NoButton)
    QApplication.sendEvent(handle, event)
    assert not event.isAccepted()
    assert committed == []


def test_a_strip_alone_cannot_grow_past_its_minimum(bar):
    minimum = bar._minimum_help_height()
    assert bar._maximum_help_height() == minimum
    bar._set_manual_height(minimum + 400)
    assert bar.height() == minimum
    assert bar._manual_height == minimum


def test_a_hidden_strip_under_a_tall_page_stays_at_its_minimum(qtbot):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    layout = QVBoxLayout(dialog)
    page = QWidget()
    page.setMinimumHeight(40)
    layout.addWidget(page)
    strip = HintBar("Hover something.")
    layout.addWidget(strip)
    dialog.resize(400, 900)
    assert not strip.isVisible()
    assert strip._maximum_help_height() == strip._minimum_help_height()

    dialog.show()
    qtbot.waitExposed(dialog)
    assert strip._maximum_help_height() > strip._minimum_help_height()


def test_four_rows_are_measured_in_the_bar_font_before_the_view_exists(bar):
    view = bar._view
    bar._view = None
    try:
        rows = bar._four_painted_rows()
        document = QTextDocument()
        document.setDocumentMargin(0)
        document.setDefaultFont(bar.font())
        document.setPlainText("Xg\nXg\nXg\nXg")
        assert rows == math.ceil(document.size().height())
        assert bar._minimum_help_height() >= rows
    finally:
        bar._view = view


def test_syncing_the_font_without_a_view_changes_nothing(bar):
    view = bar._view
    sheet = view.styleSheet()
    bar._view = None
    try:
        bar._sync_help_font()
    finally:
        bar._view = view
    assert isinstance(bar._view, QTextEdit)
    assert bar._view.styleSheet() == sheet


def test_no_widget_means_no_bar():
    assert hint_bar_of(None) is None


def test_a_window_without_a_bar_leaves_the_tooltip_in_place(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    button = QPushButton("Run", window)
    button.setToolTip("Start the run.")
    assert hint_bar_of(button) is None
    assert explain_through_the_bar(button) is False
    assert button.toolTip() == "Start the run."
    assert button.accessibleDescription() == ""
