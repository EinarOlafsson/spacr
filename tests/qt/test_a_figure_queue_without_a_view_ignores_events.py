"""The figure queue's event filter survives an event that arrives with no view.

CI on b740c741d caught `AttributeError: 'FigureQueue' object has no attribute
'_view'` raised from `FigureQueue.eventFilter` inside the Qt event loop, which
pytest-qt turned into a failure of an unrelated test. The filter is installed
on the view after the view exists, so the event reached a queue whose view was
not there -- mid-construction or mid-teardown. With no view there is nothing
to debounce, and the filter must say so instead of raising.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def test_an_event_with_no_view_is_passed_on_not_raised(qtbot):
    from PySide6.QtCore import QEvent, QObject, QSize
    from PySide6.QtGui import QResizeEvent

    from spacr.qt.widgets.figure_queue import FigureQueue

    queue = FigureQueue()
    qtbot.addWidget(queue)
    del queue._view

    other = QObject()
    resize = QResizeEvent(QSize(200, 200), QSize(100, 100))

    assert queue.eventFilter(other, resize) is False
    assert queue.eventFilter(other, QEvent(QEvent.Show)) is False


def test_the_view_resize_still_starts_the_debounce(qtbot):
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QResizeEvent

    from spacr.qt.widgets.figure_queue import FigureQueue

    queue = FigureQueue()
    qtbot.addWidget(queue)
    queue._resize_timer.stop()

    queue.eventFilter(queue._view, QResizeEvent(QSize(300, 300), QSize(100, 100)))

    assert queue._resize_timer.isActive()
