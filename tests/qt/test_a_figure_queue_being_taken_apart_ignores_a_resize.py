"""A resize of the view that reaches a queue with no debounce timer is passed on.

``test_a_figure_queue_without_a_view_ignores_events.py`` holds the half of the
guard CI caught: an event that arrived with no ``_view`` raised
``AttributeError`` inside the Qt event loop. The same guard reads the timer
the same way, because the view is built first and the timer after it, and a
teardown takes them apart in whatever order Qt frees them. A view resize that
lands in that window has nothing to debounce, and must neither raise nor
start anything.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def test_a_view_resize_with_no_timer_is_passed_on_not_raised(qtbot):
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QResizeEvent

    from spacr.qt.widgets.figure_queue import FigureQueue

    queue = FigureQueue()
    qtbot.addWidget(queue)
    timer = queue._resize_timer
    timer.stop()
    del queue._resize_timer
    try:
        handled = queue.eventFilter(
            queue._view, QResizeEvent(QSize(300, 300), QSize(100, 100)))
    finally:
        queue._resize_timer = timer

    assert handled is False
    assert not timer.isActive(), "a re-render was scheduled mid-teardown"
