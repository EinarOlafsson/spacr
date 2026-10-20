"""A native repaint remains safe when a handle's Python state is retired."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt.widgets import collapsible_splitter as cs


@pytest.mark.parametrize("orientation", [Qt.Horizontal, Qt.Vertical])
def test_a_handle_without_python_state_keeps_its_idle_pixels(qtbot, orientation):
    """Qt may repaint a live native handle after retiring its wrapper state."""
    split = cs.CollapsibleSplitter(orientation)
    qtbot.addWidget(split)
    split.add_pane(QWidget(), "Settings", mode=cs.EDGE)
    split.add_pane(QWidget(), "Preview")
    split.resize(400, 200)
    split.show()
    qtbot.waitExposed(split)
    handle = split.handle(1)
    assert not handle.underMouse()
    expected = QImage(handle.size(), QImage.Format_ARGB32_Premultiplied)
    expected.fill(Qt.transparent)
    handle.render(expected)
    original_state = dict(handle.__dict__)
    actual = QImage(handle.size(), QImage.Format_ARGB32_Premultiplied)
    actual.fill(Qt.transparent)
    problems = []
    try:
        handle.__dict__.clear()
        with qtbot.capture_exceptions() as raised:
            try:
                handle.render(actual)
            except AttributeError as exc:
                problems.append(str(exc))
        problems.extend(str(value) for _kind, value, _tb in raised)
        raised.clear()
    finally:
        handle.__dict__.update(original_state)
    assert not problems, problems
    assert actual == expected


@pytest.mark.parametrize("timer_state", ["missing", "deleted"])
def test_hiding_a_splitter_with_retired_timer_state_is_safe(qtbot, timer_state):
    """A real native Hide must not call an absent or destroyed Qt timer."""
    from shiboken6 import delete

    split = cs.CollapsibleSplitter(Qt.Horizontal)
    qtbot.addWidget(split)
    timer = split._extents_timer
    problems = []
    try:
        if timer_state == "missing":
            del split._extents_timer
        else:
            delete(timer)
        with qtbot.capture_exceptions() as raised:
            QApplication.sendEvent(split, QEvent(QEvent.Hide))
        problems.extend(str(value) for _kind, value, _tb in raised)
        raised.clear()
    finally:
        if timer_state == "missing":
            split._extents_timer = timer
    assert not problems, problems
