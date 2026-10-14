"""A native repaint remains safe when a handle's Python state is retired."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QWidget

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
