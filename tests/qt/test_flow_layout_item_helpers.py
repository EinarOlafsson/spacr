"""The flow layout's item helpers answer for every kind of item and index."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QHBoxLayout, QLabel, QSpacerItem, QVBoxLayout,
                               QWidget, QWidgetItem)
from shiboken6 import Shiboken

from spacr.qt.widgets.flow import FlowLayout, _item_python_can_track, _take_item

pytestmark = pytest.mark.qt


def test_taking_and_tracking_layout_items(qtbot, subtests):
    """_take_item returns (widget, nested layout) or (None, None); _item_python_can_track keeps items as they are."""
    host = QWidget()
    qtbot.addWidget(host)
    column = QVBoxLayout(host)

    with subtests.test(msg="an empty layout and an index past the end hold nothing"):
        assert _take_item(column, 0) == (None, None)
        column.addWidget(QLabel("only"))
        assert _take_item(column, 5) == (None, None)
        assert column.count() == 1
        _take_item(column, 0)

    with subtests.test(msg="a spacer gives nothing back and a widget is returned"):
        label = QLabel("kept")
        column.addWidget(label)
        column.addSpacing(8)
        assert _take_item(column, 1) == (None, None)
        widget, nested = _take_item(column, 0)
        assert widget is label and nested is None
        assert column.count() == 0

    with subtests.test(msg="a nested layout is returned undeleted"):
        row = QHBoxLayout()
        column.addLayout(row)
        assert _item_python_can_track(row) is row
        widget, nested = _take_item(column, 0)
        assert widget is None and nested is row
        assert Shiboken.isValid(row)

    with subtests.test(msg="python-made widget items and spacers are kept as they are"):
        label = QLabel("x")
        qtbot.addWidget(label)
        item = QWidgetItem(label)
        assert Shiboken.createdByPython(item)
        assert _item_python_can_track(item) is item
        spacer = QSpacerItem(4, 4)
        assert _item_python_can_track(spacer) is spacer

    with subtests.test(msg="a widget added to a FlowLayout sits in a python-made item"):
        other = QWidget()
        qtbot.addWidget(other)
        flow = FlowLayout(other)
        chip = QLabel("chip")
        flow.addWidget(chip)
        held = flow.itemAt(0)
        assert held.widget() is chip
        assert Shiboken.createdByPython(held)
        assert _take_item(flow, 0) == (chip, None)
        assert flow.count() == 0
