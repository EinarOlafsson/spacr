"""The flow layout's item helpers answer for every kind of item and index."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QHBoxLayout, QLabel, QSpacerItem, QVBoxLayout, QWidget, QWidgetItem
from shiboken6 import Shiboken

from spacr.qt.widgets.flow import FlowLayout, _item_python_can_track, _take_item

pytestmark = pytest.mark.qt


def test_taking_from_past_the_end_holds_nothing(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    column = QVBoxLayout(host)
    assert _take_item(column, 0) == (None, None)
    column.addWidget(QLabel("only"))
    assert _take_item(column, 5) == (None, None)
    assert column.count() == 1


def test_taking_a_widget_returns_it_and_a_spacer_returns_nothing(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    column = QVBoxLayout(host)
    label = QLabel("kept")
    column.addWidget(label)
    column.addSpacing(8)
    assert _take_item(column, 1) == (None, None)
    widget, nested = _take_item(column, 0)
    assert widget is label and nested is None
    assert column.count() == 0


def test_taking_a_nested_layout_returns_it_undeleted(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    column = QVBoxLayout(host)
    row = QHBoxLayout()
    column.addLayout(row)
    widget, nested = _take_item(column, 0)
    assert widget is None and nested is row
    assert Shiboken.isValid(row)


def test_a_python_made_widget_item_is_kept_as_it_is(qtbot):
    label = QLabel("x")
    qtbot.addWidget(label)
    item = QWidgetItem(label)
    assert Shiboken.createdByPython(item)
    assert _item_python_can_track(item) is item


def test_a_spacer_is_kept_as_it_is():
    spacer = QSpacerItem(4, 4)
    assert _item_python_can_track(spacer) is spacer


def test_a_nested_layout_item_is_kept_as_it_is(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    outer = QVBoxLayout(host)
    inner = QHBoxLayout()
    outer.addLayout(inner)
    assert _item_python_can_track(inner) is inner


def test_a_widget_added_to_the_flow_layout_is_held_in_an_item_made_here(
        qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    flow = FlowLayout(host)
    chip = QLabel("chip")
    flow.addWidget(chip)
    held = flow.itemAt(0)
    assert held.widget() is chip
    assert Shiboken.createdByPython(held)
    widget, nested = _take_item(flow, 0)
    assert widget is chip and nested is None
    assert flow.count() == 0
