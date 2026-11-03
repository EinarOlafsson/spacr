"""Flow layout walks, a layout deleted mid-pass and the wrapped-height filter."""
from __future__ import annotations

import pytest
pytest.importorskip("PySide6")
import shiboken6
from PySide6.QtCore import QEvent, QObject, QRect
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from spacr.qt.widgets import flow
from spacr.qt.widgets.flow import FlowLayout, _WrappedHeight, keep_wrapped_height

pytestmark = pytest.mark.qt


def test_layout_walks_skip_non_layout_children_and_layouts_that_hold_no_item(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    column = QVBoxLayout(host)
    label = QLabel("kept")
    column.addWidget(label)
    QObject(column)                      # a child that is not a layout
    loose = QHBoxLayout()
    loose.setParent(column)              # a child layout that is no item of column
    stray = QLabel("inside loose")
    loose.addWidget(stray)
    assert column.indexOf(loose) == -1
    assert [entry for _i, entry in flow._layout_entries(column)] == [label]
    assert flow._layout_holding(column, label) is column
    assert flow._layout_holding(column, stray) is loose
    assert flow._layout_holding(column, QLabel("elsewhere")) is None


def test_a_flow_layout_destroyed_under_a_pass_lays_out_nothing(qtbot):
    host = QWidget()
    layout = FlowLayout(host)
    layout.addWidget(QLabel("chip", host))
    shiboken6.delete(host)
    assert not shiboken6.isValid(layout)
    assert layout._do_layout(QRect(0, 0, 200, 100), True) == 0


def test_wrapped_height_filter_survives_a_label_deleted_under_it_and_installs_once(qtbot):
    class Gone:
        def heightForWidth(self, _w):
            raise RuntimeError("Internal C++ object already deleted.")

        def width(self):
            return 10

    event = QEvent(QEvent.Type.Resize)
    assert _WrappedHeight().eventFilter(Gone(), event) is False
    label = QLabel("word " * 40)
    label.setWordWrap(True)
    qtbot.addWidget(label)
    keep_wrapped_height(label)
    first = label._spacr_wrapped_height
    keep_wrapped_height(label)
    assert label._spacr_wrapped_height is first
