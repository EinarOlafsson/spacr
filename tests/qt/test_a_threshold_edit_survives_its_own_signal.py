"""A gate threshold field outlives the editingFinished that rebuilds the rows.

QLineEdit emits ``editingFinished`` from inside its own focus-out and key
handlers, and goes on using itself after the signal returns. The gates
panel answers that signal by rebuilding the threshold rows; when the rebuild
deleted the old line edits there and then, Qt went on to paint a freed
widget. That was the intermittent segmentation fault in the Gate Editor's
teardown: closing the panel moved focus out of a threshold field, the field
finished editing, and ``QWidget::focusOutEvent`` called ``update()`` on the
deleted field.
"""
from __future__ import annotations

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QVBoxLayout, QWidget

from spacr.qt.widgets.gate_editor import GateTree
from spacr.qt.widgets.gate_spec import CylinderGate, GateSet

pytestmark = pytest.mark.qt


def _tree_with_a_cylinder(qtbot, parent=None):
    tree = GateTree(parent)
    if parent is None:
        qtbot.addWidget(tree)
    gates = GateSet()
    gates.add(CylinderGate(name="c", u_column="a", v_column="b",
                           axis_column="z", u_radius=1.0, v_radius=1.0))
    tree.set_gates(gates, pd.DataFrame({"a": [0.0, 1.0, 2.0],
                                        "b": [0.0, 1.0, 2.0],
                                        "z": [0.0, 5.0, 9.0]}))
    tree._rebuild_thresholds("c")
    return tree, gates


def test_the_field_that_finished_editing_is_not_deleted_inside_its_signal(qtbot):
    tree, gates = _tree_with_a_cylinder(qtbot)
    low, high = tree._threshold_rows["z"]
    gone = []
    low.destroyed.connect(lambda *_: gone.append("low"))
    low.setText("1")
    high.setText("6")

    low.editingFinished.emit()

    assert not gone, ("the line edit was deleted while its own "
                      "editingFinished was still being delivered")
    assert gates.get("c").thresholds() == {"z": (1.0, 6.0)}
    new_low, new_high = tree._threshold_rows["z"]
    assert (new_low.text(), new_high.text()) == ("1", "6")
    qtbot.waitUntil(lambda: bool(gone), timeout=2000)


def test_closing_the_panel_from_a_typed_field_applies_it_without_a_crash(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    elsewhere = QLineEdit(host)
    layout.addWidget(elsewhere)
    tree, gates = _tree_with_a_cylinder(qtbot, host)
    layout.addWidget(tree)
    host.show()
    qtbot.waitExposed(host)
    host.activateWindow()
    qtbot.waitActive(host)
    low, high = tree._threshold_rows["z"]
    high.setText("6")
    low.setFocus(Qt.OtherFocusReason)
    qtbot.waitUntil(low.hasFocus, timeout=2000)
    QTest.keyClicks(low, "2")

    tree.hide()
    QApplication.processEvents()

    assert gates.get("c").thresholds() == {"z": (2.0, 6.0)}
    shown = tuple(edit.text() for edit in tree._threshold_rows["z"])
    assert shown == ("2", "6")
    assert elsewhere.hasFocus()
