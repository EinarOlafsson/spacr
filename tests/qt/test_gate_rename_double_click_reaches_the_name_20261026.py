"""A double-click on a gate's row in the narrow Gate table opens Rename gate.

Lesson 64 double-clicks the new gate 60 px into its row. In the real app on
gnome-shell the Gate table viewport is about 256 px wide; the Gate column was
stretched into what the three count columns left, 65 px including the
indentation and tick box, so the name read as an ellipsis and the
double-click landed on the "n" column, which asks nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QInputDialog  # noqa: E402

from spacr.qt.widgets.gate_editor import GateTree  # noqa: E402
from spacr.qt.widgets.gate_spec import GateSet, PolygonGate  # noqa: E402


def _tree(qtbot):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    gates = GateSet()
    gates.add(PolygonGate(name="tutorial_polygon", x_column="a", y_column="b",
                          vertices=((-1, -1), (1, -1), (1, 1), (-1, 1))))
    tree = GateTree()
    qtbot.addWidget(tree)
    tree.resize(258, 400)
    tree.show()
    qtbot.waitExposed(tree)
    tree.set_gates(gates, frame)
    return tree


def test_the_gate_column_shows_the_whole_name(qtbot):
    tree = _tree(qtbot)
    view = tree.tree
    item = view.topLevelItem(0)
    rect = view.visualRect(view.indexFromItem(item, 0))
    needed = view.fontMetrics().horizontalAdvance("tutorial_polygon")
    assert rect.width() >= needed


def test_double_clicking_the_row_where_the_recorder_aims_opens_rename(qtbot, monkeypatch):
    tree = _tree(qtbot)
    asked = []
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: asked.append(k.get("text")) or ("dense_core", True)))
    view = tree.tree
    item = view.topLevelItem(0)
    rect = view.visualItemRect(item)
    point = QPoint(max(4, min(rect.left() + 60, view.viewport().width() - 8)), rect.center().y())
    assert view.indexAt(point).column() == 0
    QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=point)
    QTest.mouseDClick(view.viewport(), Qt.LeftButton, pos=point)
    assert asked == ["tutorial_polygon"]
    assert "dense_core" in tree._gates.names
