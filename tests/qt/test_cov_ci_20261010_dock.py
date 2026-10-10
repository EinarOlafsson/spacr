"""Two quiet paths of the dock: leaving an unknown row, reopening mid-drag.

Pinned here:

* a hover leave for a key the dock has no row for clears the lit row and
  announces nothing;
* dragging a collapsed dock's edge reopens it, and because a drag is under
  way the reopening does not schedule the field ripple a click would.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from spacr.qt import preferences as P
from spacr.qt.widgets import dock as dock_module
from spacr.qt.widgets.dock import Dock, DockEdge

pytestmark = pytest.mark.qt

ROWS = [
    ("__home__", "Home", "Back to the tiles", "Core"),
    ("mask", "Generate Masks", "Segment objects", "Core"),
]


@pytest.fixture
def dock(qtbot, monkeypatch):
    monkeypatch.setattr(P, "get_dock_width", lambda: 0)
    monkeypatch.setattr(P, "set_dock_width", lambda width: None)
    bar = Dock(ROWS, icon_for=lambda key: None)
    qtbot.addWidget(bar)
    return bar


def test_leaving_a_key_with_no_row_announces_nothing(dock, monkeypatch):
    announced = []
    dock.module_hovered.connect(announced.append)
    lit = []
    monkeypatch.setattr(dock, "_light_only", lit.append)
    dock._on_row_hovered("mask", True)
    cancelled = []
    monkeypatch.setattr(dock._hover_help_delay, "cancel_for",
                        cancelled.append)
    dock._on_row_hovered("no-such-module", False)
    assert lit == ["mask", None]
    assert cancelled == []
    assert "no-such-module" not in announced


def test_dragging_a_collapsed_edge_reopens_it_without_a_ripple(
        qtbot, dock, monkeypatch):
    edge = DockEdge(dock)
    qtbot.addWidget(edge)
    edge.set_collapsed(True)
    scheduled = []
    monkeypatch.setattr(dock_module.QTimer, "singleShot",
                        lambda *args: scheduled.append(args))
    changes = []
    edge.collapsedChanged.connect(changes.append)

    press = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(2, 2),
                        QPointF(100, 2), Qt.LeftButton, Qt.LeftButton,
                        Qt.NoModifier)
    QApplication.sendEvent(edge, press)
    move = QMouseEvent(QEvent.Type.MouseMove, QPointF(40, 2),
                       QPointF(140, 2), Qt.NoButton, Qt.LeftButton,
                       Qt.NoModifier)
    QApplication.sendEvent(edge, move)

    assert edge._pressed_x == 100.0
    assert edge._dragged is True
    assert edge.is_collapsed() is False
    assert changes == [False]
    assert scheduled == []
