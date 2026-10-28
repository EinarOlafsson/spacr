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
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile

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


def test_the_real_home_page_receives_dock_hover_help_without_a_delay(qtbot, private_ai_profile):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QEnterEvent
    from spacr.qt.app import MainWindow

    preferences, _, _ = private_ai_profile
    previous_easy = preferences.get_easy_mode()
    window = None
    try:
        preferences.set_easy_mode(False)
        window = MainWindow()
        qtbot.addWidget(window)
        window.resize(1400, 1000)
        window.show()
        page = window._startup
        assert window._stack.currentWidget() is page
        assert page._hint_bar is not None
        window._dock_edge.set_collapsed(False)
        row = next(row for row in window._sidebar.rows() if row.key == 'measure')
        qtbot.waitUntil(lambda: row.isVisibleTo(window))
        announced = []
        window._sidebar.module_hovered.connect(announced.append)
        point = row.rect().center()
        event = QEnterEvent(QPointF(point), QPointF(row.mapTo(window, point)),
                            QPointF(row.mapToGlobal(point)))
        QApplication.sendEvent(row, event)
        assert announced == ['measure']
        assert page._hint_bar.module_key == 'measure'
        assert page._hint_bar.text() and page._hint_bar.isVisibleTo(page)
        assert row.is_hovered()
        assert window._stack.currentWidget() is page
    finally:
        try:
            if window is not None:
                window.close()
        finally:
            preferences.set_easy_mode(previous_easy)


def test_a_section_context_event_reaches_its_normal_qt_receiver_without_collapsing(dock, qtbot):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtTest import QSignalSpy

    dock.show()
    qtbot.waitExposed(dock)
    header = dock._headers['Core']
    header.setContextMenuPolicy(Qt.CustomContextMenu)
    requests = QSignalSpy(header.customContextMenuRequested)
    point = QPoint(8, 8)
    before = dock.section_is_open('Core')
    event = QContextMenuEvent(QContextMenuEvent.Mouse, point, header.mapToGlobal(point))
    QApplication.sendEvent(header, event)
    assert requests.count() == 1 and requests.at(0) == [point]
    assert dock.section_is_open('Core') == before
    assert all(row.isVisibleTo(dock) == before for row in dock.rows() if row.key != '__home__')
