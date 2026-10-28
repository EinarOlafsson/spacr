"""Dock collapse activates once per ordinary keyboard gesture."""
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QWidget
from spacr.qt.widgets.dock import Dock, DockEdge


@pytest.fixture
def edge(qtbot, monkeypatch, activate_focus_window):
    from spacr.qt.widgets import ambient
    monkeypatch.setattr(ambient, "field_ripple_for_widget", lambda *args, **kwargs: None)
    host = QWidget()
    qtbot.addWidget(host)
    dock = Dock([], parent=host)
    control = DockEdge(dock, host)
    control.show()
    host.show()
    activate_focus_window(host)
    control.setFocus()
    QApplication.processEvents()
    assert control.hasFocus()
    yield control


@pytest.mark.parametrize("key,modifiers", [
    (Qt.Key_Space, Qt.NoModifier),
    (Qt.Key_Return, Qt.NoModifier),
    (Qt.Key_Enter, Qt.NoModifier),
    (Qt.Key_Enter, Qt.KeypadModifier),
])
def test_held_activation_changes_collapse_only_once(edge, key, modifiers):
    changes = []
    edge.collapsedChanged.connect(changes.append)
    QApplication.sendEvent(edge, QKeyEvent(QEvent.KeyPress, key, modifiers))
    assert edge.is_collapsed()
    for _ in range(3):
        QApplication.sendEvent(edge, QKeyEvent(
            QEvent.KeyPress, key, modifiers, "", True, 1))
        assert edge.is_collapsed()
    assert changes == [True]
    QApplication.sendEvent(edge, QKeyEvent(QEvent.KeyRelease, key, modifiers))
    QApplication.sendEvent(edge, QKeyEvent(QEvent.KeyPress, key, modifiers))
    assert not edge.is_collapsed()
    assert changes == [True, False]


@pytest.mark.parametrize("key", [Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter])
@pytest.mark.parametrize("modifiers", [
    Qt.ControlModifier, Qt.AltModifier, Qt.ShiftModifier, Qt.MetaModifier,
])
def test_modified_activation_does_not_collapse(edge, key, modifiers):
    changes = []
    edge.collapsedChanged.connect(changes.append)
    QApplication.sendEvent(edge, QKeyEvent(QEvent.KeyPress, key, modifiers))
    assert not edge.is_collapsed()
    assert changes == []
