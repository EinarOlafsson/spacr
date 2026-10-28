"""Assistive clients can inspect and operate the dock's collapse control."""

import json

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QAccessible, QAccessibleActionInterface
from PySide6.QtWidgets import QWidget

from spacr.qt.widgets.dock import Dock, DockEdge


@pytest.fixture
def edge(qtbot, monkeypatch):
    from spacr.qt.widgets import ambient

    monkeypatch.setattr(ambient, "field_ripple_for_widget", lambda *args, **kwargs: None)
    host = QWidget()
    qtbot.addWidget(host)
    dock = Dock([], parent=host)
    control = DockEdge(dock, host)
    control.show()
    host.show()
    yield control


def test_primary_role_and_collapse_state_describe_the_control(edge):
    interface = QAccessible.queryAccessibleInterface(edge)
    state = interface.state()
    print(json.dumps({"role": interface.role().name,
                      "actions": interface.actionInterface().actionNames(),
                      "expandable": bool(state.expandable),
                      "expanded": bool(state.expanded),
                      "collapsed": bool(state.collapsed)}))
    assert interface.role() == QAccessible.Button
    assert state.expandable and state.expanded and not state.collapsed
    edge.set_collapsed(True)
    assert interface.state().collapsed and not interface.state().expanded
    edge.set_collapsed(False)
    assert interface.state().expanded and not interface.state().collapsed


def test_accessible_press_uses_existing_collapse_signal_without_resizing(edge):
    interface = QAccessible.queryAccessibleInterface(edge)
    action = interface.actionInterface()
    assert action is not None
    assert QAccessibleActionInterface.pressAction() in action.actionNames()
    width = edge._dock.width()
    changed = []
    edge.collapsedChanged.connect(changed.append)
    action.doAction(QAccessibleActionInterface.pressAction())
    assert edge.is_collapsed() and interface.state().collapsed
    action.doAction(QAccessibleActionInterface.pressAction())
    assert not edge.is_collapsed() and interface.state().expanded
    assert changed == [True, False]
    assert edge._dock.width() == width


def test_disabled_accessible_control_cannot_collapse(edge):
    interface = QAccessible.queryAccessibleInterface(edge)
    action = interface.actionInterface()
    edge.setEnabled(False)
    changed = []
    edge.collapsedChanged.connect(changed.append)
    assert interface.state().disabled
    assert action.actionNames() == []
    action.doAction(QAccessibleActionInterface.pressAction())
    assert not edge.is_collapsed() and changed == []


@pytest.mark.parametrize("route", ["setter", "mouse", "keyboard"])
def test_real_collapse_changes_notify_assistive_clients(
        edge, route, monkeypatch, qtbot, activate_focus_window):
    events = []
    original = QAccessible.updateAccessibility

    def record(event):
        if event.type() == QAccessible.StateChanged and event.object() is edge:
            changes = event.changedStates()
            events.append((bool(changes.expanded), bool(changes.collapsed),
                           edge.is_collapsed()))
        original(event)

    monkeypatch.setattr(QAccessible, "updateAccessibility", record)
    changed = []
    edge.collapsedChanged.connect(changed.append)
    edge.set_collapsed(True)
    edge.set_collapsed(True)
    if route == "setter":
        edge.set_collapsed(False)
    elif route == "mouse":
        qtbot.mouseClick(edge, Qt.LeftButton)
    else:
        activate_focus_window(edge)
        edge.setFocus(Qt.OtherFocusReason)
        qtbot.waitUntil(edge.hasFocus)
        qtbot.keyClick(edge, Qt.Key_Space)
    assert changed == [True, False]
    assert events == [(True, True, True), (True, True, False)]


def test_existing_name_description_and_parent_are_preserved(edge):
    interface = QAccessible.queryAccessibleInterface(edge)
    assert interface.text(QAccessible.Name) == edge.accessibleName()
    before = interface.text(QAccessible.Description)
    assert before == edge.accessibleDescription()
    edge.set_collapsed(True)
    assert interface.text(QAccessible.Description) == edge.accessibleDescription()
    assert interface.text(QAccessible.Description) != before
    edge.setAccessibleName("Caller supplied context")
    assert interface.text(QAccessible.Name) == "Caller supplied context"
    assert interface.parent().object() is edge.parentWidget()


def test_cached_interface_is_stable_and_removed_with_its_widget(edge):
    interface = QAccessible.queryAccessibleInterface(edge)
    identifier = QAccessible.uniqueId(interface)
    assert QAccessible.uniqueId(QAccessible.queryAccessibleInterface(edge)) == identifier
    edge.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert QAccessible.accessibleInterface(identifier) is None


def test_ordinary_widget_accessibility_is_unchanged(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    interface = QAccessible.queryAccessibleInterface(widget)
    assert interface.role() == QAccessible.Client
    assert not interface.state().expandable
    assert QAccessibleActionInterface.pressAction() not in interface.actionInterface().actionNames()


def test_assistive_focus_does_not_toggle_or_resize_the_dock(edge, qtbot, activate_focus_window):
    from PySide6.QtWidgets import QPushButton

    other = QPushButton('Other control', edge.parentWidget())
    other.move(100, 100)
    other.show()
    activate_focus_window(edge)
    other.setFocus(Qt.OtherFocusReason)
    qtbot.waitUntil(other.hasFocus)
    action = QAccessible.queryAccessibleInterface(edge).actionInterface()
    focus = QAccessibleActionInterface.setFocusAction()
    assert focus in action.actionNames()
    width = edge._dock.width()
    changed = []
    edge.collapsedChanged.connect(changed.append)
    action.doAction(focus)
    qtbot.waitUntil(edge.hasFocus)
    action.doAction(focus)
    assert not edge.is_collapsed() and changed == []
    assert edge._dock.width() == width
    qtbot.keyClick(edge, Qt.Key_Tab)
    assert other.hasFocus()
