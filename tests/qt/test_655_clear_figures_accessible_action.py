"""Assistive clients can identify and press the real Clear figures control."""

import gc

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QAccessible, QAccessibleActionInterface
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from spacr.qt import i18n
from spacr.qt.widgets.figure_queue import _ClearFiguresLabel


@pytest.fixture
def control(qtbot):
    host = QWidget()
    layout = QVBoxLayout(host)
    label = _ClearFiguresLabel(host)
    layout.addWidget(label)
    qtbot.addWidget(host)
    host.resize(300, 100)
    host.show()
    yield label


def test_clear_control_has_button_role_and_current_name(control):
    interface = QAccessible.queryAccessibleInterface(control)
    assert interface.role() == QAccessible.Button
    assert interface.state().focusable
    assert not interface.state().checkable
    for language in ("en", "ko"):
        i18n.retranslate_widget_tree(control.parentWidget(), language)
        assert interface.text(QAccessible.Name) == control.text()
    control.setAccessibleName("Clear the current result queue")
    assert interface.text(QAccessible.Name) == "Clear the current result queue"


def test_accessible_press_emits_one_existing_click_and_flash(control):
    action = QAccessible.queryAccessibleInterface(control).actionInterface()
    assert action is not None
    assert QAccessibleActionInterface.pressAction() in action.actionNames()
    clicks = []
    control.clicked.connect(lambda: clicks.append(True))
    action.doAction(QAccessibleActionInterface.pressAction())
    assert clicks == [True]
    assert control._flash.active
    action.doAction("unrecognized-action")
    assert clicks == [True]


def test_disabled_accessible_press_cannot_clear_or_flash(control):
    interface = QAccessible.queryAccessibleInterface(control)
    action = interface.actionInterface()
    assert action is not None
    control.setEnabled(False)
    clicks = []
    control.clicked.connect(lambda: clicks.append(True))
    assert interface.state().disabled
    assert action.actionNames() == []
    action.doAction(QAccessibleActionInterface.pressAction())
    assert clicks == []
    assert not control._flash.active


def test_adapter_keeps_normal_navigation_and_qt_cache_ownership(control):
    interface = QAccessible.queryAccessibleInterface(control)
    identifier = QAccessible.uniqueId(interface)
    assert interface.parent().object() is control.parentWidget()
    del interface
    gc.collect()
    interface = QAccessible.queryAccessibleInterface(control)
    assert QAccessible.uniqueId(interface) == identifier
    control.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert QAccessible.accessibleInterface(identifier) is None


@pytest.mark.parametrize("kind,role", [(QLabel, QAccessible.StaticText),
                                      (QPushButton, QAccessible.Button)])
def test_unrelated_widgets_keep_native_accessibility(qtbot, kind, role):
    widget = kind("Ordinary control")
    qtbot.addWidget(widget)
    interface = QAccessible.queryAccessibleInterface(widget)
    assert interface.role() == role
    assert interface.text(QAccessible.Name) == "Ordinary control"
