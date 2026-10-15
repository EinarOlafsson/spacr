"""Assistive clients can inspect and adjust the help strip's actual height."""

import gc

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QAccessible, QAccessibleActionInterface
from PySide6.QtWidgets import QApplication, QFrame

from spacr.qt import i18n
from tests.qt.test_655_hint_resize_keyboard import help_window as help_window


def test_grip_exposes_actual_height_range_and_pixel_step(help_window):
    _, _, bar, _ = help_window
    interface = QAccessible.queryAccessibleInterface(bar._resize_handle)
    assert interface.role() == QAccessible.Grip
    assert interface.state().focusable and interface.state().sizeable
    value = interface.valueInterface()
    assert value is not None
    assert value.currentValue() == bar.height() == value.minimumValue()
    assert value.maximumValue() > value.currentValue()
    assert value.minimumStepSize() == 1
    assert interface.text(QAccessible.Value) == str(bar.height())


def test_accessible_actions_resize_and_commit_once(help_window):
    _, _, bar, _ = help_window
    action = QAccessible.queryAccessibleInterface(bar._resize_handle).actionInterface()
    assert QAccessibleActionInterface.increaseAction() in action.actionNames()
    original = bar.height()
    step = bar.fontMetrics().lineSpacing()
    commits = []
    bar.helpHeightCommitted.connect(commits.append)
    action.doAction(QAccessibleActionInterface.increaseAction())
    assert bar.height() == original + step
    action.doAction(QAccessibleActionInterface.decreaseAction())
    assert bar.height() == original
    assert commits == [original + step, original]
    assert action.keyBindingsForAction(QAccessibleActionInterface.increaseAction()) == ['Up']
    assert action.keyBindingsForAction(QAccessibleActionInterface.decreaseAction()) == ['Down']


def test_value_setter_uses_real_bounds_and_persistence(help_window):
    _, _, bar, _ = help_window
    value = QAccessible.queryAccessibleInterface(bar._resize_handle).valueInterface()
    assert value is not None
    commits = []
    bar.helpHeightCommitted.connect(commits.append)
    desired = value.minimumValue() + 1
    value.setCurrentValue(float(desired))
    value.setCurrentValue(desired)
    assert bar.height() == desired and bar._manual_height == desired
    assert commits == [desired]
    value.setCurrentValue(value.maximumValue() + 1)
    value.setCurrentValue(value.minimumValue() - 1)
    assert bar.height() == desired and commits == [desired]


@pytest.mark.parametrize('bad', [None, True, '120', float('nan'), float('inf'), 100.5])
def test_non_pixel_values_do_not_resize(help_window, bad):
    _, _, bar, _ = help_window
    value = QAccessible.queryAccessibleInterface(bar._resize_handle).valueInterface()
    assert value is not None
    original = bar.height()
    commits = []
    bar.helpHeightCommitted.connect(commits.append)
    value.setCurrentValue(bad)
    assert bar.height() == original and commits == []


def test_boundary_and_disabled_actions_do_not_change_height(help_window):
    _, _, bar, _ = help_window
    handle = bar._resize_handle
    interface = QAccessible.queryAccessibleInterface(handle)
    value, action = interface.valueInterface(), interface.actionInterface()
    assert value is not None
    lower, upper = value.minimumValue(), value.maximumValue()
    commits = []
    bar.helpHeightCommitted.connect(commits.append)
    assert QAccessibleActionInterface.decreaseAction() not in action.actionNames()
    action.doAction(QAccessibleActionInterface.decreaseAction())
    assert commits == []
    value.setCurrentValue(upper)
    QApplication.processEvents()
    assert bar.height() == upper
    assert QAccessibleActionInterface.increaseAction() not in action.actionNames()
    action.doAction(QAccessibleActionInterface.increaseAction())
    assert commits == [upper]
    bar.setEnabled(False)
    assert interface.state().disabled and action.actionNames() == []
    value.setCurrentValue(lower)
    action.doAction(QAccessibleActionInterface.decreaseAction())
    assert bar.height() == upper and commits == [upper]


def test_value_notifications_follow_real_resizes_only(help_window, qtbot, monkeypatch):
    _, _, bar, _ = help_window
    handle = bar._resize_handle
    interface = QAccessible.queryAccessibleInterface(handle)
    value = interface.valueInterface()
    assert value is not None
    notices = []
    original_update = QAccessible.updateAccessibility

    def observe(event):
        if event.type() == QAccessible.ValueChanged and event.object() is handle:
            notices.append((event.value(), value.currentValue()))
        original_update(event)

    monkeypatch.setattr(QAccessible, 'updateAccessibility', observe)
    handle.setFocus(Qt.TabFocusReason)
    original = bar.height()
    qtbot.keyClick(handle, Qt.Key_Up)
    QApplication.processEvents()
    grown = bar.height()
    assert notices == [(grown, grown)]
    value.setCurrentValue(grown)
    assert len(notices) == 1
    bar._set_manual_height(original)
    QApplication.processEvents()
    assert notices == [(grown, grown), (original, original)]


def test_range_tracks_layout_and_names_follow_normal_language_pass(help_window):
    host, _, bar, _ = help_window
    interface = QAccessible.queryAccessibleInterface(bar._resize_handle)
    value = interface.valueInterface()
    assert value is not None
    maximum = value.maximumValue()
    host.resize(host.width(), host.height() + 120)
    QApplication.processEvents()
    assert value.maximumValue() > maximum
    for language in ('ko', 'en'):
        i18n.retranslate_widget_tree(host, language)
        assert interface.text(QAccessible.Name) == bar._resize_handle.accessibleName()
        assert interface.text(QAccessible.Description) == bar._resize_handle.accessibleDescription()
        assert interface.text(QAccessible.Name)


def test_value_pointer_survives_collection_and_qt_retires_the_adapter(help_window):
    host, _, bar, _ = help_window
    handle = bar._resize_handle
    interface = QAccessible.queryAccessibleInterface(handle)
    value = interface.valueInterface()
    assert value is not None
    identifier = QAccessible.uniqueId(interface)
    assert interface.parent().object() is bar
    gc.collect()
    assert value.currentValue() == bar.height()
    assert QAccessible.queryAccessibleInterface(handle).valueInterface().currentValue() == bar.height()
    host.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert QAccessible.accessibleInterface(identifier) is None


def test_ordinary_frame_accessibility_is_unchanged(qtbot):
    frame = QFrame()
    frame.setFrameShape(QFrame.HLine)
    qtbot.addWidget(frame)
    interface = QAccessible.queryAccessibleInterface(frame)
    assert interface.role() == QAccessible.Border
    assert interface.valueInterface() is None
