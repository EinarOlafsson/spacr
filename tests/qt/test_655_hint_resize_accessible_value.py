"""Assistive clients can resize the help grip without an unsafe value-interface cast."""

import gc

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QAccessible, QAccessibleActionInterface
from PySide6.QtWidgets import QApplication, QFrame

from spacr.qt import i18n
from tests.qt.test_655_hint_resize_keyboard import help_window as help_window


def test_grip_role_and_truthful_value_text_without_custom_value_interface(help_window):
    _, _, bar, _ = help_window
    interface = QAccessible.queryAccessibleInterface(bar._resize_handle)
    assert interface.role() == QAccessible.Grip
    assert interface.state().focusable and interface.state().sizeable
    assert interface.valueInterface() is None
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


def test_boundary_and_disabled_actions_do_not_change_height(help_window):
    _, _, bar, _ = help_window
    handle = bar._resize_handle
    interface = QAccessible.queryAccessibleInterface(handle)
    action = interface.actionInterface()
    upper = bar._maximum_help_height()
    commits = []
    bar.helpHeightCommitted.connect(commits.append)
    assert QAccessibleActionInterface.decreaseAction() not in action.actionNames()
    action.doAction(QAccessibleActionInterface.decreaseAction())
    assert commits == []
    bar._set_manual_height(upper)
    QApplication.processEvents()
    assert bar.height() == upper
    assert QAccessibleActionInterface.increaseAction() not in action.actionNames()
    action.doAction(QAccessibleActionInterface.increaseAction())
    assert commits == []
    bar.setEnabled(False)
    assert interface.state().disabled and action.actionNames() == []
    action.doAction(QAccessibleActionInterface.decreaseAction())
    assert bar.height() == upper and commits == []


def test_value_text_tracks_keyboard_and_existing_manual_resizes(help_window, qtbot):
    _, _, bar, _ = help_window
    handle = bar._resize_handle
    interface = QAccessible.queryAccessibleInterface(handle)
    handle.setFocus(Qt.TabFocusReason)
    original = bar.height()
    qtbot.keyClick(handle, Qt.Key_Up)
    QApplication.processEvents()
    assert bar.height() > original
    assert interface.text(QAccessible.Value) == str(bar.height())
    bar._set_manual_height(original)
    QApplication.processEvents()
    assert interface.text(QAccessible.Value) == str(original)


def test_actions_track_layout_and_names_follow_normal_language_pass(help_window):
    host, _, bar, _ = help_window
    interface = QAccessible.queryAccessibleInterface(bar._resize_handle)
    action = interface.actionInterface()
    bar._set_manual_height(bar._maximum_help_height())
    QApplication.processEvents()
    assert QAccessibleActionInterface.increaseAction() not in action.actionNames()
    host.resize(host.width(), host.height() + 120)
    QApplication.processEvents()
    assert QAccessibleActionInterface.increaseAction() in action.actionNames()
    for language in ('ko', 'en'):
        i18n.retranslate_widget_tree(host, language)
        assert interface.text(QAccessible.Name) == bar._resize_handle.accessibleName()
        assert interface.text(QAccessible.Description) == bar._resize_handle.accessibleDescription()
        assert interface.text(QAccessible.Name)


def test_qt_owns_adapter_through_collection_and_retires_it_with_widget(help_window):
    host, _, bar, _ = help_window
    handle = bar._resize_handle
    interface = QAccessible.queryAccessibleInterface(handle)
    identifier = QAccessible.uniqueId(interface)
    assert interface.parent().object() is bar
    gc.collect()
    assert interface.text(QAccessible.Value) == str(bar.height())
    assert QAccessible.queryAccessibleInterface(handle).actionInterface() is not None
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


def test_assistive_focus_does_not_resize_or_commit_and_tab_can_leave(help_window, qtbot):
    _, before, bar, _ = help_window
    handle = bar._resize_handle
    assert before.hasFocus()
    action = QAccessible.queryAccessibleInterface(handle).actionInterface()
    focus = QAccessibleActionInterface.setFocusAction()
    assert focus in action.actionNames()
    assert action.keyBindingsForAction(focus) == []
    height = bar.height()
    commits = []
    bar.helpHeightCommitted.connect(commits.append)
    action.doAction(focus)
    qtbot.waitUntil(handle.hasFocus)
    action.doAction(focus)
    assert bar.height() == height and commits == []
    qtbot.keyClick(handle, Qt.Key_Tab)
    assert bar._view.hasFocus()


def test_retained_resize_callback_cannot_commit_after_bar_is_disabled(help_window):
    from pathlib import Path
    from spacr.qt import preferences

    _, _, bar, _ = help_window
    resize = bar._resize_handle._resize_to
    height = bar.height()
    requested = height + bar.fontMetrics().lineSpacing()
    assert requested <= bar._maximum_help_height()
    stored = preferences._settings()
    key = preferences._KEY_PREFERENCES_HELP_HEIGHT
    stored.setValue(key, height)
    stored.sync()
    settings_path = Path(stored.fileName())
    before = settings_path.read_bytes()
    commits = []
    bar.helpHeightCommitted.connect(commits.append)
    bar.helpHeightCommitted.connect(lambda value: stored.setValue(key, value))
    bar.setEnabled(False)
    resize(requested)
    stored.sync()
    assert bar.height() == height and commits == []
    assert int(stored.value(key)) == height
    assert settings_path.read_bytes() == before
    bar.setEnabled(True)
    resize(requested)
    stored.sync()
    assert bar.height() == requested and commits == [requested]
    assert int(stored.value(key)) == requested
