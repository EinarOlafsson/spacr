"""Assistive clients can identify, inspect and operate the shared toggle."""

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QAccessible, QAccessibleActionInterface
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from spacr.qt import i18n
from spacr.qt.widgets.ai_toggle_label import AiToggleLabel


@pytest.fixture(params=["AI", "Live"])
def toggle(request, qtbot):
    host = QWidget()
    layout = QVBoxLayout(host)
    label = AiToggleLabel(host, text=request.param)
    layout.addWidget(label)
    i18n.retranslate_widget_tree(host, "en")
    qtbot.addWidget(host)
    host.resize(360, 100)
    host.show()
    yield label


def test_exposes_checkbox_role_and_live_check_state(toggle):
    interface = QAccessible.queryAccessibleInterface(toggle)
    assert interface.role() == QAccessible.CheckBox
    state = interface.state()
    assert state.checkable and not state.checked and not state.readOnly
    toggle.setChecked(True)
    assert interface.state().checked
    toggle.setChecked(False)
    assert not interface.state().checked
    toggle.setEnabled(False)
    assert interface.state().disabled


def test_accessibility_action_uses_the_existing_toggle_signal(toggle):
    interface = QAccessible.queryAccessibleInterface(toggle)
    action = interface.actionInterface()
    assert action is not None
    assert QAccessibleActionInterface.toggleAction() in action.actionNames()
    changed = []
    toggle.toggled.connect(changed.append)
    action.doAction(QAccessibleActionInterface.toggleAction())
    assert toggle.isChecked() and interface.state().checked
    action.doAction(QAccessibleActionInterface.toggleAction())
    assert not toggle.isChecked() and not interface.state().checked
    assert changed == [True, False]


def test_disabled_accessible_action_cannot_toggle(toggle):
    action = QAccessible.queryAccessibleInterface(toggle).actionInterface()
    assert action is not None
    toggle.setEnabled(False)
    changed = []
    toggle.toggled.connect(changed.append)
    assert QAccessibleActionInterface.toggleAction() not in action.actionNames()
    action.doAction(QAccessibleActionInterface.toggleAction())
    assert not toggle.isChecked()
    assert changed == []


def test_real_changes_emit_checked_state_notifications(toggle, monkeypatch, qtbot):
    events = []
    original = QAccessible.updateAccessibility

    def record(event):
        if event.type() == QAccessible.StateChanged and event.object() is toggle:
            events.append((bool(event.changedStates().checked), toggle.isChecked()))
        original(event)

    monkeypatch.setattr(QAccessible, "updateAccessibility", record)
    changed = []
    toggle.toggled.connect(changed.append)
    toggle.setChecked(True)
    toggle.setChecked(True)
    qtbot.mouseClick(toggle, Qt.LeftButton)
    assert changed == [True, False]
    assert events == [(True, True), (True, False)]


def test_accessible_name_follows_language_and_preserves_explicit_name(toggle):
    interface = QAccessible.queryAccessibleInterface(toggle)
    for language in ("en", "ko"):
        i18n.retranslate_widget_tree(toggle.parentWidget(), language)
        assert interface.text(QAccessible.Name) == toggle.text()
    toggle.setAccessibleName("Caller supplied context")
    toggle.setText("A long logical toggle label that will be elided")
    toggle.setFixedWidth(65)
    assert toggle.displayed_text() != toggle.text()
    assert interface.text(QAccessible.Name) == "Caller supplied context"
    toggle.setAccessibleName("")
    assert interface.text(QAccessible.Name) == toggle.text()


def test_adapter_preserves_parent_and_releases_cached_interface(toggle):
    interface = QAccessible.queryAccessibleInterface(toggle)
    identifier = QAccessible.uniqueId(interface)
    assert interface.parent().object() is toggle.parentWidget()
    assert QAccessible.uniqueId(QAccessible.queryAccessibleInterface(toggle)) == identifier
    toggle.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert QAccessible.accessibleInterface(identifier) is None


def test_ordinary_label_accessibility_is_unchanged(qtbot):
    label = QLabel("Ordinary label")
    qtbot.addWidget(label)
    interface = QAccessible.queryAccessibleInterface(label)
    assert interface.role() == QAccessible.StaticText
    assert not interface.state().checkable
    assert interface.text(QAccessible.Name) == "Ordinary label"


def test_other_accessible_texts_come_from_the_widget(toggle):
    """Only the name is the label's own; a description is the widget's."""
    toggle.setAccessibleDescription("Turns spaCR AI on or off")
    interface = QAccessible.queryAccessibleInterface(toggle)
    assert interface.text(QAccessible.Description) == "Turns spaCR AI on or off"
    assert interface.text(QAccessible.Name) == toggle.text()


def test_a_non_toggle_action_reaches_the_widget_without_toggling(
        toggle, qtbot, activate_focus_window):
    """Moving focus by an assistive client focuses the label and leaves it off."""
    activate_focus_window(toggle)
    action = QAccessible.queryAccessibleInterface(toggle).actionInterface()
    focus = QAccessibleActionInterface.setFocusAction()
    assert focus in action.actionNames()
    changed = []
    toggle.toggled.connect(changed.append)
    action.doAction(focus)
    qtbot.waitUntil(toggle.hasFocus)
    assert not toggle.isChecked() and changed == []
