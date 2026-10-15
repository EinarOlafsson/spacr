"""Keyboard access preserves the shared AI/Live label behavior."""
import pytest
from PySide6.QtCore import QEvent, QSettings, Qt
from PySide6.QtGui import QAccessible, QColor, QKeyEvent
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
from spacr.qt import i18n, preferences, theme
from spacr.qt.widgets.ai_toggle_label import AiToggleLabel


@pytest.fixture(params=['AI', 'Live'])
def toggle_row(request, qtbot, monkeypatch, tmp_path):
    settings = tmp_path / 'preferences.ini'
    monkeypatch.setattr(preferences, '_settings',
                        lambda: QSettings(str(settings), QSettings.IniFormat))
    monkeypatch.setattr(preferences, '_SAFE_MODE', False)
    preferences.set_strong_focus(False)
    theme._forget_painting_theme()
    host = QWidget()
    layout = QVBoxLayout(host)
    before = QPushButton('Before')
    label = AiToggleLabel(text=request.param)
    after = QPushButton('After')
    for child in (before, label, after):
        layout.addWidget(child)
    i18n.retranslate_widget_tree(host, 'en')
    qtbot.addWidget(host)
    host.resize(360, 180)
    host.show()
    host.activateWindow()
    before.setFocus(Qt.TabFocusReason)
    qtbot.waitUntil(before.hasFocus)
    yield host, before, label, after
    theme._forget_painting_theme()


def test_tab_can_reach_and_leave_the_toggle(toggle_row, qtbot):
    _, before, label, after = toggle_row
    reached = []
    for _ in range(8):
        qtbot.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
        reached.append(QApplication.focusWidget())
    assert label in reached, (label.text(), label.focusPolicy(), reached)
    assert before in reached and after in reached


@pytest.mark.parametrize('key', [Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter])
def test_actual_activation_key_toggles_once_when_given_focus(toggle_row, qtbot, key):
    _, _, label, _ = toggle_row
    changed = []
    label.toggled.connect(changed.append)
    label.setFocus(Qt.TabFocusReason)
    qtbot.waitUntil(label.hasFocus)
    qtbot.keyClick(label, key)
    assert label.isChecked()
    assert changed == [True]


@pytest.mark.parametrize('language', ['en', 'ko'])
def test_native_accessible_name_matches_translated_label(toggle_row, language):
    host, _, label, _ = toggle_row
    i18n.retranslate_widget_tree(host, language)
    interface = QAccessible.queryAccessibleInterface(label)
    assert interface is not None
    assert interface.text(QAccessible.Name) == label.text()
    assert interface.text(QAccessible.Name).strip()


def test_mouse_click_still_proves_the_toggle_is_actionable(toggle_row, qtbot):
    _, _, label, _ = toggle_row
    changed = []
    label.toggled.connect(changed.append)
    qtbot.mouseClick(label, Qt.LeftButton)
    assert label.isChecked()
    assert changed == [True]


@pytest.mark.parametrize('theme_name', ['dark', 'light', 'high_contrast'])
def test_focus_is_visible_and_tracks_live_stronger_outline(toggle_row, theme_name):
    host, before, label, _ = toggle_row
    preferences.set_theme_choice(theme_name)
    preferences.set_strong_focus(False)
    theme._forget_painting_theme()
    label._refresh_style()
    before.setFocus(Qt.TabFocusReason)
    plain = label.grab().toImage()
    label.setFocus(Qt.TabFocusReason)
    assert label.hasFocus()
    ordinary = label.grab().toImage()
    assert ordinary != plain
    preferences.set_strong_focus(True)
    host.setStyleSheet(theme._the_focus_ring(theme_name))
    QApplication.processEvents()
    assert label.hasFocus()
    stronger = label.grab().toImage()
    colour = QColor(theme.focus_ring_colour(theme_name))
    assert [stronger.pixelColor(stronger.width() // 2, y)
            for y in range(theme.FOCUS_RING_PX)] == [colour] * theme.FOCUS_RING_PX
    assert stronger != ordinary
    preferences.set_strong_focus(False)
    host.setStyleSheet(theme._the_focus_ring(theme_name))
    QApplication.processEvents()
    assert label.hasFocus()
    assert label.grab().toImage() == ordinary
    before.setFocus(Qt.TabFocusReason)
    assert label.grab().toImage() == plain


def test_modified_and_repeated_keys_do_not_change_state(toggle_row, qtbot):
    _, _, label, _ = toggle_row
    changes = []
    label.toggled.connect(changes.append)
    label.setFocus(Qt.TabFocusReason)
    for key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
        for modifier in (Qt.ControlModifier, Qt.AltModifier, Qt.ShiftModifier,
                         Qt.MetaModifier, Qt.KeypadModifier | Qt.ControlModifier,
                         Qt.KeypadModifier | Qt.AltModifier,
                         Qt.KeypadModifier | Qt.ShiftModifier,
                         Qt.KeypadModifier | Qt.MetaModifier):
            qtbot.keyClick(label, key, modifier)
        repeat = QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, '', True, 1)
        QApplication.sendEvent(label, repeat)
        assert repeat.isAccepted()
    keypad_repeat = QKeyEvent(QEvent.KeyPress, Qt.Key_Enter,
                              Qt.KeypadModifier, '', True, 1)
    QApplication.sendEvent(label, keypad_repeat)
    assert keypad_repeat.isAccepted()
    qtbot.keyClick(label, Qt.Key_F6)
    for key in (Qt.Key_Space, Qt.Key_Return):
        qtbot.keyClick(label, key, Qt.KeypadModifier)
    assert not label.isChecked()
    assert changes == []


def test_unfocused_or_disabled_toggle_does_not_activate(toggle_row, qtbot):
    _, before, label, _ = toggle_row
    changes = []
    label.toggled.connect(changes.append)
    before.setFocus(Qt.TabFocusReason)
    qtbot.keyClick(label, Qt.Key_Space)
    assert not label.isChecked()
    label.setEnabled(False)
    qtbot.keyClick(label, Qt.Key_Space)
    assert not label.isChecked()
    assert changes == []


def test_keyboard_signal_state_and_language_round_trip(toggle_row, qtbot):
    host, _, label, _ = toggle_row
    changes = []
    label.toggled.connect(changes.append)
    label.setFocus(Qt.TabFocusReason)
    qtbot.keyClick(label, Qt.Key_Space)
    assert label.isChecked()
    i18n.retranslate_widget_tree(host, 'ko')
    assert label.hasFocus()
    assert label.isChecked()
    assert changes == [True]
    assert QAccessible.queryAccessibleInterface(label).text(QAccessible.Name) == label.text()
    qtbot.keyClick(label, Qt.Key_Return)
    assert not label.isChecked()
    assert changes == [True, False]
    label.setChecked(False)
    assert changes == [True, False]


def test_focus_paint_does_not_read_preferences(toggle_row, monkeypatch):
    host, _, label, _ = toggle_row
    preferences.set_strong_focus(True)
    host.setStyleSheet(theme._the_focus_ring('dark'))
    QApplication.processEvents()
    label.setFocus(Qt.TabFocusReason)

    def forbidden():
        raise AssertionError('The paint path read the preference store')

    monkeypatch.setattr(preferences, 'get_strong_focus', forbidden)
    assert not label.grab().isNull()


def test_numeric_keypad_enter_toggles_once(toggle_row, qtbot):
    _, _, label, _ = toggle_row
    changes = []
    label.toggled.connect(changes.append)
    label.setFocus(Qt.TabFocusReason)
    qtbot.keyClick(label, Qt.Key_Enter, Qt.KeypadModifier)
    assert label.isChecked()
    assert changes == [True]
