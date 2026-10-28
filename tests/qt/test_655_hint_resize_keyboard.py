"""The help grip offers bounded keyboard resizing without trapping focus."""
import pytest
from PySide6.QtCore import QEvent, QSettings, Qt
from PySide6.QtGui import QColor, QKeyEvent
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
from spacr.qt import preferences, theme
from spacr.qt.widgets.hint_bar import HintBar


@pytest.fixture
def help_window(qtbot, monkeypatch, tmp_path, activate_focus_window):
    settings = tmp_path / 'preferences.ini'
    monkeypatch.setattr(preferences, '_settings',
                        lambda: QSettings(str(settings), QSettings.IniFormat))
    monkeypatch.setattr(preferences, '_SAFE_MODE', False)
    preferences.set_strong_focus(False)
    theme._forget_painting_theme()
    host = QWidget()
    layout = QVBoxLayout(host)
    page = QWidget()
    page_layout = QVBoxLayout(page)
    before = QPushButton('Before')
    page_layout.addWidget(before)
    page_layout.addStretch()
    bar = HintBar('Help text.')
    after = QPushButton('After')
    for child in (page, bar, after):
        layout.addWidget(child)
    qtbot.addWidget(host)
    host.resize(420, 700)
    host.show()
    activate_focus_window(host)
    before.setFocus(Qt.TabFocusReason)
    qtbot.waitUntil(before.hasFocus)
    QApplication.processEvents()
    yield host, before, bar, after
    theme._forget_painting_theme()


def test_tab_reaches_grip_and_leaves_for_help_text(help_window, qtbot):
    _, before, bar, after = help_window
    qtbot.keyClick(before, Qt.Key_Tab)
    assert bar._resize_handle.hasFocus()
    qtbot.keyClick(bar._resize_handle, Qt.Key_Tab)
    assert bar._view.hasFocus()
    qtbot.keyClick(bar._view, Qt.Key_Tab)
    assert after.hasFocus()
    qtbot.keyClick(after, Qt.Key_Backtab)
    qtbot.keyClick(bar._view, Qt.Key_Backtab)
    assert bar._resize_handle.hasFocus()


def test_arrows_change_height_and_emit_commits(help_window, qtbot):
    _, _, bar, _ = help_window
    handle = bar._resize_handle
    handle.setFocus(Qt.TabFocusReason)
    original = bar.height()
    step = bar.fontMetrics().lineSpacing()
    commits = []
    bar.helpHeightCommitted.connect(commits.append)
    qtbot.keyClick(handle, Qt.Key_Up)
    assert bar.height() == original + step
    assert bar._manual_height == bar.height()
    qtbot.keyClick(handle, Qt.Key_Down)
    assert bar.height() == original
    assert commits == [original + step, original]


@pytest.mark.parametrize('scheme', ['dark', 'light', 'high_contrast'])
def test_focus_visible_and_stronger_on_already_open_grip(help_window, scheme):
    host, before, bar, _ = help_window
    handle = bar._resize_handle
    preferences.set_theme_choice(scheme)
    preferences.set_strong_focus(False)
    theme._forget_painting_theme()
    host.setStyleSheet(theme._the_focus_ring(scheme))
    QApplication.sendEvent(handle, QEvent(QEvent.StyleChange))
    before.setFocus(Qt.TabFocusReason)
    plain = handle.grab().toImage()
    handle.setFocus(Qt.TabFocusReason)
    ordinary = handle.grab().toImage()
    assert ordinary != plain
    preferences.set_strong_focus(True)
    host.setStyleSheet(theme._the_focus_ring(scheme))
    QApplication.processEvents()
    stronger = handle.grab().toImage()
    colour = QColor(theme.focus_ring_colour(scheme))
    assert [stronger.pixelColor(stronger.width() // 2, y)
            for y in range(theme.FOCUS_RING_PX)] == [colour] * theme.FOCUS_RING_PX
    assert stronger != ordinary
    preferences.set_strong_focus(False)
    host.setStyleSheet(theme._the_focus_ring(scheme))
    QApplication.processEvents()
    assert handle.hasFocus()
    assert handle.grab().toImage() == ordinary
    before.setFocus(Qt.TabFocusReason)
    assert handle.grab().toImage() == plain


def test_bounds_do_not_emit_unchanged_height(help_window, qtbot):
    _, _, bar, _ = help_window
    handle = bar._resize_handle
    handle.setFocus(Qt.TabFocusReason)
    commits = []
    bar.helpHeightCommitted.connect(commits.append)
    minimum = bar._minimum_help_height()
    qtbot.keyClick(handle, Qt.Key_Down)
    assert bar.height() == minimum and commits == []
    maximum = bar._maximum_help_height()
    bar._set_manual_height(maximum)
    QApplication.processEvents()
    qtbot.keyClick(handle, Qt.Key_Up)
    assert bar.height() == maximum and commits == []
    qtbot.keyClick(handle, Qt.Key_Down)
    assert bar.height() < maximum
    assert commits == [bar.height()]


def test_held_arrows_repeat_at_font_aware_step(help_window):
    _, _, bar, _ = help_window
    font = bar.font()
    font.setPointSize(font.pointSize() + 5)
    bar.setFont(font)
    QApplication.processEvents()
    handle = bar._resize_handle
    handle.setFocus(Qt.TabFocusReason)
    original = bar.height()
    step = bar.fontMetrics().lineSpacing()
    commits = []
    bar.helpHeightCommitted.connect(commits.append)
    for count in (1, 2):
        event = QKeyEvent(QEvent.KeyPress, Qt.Key_Up, Qt.NoModifier, '', True, 1)
        QApplication.sendEvent(handle, event)
        assert event.isAccepted()
        assert bar.height() == original + count * step
    assert commits == [original + step, original + 2 * step]


def test_unrelated_modifiers_and_keys_leave_height(help_window, qtbot):
    _, _, bar, _ = help_window
    handle = bar._resize_handle
    handle.setFocus(Qt.TabFocusReason)
    original = bar.height()
    commits = []
    bar.helpHeightCommitted.connect(commits.append)
    for key in (Qt.Key_Up, Qt.Key_Down):
        for modifier in (Qt.ControlModifier, Qt.AltModifier,
                         Qt.ShiftModifier, Qt.MetaModifier):
            qtbot.keyClick(handle, key, modifier)
    for key in (Qt.Key_Space, Qt.Key_Left, Qt.Key_Right):
        qtbot.keyClick(handle, key)
    assert bar.height() == original and commits == []


def test_paint_reads_cached_focus_style(help_window, monkeypatch):
    host, _, bar, _ = help_window
    preferences.set_strong_focus(True)
    host.setStyleSheet(theme._the_focus_ring('dark'))
    QApplication.processEvents()
    bar._resize_handle.setFocus(Qt.TabFocusReason)

    def forbidden():
        raise AssertionError('painting read the preference store')

    monkeypatch.setattr(preferences, 'get_strong_focus', forbidden)
    assert not bar._resize_handle.grab().isNull()
