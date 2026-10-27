"""Easy input work leaves the live field visible and empty input truly empty."""
import threading

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import Qt

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


@pytest.mark.parametrize(('theme', 'pending'), [('dark', 'inspection'), ('light', 'regex')])
def test_clearing_easy_input_retires_pending_work_without_guidance(
        qtbot, qapp, private_ai_profile, tmp_path, monkeypatch, theme, pending):
    from spacr.qt import theme as styling
    from spacr.qt.app import MainWindow
    from spacr.qt.widgets import home

    preferences, _, _ = private_ai_profile
    preferences._set_spacr_ai_enabled(False)
    preferences.set_easy_mode(True)
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(False)
    preferences.set_theme(theme)
    previous_sheet = qapp.styleSheet()
    qapp.setStyleSheet(styling.stylesheet(theme))
    image = tmp_path / 'scan_101_C1.tif'
    tifffile.imwrite(image, np.arange(1200, dtype=np.uint16).reshape(30, 40))
    original_bytes = image.read_bytes()
    entered, release = threading.Event(), threading.Event()
    blocked = []
    original_inspect = home._easy_inspect_path
    original_regex = home._easy_validate_regex

    def inspect(value):
        if blocked and pending == 'inspection':
            entered.set()
            assert release.wait(5)
        return original_inspect(value)

    def validate(result, pattern, cancelled=None):
        entered.set()
        assert release.wait(5)
        assert cancelled is not None and cancelled()
        return original_regex(result, pattern, cancelled=cancelled)

    monkeypatch.setattr(home, '_easy_inspect_path', inspect)
    monkeypatch.setattr(home, '_easy_validate_regex', validate)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1280, 900)
    window.show()
    page = window._startup
    try:
        qtbot.waitUntil(lambda: window.window_backdrop() is not None, timeout=5000)
        page.path_edit.setFocus()
        qtbot.keyClicks(page.path_edit, str(image))
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(lambda: not page._runner.is_busy() and bool(page.field.cards), timeout=10000)
        assert window._stack.currentWidget() is page
        assert page._last_result['path'] == str(image)
        assert any(isinstance(card, home._EasyRegexCard) for card in page.field.cards)
        if pending == 'inspection':
            blocked.append(True)
            qtbot.keyClick(page.path_edit, Qt.Key_Return)
        else:
            card = next(card for card in page.field.cards if isinstance(card, home._EasyRegexCard))
            qtbot.waitUntil(lambda: not card._growing, timeout=7000)
            card.pattern_edit.setPlainText(r'scan_(?P<fieldID>[0-9]+)_C(?P<chanID>[0-9]+)\.tif$')
            card.click()
        qtbot.waitUntil(entered.is_set, timeout=5000)
        assert page._runner.is_busy()
        page.path_edit.setFocus()
        page.path_edit.selectAll()
        qtbot.keyClick(page.path_edit, Qt.Key_Backspace)
        assert page.path_edit.text() == ''
        assert not page._runner.is_busy() and not page._debounce.isActive()
        assert not page.field.cards and not page.field._animations
        assert page._last_result is None and page._settings_overrides == {}
        assert page._regex_cancel is None
        assert page.status.text() == '' and not page.status.isVisible()
        assert page.path_edit.accessibleDescription() == '' and page.path_edit.toolTip() == ''
        assert page.path_edit.isVisible() and page._workflow is None
        release.set()
        qtbot.waitUntil(lambda: page._runner.active_jobs() == 0, timeout=10000)
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        assert not page._debounce.isActive() and not page._runner.active_jobs()
        assert not page.field.cards and page.status.text() == ''
        assert window._stack.currentWidget() is page
        assert image.read_bytes() == original_bytes
    finally:
        release.set()
        window.close()
        qapp.setStyleSheet(previous_sheet)
