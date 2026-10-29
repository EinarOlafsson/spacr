"""Native window updates retain local input and stop maintenance after closing."""
import threading

import numpy as np
from PySide6.QtWidgets import QDialog, QLabel

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


def _window(qtbot, private_ai_profile):
    from spacr.qt.app import MainWindow

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_ambient_enabled(False)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.show()
    return window, preferences


def test_portable_navigator_keeps_inspected_input_when_preferences_rebuild_home(
        qtbot, private_ai_profile, tmp_path, monkeypatch):
    from spacr.logging_util import _spacr_home

    monkeypatch.setenv('SPACR_PORTABLE', str(tmp_path / 'portable'))
    image = tmp_path / 'field.npy'
    np.save(image, np.zeros((8, 9, 3), dtype=np.uint16))
    before = image.read_bytes()
    window, preferences = _window(qtbot, private_ai_profile)
    try:
        page = window._startup
        page.accept_paths([str(image)])
        qtbot.waitUntil(lambda: page._last_result is not None and not page._runner.is_busy(), timeout=5000)
        actions = list(page._last_result['actions'])
        preferences.set_font_scale(1.05)
        assert window.refresh_after_preferences({'font_scale'}) == 'rebuilt'
        rebuilt = window._startup
        assert rebuilt is not page and rebuilt.path_edit.text() == str(image)
        qtbot.waitUntil(lambda: rebuilt._last_result is not None and not rebuilt._runner.is_busy(), timeout=5000)
        assert rebuilt._last_result['actions'] == actions
        inspected = []
        def inspect_dialog(dialog):
            text = '\n'.join(label.text() for label in dialog.findChildren(QLabel))
            assert 'Portable mode:' in text and str(_spacr_home()) in text
            inspected.append(True)
            return QDialog.Accepted
        monkeypatch.setattr(QDialog, 'exec', inspect_dialog)
        window._show_about()
        assert inspected and image.read_bytes() == before
        assert window._stack.currentWidget() is rebuilt
    finally:
        window.close()


def test_closing_window_rejects_late_news_and_does_not_start_more_runtime_maintenance(
        qtbot, private_ai_profile, monkeypatch):
    from spacr.qt import app

    window, _ = _window(qtbot, private_ai_profile)
    runs = []
    monkeypatch.setattr(app, '_prune_retained_runtime_copies_quietly',
                        lambda: runs.append(threading.get_ident()))
    window._prune_retained_runtime_copies()
    qtbot.waitUntil(lambda: len(runs) == 1, timeout=5000)
    assert runs[0] != threading.get_ident()
    news = list(window._startup._navigator_aside._news._releases)
    assert window.close() and window._closing
    window._on_news_ready([{'tag': 'v999.0.0', 'name': 'late', 'body': 'late'}])
    window._prune_retained_runtime_copies()
    assert len(runs) == 1
    assert window._startup._navigator_aside._news._releases == news
