"""Local state-machine guards; genuine Cocoa acceptance belongs to native CI."""
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QDialog

from spacr.qt.startup_benchmark import _DistributionSmokeController as Controller

_REAL_DIALOG_EXEC = QDialog.exec


def test_cocoa_probe_cannot_be_passed_by_an_offscreen_qt_platform(monkeypatch):
    import sys

    monkeypatch.setattr(sys, 'platform', 'darwin')
    fake = SimpleNamespace(app=SimpleNamespace(platformName=lambda: 'offscreen'))
    with pytest.raises(RuntimeError, match='requires macOS Cocoa'):
        Controller._start_native_menu_check(fake)


def test_cocoa_probe_checks_the_requested_in_window_menu_when_configured(monkeypatch):
    import sys

    monkeypatch.setattr(sys, 'platform', 'darwin')
    calls = []
    fake = SimpleNamespace(app=SimpleNamespace(platformName=lambda: 'cocoa'),
        window=SimpleNamespace(menuBar=lambda: SimpleNamespace(isNativeMenuBar=lambda: False)),
        _start_window_menu_check=lambda: calls.append('actual window menu'))
    Controller._start_native_menu_check(fake)
    assert calls == ['actual window menu']


@pytest.mark.parametrize('missing', [None, 'preferences_opened', 'preferences_closed', 'quit_dispatched'])
def test_only_a_complete_native_action_sequence_can_accept_application_quit(missing):
    menu = dict(preferences_opened=True, preferences_closed=True,
                quit_dispatched=True, quit_observed=False)
    if missing:
        menu[missing] = False
    events = []
    fake = SimpleNamespace(phase='native-menu-quitting', record={'native_menu': menu},
        timer=SimpleNamespace(stop=lambda: events.append('timer stopped')),
        _write=lambda: events.append('receipt written'))
    Controller._quitting(fake)
    assert fake.record['status'] == ('failed' if missing else 'passed')
    assert menu['quit_observed'] is (missing is None)
    assert ('timer stopped' in events) is (missing is None)
    assert events[-1] == 'receipt written'


def test_a_native_action_that_does_not_open_preferences_is_rejected():
    calls = []
    fake = SimpleNamespace(phase='native-menu-opening', _native_menu=123,
        record={'native_menu': {'mode': 'system-menu-on-cocoa'}},
        _native_actions={'preferences': 2},
        _cocoa_message=lambda *args: calls.append(args),
        _pipeline_failed=lambda reason: calls.append(reason))
    Controller._invoke_native_preferences(fake)
    assert calls[0][1] == 'performActionForItemAtIndex:'
    assert 'did not open the verified dialog' in calls[1]


@pytest.fixture
def window_menu_probe(qtbot, qt_theme_applied, tmp_path):
    """Use the actual application bar; this remains a local controller test."""
    from PySide6.QtWidgets import QApplication
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1280, 720)
    window.show()
    qtbot.waitExposed(window)
    probe = SimpleNamespace(window=window, app=QApplication.instance(), record={},
        output=tmp_path / 'smoke.json', _write=lambda: None,
        _invoke_native_preferences=lambda: None)
    return probe


def test_unified_menu_witness_checks_actual_actions_and_window_buttons(window_menu_probe):
    """The witness records the real requested unified bar, with all three controls."""
    probe = window_menu_probe
    Controller._start_window_menu_check(probe)
    menu = probe.record['native_menu']
    assert menu['mode'] == 'in-window-on-cocoa'
    assert set(menu['window_controls']) == {'MinimiseWindow', 'FullScreenToggle', 'CloseWindow'}
    assert probe._window_menu_actions['preferences'] is probe.window._act_preferences
    assert probe._window_menu_actions['quit'] is probe.window._act_quit
    assert menu['preferences_opened'] is False


@pytest.mark.parametrize('fault', ['hidden_bar', 'hidden_control', 'hidden_preferences', 'wrong_role'])
def test_unified_menu_witness_rejects_missing_user_routes(window_menu_probe, fault):
    """Missing visible controls and misassigned actions must fail before dispatch."""
    from PySide6.QtGui import QAction

    probe = window_menu_probe
    if fault == 'hidden_bar':
        probe.window.menuBar().hide()
    elif fault == 'hidden_control':
        probe.window._close_button.hide()
    elif fault == 'hidden_preferences':
        probe.window._act_preferences.setVisible(False)
    else:
        probe.window._act_preferences.setMenuRole(QAction.MenuRole.NoRole)
    with pytest.raises(RuntimeError, match='not visible|not fully visible|actual preferences'):
        Controller._start_window_menu_check(probe)


def test_unified_menu_click_opens_the_real_preferences_dialog(window_menu_probe, monkeypatch):
    """Click actual menu geometry and observe Preferences; native CI proves Cocoa."""
    from PySide6.QtCore import QTimer
    from spacr.qt.preferences import _preferences_window_class

    monkeypatch.setattr(_preferences_window_class(), 'exec', _REAL_DIALOG_EXEC)

    probe = window_menu_probe
    Controller._start_window_menu_check(probe)
    observed = []

    def close_preferences():
        """Observe and close the actual modal dialog without replacing its route."""
        dialog = probe.app.activeModalWidget()
        if isinstance(dialog, _preferences_window_class()):
            observed.append(type(dialog).__name__)
            Controller._poll_native_menu_check(probe)

    timer = QTimer(probe.window)
    timer.timeout.connect(close_preferences)
    timer.start(20)
    deadline = QTimer(probe.window)
    deadline.setSingleShot(True)
    deadline.timeout.connect(lambda: probe.app.activeModalWidget().reject()
                             if probe.app.activeModalWidget() else None)
    deadline.start(10000)
    try:
        Controller._click_window_menu_action(probe, 'preferences')
    finally:
        timer.stop()
        deadline.stop()
    assert observed == ['_PreferencesWindow']
    assert probe.record['native_menu']['preferences_opened'] is True
    assert probe.phase == 'native-menu-closing'
    assert (probe.output.parent / 'native-menu-preferences.png').is_file()
