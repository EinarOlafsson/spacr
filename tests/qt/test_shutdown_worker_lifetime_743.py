"""Native worker ownership survives a deferred window close within one budget."""
import io
import json
import threading

import pytest
from PySide6.QtWidgets import QMessageBox

from spacr.qt import bridge
from spacr.qt.app import MainWindow
from spacr.qt.job_runner import JobRunner


def test_native_background_worker_defers_close_then_finished_work_allows_exit(qtbot, qt_theme_applied, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    window = MainWindow()
    qtbot.addWidget(window)
    runner = JobRunner(window, app_key='graph_builder', threaded=True)
    answers = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: answers.append(args[2]))
    original_retire = bridge.retire_live_work
    monkeypatch.setattr(bridge, 'retire_live_work', lambda timeout_ms: original_retire(min(timeout_ms, 30)))

    def blocked():
        entered.set()
        assert release.wait(10)

    try:
        runner.submit(blocked)
        qtbot.waitUntil(entered.is_set, timeout=5000)
        assert window.close() is False
        assert window._closing is False
        assert bridge.parked_thread_count() >= 1
        assert len(answers) == 1 and 'No worker was force-terminated' in answers[0]
        assert not window._screens
        from spacr import crashreport

        capture = io.StringIO()
        monkeypatch.setattr(crashreport, '_NATIVE_CAPTURE_HANDLE', capture)
        detail = bridge._record_shutdown_workers('wait expired')
        assert 'shutdown_cancelled=on' in detail
        assert 'shutdown_module=spacr.qt.job_runner' in detail
        assert str(__file__) not in capture.getvalue()
        contexts = [json.loads(line.split('=== spaCR context ', 1)[1].rsplit(' ===', 1)[0])
                    for line in capture.getvalue().splitlines()]
        note, _screen = crashreport._native_crash_note({'known': False}, contexts, [])
        assert 'wait expired' in note and 'spacr.qt.job_runner' in note
        assert 'cancelled on' in note
        release.set()
        qtbot.waitUntil(lambda: bridge.prune_parked_threads() == 0, timeout=5000)
        assert window.close()
        assert window._closing is True
    finally:
        release.set()
        runner.shutdown(1000)
        bridge.prune_parked_threads()


def test_import_and_parking_share_one_exit_hook(monkeypatch):
    registered = []
    monkeypatch.setattr(bridge.atexit, 'register', registered.append)
    monkeypatch.setattr(bridge, '_PARKED_EXIT_HOOK_INSTALLED', False)
    bridge._install_parked_exit_hook()
    bridge._install_parked_exit_hook()
    assert registered == [bridge._retire_live_work_at_exit]


def test_interrupted_exit_wait_records_phase_before_propagating_user_interrupt(monkeypatch):
    recorded = []

    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(bridge, 'wait_for_parked_threads', interrupt)
    monkeypatch.setattr(bridge, '_record_shutdown_workers', recorded.append)
    with pytest.raises(KeyboardInterrupt):
        bridge._drain_parked_threads_at_exit()
    assert recorded == ['wait interrupted']


def test_shutdown_native_context_rejects_paths_and_personal_identifiers(monkeypatch):
    from spacr import crashreport

    monkeypatch.setattr(crashreport, '_identity_scrub', lambda value: value.replace('privateperson', '<USER>'))
    values = crashreport._native_context_values({
        'shutdown_module': '/home/privateperson/private experiment',
        'shutdown_function': 'privateperson', 'shutdown_worker': 1,
        'shutdown_phase': 'wait interrupted', 'shutdown_line': 12,
        'shutdown_cancelled': True, 'shutdown_interruption': False,
        'settings': {'password': 'private'}, 'source': 'private.csv'})
    assert values == {'shutdown_phase': 'wait interrupted', 'shutdown_worker': '1',
                      'shutdown_line': '12', 'shutdown_cancelled': 'on',
                      'shutdown_interruption': 'off'}
