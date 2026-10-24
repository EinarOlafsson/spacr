"""The Invasion recorder retains Run/completion scenes across Qt lifecycle edges."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[3]),
                str(Path(__file__).resolve().parents[1])]

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from capture_acceptance import assess_pipeline
from capture_invasion import run_invasion_pipeline
from spacr.qt.widgets.console_panel import ConsolePanel


@pytest.mark.parametrize('mode', ['fast_ok', 'fast_failed', 'deferred_ok', 'deferred_failed', 'refused', 'stale'])
def test_actual_qt_run_and_completion_boundaries_do_not_require_a_live_worker(mode):
    app = QApplication.instance() or QApplication([])
    console = ConsolePanel('Invasion', follow_log=False, chat=False)
    run, stop = QPushButton('Run'), QPushButton('Stop')
    screen = SimpleNamespace(_console=console, _btn_run=run, _btn_stop=stop,
        _run_started_at=1, _entry_import=None, running=False)
    screen._worker_thread_is_running = lambda: screen.running
    console.append_notice('✓ Finished\n')  # Previous run cannot satisfy this click.
    events = []
    scenes = SimpleNamespace(
        begin=lambda name: events.append(('begin', name)),
        end=lambda name, target: events.append(('end', name, target is console)))

    def finish():
        if mode.endswith('failed'):
            console.append_error('actual new pipeline error')
            console.append_notice('✗ Failed — see traceback above\n')
        else:
            console.append_notice('✓ Finished\n')
        screen.running = False
        screen._worker = None
        run.setEnabled(True)
        events.append(('finished',))

    def begin():
        screen._entry_import = None
        screen._run_started_at = 2
        screen._worker = object()
        screen.running = True
        console.append_stdout('new Invasion run\n')
        QTimer.singleShot(350, finish)

    def clicked():
        events.append(('click',))
        if mode == 'refused':
            return
        if mode == 'stale':
            console.append_notice('✓ Finished\n')
            return
        run.setEnabled(False)
        if mode.startswith('deferred'):
            assert not hasattr(screen, '_worker')
            screen._entry_import = object()
            QTimer.singleShot(20, begin)
        else:
            screen._run_started_at = 2
            finish()  # Completion and worker cleanup before click returns.
    run.clicked.connect(clicked)
    settle = lambda seconds: QTest.qWait(max(1, int(seconds * 1000)))
    try:
        if mode in ('refused', 'stale'):
            with pytest.raises(TimeoutError, match='new completed run'):
                run_invasion_pipeline(screen, scenes, settle, .45)
        else:
            outcome = run_invasion_pipeline(screen, scenes, settle, 2)
            expected_ok = mode.endswith('_ok')
            assert outcome['finished'] and outcome['ok'] is expected_ok
            assert bool(outcome['errors']) is not expected_ok
            assert assess_pipeline(outcome, [], 1)['accepted'] is expected_ok
            assert screen._worker is None
        boundary_events = [event for event in events if event[0] != 'finished']
        assert boundary_events == [('begin', '13_actual_run'), ('click',),
                                   ('end', '13_actual_run', True), ('begin', '14_actual_completion')]
        if mode.startswith('deferred'):
            assert events.index(('begin', '14_actual_completion')) < events.index(('finished',))
    finally:
        for widget in (console, run, stop):
            widget.close()
            widget.deleteLater()
        app.processEvents()
