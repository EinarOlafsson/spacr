"""Observe Qt-delivered completion without requiring a surviving worker."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[3]),
                str(Path(__file__).resolve().parents[1])]

import pytest
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from capture_plaque import observe_plaque_run
from capture_acceptance import assess_pipeline
from spacr.qt.i18n import tr
from spacr.qt.widgets.console_panel import ConsolePanel


@pytest.fixture
def screen():
    app = QApplication.instance() or QApplication([])
    console = ConsolePanel('Plaque', follow_log=False, chat=False)
    button = QPushButton('Run')
    state = SimpleNamespace(_console=console, _btn_run=button,
        _entry_import=None, running=False)
    state._worker_thread_is_running = lambda: state.running
    yield app, state
    console.close()
    button.close()
    console.deleteLater()
    button.deleteLater()
    app.processEvents()


def click(state):
    QTest.mouseClick(state._btn_run, Qt.LeftButton)


def settle(seconds):
    QTest.qWait(max(1, int(seconds * 1000)))


@pytest.mark.parametrize('immediate', [False, True])
@pytest.mark.parametrize('ok', [False, True])
def test_real_console_after_deferred_or_already_finished_run(screen, immediate, ok):
    _, state = screen
    # An older successful run must not supply this run's outcome.
    state._console.append_notice('✓ Finished\n')
    state._run_started_at = 1

    def finish():
        if not ok:
            state._console.append_error('actual worker traceback')
        state._console.append_notice('✓ Finished\n' if ok else
                                    '✗ Failed — see traceback above\n')
        state.running = False
        state._worker = None  # Current AppScreen clears this reference.
        state._btn_run.setEnabled(True)

    def begin():
        state._entry_import = None
        state._run_started_at = 2
        state.running = True
        state._worker = object()
        state._console.append_stdout('actual pipeline output\n')
        if immediate:
            finish()
        else:
            QTimer.singleShot(20, finish)

    def requested():
        state._btn_run.setEnabled(False)
        state._entry_import = object()
        if immediate:
            begin()  # Completes and clears worker before click returns.
        else:
            assert not hasattr(state, '_worker')
            QTimer.singleShot(20, begin)

    state._btn_run.clicked.connect(requested)
    outcome = observe_plaque_run(state, lambda: click(state), settle, 2)
    assert state._worker is None
    assert outcome['finished'] and outcome['ok'] is ok
    assert bool(outcome['errors']) is not ok
    assert assess_pipeline(outcome, [], 0, requires_figure=False)['accepted'] is ok


@pytest.mark.parametrize('mode', ['refused', 'no_terminal', 'still_running', 'stopped', 'replaced'])
def test_idle_or_stale_notices_cannot_establish_new_success(screen, mode):
    _, state = screen
    state._console.append_notice('✓ Finished\n')
    state._run_started_at = 1

    def requested():
        if mode == 'refused':
            return  # Validation/project/confirmation refusal did not run.
        state._run_started_at = 2
        if mode == 'still_running':
            state._console.append_notice('✓ Finished\n')
            state.running = True
        elif mode == 'stopped':
            state._console.append_notice('■ Stopped safely at a field, trial, or job boundary\n')
        elif mode == 'replaced':
            block = next(state._console._pipeline_console_blocks())[0]
            block.setPlainText('replacement\n' + tr('✓ Finished'))

    state._btn_run.clicked.connect(requested)
    with pytest.raises(TimeoutError, match='new completed run'):
        observe_plaque_run(state, lambda: click(state), settle, .15)


def test_terminal_notice_waits_for_thread_cleanup(screen):
    _, state = screen
    def requested():
        state._run_started_at = 2
        state.running = True
        state._console.append_notice('✓ Finished\n')
        QTimer.singleShot(30, lambda: setattr(state, 'running', False))
    state._btn_run.clicked.connect(requested)
    outcome = observe_plaque_run(state, lambda: click(state), settle, 2)
    assert outcome['ok'] and not state.running
