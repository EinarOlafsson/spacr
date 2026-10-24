"""Real Classify import and Qt completion edges, without training a model."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[3]),
                str(Path(__file__).resolve().parents[1])]

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from capture_classify_ml import bounded_settings, run_ml_pipeline
from capture_acceptance import run_recorded_pipeline
from capture_settings import import_capture_settings
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.widgets.console_panel import ConsolePanel


def test_real_classify_import_applies_nested_class_editor_without_running_model(tmp_path):
    app = QApplication.instance() or QApplication([])
    screen = AppScreen('classify_merged')
    screen.resize(1500, 1000)
    screen.show()
    QTest.qWait(100)
    settings = bounded_settings()
    try:
        assert type(screen._settings_model._widgets['classes']).__name__ == 'ClassEditorWidget'
        current = screen._settings_model.collect()
        assert not {'classifier_family', 'dataset_mode'} & set(screen._form_shaping_keys())
        assert not screen._bulk_apply_changes_form_shape(settings, current)
        proof = import_capture_settings(app, screen, tmp_path, settings, None,
            lambda seconds: QTest.qWait(max(1, int(seconds * 1000))),
            filename='classify_ml_settings.csv')
        actual = screen._settings_model.collect()
        assert all(actual[key] == value for key, value in settings.items())
        assert actual['classes'] == settings['classes']
        assert proof['actual_file_picker'] and Path(proof['path']).name == 'classify_ml_settings.csv'
        assert 'Loaded ' in screen._console.as_text()
        assert not screen._worker_thread_is_running()
        assert not hasattr(screen, '_run_started_at')
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()


@pytest.mark.parametrize('mode', ['fast_ok', 'fast_failed', 'deferred_ok', 'deferred_failed', 'refused', 'stale'])
def test_ml_run_observer_retains_actual_outcomes_before_any_finished_scene(mode):
    app = QApplication.instance() or QApplication([])
    console = ConsolePanel('Classify', follow_log=False, chat=False)
    run, stop = QPushButton('Run'), QPushButton('Stop')
    screen = SimpleNamespace(_console=console, _btn_run=run, _btn_stop=stop,
        _run_started_at=1, _entry_import=None, running=False)
    screen._worker_thread_is_running = lambda: screen.running
    console.append_notice('✓ Finished\n')
    events = []
    capture = lambda name: events.append(name)

    def finished():
        screen.running = False
        screen._worker = None
        run.setEnabled(True)
        if mode.endswith('failed'):
            console.append_error('actual new ML error')
            console.append_notice('✗ Failed — see traceback above\n')
        else:
            console.append_notice('✓ Finished\n')

    def begin():
        screen._entry_import = None
        screen._run_started_at = 2
        screen._worker = object()
        screen.running = True
        console.append_stdout('new ML pipeline\n')
        QTimer.singleShot(1150, finished)

    def requested():
        if mode == 'refused':
            return
        if mode == 'stale':
            console.append_notice('✓ Finished\n')
            return
        run.setEnabled(False)
        if mode.startswith('deferred'):
            screen._entry_import = object()
            QTimer.singleShot(20, begin)
        else:
            screen._run_started_at = 2
            finished()
    run.clicked.connect(requested)
    try:
        settle = lambda seconds: QTest.qWait(max(1, int(seconds * 1000)))
        if mode in ('refused', 'stale'):
            with pytest.raises(TimeoutError, match='new completed run'):
                run_ml_pipeline(screen, capture, settle, 1.1)
        else:
            outcome = run_ml_pipeline(screen, capture, settle, 3)
            assert outcome['finished'] and outcome['ok'] is mode.endswith('_ok')
            assert bool(outcome['errors']) is mode.endswith('failed')
            assert screen._worker is None and not screen.running
        # This helper never starts the common 23_batch_finished scene before
        # observing completion; the normal capture_refresh tail owns that scene.
        assert events == ['21_batch_running']
    finally:
        for widget in (console, run, stop):
            widget.close()
            widget.deleteLater()
        app.processEvents()


@pytest.mark.parametrize('module', ['mask', 'map_barcodes', 'regression', 'umap'])
@pytest.mark.parametrize('batch_clip,running_clip', [(True, False), (False, True)])
def test_shared_mapping_observer_preserves_run_click_and_recording_boundaries(batch_clip, running_clip, module):
    app = QApplication.instance() or QApplication([])
    console = ConsolePanel(module, follow_log=False, chat=False)
    run, stop = QPushButton('Run'), QPushButton('Stop')
    screen = SimpleNamespace(app_key=module, _console=console, _btn_run=run, _btn_stop=stop,
        _entry_import=None, running=False)
    screen._worker_thread_is_running = lambda: screen.running
    events = []
    class Capture:
        def __call__(self, name):
            events.append(name)
        def end(self):
            events.append('end recording')
    capture = Capture()
    def requested():
        events.append('actual Run click')
        screen._run_started_at = 1
        screen._worker = None  # This real Qt click completes before returning.
        console.append_notice('✓ Finished\n')
    run.clicked.connect(requested)
    try:
        outcome = run_recorded_pipeline(screen, capture,
            lambda seconds: QTest.qWait(max(1, int(seconds * 1000))), 5,
            batch_clip=batch_clip, running_clip=running_clip)
        assert outcome['finished'] and outcome['ok']
        assert events == (['actual Run click', 'end recording', '21_batch_running']
                          if batch_clip else ['actual Run click', 'end recording'])
        assert '23_batch_finished' not in events
    finally:
        for widget in (console, run, stop):
            widget.close()
            widget.deleteLater()
        app.processEvents()
