"""Real recipe previews stay responsive and retire obsolete workers safely."""
import threading

import pandas as pd
import pytest
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtWidgets import QDialog, QPushButton

from spacr.condition_annotations import new_definition, source_context
from spacr.qt import bridge
from spacr.qt.widgets import condition_annotation_dialog as module

PATTERN = r'^(?P<cell_type>[^_]+)_(?P<replicate>rep\d+)_(?P<timepoint>\d+h)_(?P<drug>.+)\.tif$'
OUTPUTS = ['cell_type', 'replicate', 'timepoint', 'drug', 'treatment', 'condition']


def _fixture():
    frame = pd.DataFrame({
        'original_filename': ['HeLa_rep2_24h_DMSO.tif', 'U2OS_rep12_48h_Drug-A.tif',
                              'HeLa_rep2_24h_Drug.B.tif', 'unmatched.tif', None],
        'measurement': [3, 8, 5, 1, 2],
    }, index=[7, 7, 2, 2, 0])
    source = source_context()
    definition = new_definition(frame, source)
    definition.pop('column')
    definition.pop('conditions')
    definition.update(version=3, columns=[
        dict(column=name, kind='extract', metadata_column='original_filename',
             pattern=PATTERN, group=name) for name in OUTPUTS[:4]
    ] + [dict(column='treatment', kind='rules', conditions=[
        dict(name='control', criteria=[dict(metadata_column='drug', operator='equals', value='DMSO')]),
        dict(name='treated', match='all', criteria=[
            dict(metadata_column='drug', operator='starts_with', value='Drug'),
            dict(metadata_column='drug', operator='not_contains', value='.B'),
        ]),
    ]), dict(column='condition', kind='template', parts=[
        dict(kind='column', column='cell_type'), dict(kind='text', text=' / '),
        dict(kind='column', column='treatment'), dict(kind='text', text='_'),
        dict(kind='column', column='replicate'), dict(kind='text', text='_'),
        dict(kind='column', column='timepoint'),
    ])])
    return frame, source, definition


def _dialog(qtbot):
    frame, source, definition = _fixture()
    dialog = module.ConditionAnnotationDialog(frame, source, definition=definition, threaded=True)
    qtbot.addWidget(dialog)
    dialog.show()
    return dialog, frame


def _assert_outputs(dialog, original):
    result = dialog.result_frame
    assert result is not None, dialog.status.text()
    assert list(result.columns) == list(original.columns) + OUTPUTS
    expected = [
        ['HeLa', 'U2OS', 'HeLa', '', ''],
        ['rep2', 'rep12', 'rep2', '', ''],
        ['24h', '48h', '24h', '', ''],
        ['DMSO', 'Drug-A', 'Drug.B', '', ''],
        ['control', 'treated', '', '', ''],
        ['HeLa / control_rep2_24h', 'U2OS / treated_rep12_48h', '', '', ''],
    ]
    for name, values in zip(OUTPUTS, expected):
        assert result[name].fillna('').tolist() == values
    pd.testing.assert_frame_equal(result[list(original.columns)], original)
    pd.testing.assert_frame_equal(dialog.frame, original)
    assert result.attrs['condition_annotation'] == dialog.configuration()


def test_recipe_preview_runs_off_thread_and_apply_keeps_every_output(qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    actual_preview = module.preview
    worker_threads = []

    def blocked(*args):
        worker_threads.append(QThread.currentThread())
        entered.set()
        assert release.wait(5)
        return actual_preview(*args)

    monkeypatch.setattr(module, 'preview', blocked)
    dialog, original = _dialog(qtbot)
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        assert all(thread != dialog.thread() for thread in worker_threads)
        assert not dialog.apply_button.isEnabled()
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        qtbot.waitUntil(lambda: bool(heartbeat), timeout=1000)
        assert not release.is_set()  # GUI events continued during real worker execution.
    finally:
        release.set()
    qtbot.waitUntil(dialog.apply_button.isEnabled, timeout=5000)
    _assert_outputs(dialog, original)
    threads = [thread for thread, _ in dialog._jobs._jobs.values()]
    qtbot.mouseClick(dialog.apply_button, Qt.LeftButton)
    assert dialog.result() == QDialog.Accepted
    assert dialog._jobs.active_jobs() == dialog._jobs.pending_jobs() == 0
    assert all(bridge.thread_has_stopped(thread) for thread in threads)
    _assert_outputs(dialog, original)


@pytest.mark.parametrize('obsolete_invalid', [False, True], ids=['stale-values', 'stale-error'])
def test_edit_during_recipe_preview_rejects_obsolete_values_and_errors(qtbot, monkeypatch, obsolete_invalid):
    entered, release = threading.Event(), threading.Event()
    actual_preview = module.preview

    def blocked(frame, definition, source):
        first = definition['columns'][0]
        if first['pattern'] == '[' or first['group'] == 'drug':
            entered.set()
            assert release.wait(5)
        return actual_preview(frame, definition, source)

    monkeypatch.setattr(module, 'preview', blocked)
    dialog, original = _dialog(qtbot)
    qtbot.waitUntil(dialog.apply_button.isEnabled, timeout=5000)
    try:
        if obsolete_invalid:
            dialog.extract_pattern.setText('[')
        else:
            dialog.extract_group.setCurrentText('drug')
        qtbot.mouseClick(dialog.preview_button, Qt.LeftButton)
        qtbot.waitUntil(entered.is_set, timeout=5000)
        dialog.extract_pattern.setText(PATTERN)
        dialog.extract_group.setCurrentText('cell_type')
        assert not dialog.apply_button.isEnabled()
        assert dialog.result_frame is None
        qtbot.mouseClick(dialog.preview_button, Qt.LeftButton)
        qtbot.waitUntil(dialog.apply_button.isEnabled, timeout=5000)
        _assert_outputs(dialog, original)
        status = dialog.status.text()
    finally:
        release.set()
    qtbot.waitUntil(lambda: dialog._jobs.active_jobs() == 0, timeout=5000)
    assert dialog._jobs.pending_jobs() == 0
    assert dialog.apply_button.isEnabled()
    assert dialog.status.text() == status
    _assert_outputs(dialog, original)
    qtbot.mouseClick(dialog.apply_button, Qt.LeftButton)
    assert dialog.result() == QDialog.Accepted


@pytest.mark.parametrize('action', ['cancel', 'close'])
def test_closing_active_recipe_preview_retires_workers_without_late_delivery(qtbot, monkeypatch, action):
    entered, release = threading.Event(), threading.Event()
    actual_preview = module.preview

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return actual_preview(*args)

    monkeypatch.setattr(module, 'preview', blocked)
    dialog, original = _dialog(qtbot)
    threads = []
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        threads = [thread for thread, _ in dialog._jobs._jobs.values()]
        assert threads
        # Exercise the real timeout/parking path without a three-second GUI wait.
        shutdown = dialog._jobs.shutdown
        monkeypatch.setattr(dialog._jobs, 'shutdown', lambda: shutdown(timeout_ms=10))
        if action == 'cancel':
            cancel = next(button for button in dialog.findChildren(QPushButton)
                          if button.text() == module.tr('Cancel'))
            qtbot.mouseClick(cancel, Qt.LeftButton)
        else:
            dialog.close()
        assert not dialog.isVisible()
        assert dialog.result() == QDialog.Rejected
        assert not dialog._timer.isActive()
        assert dialog._jobs.active_jobs() == dialog._jobs.pending_jobs() == 0
        assert any(not bridge.thread_has_stopped(thread) for thread in threads)
    finally:
        release.set()
        qtbot.waitUntil(lambda: all(bridge.thread_has_stopped(thread) for thread in threads), timeout=5000)
        bridge.prune_parked_threads()
    assert all(pair[0] not in threads for pair in bridge._PARKED_THREADS)
    assert dialog.result_frame is None
    assert dialog.source_model.preview_columns == {}
    assert dialog.source_model.preview_values is None
    pd.testing.assert_frame_equal(dialog.frame, original)
