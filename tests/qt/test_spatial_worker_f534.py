"""Spatial button jobs snapshot inputs and keep writes alive safely on close."""
import threading

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt, QThread, QTimer
from PySide6.QtWidgets import QVBoxLayout, QWidget

from spacr import ops_engine
from spacr.qt import bridge
from spacr.qt.screens.map_barcodes import _SpatialTranscriptomicsPanel


def _panel(qtbot, tmp_path, host=None):
    panel = _SpatialTranscriptomicsPanel(host, host)
    qtbot.addWidget(panel)
    panel.folder.setText(str(tmp_path / 'platform'))
    panel.masks['cell'].setText(str(tmp_path / 'mask.npy'))
    panel.db.setText(str(tmp_path / 'measurements.db'))
    panel.show()
    return panel


@pytest.fixture
def engine(monkeypatch):
    bundle = {'platform': 'xenium', 'genes': ['A', 'B'],
              'transcripts': pd.DataFrame({'gene': ['A', 'A', 'B']})}
    registered = {'image': np.zeros((4, 4)), 'xy': np.zeros((3, 2)),
                  'radius': 0, 'registration': {'method': 'fixture'}}
    monkeypatch.setattr(ops_engine, '_st_read_bundle', lambda *_a, **_kw: bundle)
    monkeypatch.setattr(ops_engine, '_st_register', lambda *_a, **_kw: registered)
    monkeypatch.setattr(ops_engine, '_st_gene_values', lambda *_a: np.ones(3))
    drawn = []
    monkeypatch.setattr(ops_engine, '_st_draw_overlay', lambda *_a, **_kw: drawn.append(QThread.currentThread()))
    return bundle, registered, drawn


def _idle(qtbot, panel):
    qtbot.waitUntil(lambda: panel._busy is None and panel._jobs.active_jobs() == 0, timeout=5000)


@pytest.mark.parametrize('fails', [False, True], ids=['obsolete-success', 'obsolete-error'])
def test_load_button_keeps_gui_alive_and_rejects_results_after_input_edit(qtbot, monkeypatch, tmp_path, engine, fails):
    panel = _panel(qtbot, tmp_path)
    entered, release = threading.Event(), threading.Event()
    read = ops_engine._st_read_bundle
    threads = []

    def blocked(*args, **kwargs):
        threads.append(QThread.currentThread())
        entered.set()
        assert release.wait(5)
        if fails:
            raise ValueError('obsolete folder')
        return read(*args, **kwargs)

    monkeypatch.setattr(ops_engine, '_st_read_bundle', blocked)
    try:
        qtbot.mouseClick(panel.load_button, Qt.LeftButton)
        qtbot.waitUntil(entered.is_set, timeout=5000)
        heartbeats = []
        QTimer.singleShot(0, lambda: heartbeats.append(True))
        qtbot.waitUntil(lambda: bool(heartbeats), timeout=1000)
        assert all(thread != panel.thread() for thread in threads)
        assert not panel.assign_button.isEnabled()
        assert panel.folder.isEnabled()
        panel.image.setText('changed-image.tif')
        panel.image.clear()  # Returning to the same key must still discard this job.
    finally:
        release.set()
    _idle(qtbot, panel)
    assert panel._bundle is panel._registered is panel._loaded_key is None
    assert not engine[2]
    assert not panel.status.text()
    monkeypatch.setattr(ops_engine, '_st_read_bundle', read)
    qtbot.mouseClick(panel.load_button, Qt.LeftButton)
    _idle(qtbot, panel)
    assert panel._bundle is engine[0]
    assert panel.gene.currentText() == 'A'
    assert engine[2] == [panel.thread()]


def test_assignment_uses_one_snapshot_and_failure_allows_retry(qtbot, monkeypatch, tmp_path, engine):
    panel = _panel(qtbot, tmp_path)
    assert panel.load()  # Existing synchronous API remains available.
    entered, release = threading.Event(), threading.Event()
    calls = []
    original_path = panel.db.text()

    def assign(request, **kwargs):
        calls.append((request, kwargs, QThread.currentThread()))
        entered.set()
        assert release.wait(5)
        if len(calls) == 1:
            raise OSError('database is read-only')
        return {'objects': {'cell': {'objects': 3}}, 'db': request['db'], 'output': str(tmp_path)}

    monkeypatch.setattr(ops_engine, '_st_run', assign)
    try:
        qtbot.mouseClick(panel.assign_button, Qt.LeftButton)
        qtbot.waitUntil(entered.is_set, timeout=5000)
        assert not panel.db.isEnabled()
        panel._start_run()  # A queued activation cannot start another write.
        assert panel.run() is None
        assert panel.load() is False
        panel.db.setText(str(tmp_path / 'next.db'))
        heartbeats = []
        QTimer.singleShot(0, lambda: heartbeats.append(True))
        qtbot.waitUntil(lambda: bool(heartbeats), timeout=1000)
    finally:
        release.set()
    _idle(qtbot, panel)
    assert len(calls) == 1 and calls[0][0]['db'] == original_path
    assert calls[0][2] != panel.thread()
    assert calls[0][1]['bundle'] is engine[0]
    assert 'database is read-only' in panel.status.text()
    assert panel.assign_button.isEnabled() and panel.db.isEnabled()
    qtbot.mouseClick(panel.assign_button, Qt.LeftButton)
    _idle(qtbot, panel)
    assert len(calls) == 2 and panel.summary['db'].endswith('next.db')


@pytest.mark.parametrize('action', ['close', 'delete'])
def test_abandoned_load_never_starts_assignment_or_delivers_to_destroyed_panel(qtbot, monkeypatch, tmp_path, engine, action):
    panel = _panel(qtbot, tmp_path)
    entered, release = threading.Event(), threading.Event()
    read = ops_engine._st_read_bundle
    writes = []

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return read(*args, **kwargs)

    monkeypatch.setattr(ops_engine, '_st_read_bundle', blocked)
    monkeypatch.setattr(ops_engine, '_st_run', lambda *_a, **_kw: writes.append(True))
    jobs = panel._jobs
    shutdown = jobs.shutdown
    monkeypatch.setattr(jobs, 'shutdown', lambda: shutdown(timeout_ms=10))
    threads = []
    try:
        qtbot.mouseClick(panel.assign_button, Qt.LeftButton)
        qtbot.waitUntil(entered.is_set, timeout=5000)
        threads = [thread for thread, _ in jobs._jobs.values()]
        if action == 'delete':
            panel.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        else:
            panel.close()
            qtbot.waitUntil(lambda: panel._closed, timeout=1000)
        assert jobs.active_jobs() == jobs.pending_jobs() == 0
    finally:
        release.set()
        qtbot.waitUntil(lambda: all(bridge.thread_has_stopped(thread) for thread in threads), timeout=5000)
        bridge.prune_parked_threads()
    assert not writes and not engine[2]
    assert all(pair[0] not in threads for pair in bridge._PARKED_THREADS)


def test_host_close_veto_preserves_job_and_accepted_close_drains(qtbot, monkeypatch, tmp_path, engine):
    class Host(QWidget):
        veto = True

        def closeEvent(self, event):  # noqa: N802
            event.ignore() if self.veto else event.accept()

    host = Host()
    qtbot.addWidget(host)
    panel = _panel(qtbot, tmp_path, host)
    QVBoxLayout(host).addWidget(panel)
    host.show()
    entered, release = threading.Event(), threading.Event()
    read = ops_engine._st_read_bundle

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return read(*args, **kwargs)

    monkeypatch.setattr(ops_engine, '_st_read_bundle', blocked)
    shutdown = panel._jobs.shutdown
    monkeypatch.setattr(panel._jobs, 'shutdown', lambda: shutdown(timeout_ms=10))
    threads = []
    try:
        qtbot.mouseClick(panel.load_button, Qt.LeftButton)
        qtbot.waitUntil(entered.is_set, timeout=5000)
        threads = [thread for thread, _ in panel._jobs._jobs.values()]
        host.hide()
        host.close()
        QCoreApplication.processEvents()
        assert not panel._closed  # Hidden+vetoed is not evidence of accepted close.
        # A hidden host accepting close without deletion has no visible transition;
        # its still-owned job may finish until definitive deletion shuts it down.
        host.show()
        host.close()
        QCoreApplication.processEvents()
        assert host.isVisible() and not panel._closed
        assert panel._jobs.pending_jobs() == 1
        host.veto = False
        host.close()
        qtbot.waitUntil(lambda: panel._closed, timeout=1000)
    finally:
        release.set()
        qtbot.waitUntil(lambda: all(bridge.thread_has_stopped(thread) for thread in threads), timeout=5000)
        bridge.prune_parked_threads()
    assert not engine[2]


def test_reopen_waits_for_parked_write_then_uses_fresh_stop_token(qtbot, monkeypatch, tmp_path, engine):
    panel = _panel(qtbot, tmp_path)
    assert panel.load()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def assign(request, **kwargs):
        calls.append(request['db'])
        entered.set()
        assert release.wait(5)
        return {'objects': {'cell': {'objects': 3}}, 'db': request['db'], 'output': str(tmp_path)}

    monkeypatch.setattr(ops_engine, '_st_run', assign)
    shutdown = panel._jobs.shutdown
    monkeypatch.setattr(panel._jobs, 'shutdown', lambda: shutdown(timeout_ms=10))
    threads = []
    old_stop = panel._stop
    try:
        qtbot.mouseClick(panel.assign_button, Qt.LeftButton)
        qtbot.waitUntil(entered.is_set, timeout=5000)
        threads = [thread for thread, _ in panel._jobs._jobs.values()]
        panel.close()
        qtbot.waitUntil(lambda: panel._closed, timeout=1000)
        panel.show()
        QCoreApplication.processEvents()
        assert panel._closed and not panel.assign_button.isEnabled()
        assert panel._stop is old_stop and old_stop.is_set()
        panel._start_run()
        assert len(calls) == 1
    finally:
        release.set()
        qtbot.waitUntil(lambda: all(bridge.thread_has_stopped(thread) for thread in threads), timeout=5000)
        bridge.prune_parked_threads()
    qtbot.waitUntil(lambda: not panel._closed, timeout=1000)
    assert panel._stop is not old_stop and not panel._stop.is_set()
    assert old_stop.is_set()
    assert panel.summary is None  # The closed generation cannot publish its result.
    qtbot.mouseClick(panel.assign_button, Qt.LeftButton)
    _idle(qtbot, panel)
    assert len(calls) == 2 and panel.summary is not None


def test_overlay_error_surfaces_and_does_not_leave_buttons_busy(qtbot, monkeypatch, tmp_path, engine):
    panel = _panel(qtbot, tmp_path)

    def fail(*args, **kwargs):
        raise ValueError('cannot render overlay')

    monkeypatch.setattr(ops_engine, '_st_draw_overlay', fail)
    qtbot.mouseClick(panel.load_button, Qt.LeftButton)
    _idle(qtbot, panel)
    assert 'cannot render overlay' in panel.status.text()
    assert panel._loaded_key is None
    assert panel.load_button.isEnabled() and panel.assign_button.isEnabled()


def test_real_xenium_button_assignment_writes_expected_counts(qtbot, monkeypatch, tmp_path):
    pytest.importorskip('pyarrow')
    from spacr.tabular import read_table
    from tests.test_spatial_transcriptomics import _masks, xenium_bundle

    panel = _panel(qtbot, tmp_path)
    folder = xenium_bundle(tmp_path / 'xenium')
    cell, _, _ = _masks()
    mask = tmp_path / 'plate1_r1_c1_f1.npy'
    np.save(mask, cell)
    panel.folder.setText(folder)
    panel.masks['cell'].setText(str(mask))
    run = ops_engine._st_run
    threads = []

    def observed(request, **kwargs):
        threads.append(QThread.currentThread())
        # AnnData is unrelated to button scheduling; retain real DB and figure exports.
        return run(dict(request, anndata=False), **kwargs)

    monkeypatch.setattr(ops_engine, '_st_run', observed)
    qtbot.mouseClick(panel.assign_button, Qt.LeftButton)
    _idle(qtbot, panel)
    assert panel.summary is not None, panel.status.text()
    assert threads and threads[0] != panel.thread()
    wide = read_table(panel.db.text(), table='cell_expression', report=None)
    assert len(wide) == 20
    assert wide.expr_GENE_FLAT.tolist() == [10] * 20
    assert sorted(wide.expr_GENE_UP.tolist()) == [1] * 10 + [25] * 10
    assert panel.summary['prcf'] == 'plate1_r1_c1_f1'
    assert panel.figure.axes
