"""Lifetime of the download and folder-scan thread/worker pairs."""
from __future__ import annotations

import gc
import threading
import time
import weakref

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Signal
from PySide6.QtWidgets import QWidget

from spacr.qt import hf_download as hf


class _SlowWorker(QObject):
    """A download worker that waits for a gate and fetches nothing."""

    progress = Signal(str, int, int)
    info = Signal(str)
    finished = Signal(bool, str, str, str)

    def __init__(self, dest):
        super().__init__()
        self.gate = threading.Event()
        self._cancel = False

    def cancel(self):
        self._cancel = True
        self.gate.set()

    def run(self):
        self.gate.wait(10)
        self.finished.emit(True, "d", "s", "")


def _record_threads(monkeypatch):
    """Record a weak reference to every QThread the download creates."""
    from PySide6.QtCore import QThread

    refs = []

    def make(*args):
        thread = QThread(*args)
        refs.append(weakref.ref(thread))
        return thread

    monkeypatch.setattr(hf, "QThread", make)
    return refs


def _pump(qapp, until, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        qapp.processEvents()
        if until():
            return True
        time.sleep(0.005)
    return False


def test_a_download_outlives_its_widget_and_is_freed_by_refcount(qapp, tmp_path, monkeypatch):
    """Destroying the caller mid-download neither aborts nor leaks the pair."""
    made = []
    refs = _record_threads(monkeypatch)

    def factory(dest):
        worker = _SlowWorker(dest)
        made.append(worker)
        return worker

    parent = QWidget()
    results = []
    gc.disable()
    try:
        hf.download_toxo_mito_demo(parent, tmp_path,
                                   lambda r, e: results.append((r, e)),
                                   worker_factory=factory)
        worker = made[0]
        thread_ref = refs[0]
        assert thread_ref().isRunning()
        assert thread_ref().parent() is None
        worker_ref = weakref.ref(worker)
        del worker
        made.clear()
        parent.deleteLater()
        del parent
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert _pump(qapp, lambda: not hf._LIVE_DOWNLOADS)
        assert not hf._LIVE_DOWNLOADS
        _pump(qapp, lambda: worker_ref() is None and thread_ref() is None, 2)
        assert worker_ref() is None
        assert thread_ref() is None
    finally:
        gc.enable()


def test_a_finished_download_releases_the_pair_without_gc(qapp, tmp_path, monkeypatch):
    """After completion the worker and thread die by reference count."""
    made = []
    refs = _record_threads(monkeypatch)

    def factory(dest):
        worker = _SlowWorker(dest)
        made.append(worker)
        return worker

    parent = QWidget()
    results = []
    gc.disable()
    try:
        hf.download_toxo_mito_demo(parent, tmp_path,
                                   lambda r, e: results.append((r, e)),
                                   worker_factory=factory)
        worker = made[0]
        thread_ref = refs[0]
        worker_ref = weakref.ref(worker)
        worker.gate.set()
        del worker
        made.clear()
        assert _pump(qapp, lambda: bool(results))
        assert _pump(qapp, lambda: not hf._LIVE_DOWNLOADS)
        _pump(qapp, lambda: worker_ref() is None and thread_ref() is None, 2)
        assert results and results[0][0] is not None
        assert worker_ref() is None
        assert thread_ref() is None
    finally:
        gc.enable()
        parent.deleteLater()


def test_a_finished_folder_scan_is_released_by_refcount(qapp, tmp_path):
    """The scan's worker and thread are freed without the collector."""
    from spacr.qt.screens import settings_model as sm

    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "r01c01f01p01-ch1sk1fk1fl1.tiff").write_bytes(b"0")
    field = sm._MetadataTypeField(default="opera_phenix",
                                  source_folder=lambda: str(folder),
                                  custom_regex=lambda: None)
    field.set_threaded(True)
    gc.disable()
    try:
        field.test_on_the_folder()
        thread_ref = weakref.ref(field._thread)
        worker_ref = weakref.ref(field._worker)
        assert _pump(qapp, lambda: field._thread is None)
        assert _pump(qapp, lambda: not sm._LIVE_FOLDER_SCANS)
        _pump(qapp, lambda: worker_ref() is None and thread_ref() is None, 2)
        assert not sm._LIVE_FOLDER_SCANS
        assert worker_ref() is None
        assert thread_ref() is None
    finally:
        gc.enable()
