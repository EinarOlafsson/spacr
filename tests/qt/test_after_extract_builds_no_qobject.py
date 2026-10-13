"""``after_extract`` runs on the download thread and must construct no QObject.

Reported 2026-09-01, downloading the Measure test set::

    _MeasureExampleWorker was CONSTRUCTED on 'Dummy-6', so it lives there and
    every later touch from the GUI thread is illegal.
      hf_download.py:891 in after_extract
      _MeasureExampleWorker(dest)._expand_arrays(Path(dest) / "merged")

`thread_guard` was right. The worker was built purely to borrow a helper that
never read ``self``, so there was never an object to need. These tests pin
both halves: the array expansion still works, and reaching it constructs
nothing that has thread affinity.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest


def test_expanding_arrays_needs_no_worker(tmp_path):
    """The helper is a module function and does the whole job on its own."""
    from spacr.qt.hf_download import expand_measure_arrays

    merged = tmp_path / "merged"
    merged.mkdir()
    np.savez_compressed(merged / "field.npz", image=np.arange(6).reshape(2, 3))

    expand_measure_arrays(merged)

    assert (merged / "field.npy").is_file()
    assert not (merged / "field.npz").exists(), (
        "the archive is removed: keeping both doubles the disk cost")
    assert np.load(merged / "field.npy").tolist() == [[0, 1, 2], [3, 4, 5]]


def test_after_extract_constructs_no_qobject_on_the_worker_thread(tmp_path):
    """The reported defect, driven on a real non-GUI thread.

    ``after_extract`` is called by ``run`` on the download thread. Any QObject
    built there acquires that thread's affinity, which is what the guard
    reported. This runs it on a genuine ``threading.Thread`` and fails if a
    QObject is constructed at all.
    """
    from PySide6.QtCore import QObject

    from spacr.qt.hf_download import _MeasureTarWorker

    merged = tmp_path / "merged"
    merged.mkdir()
    np.savez_compressed(merged / "field.npz", image=np.zeros((2, 2)))

    built = []
    original = QObject.__init__

    def spy(self, *a, **k):
        built.append(type(self).__name__)
        return original(self, *a, **k)

    worker = _MeasureTarWorker.__new__(_MeasureTarWorker)
    errors = []

    def run_it():
        try:
            QObject.__init__ = spy
            try:
                worker.after_extract(str(tmp_path))
            finally:
                QObject.__init__ = original
        except Exception as exc:                             # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=run_it, name="Dummy-test")
    thread.start()
    thread.join(timeout=30)

    assert not errors, errors
    assert not built, (
        f"after_extract constructed {built} on the download thread; "
        "that is the affinity bug thread_guard reported")
    assert (merged / "field.npy").is_file(), "the work still has to happen"


def test_the_guard_would_have_caught_the_old_shape(tmp_path):
    """Not vacuous: constructing the worker really does trip the spy.

    Without this, the test above would pass just as happily if
    ``after_extract`` did nothing at all.
    """
    from PySide6.QtCore import QObject

    from spacr.qt.hf_download import _MeasureExampleWorker

    built = []
    original = QObject.__init__

    def spy(self, *a, **k):
        built.append(type(self).__name__)
        return original(self, *a, **k)

    QObject.__init__ = spy
    try:
        _MeasureExampleWorker(tmp_path)
    finally:
        QObject.__init__ = original

    assert "_MeasureExampleWorker" in built
