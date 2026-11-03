"""Closing a window never walks the heap while a background import runs.

``spacr.qt.bridge._loose_qthreads`` finds stray QThreads through
``gc.get_objects()``. That list also holds the tuples other threads are
still building, and a resize of one then raises ``SystemError: bad argument
to internal function``. Qt shard 3 of CI run 41607985886 hit it inside the
Mask screen's background ``import spacr.core``: torch's device-backend entry
points failed half way through ``import torch``, torch refuses a second
start in one process, and two later tests in the same worker raised
"`torch` can only be initialized once per process" on ``import spacr.utils``.
The test that closed its window was
``tests/qt/test_partial_field_presentation_cache_20261014.py::
test_the_real_mask_backdrop_uses_the_compatible_native_cache``.
"""
from __future__ import annotations

import logging
import threading

import pytest

pytest.importorskip("PySide6")

from spacr.qt import bridge                                          # noqa: E402

pytestmark = pytest.mark.qt


class _Importer:
    """A thread registered as a background importer until it is released."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        with bridge._importing_in_the_background():
            self.entered.set()
            self.release.wait(30)

    def __enter__(self):
        self.thread.start()
        assert self.entered.wait(10)
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.thread.join(10)
        assert not self.thread.is_alive()


def _importer_registered() -> bool:
    with bridge._BACKGROUND_IMPORTERS_LOCK:
        return bool(bridge._BACKGROUND_IMPORTERS)


def test_the_heap_walk_waits_for_the_background_import_to_end(monkeypatch):
    walks = []

    def walk():
        walks.append(_importer_registered())
        return []

    monkeypatch.setattr(bridge, "_loose_qthreads", walk)
    with _Importer() as importer:
        threading.Timer(0.2, importer.release.set).start()
        assert bridge._stop_loose_qthreads(10000) == 0
    assert walks == [False]


def test_an_import_outlasting_the_budget_leaves_the_heap_unwalked(monkeypatch, caplog):
    walks = []
    monkeypatch.setattr(bridge, "_loose_qthreads", lambda: walks.append(1) or [])
    with _Importer():
        with caplog.at_level(logging.WARNING, logger=bridge.LOG.name):
            assert bridge._stop_loose_qthreads(100) == 0
        assert _importer_registered()
    assert walks == []
    assert any("background import is still running" in record.getMessage()
               for record in caplog.records)


def test_without_a_background_import_the_heap_is_walked_at_once(monkeypatch):
    walks = []
    monkeypatch.setattr(bridge, "_loose_qthreads", lambda: walks.append(1) or [])
    assert not _importer_registered()
    assert bridge._stop_loose_qthreads(0) == 0
    assert walks == [1]


def test_the_importing_thread_itself_does_not_wait_for_itself(monkeypatch):
    walks = []
    monkeypatch.setattr(bridge, "_loose_qthreads", lambda: walks.append(1) or [])
    with bridge._importing_in_the_background():
        assert bridge._stop_loose_qthreads(0) == 0
    assert walks == [1]
