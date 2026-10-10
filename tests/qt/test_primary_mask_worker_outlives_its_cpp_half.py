"""A primary-mask worker's wrapper must still exist when Qt deletes its C++ half.

Item 43, hosted coverage shards 0 and 7: a pytest-xdist worker died with
SIGSEGV inside ``Shiboken::BindingManager::unregisterWrapper`` called from
``QThread::event`` (a deferred delete) while a Make Masks test waited for a
primary-mask snapshot. The deleted object was a ``_SourceWorker``.

Shiboken's ``Object::destroy`` first removes the object from its Qt parent,
which drops the parent's reference, and then calls ``releaseWrapper(self)``
whenever any wrapper is registered at that C++ address. If the parent held
the last reference, ``self`` is already freed there; a stale registration at
a reused address then makes the call a use-after-free. Whether that stale
entry exists depends on everything the process did before, which is why only
loaded hosted shards crashed.

The address reuse cannot be arranged from a test, but the precondition can be
checked exactly: when ``destroyed`` fires, the wrapper is either alive (safe)
or already freed (the crash window). Both ways a worker is deleted are covered.
"""
from __future__ import annotations

import threading
import weakref

import numpy as np
import pytest

pytest.importorskip('PySide6')
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from spacr.qt.widgets import primary_mask_selector as module
from spacr.tiff_io import write_tiff

pytestmark = pytest.mark.qt


def _watch(worker):
    """Record, at C++ destruction, whether the Python wrapper still exists."""
    reference = weakref.ref(worker)
    seen = []
    worker.destroyed.connect(lambda *_args: seen.append(reference() is not None))
    return seen


def _flush_deletes():
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_a_finished_load_deletes_its_worker_while_the_wrapper_is_held(qtbot, tmp_path):
    selector = module.PrimaryMaskSelector()
    qtbot.addWidget(selector)
    path = tmp_path / 'primary.tif'
    write_tiff(path, np.full((10, 10), 7, dtype=np.uint16))
    selector.bind_field(tmp_path / 'field.tif', (10, 10), tmp_path / 'masks/field.tif')
    selector.path.setText(str(path))
    selector._source_changed()
    seen = _watch(selector._worker)
    qtbot.waitUntil(lambda: selector.snapshot is not None, timeout=10000)
    _flush_deletes()
    assert seen == [True]
    selector.shutdown()


def test_a_drained_worker_deleted_with_its_selector_keeps_its_wrapper(
        qtbot, tmp_path, monkeypatch):
    entered, finish = threading.Event(), threading.Event()

    def blocked(**_request):
        entered.set()
        assert finish.wait(5)
        raise OSError('source disconnected')

    monkeypatch.setattr(module, 'read_primary_source', blocked)
    selector = module.PrimaryMaskSelector()
    selector.path.setText(str(tmp_path))
    selector.bind_field(tmp_path / 'one.tif', (10, 10), tmp_path / 'mask.tif')
    qtbot.waitUntil(entered.is_set)
    seen = _watch(selector._worker)
    finish.set()
    selector.shutdown()
    selector.deleteLater()
    del selector
    _flush_deletes()
    assert seen == [True]


def test_retired_wrappers_are_released_once_their_cpp_half_is_gone(qtbot, tmp_path):
    selector = module.PrimaryMaskSelector()
    qtbot.addWidget(selector)
    path = tmp_path / 'primary.tif'
    write_tiff(path, np.full((10, 10), 7, dtype=np.uint16))
    selector.bind_field(tmp_path / 'field.tif', (10, 10), tmp_path / 'masks/field.tif')
    selector.path.setText(str(path))
    for _round in range(3):
        selector._source_changed()
        qtbot.waitUntil(lambda: selector.snapshot is not None, timeout=10000)
        _flush_deletes()
    assert sum(isValid(kept) for kept in module._RETIRED) <= 1
    assert len(module._RETIRED) <= 1
    selector.shutdown()
