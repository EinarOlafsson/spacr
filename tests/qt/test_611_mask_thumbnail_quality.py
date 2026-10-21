"""Source detail, live refresh and immutable mask data for item 611."""
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QPoint

from spacr import channel_sorting as cs
from spacr.qt import mask_thumbnail_quality as quality
from spacr.qt.widgets.channel_sort_dialog import _ThumbWorker, _grey_icon
from spacr.qt.widgets.organize_for_measure import OrganizeForMeasureDialog, _Column


@pytest.fixture(autouse=True)
def isolated_quality():
    previous = quality.get_quality()
    quality.set_quality("low")
    yield
    quality.set_quality(previous)


@pytest.fixture
def source(tmp_path):
    image = np.zeros((1024, 2048), np.uint16)
    image[:, ::2] = 10000
    image[100:300, 400:800] += 20000
    mask = np.zeros_like(image)
    mask[100:300, 400:800] = 321
    tifffile.imwrite(tmp_path / "field.tif", image)
    (tmp_path / "masks").mkdir()
    tifffile.imwrite(tmp_path / "masks/field.tif", mask)
    return tmp_path, image, mask


def test_quality_reads_original_detail_and_preserves_labels(qapp, source):
    folder, original, labels = source
    before = (folder / "masks/field.tif").read_bytes()
    results = []
    for name, side in quality.QUALITY_SIZES.items():
        quality.set_quality(name)
        worker = _ThumbWorker(str(folder), ["field.tif"])
        got = []
        worker.ready.connect(lambda _name, image, mask: got.append((image, mask)))
        worker.run()
        image, mask = got[0]
        assert image.shape == mask.shape == (side // 2, side)
        assert set(np.unique(mask)) <= {0, 255}
        # Both sample precisely the same positions, preserving mask/image alignment.
        step = 2048 // side
        assert np.array_equal(mask > 0, labels[::step, ::step] > 0)
        results.append(image)
    assert np.array_equal(results[0], cs.thumbnail(str(folder / "field.tif"), 64))
    assert np.array_equal(tifffile.imread(folder / "field.tif"), original)
    assert (folder / "masks/field.tif").read_bytes() == before


def test_quality_never_upsamples_small_source(qapp, tmp_path):
    path = tmp_path / "small.tif"
    tifffile.imwrite(path, np.arange(200, dtype=np.uint16).reshape(10, 20))
    for side in quality.QUALITY_SIZES.values():
        assert cs.thumbnail(str(path), side).shape == (10, 20)


def test_quality_persists_and_synchronizes_open_controls(qtbot):
    from spacr.qt.prefs import _s
    _s().remove(quality.QUALITY_KEY)
    assert quality.get_quality() == "low"
    first, second = quality.quality_combo(), quality.quality_combo()
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    first.setCurrentIndex(first.findData("high"))
    assert second.currentData() == quality.get_quality() == "high"
    third = quality.quality_combo()
    qtbot.addWidget(third)
    assert third.currentData() == "high"
    _s().setValue(quality.QUALITY_KEY, "unknown")
    assert quality.get_quality() == "low"


def test_live_quality_and_file_edits_refresh_without_resizing(qtbot, source):
    folder, _image, _mask = source
    dialog = OrganizeForMeasureDialog()
    qtbot.addWidget(dialog)
    dialog.columns = [_Column("channel"), _Column("mask", "cell")]
    image = str(folder / "field.tif")
    mask = str(folder / "masks/field.tif")
    dialog.rows = [[image, mask]]
    dialog.row_keys = [None]
    dialog._set_view("image", remember=False)
    dialog.show()
    qtbot.waitUntil(lambda: len(dialog.delegate.pixmaps) == 2)
    geometry = (dialog.table.rowHeight(0), dialog.table.columnWidth(0))
    before = dialog.delegate.pixmaps[image].cacheKey()
    quality.set_quality("high")
    qtbot.waitUntil(lambda: image in dialog.delegate.pixmaps
                   and dialog.delegate.pixmaps[image].cacheKey() != before)
    assert geometry == (dialog.table.rowHeight(0), dialog.table.columnWidth(0))
    qtbot.waitUntil(lambda: mask in dialog.delegate.pixmaps)
    old = dialog.delegate.pixmaps[mask].cacheKey()
    tifffile.imwrite(mask, np.ones((1024, 2048), np.uint16) * 712)
    qtbot.waitUntil(lambda: mask in dialog.delegate.pixmaps
                   and dialog.delegate.pixmaps[mask].cacheKey() != old, timeout=5000)
    assert len(dialog._thumb_workers) <= 1
    dialog.close()



def test_fast_quality_changes_keep_one_worker_and_reject_stale_results(
        qtbot, monkeypatch, source):
    import threading

    folder, _image, _mask = source
    original = cs.thumbnail
    started, release = threading.Event(), threading.Event()
    calls = []

    def slow_thumbnail(path, size=96):
        calls.append(size)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        return original(path, size)

    monkeypatch.setattr(cs, "thumbnail", slow_thumbnail)
    dialog = OrganizeForMeasureDialog()
    qtbot.addWidget(dialog)
    path = str(folder / "field.tif")
    dialog.columns = [_Column("channel")]
    dialog.rows = [[path]]
    dialog.row_keys = [None]
    dialog._set_view("image", remember=False)
    try:
        qtbot.waitUntil(started.is_set)
        quality.set_quality("medium")
        quality.set_quality("high")
        assert len(dialog._thumb_workers) == 1
        assert not dialog.delegate.pixmaps
    finally:
        release.set()
    qtbot.waitUntil(lambda: path in dialog.delegate.pixmaps)
    assert calls == [64, 1024]
    assert dialog.delegate.pixmaps[path].width() > 64
    dialog.close()


def test_thumbnail_cache_has_finite_budget(qtbot):
    dialog = OrganizeForMeasureDialog()
    qtbot.addWidget(dialog)
    image = np.ones((32, 32), np.uint8)
    for index in range(520):
        dialog._take_thumbnail(f"missing-{index}.tif", image, None)
    assert len(dialog.delegate.pixmaps) == 512
    assert "missing-0.tif" not in dialog.delegate.pixmaps
    assert "missing-519.tif" in dialog.delegate.pixmaps
    dialog.close()


def test_thumbnail_loading_is_limited_to_visible_rows(qtbot, tmp_path):
    for index in range(100):
        tifffile.imwrite(tmp_path / f"{index}.tif", np.ones((16, 32), np.uint16))
    dialog = OrganizeForMeasureDialog()
    qtbot.addWidget(dialog)
    dialog.columns = [_Column("channel")]
    dialog.rows = [[str(tmp_path / f"{index}.tif")] for index in range(100)]
    dialog.row_keys = [None] * 100
    dialog.show()
    dialog._set_view("image", remember=False)
    qtbot.waitUntil(lambda: bool(dialog.delegate.pixmaps))
    assert len(dialog.delegate.pixmaps) < 100
    last = dialog.rows[-1][0]
    assert last not in dialog.delegate.pixmaps
    dialog.table.scrollToBottom()
    qtbot.waitUntil(lambda: last in dialog.delegate.pixmaps)
    dialog.close()


@pytest.fixture
def blocked_thumbnail_dialog(qtbot, monkeypatch, source):
    import threading

    from PySide6.QtCore import QCoreApplication, QEvent, QThread
    from shiboken6 import isValid

    started, release = threading.Event(), threading.Event()

    def blocked_thumbnail(path, size=96):
        started.set()
        release.wait(5)
        return np.ones((8, 8), np.uint8)

    monkeypatch.setattr(cs, "thumbnail", blocked_thumbnail)
    folder, _image, _mask = source
    dialog = OrganizeForMeasureDialog()
    dialog.columns = [_Column("channel")]
    dialog.rows = [[str(folder / "field.tif")]]
    dialog.row_keys = [None]
    worker = None
    try:
        dialog._set_view("image", remember=False)
        qtbot.waitUntil(started.is_set)
        worker = dialog._thumb_workers[0]
        yield dialog, worker, release
    finally:
        release.set()
        if worker is not None and isValid(worker):
            worker.stop()
            assert QThread.wait(worker, 3000)
        if isValid(dialog):
            dialog._stop_thumbs()
            dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        from spacr.qt.bridge import prune_parked_threads
        prune_parked_threads()


def test_hidden_dialog_close_stops_thumbnail_worker(
        blocked_thumbnail_dialog, monkeypatch):
    dialog, worker, release = blocked_thumbnail_dialog
    original_stop = worker.stop
    stopped = []

    def stop():
        stopped.append(True)
        original_stop()
        release.set()

    monkeypatch.setattr(worker, "stop", stop)
    assert not dialog.isVisible()
    dialog.close()
    assert dialog._thumb_closed
    assert stopped == [True]
    assert not worker.isRunning()


def test_slow_thumbnail_worker_survives_dialog_deletion(
        blocked_thumbnail_dialog, monkeypatch):
    from PySide6.QtCore import QCoreApplication, QEvent
    from shiboken6 import isValid

    from spacr.qt import bridge

    dialog, worker, _release = blocked_thumbnail_dialog
    waits = []

    def timed_out_wait(timeout):
        waits.append(timeout)
        return False

    monkeypatch.setattr(worker, "wait", timed_out_wait)
    dialog.close()
    assert dialog._thumb_closed
    dialog._stop_thumbs()
    assert waits == [5000]
    assert worker.isRunning()
    assert worker.parent() is None
    assert any(thread is worker for thread, _owner in bridge._PARKED_THREADS)
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not isValid(dialog)
    assert isValid(worker)
    assert worker.isRunning()
