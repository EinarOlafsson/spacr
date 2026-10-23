"""Save mask writes on the mask writer thread, so a slow disk never holds the window.

Lesson 14 lost 3.1 s of GUI time to one TIFF write on a busy disk. A save
that lands within ``_MASK_SAVE_WAIT_SECONDS`` still reports at once; a slower
one finishes in the background, and every reader of that file waits for it.
"""
from __future__ import annotations

import json
import threading

import imageio.v2 as imageio
import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QSettings, Qt

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm


@pytest.fixture
def opened(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    from spacr.qt import prefs

    monkeypatch.setattr(engine, "_MASK_SAVE_WAIT_SECONDS", 0.1)

    settings_path = tmp_path / "editor.ini"
    monkeypatch.setattr(
        prefs, "_s", lambda: QSettings(str(settings_path), QSettings.IniFormat))
    folder = tmp_path / "fields"
    folder.mkdir()
    for index in range(2):
        imageio.imwrite(folder / f"field_{index}.tif",
                        np.full((64, 64), index + 1, dtype=np.uint16))
    screen = mm.MakeMasksScreen()
    qtbot.addWidget(screen)
    assert screen._open_folder(str(folder))
    qtbot.waitUntil(lambda: screen._canvas.mask is not None)
    yield screen, folder
    mm._wait_for_mask_writes(timeout=10)
    screen._magnifier.close()
    screen.close_folded()


def _saved(folder, index):
    return engine.mask_save_path(str(folder), f"field_{index}.tif")


def _held_save(monkeypatch):
    """Make every mask save wait for the returned event, recording its thread."""
    release, threads = threading.Event(), []
    real = engine.save_mask

    def held(*args, **kwargs):
        threads.append(threading.current_thread().name)
        assert release.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "save_mask", held)
    return release, threads


def test_a_reader_timeout_leaves_the_pending_mask_write_intact(tmp_path):
    path = tmp_path / "mask.tif"
    other = tmp_path / "other.tif"
    entered, release = threading.Event(), threading.Event()
    labels = np.array([[0, 7], [7, 0]], dtype=np.uint16)
    request = {"paths": {str(path)}}

    def held_write(_request):
        entered.set()
        assert release.wait(10)
        tifffile.imwrite(path, labels)
        return str(path)

    mm._queue_mask_write(request, held_write)
    try:
        assert entered.wait(10)
        assert mm._mask_writes_pending(path)
        assert not mm._wait_for_mask_writes(path, timeout=0)
        assert mm._wait_for_mask_writes(other, timeout=0)
        assert not request["done"].is_set()
        assert not path.exists()
    finally:
        release.set()
        assert request["done"].wait(10)

    assert mm._wait_for_mask_writes(path, timeout=1)
    assert not mm._mask_writes_pending(path)
    assert request["error"] is None
    assert request["result"] == str(path)
    np.testing.assert_array_equal(tifffile.imread(path), labels)


def test_a_fast_save_is_written_by_the_writer_and_reported_at_once(opened, monkeypatch):
    screen, folder = opened
    threads = []
    real = engine.save_mask

    def recorded(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "save_mask", recorded)
    monkeypatch.setattr(engine, "_MASK_SAVE_WAIT_SECONDS", 30.0)
    screen._canvas.mask[12:20, 12:20] = 7
    screen._btn_save.click()
    assert threads == ["spacr-mask-writer"]
    assert screen._status_label.text().startswith("Saved")
    assert np.all(tifffile.imread(_saved(folder, 0))[12:20, 12:20] > 0)


def test_logged_brush_saves_keep_the_mask_and_ledger_artifact_together(
        opened, qtbot):
    screen, folder = opened
    screen.resize(1000, 800)
    screen.show()
    qtbot.waitUntil(lambda: screen._canvas.pixmap() is not None)
    qtbot.mouseClick(screen._btn_brush, Qt.LeftButton)
    path = _saved(folder, 0)
    assert screen._log.artifact == path
    assert not screen._log.edits

    for count, image_point in enumerate(((14, 14), (48, 48)), start=1):
        before = screen._canvas.mask.copy()
        point = screen._canvas._image_to_canvas(*image_point)
        assert point is not None
        qtbot.mouseClick(screen._canvas, Qt.LeftButton, pos=point)
        changed = int(np.count_nonzero(screen._canvas.mask != before))
        assert changed > 0
        assert len(screen._log.edits) == count
        assert screen._log.edits[-1].kind == "paint"
        assert screen._log.edits[-1].n_changed == changed
        expected = engine.canonical_labels(screen._canvas.mask)

        qtbot.mouseClick(screen._btn_save, Qt.LeftButton)
        assert mm._wait_for_mask_writes(path, timeout=10)
        qtbot.waitUntil(lambda: screen._status_label.text().startswith("Saved"),
                        timeout=10000)
        np.testing.assert_array_equal(tifffile.imread(path), expected)
        with open(path + ".curation.json", encoding="utf-8") as handle:
            curation_log = json.load(handle)
        assert curation_log["artifact"] == path
        assert len(curation_log["edits"]) == count
        assert [edit["kind"] for edit in curation_log["edits"]] == ["paint"] * count
        assert screen._log.artifact == path


def test_a_slow_save_returns_at_once_and_reports_when_it_lands(opened, monkeypatch, qtbot):
    screen, folder = opened
    release, threads = _held_save(monkeypatch)
    screen._canvas.mask[12:20, 12:20] = 5
    path = screen._save_mask()
    assert path == _saved(folder, 0)
    assert threads == ["spacr-mask-writer"]
    assert mm._mask_writes_pending(path)
    assert not screen._status_label.text().startswith("Saved")
    release.set()
    qtbot.waitUntil(lambda: screen._status_label.text().startswith("Saved"), timeout=10000)
    assert not mm._mask_writes_pending()
    assert np.all(tifffile.imread(path)[12:20, 12:20] > 0)


def test_coming_back_to_a_field_reads_the_mask_its_save_is_still_writing(
        opened, monkeypatch, qtbot):
    screen, folder = opened
    release, _threads = _held_save(monkeypatch)
    screen._canvas.mask[12:20, 12:20] = 9
    edited = engine.canonical_labels(screen._canvas.mask)
    assert screen._save_mask() == _saved(folder, 0)
    screen._btn_next.click()
    assert screen._current_index == 1
    assert screen._loading, "the next field must wait for the save, not read past it"
    threading.Timer(0.3, release.set).start()
    qtbot.waitUntil(lambda: not screen._loading, timeout=10000)
    screen._btn_prev.click()
    qtbot.waitUntil(lambda: not screen._loading and screen._current_index == 0, timeout=10000)
    np.testing.assert_array_equal(screen._canvas.mask, edited)


def test_closing_the_screen_waits_for_the_save_in_progress(opened, monkeypatch):
    screen, folder = opened
    release, _threads = _held_save(monkeypatch)
    screen._canvas.mask[12:20, 12:20] = 3
    screen._save_mask()
    threading.Timer(0.3, release.set).start()
    screen.close()
    assert not mm._mask_writes_pending()
    assert np.all(tifffile.imread(_saved(folder, 0))[12:20, 12:20] > 0)
    assert screen._status_label.text().startswith("Saved")


def test_a_failed_background_save_is_reported_and_tried_again(opened, monkeypatch, qtbot):
    screen, folder = opened
    warnings = []
    monkeypatch.setattr(screen, "_warn", lambda *args: warnings.append(args))
    release = threading.Event()
    real = engine.save_mask
    calls = []

    def failing_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            assert release.wait(10)
            raise OSError("disk full")
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "save_mask", failing_once)
    before = screen._last_saved_mask.copy()
    screen._canvas.mask[12:20, 12:20] = 4
    assert screen._save_mask() == _saved(folder, 0)
    release.set()
    qtbot.waitUntil(lambda: bool(warnings), timeout=10000)
    assert warnings == [("Save failed", "disk full")]
    np.testing.assert_array_equal(screen._last_saved_mask, before)
    screen._btn_save_on_navigation.trigger()
    assert screen._save_mask_if_needed()
    assert mm._wait_for_mask_writes(timeout=10)
    assert len(calls) == 2
    assert np.all(tifffile.imread(_saved(folder, 0))[12:20, 12:20] > 0)


def test_a_fast_failure_still_stops_the_save_at_once(opened, monkeypatch):
    screen, _folder = opened
    warnings = []
    monkeypatch.setattr(screen, "_warn", lambda *args: warnings.append(args))

    def broken(*args, **kwargs):
        raise OSError("read-only folder")

    monkeypatch.setattr(engine, "save_mask", broken)
    screen._canvas.mask[12:20, 12:20] = 2
    assert screen._save_mask() is None
    assert warnings == [("Save failed", "read-only folder")]


def test_atomic_replace_failure_during_close_preserves_file_unsaved_mask_and_history(
        opened, monkeypatch):
    import os
    from pathlib import Path
    import time

    screen, folder = opened
    target = Path(engine.save_mask(str(folder), 'field_0.tif', screen._canvas.mask))
    prior_file = target.read_bytes()
    source = folder / 'field_0.tif'
    source_bytes = source.read_bytes()
    previous_saved = screen._last_saved_mask.copy()
    screen._canvas.mask[12:20, 12:20] = 9
    screen._history.push(screen._canvas.mask)
    unsaved = screen._canvas.mask.copy()

    def history_state():
        return [[(mask.dtype.str, mask.shape, mask.tobytes()) for mask in stack]
                for stack in (screen._history._undo, screen._history._redo)]

    history = history_state()
    warnings = []
    monkeypatch.setattr(screen, '_warn', lambda *args: warnings.append(args))
    entered, release = threading.Event(), threading.Event()
    real_replace, real_fsync = engine.os.replace, engine.os.fsync
    synced = set()
    temporary_paths = []

    def fsync(descriptor):
        result = real_fsync(descriptor)
        identity = os.fstat(descriptor)
        synced.add((identity.st_dev, identity.st_ino))
        return result

    def replace(source_path, destination):
        if Path(destination) != target:
            return real_replace(source_path, destination)
        temporary = Path(source_path)
        assert temporary.parent == target.parent and temporary.name.startswith('.spacr-mask-')
        identity = temporary.stat()
        assert (identity.st_dev, identity.st_ino) in synced
        np.testing.assert_array_equal(tifffile.imread(temporary), engine.canonical_labels(unsaved))
        temporary_paths.append(temporary)
        entered.set()
        assert release.wait(10)
        raise PermissionError('owned mask replacement refused')

    monkeypatch.setattr(engine.os, 'fsync', fsync)
    monkeypatch.setattr(engine.os, 'replace', replace)
    try:
        assert screen._save_mask() == str(target)
        assert entered.wait(10) and mm._mask_writes_pending(target)
        request = screen._mask_write_requests[-1]
        timer = threading.Timer(.3, release.set)
        timer.start()
        started = time.monotonic()
        screen.close()
        assert time.monotonic() - started < 10
        timer.join(timeout=1)
    finally:
        release.set()
    assert request['done'].is_set() and request['finished']
    assert isinstance(request['error'], PermissionError)
    assert not mm._mask_writes_pending() and not screen._mask_write_requests
    assert mm._MASK_WRITE_STATE['running'] is None and not mm._MASK_WRITE_QUEUE
    assert target.read_bytes() == prior_file and source.read_bytes() == source_bytes
    assert temporary_paths and all(not path.exists() for path in temporary_paths)
    assert not list(target.parent.glob('.spacr-mask-*.tif'))
    np.testing.assert_array_equal(screen._canvas.mask, unsaved)
    np.testing.assert_array_equal(screen._last_saved_mask, previous_saved)
    assert screen._canvas.mask.dtype == unsaved.dtype and history_state() == history
    assert not np.array_equal(screen._canvas.mask, screen._last_saved_mask)
    assert not screen._save_would_change_nothing()
    assert warnings == [('Save failed', 'owned mask replacement refused')]
    assert not screen._status_label.text().startswith('Saved')
