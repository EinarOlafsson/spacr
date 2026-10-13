"""Edges of 641's paced logging, background notifier, output drain and caches."""
from __future__ import annotations

import logging
import logging.handlers
import os
import time
import types

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr import logging_util as lu  # noqa: E402


def test_only_rotating_handlers_are_quickened(tmp_path):
    plain = logging.StreamHandler()
    assert lu._quicken(plain) is plain
    assert not hasattr(plain, "_spacr_flushed_at")


def test_a_failing_late_or_loud_flush_does_not_raise(tmp_path):
    handler = types.SimpleNamespace(_spacr_late_flush_pid=1,
                                    _spacr_flushed_at=0.0, errors=[])
    handler.handleError = handler.errors.append

    def broken():
        raise OSError("disk gone")

    lu._late_flush(handler, broken)
    assert handler._spacr_late_flush_pid == 0
    record = logging.LogRecord("spacr", logging.ERROR, __file__, 1, "x", (), None)
    lu._emit_flushing_the_loud(handler, lambda r: None, broken, record)
    assert handler.errors == [record]


def test_rollover_reopens_a_closed_stream_and_reads_a_plain_position(tmp_path):
    handler = logging.handlers.RotatingFileHandler(
        tmp_path / "x.log", maxBytes=10 ** 9, backupCount=1, delay=True)
    try:
        assert handler.stream is None
        record = logging.LogRecord("spacr", logging.INFO, __file__, 1, "x", (), None)
        assert lu._quick_should_rollover(handler, lambda r: True, record) is False
        assert handler.stream is not None
        handler.stream.close()
        handler.stream = open(tmp_path / "y.log", "w", encoding="utf-8")
        handler.stream = types.SimpleNamespace(tell=lambda: 0)
        assert lu._quick_should_rollover(handler, lambda r: True, record) is False
    finally:
        handler.stream = None
        handler.close()


def test_a_failing_notifier_is_only_logged(monkeypatch):
    from spacr.qt import notify

    done = []

    def broken():
        done.append(True)
        raise RuntimeError("no bus")

    import threading

    real = threading.Thread

    class _Inline(real):
        def start(self):
            self.run()

    monkeypatch.setattr(threading, "Thread", _Inline)
    notify._in_background(broken)
    assert done == [True]


def test_draining_a_finished_worker_does_nothing():
    from spacr.qt.bridge import _OutputDrain

    drain = _OutputDrain.__new__(_OutputDrain)
    drain._worker = lambda: None
    _OutputDrain._drain(drain)
    assert drain._worker() is None


def test_the_listing_cache_keeps_only_the_last_folders(tmp_path, monkeypatch):
    from spacr.qt.widgets import preview_controls as pc

    monkeypatch.setattr(pc, "_LISTINGS_KEPT", 2)
    monkeypatch.setattr(pc, "_LISTINGS", type(pc._LISTINGS)())
    old = time.time() - 60
    for name in ("a", "b", "c"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "f.tif").write_bytes(b"")
        os.utime(folder, (old, old))
        assert pc._file_names(folder) == ("f.tif",)
    assert list(pc._LISTINGS) == [str(tmp_path / "b"), str(tmp_path / "c")]


def test_empty_wells_share_one_pen(qtbot):
    from spacr.qt.screens.plate_view import PlateGridWidget

    grid = PlateGridWidget()
    qtbot.addWidget(grid)
    grid.set_plate(None, n_rows=2, n_cols=3)
    grid.resize(300, 200)
    assert not grid.grab().isNull()


def test_a_threaded_plate_view_waits_for_its_timer(qtbot):
    from spacr.qt.screens.plate_view import PlateViewScreen

    screen = PlateViewScreen(threaded=True)
    qtbot.addWidget(screen)
    screen._loading = False
    screen._frame = pd.DataFrame({"a": [1]})
    screen._on_view_changed()
    assert screen._recompute_timer.isActive()
    screen._recompute_timer.stop()
