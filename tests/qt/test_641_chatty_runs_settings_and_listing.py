"""Item 641, second pass: guards for chatty runs, preferences and listings.

* A log file flushes a few times a second, not once per record, and a
  warning still reaches the disk at once.
* The master log and the per-level file format a record once between them,
  and a record never visits the files of the levels above it.
* Records logged on a worker thread reach the console in a few batches, in
  order, with a warning delivered at once.
* The animated backdrop slows to its run rate while a pipeline holds the
  interpreter, and the desktop notifier never holds the interface.
* Saving many preferences opens one store and writes it once.
* Loading a preview reads a settled folder once, not once per helper.

Counted, never timed, except the notifier, whose ceiling is ten times the
measured value.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import threading
import time

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _counting_stream(handler):
    calls = []
    stream = handler.stream if handler.stream is not None else handler._open()
    handler.stream = stream
    real = stream.flush

    class _Stream:
        def __getattr__(self, name):
            return getattr(stream, name)

        def flush(self):
            calls.append(1)
            real()

    handler.stream = _Stream()
    return calls


def test_a_log_file_flushes_by_the_clock_and_at_once_for_a_warning(tmp_path):
    from spacr.logging_util import _quicken

    path = tmp_path / "run.log"
    handler = _quicken(logging.handlers.RotatingFileHandler(
        path, maxBytes=10_000_000, backupCount=1, encoding="utf-8"))
    flushes = _counting_stream(handler)
    log = logging.getLogger("spacr.test641.flush")
    log.propagate = False
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        for i in range(2000):
            log.info("field %d measured", i)
        assert len(flushes) < 50, f"{len(flushes)} flushes for 2000 records"
        log.warning("the plate is missing a well")
        assert "the plate is missing a well" in path.read_text("utf-8")
        log.info("the last quiet line")
        deadline = time.monotonic() + 5
        while ("the last quiet line" not in path.read_text("utf-8")
               and time.monotonic() < deadline):
            time.sleep(0.05)
        assert "the last quiet line" in path.read_text("utf-8")
    finally:
        log.removeHandler(handler)
        handler.close()


def test_the_master_and_level_files_format_a_record_once(monkeypatch):
    from spacr.logging_util import FILE_FORMAT, _CompactTraceFormat

    calls = []
    real = logging.Formatter.format
    monkeypatch.setattr(logging.Formatter, "format",
                        lambda self, record: calls.append(1)
                        or real(self, record))
    record = logging.LogRecord("spacr.x", logging.INFO, __file__, 1,
                               "field %d", (3,), None)
    first = _CompactTraceFormat(FILE_FORMAT).format(record)
    second = _CompactTraceFormat(FILE_FORMAT).format(record)
    assert first == second and "field 3" in first
    assert len(calls) == 1


def test_a_level_file_is_not_asked_about_quieter_records(tmp_path,
                                                         monkeypatch):
    from spacr import logging_util as lu

    monkeypatch.setattr(lu, "_LEVEL_HANDLERS", {})
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        lu._install_level_handlers(tmp_path / "spacr.log", lu.LEVELS)
        for level, handler in lu._LEVEL_HANDLERS.items():
            assert handler.level == level
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()


def test_worker_records_reach_the_console_in_few_batches_in_order(qtbot):
    from spacr.qt.logging_util import QtLogHandler

    handler = QtLogHandler()
    batches = []
    handler.records_ready.connect(batches.append)
    log = logging.getLogger("spacr.test641.batch")
    log.propagate = False
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    def work():
        for i in range(2000):
            log.info("line %d", i)
        log.warning("loud")

    try:
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        qtbot.waitUntil(lambda: any(
            text.endswith("loud\n") for batch in batches
            for text, _level in batch), timeout=5000)
    finally:
        log.removeHandler(handler)
    texts = [text for batch in batches for text, _level in batch]
    assert [t.rsplit(": ", 1)[-1] for t in texts] == (
        [f"line {i}\n" for i in range(2000)] + ["loud\n"])
    assert len(batches) < 100, f"{len(batches)} batches for 2001 records"


def test_the_backdrop_slows_while_a_run_holds_the_interpreter(qtbot):
    from spacr.qt import gil_priority
    from spacr.qt.widgets.ambient import _RUN_FPS, AmbientWidget

    widget = AmbientWidget(fps=24)
    qtbot.addWidget(widget)
    widget._follow_the_run()
    assert widget._rate() == 24
    with gil_priority.responsive_gui():
        widget._follow_the_run()
        assert widget._rate() == _RUN_FPS
        assert widget._timer.interval() == 1000 // _RUN_FPS
    widget._follow_the_run()
    assert widget._rate() == 24


def test_the_desktop_notifier_never_holds_the_interface(monkeypatch):
    from spacr.qt import notify

    started = threading.Event()

    def slow_run(*_args, **_kwargs):
        started.set()
        time.sleep(1.0)

    monkeypatch.setattr(notify.platform, "system", lambda: "Linux")
    monkeypatch.setattr(notify.shutil, "which", lambda name: "/bin/true")
    monkeypatch.setattr(notify.subprocess, "run", slow_run)
    t = time.perf_counter()
    assert notify.notify("Done", "1 plate") is True
    assert time.perf_counter() - t < 0.5
    assert started.wait(5)


def test_saving_many_preferences_opens_one_store(monkeypatch):
    from PySide6.QtCore import QSettings

    from spacr.qt import preferences

    opened = []

    class Counting(QSettings):
        def __init__(self, *args):
            opened.append(args)
            super().__init__(*args)

    monkeypatch.setattr(preferences, "QSettings", Counting)
    with preferences._one_store():
        for size in range(8, 28):
            preferences.set_figure_text_size(size)
    assert len(opened) == 1
    assert preferences.get_figure_text_size() == 27


def test_a_settled_folder_is_listed_once_per_preview_load(tmp_path,
                                                         monkeypatch):
    from spacr.qt.widgets import live_preview, preview_controls

    for well in ("A01", "A02", "B01"):
        for channel in (1, 2):
            (tmp_path / f"plate1_{well}_s1_w{channel}.tif").write_bytes(b"")
    old = time.time() - 3600
    os.utime(tmp_path, (old, old))
    monkeypatch.setattr(preview_controls, "_LISTINGS",
                        type(preview_controls._LISTINGS)())
    listed = []
    real = os.scandir
    monkeypatch.setattr(os, "scandir",
                        lambda *a, **k: listed.append(a) or real(*a, **k))
    first = live_preview.first_supported_image(tmp_path)
    sets, _channels = preview_controls.enumerate_image_sets(
        tmp_path, (".tif",))
    assert first is not None and first.name == "plate1_A01_s1_w1.tif"
    assert sum(len(s.channels) for s in sets) == 6
    assert len(listed) == 1, f"listed {len(listed)} times"
    (tmp_path / "plate1_B02_s1_w1.tif").write_bytes(b"")
    sets, _channels = preview_controls.enumerate_image_sets(
        tmp_path, (".tif",))
    assert sum(len(s.channels) for s in sets) == 7
