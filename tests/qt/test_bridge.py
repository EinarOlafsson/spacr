"""PipelineWorker + stream redirector tests."""
from __future__ import annotations

import io
import sys

import pytest

from PySide6.QtCore import QThread, QCoreApplication

from spacr.qt.bridge import (
    PipelineWorker,
    _StreamRedirector,
    make_thread,
)


def test_stream_redirector_emits_line_by_line():
    received = []
    r = _StreamRedirector(received.append)
    r.write("hello ")
    assert received == []          # no newline yet
    r.write("world\n")
    assert received == ["hello world\n"]
    r.write("multi\nline\nchunk\n")
    assert received[-3:] == ["multi\n", "line\n", "chunk\n"]


def test_stream_redirector_flush_emits_remainder():
    received = []
    r = _StreamRedirector(received.append)
    r.write("no newline yet")
    r.flush()
    assert received == ["no newline yet"]


def test_pipeline_worker_success(qtbot, qt_theme_applied):
    def _fn(settings):
        print("running")
        return {"ok": True}

    worker = PipelineWorker(_fn, {})
    lines = []
    worker.line_ready.connect(lines.append)
    finished = []
    worker.finished.connect(lambda ok: finished.append(ok))
    worker.run()
    assert finished == [True]
    assert any("running" in l for l in lines)


def test_pipeline_worker_captures_exception(qtbot, qt_theme_applied):
    def _fn(settings):
        raise RuntimeError("boom")

    worker = PipelineWorker(_fn, {})
    errors = []
    worker.error.connect(errors.append)
    finished = []
    worker.finished.connect(lambda ok: finished.append(ok))
    worker.run()
    assert finished == [False]
    assert len(errors) == 1
    assert "boom" in errors[0]


def test_make_thread_returns_thread_and_worker():
    thread, worker = make_thread(lambda s: None, {})
    try:
        assert isinstance(thread, QThread)
        assert isinstance(worker, PipelineWorker)
    finally:
        thread.deleteLater()
        worker.deleteLater()
