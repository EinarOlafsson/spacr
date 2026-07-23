"""Tests for the Batch F diagnostics fixes:

* ``_StreamRedirector`` flushes partial buffers past the chunk cap.
* ``PipelineWorker.run`` starts an idle-flush pump so the console
  never sits silent for long.
* ``log_call`` decorator emits entry/return traces when verbose mode
  is on and is a pass-through no-op when it's off.
* ``log_button_press`` respects the same on/off gate.
* ``resolve_pipeline_entry`` returns wrapped functions.
"""
from __future__ import annotations

import logging
import time

import pytest


# ---------------------------------------------------------------------------
# Stream redirector chunk cap
# ---------------------------------------------------------------------------

class TestStreamRedirector:
    def test_full_line_flushes_immediately(self):
        from spacr.qt.bridge import _StreamRedirector
        seen = []
        r = _StreamRedirector(seen.append)
        r.write("hello\n")
        assert seen == ["hello\n"]

    def test_partial_line_buffers_until_newline(self):
        from spacr.qt.bridge import _StreamRedirector
        seen = []
        r = _StreamRedirector(seen.append)
        r.write("no newline")
        assert seen == []
        r.write(" yet\nsecond\n")
        assert seen == ["no newline yet\n", "second\n"]

    def test_chunk_cap_flushes_giant_partial(self):
        from spacr.qt.bridge import _StreamRedirector
        seen = []
        r = _StreamRedirector(seen.append)
        r.write("x" * 5000)   # > 1024 chars, no newline
        assert seen, "chunk-cap should have flushed the buffer"

    def test_idle_flush_emits_pending(self):
        from spacr.qt.bridge import _StreamRedirector
        seen = []
        r = _StreamRedirector(seen.append)
        r.write("pending")
        assert seen == []
        r.idle_flush()
        assert seen == ["pending"]


# ---------------------------------------------------------------------------
# Verbose-logger gating
# ---------------------------------------------------------------------------

@pytest.fixture
def _isolated_prefs(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope,
                       str(tmp_path))
    yield


class TestLogCallDecorator:
    def test_no_op_when_verbose_off(self, _isolated_prefs, caplog):
        from spacr.qt.verbose_logger import (
            apply_verbose_logging, log_call,
        )
        apply_verbose_logging(False)
        @log_call
        def add(a, b): return a + b
        with caplog.at_level(logging.DEBUG, logger="spacr.trace"):
            assert add(1, 2) == 3
        assert not any("add" in r.message for r in caplog.records)

    def test_emits_entry_and_return_when_on(self, _isolated_prefs,
                                                 caplog):
        from spacr.qt.verbose_logger import (
            apply_verbose_logging, log_call,
        )
        apply_verbose_logging(True)
        try:
            @log_call
            def add(a, b): return a + b
            with caplog.at_level(logging.DEBUG, logger="spacr.trace"):
                assert add(1, 2) == 3
            messages = " ".join(r.message for r in caplog.records)
            assert "add" in messages
            assert "-> 3" in messages
        finally:
            apply_verbose_logging(False)

    def test_records_exception(self, _isolated_prefs, caplog):
        from spacr.qt.verbose_logger import (
            apply_verbose_logging, log_call,
        )
        apply_verbose_logging(True)
        try:
            @log_call
            def boom(): raise ValueError("nope")
            with caplog.at_level(logging.DEBUG, logger="spacr.trace"):
                with pytest.raises(ValueError):
                    boom()
            messages = " ".join(r.message for r in caplog.records)
            assert "RAISED" in messages and "ValueError" in messages
        finally:
            apply_verbose_logging(False)


class TestLogButtonPress:
    def test_no_op_when_verbose_off(self, _isolated_prefs, caplog):
        from spacr.qt.verbose_logger import (
            apply_verbose_logging, log_button_press,
        )
        apply_verbose_logging(False)
        with caplog.at_level(logging.DEBUG, logger="spacr.trace"):
            log_button_press("mask.Run", {"src": "/tmp/x"})
        assert not caplog.records

    def test_emits_when_on(self, _isolated_prefs, caplog):
        from spacr.qt.verbose_logger import (
            apply_verbose_logging, log_button_press,
        )
        apply_verbose_logging(True)
        try:
            with caplog.at_level(logging.DEBUG, logger="spacr.trace"):
                log_button_press("mask.Run", {"src": "/tmp/x"})
            messages = " ".join(r.message for r in caplog.records)
            assert "button:mask.Run" in messages
            assert "/tmp/x" in messages
        finally:
            apply_verbose_logging(False)


# ---------------------------------------------------------------------------
# resolve_pipeline_entry wraps functions
# ---------------------------------------------------------------------------

class TestResolveWraps:
    def test_mask_entry_is_wrapped(self):
        from spacr.qt.bridge import resolve_pipeline_entry
        fn = resolve_pipeline_entry("mask")
        assert fn is not None
        # wrapped functools.wraps → __wrapped__ points at the original
        assert getattr(fn, "__wrapped__", None) is not None
        # Underlying function is preprocess_generate_masks
        assert fn.__wrapped__.__name__ == "preprocess_generate_masks"

    def test_unknown_app_returns_none(self):
        from spacr.qt.bridge import resolve_pipeline_entry
        assert resolve_pipeline_entry("not-a-real-app") is None


# ---------------------------------------------------------------------------
# Console shows entry-name breadcrumb on Run
# ---------------------------------------------------------------------------

class TestRunBreadcrumb:
    """The AppScreen's Run handler now emits a more informative
    console line — 'Starting mask (preprocess_generate_masks) with
    src=… + N settings…'. Rather than construct the full AppScreen
    (heavy + brittle in the test suite), verify the code path directly
    by checking that the resolved entry point exposes __qualname__ so
    the breadcrumb string can name it."""

    def test_entry_qualname_is_accessible(self):
        from spacr.qt.bridge import resolve_pipeline_entry
        entry = resolve_pipeline_entry("mask")
        # The log_call wrapper preserves __qualname__ via functools.wraps
        qname = getattr(entry, "__qualname__", "")
        assert "preprocess_generate_masks" in qname
