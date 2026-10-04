"""Verbose logger edges: a pre-attached file handler, no root sink, long reprs."""
from __future__ import annotations

import logging
import sys

import pytest

pytest.importorskip("PySide6")

from spacr.qt import verbose_logger as vl  # noqa: E402


def test_a_new_file_handler_already_on_a_child_is_moved_to_the_root(
        monkeypatch, tmp_path):
    handler = logging.FileHandler(tmp_path / "x.log")
    child = logging.getLogger("spacr.qt")
    child.addHandler(handler)
    monkeypatch.setattr(vl, "_file_handler", None)
    monkeypatch.setattr(vl, "current_log_file", lambda: tmp_path / "x.log")
    monkeypatch.setattr(vl, "_quicken", lambda made: (made.close(), handler)[1])
    sink = logging.getLogger(vl._SINK_LOGGER)
    try:
        assert vl._ensure_file_handler() is handler
        assert handler not in child.handlers and handler in sink.handlers
    finally:
        sink.removeHandler(handler)
        child.removeHandler(handler)
        handler.close()


def test_without_a_root_sink_module_every_record_passes(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.qt.logging_util", None)
    record = logging.LogRecord("spacr", logging.INFO, __file__, 1, "m", (), None)
    assert vl._NotAlreadyShownByTheRootSink().filter(record) is True


def test_a_long_repr_is_cut_in_the_middle():
    class _Long:
        def __repr__(self):
            return "a" * 50 + "z" * 50

    brief = vl._BriefRepr()
    brief.maxother = 11
    text = brief.repr_instance(_Long(), 1)
    assert text == "aaaa...zzzz"
