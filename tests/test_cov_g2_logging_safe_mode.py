"""Safe mode writes every log flush at once: it starts no timer thread."""
from __future__ import annotations

import sys
import threading
import types

from spacr import logging_util as lu


class _Handler:
    _spacr_flushed_at = 0.0
    _spacr_late_flush_pid = 0


def test_safe_mode_flushes_now_and_starts_no_thread(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences",
                        types.SimpleNamespace(in_safe_mode=lambda: True))
    started = []
    monkeypatch.setattr(threading.Thread, "start",
                        lambda self: started.append(self.name))
    flushed = []
    handler = _Handler()
    for _ in range(3):
        lu._paced_flush(handler, lambda: flushed.append(1))
    assert flushed == [1, 1, 1] and started == []


def test_an_unreadable_safe_mode_flag_counts_as_off(monkeypatch):
    def broken():
        raise RuntimeError("half imported")

    monkeypatch.setitem(sys.modules, "spacr.qt.preferences",
                        types.SimpleNamespace(in_safe_mode=broken))
    assert lu._in_safe_mode() is False
    monkeypatch.delitem(sys.modules, "spacr.qt.preferences")
    assert lu._in_safe_mode() is False
