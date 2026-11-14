"""Table-driven test of the Qt log sink ``QtLogHandler.emit``.

``emit`` runs inside every ``logging`` call in the process, so its guards are
load-bearing: a record is relayed with its exact formatted line and level; it is
dropped while a console panel is mid-write (re-entering one is a documented
segfault); a latch that cannot be consulted lets the record through; and a
formatter that fails routes to ``handleError`` instead of propagating.
"""
from __future__ import annotations

import builtins
import logging
import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt import logging_util as lu  # noqa: E402
from spacr.qt import verbose_logger  # noqa: E402


def _record(message, level):
    record = logging.LogRecord("spacr.test", level, __file__, 1, message,
                               None, None)
    record.created = 0.0
    record.msecs = 0.0
    return record


class _Exploding(logging.Formatter):
    def format(self, record):
        raise ValueError("cannot format this")


def test_emit_relays_drops_or_hands_off_each_record(subtests, monkeypatch):
    """Exact (record_ready, records_ready, handleError) outcome per guard."""
    real_import = builtins.__import__

    def latch_busy(patch):
        patch.setattr(verbose_logger, "console_write_in_progress", lambda: True)

    def latch_idle(patch):
        patch.setattr(verbose_logger, "console_write_in_progress", lambda: False)

    def latch_missing(patch):
        def block(name, globals=None, locals=None, fromlist=(), level=0):
            if "verbose_logger" in name or "verbose_logger" in (fromlist or ()):
                raise ImportError("no latch here")
            return real_import(name, globals, locals, fromlist, level)
        patch.setattr(builtins, "__import__", block)

    def latch_raises(patch):
        def boom():
            raise RuntimeError("latch broke")
        patch.setattr(verbose_logger, "console_write_in_progress", boom)

    cases = [
        # (id, setup, formatter, level, message, relayed?, handled?)
        ("a warning reaches the panel with its level", latch_idle, None,
         logging.WARNING, "a thing happened", True, False),
        ("an info line reaches the panel with its level", latch_idle, None,
         logging.INFO, "plain news", True, False),
        ("nothing is emitted while a console panel is writing", latch_busy,
         None, logging.ERROR, "dropped on the floor", False, False),
        ("a missing latch does not stop the record", latch_missing, None,
         logging.INFO, "still delivered", True, False),
        ("a latch that raises does not stop the record", latch_raises, None,
         logging.INFO, "delivered anyway", True, False),
        ("a formatting failure is handled, not raised", latch_idle,
         _Exploding(), logging.INFO, "never formatted", False, True),
    ]
    for case_id, setup, formatter, level, message, relayed, handled in cases:
        with subtests.test(msg=case_id):
            handler = lu.QtLogHandler(level=logging.DEBUG)
            single, batches, errors = [], [], []
            handler.record_ready.connect(
                lambda text, lvl: single.append((text, lvl)))
            handler.records_ready.connect(lambda rows: batches.append(rows))
            if formatter is not None:
                handler.setFormatter(formatter)
            record = _record(message, level)
            with monkeypatch.context() as patch:
                patch.setattr(type(handler), "handleError",
                              lambda self, r: errors.append(r))
                setup(patch)
                handler.emit(record)                 # must not raise
            name = logging.getLevelName(level)
            stamp = time.strftime("%H:%M:%S", time.localtime(0))
            line = f"{stamp} [{name}] spacr.test: {message}\n"
            want = [(line, level)] if relayed else []
            assert single == want
            assert batches == ([want] if relayed else [])
            assert errors == ([record] if handled else [])
