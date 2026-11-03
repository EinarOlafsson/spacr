"""The thread stream router answers a closed wrapped stream without raising.

colorama's atexit reset asks sys.stdout.isatty() after the test runner has
closed the captured stream underneath the router; that surfaced as eight
"I/O operation on closed file" tracebacks per run.
"""
from __future__ import annotations

import io

from spacr.qt.bridge import _ThreadStreamRouter


def test_a_closed_original_stream_is_not_a_terminal_and_does_not_raise():
    stream = io.StringIO()
    router = _ThreadStreamRouter(stream)
    stream.close()
    assert router.isatty() is False
    assert router.write("text") == 0
    router.flush()


def test_a_finalizing_interpreter_reaches_the_original_without_the_lock(monkeypatch):
    """While Python exits a write skips the routing table and its lock."""
    import sys
    import threading

    original = io.StringIO()
    console = io.StringIO()
    router = _ThreadStreamRouter(original)
    router._targets[threading.get_ident()] = [console]

    class _HeldForever:
        def __enter__(self):
            raise AssertionError("the routing lock was taken while finalizing")

        def __exit__(self, *exc):
            return False

    router._lock = _HeldForever()
    monkeypatch.setattr(sys, "is_finalizing", lambda: True)

    assert router._target() is original
