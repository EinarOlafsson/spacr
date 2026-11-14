"""Desktop notification backends: what command each platform actually runs.

``notify`` shells out, so the subprocess call is intercepted and its argument
vector asserted rather than fired -- a test run must not put a toast on the
developer's desktop. Everything else (backend selection, escaping, the
tray fallback, the finished-pipeline wrapper) runs for real.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt import notify as notify_module


def test_background_notifier_failure_is_logged_without_escaping(
        monkeypatch, caplog):
    """A desktop backend may fail after the caller has already returned."""
    import threading

    launched = []

    class ImmediateThread:
        def __init__(self, target, name, daemon):
            assert name == "spacr-notify"
            assert daemon is True
            self.target = target

        def start(self):
            launched.append(True)
            self.target()

    def broken_backend(message):
        assert message == "finished"
        raise OSError("desktop bus vanished")

    monkeypatch.setattr(threading, "Thread", ImmediateThread)
    with caplog.at_level("DEBUG", logger="spacr.qt.notify"):
        notify_module._in_background(broken_backend, "finished")
    assert launched == [True]
    assert "desktop bus vanished" in caplog.text


def test_esc_tolerates_none():
    """``_esc(None)`` is an empty string, not a crash on ``.replace``."""
    assert notify_module._esc(None) == ""
    assert notify_module._esc("") == ""
    assert notify_module._esc('a"b') == 'a\\"b'


# --------------------------------------------------------------------------
# tray fallback
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# announce_pipeline_finished
# --------------------------------------------------------------------------
