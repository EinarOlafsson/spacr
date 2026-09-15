"""The destroy-time join takes the quit hook down only where there is one to take.

``_join_on_destroy`` joins a backdrop's render thread when Qt frees the widget
and, since the quit-hook fix, also takes the widget's hook off the
application's ``aboutToQuit`` -- a backdrop freed with its screen never runs
``shutdown``, which is where that used to happen
(``test_cov_r8_fractal_app_hooks.py`` holds that half against a real app).

The handler runs while Qt is tearing things down, which is also when the
application may already be gone or half-destroyed. Joining the thread is the
part that prevents a crash, so neither of those may stop it or raise out of
the destruction.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from spacr.qt.widgets import fractal_travel as F

pytestmark = pytest.mark.qt


class _DestroyedSignal:
    def __init__(self):
        self.handler = None

    def connect(self, handler):
        self.handler = handler


class _Widget:
    def __init__(self):
        self.destroyed = _DestroyedSignal()


class _Thread:
    def __init__(self):
        self.quits = 0
        self.waited = None

    def quit(self):
        self.quits += 1

    def wait(self, milliseconds):
        self.waited = milliseconds
        return True


def _free(monkeypatch, application):
    """Free a watched widget while ``QApplication.instance()`` is ``application``."""
    widget, thread = _Widget(), _Thread()
    F._join_on_destroy(widget, thread, quit_hook=lambda: None)
    monkeypatch.setattr(QApplication, "instance",
                        staticmethod(lambda: application))
    widget.destroyed.handler()                 # must not raise
    monkeypatch.undo()
    return thread


def test_a_backdrop_freed_after_the_application_still_joins_its_thread(
        qapp, monkeypatch):
    thread = _free(monkeypatch, None)
    assert (thread.quits, thread.waited) == (1, 5000)


def test_an_application_that_refuses_the_disconnect_does_not_raise(
        qapp, monkeypatch):
    attempts = []

    def half_destroyed(hook):
        attempts.append(hook)
        raise RuntimeError("Internal C++ object (QApplication) already deleted.")

    application = SimpleNamespace(
        aboutToQuit=SimpleNamespace(disconnect=half_destroyed))
    thread = _free(monkeypatch, application)

    assert len(attempts) == 1, "the hook was never offered to the application"
    assert (thread.quits, thread.waited) == (1, 5000)
