"""A call routed to the GUI thread returns its error, or runs in place.

A worker that needs an answer only the GUI thread may compute posts the
call and waits. An exception raised there comes back to the worker as the
same exception; a relay already torn down (the application is closing)
runs the call where it was asked.
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")

from spacr.qt import bridge  # noqa: E402

pytestmark = pytest.mark.qt


def _from_a_worker(fn, qtbot):
    """Call ``bridge._call_on_the_gui_thread(fn)`` from a plain thread."""
    outcome = {}

    def ask():
        try:
            outcome["value"] = bridge._call_on_the_gui_thread(fn, timeout_s=10)
        except BaseException as error:  # noqa: BLE001
            outcome["error"] = error

    worker = threading.Thread(target=ask)
    worker.start()
    qtbot.waitUntil(lambda: not worker.is_alive(), timeout=10000)
    return outcome


def test_an_error_raised_on_the_gui_thread_reaches_the_worker(qtbot, monkeypatch):
    monkeypatch.setattr(bridge, "_GUI_CALLS", [])
    monkeypatch.setattr("spacr.workspace._STATE_DISPATCHER", None, raising=False)
    bridge._install_the_workspace_dispatcher()
    asked_on = []

    def refuse():
        asked_on.append(threading.current_thread() is threading.main_thread())
        raise KeyError("no such section")

    outcome = _from_a_worker(refuse, qtbot)
    assert isinstance(outcome.get("error"), KeyError)
    assert asked_on == [True]


def test_a_torn_down_relay_runs_the_call_in_place(qtbot, monkeypatch):
    class Gone:
        class _asked:
            @staticmethod
            def emit(job):
                raise RuntimeError("Internal C++ object already deleted")

    monkeypatch.setattr(bridge, "_GUI_CALLS", [Gone()])
    asked_on = []
    outcome = _from_a_worker(lambda: asked_on.append(threading.current_thread().name) or 7, qtbot)
    assert outcome == {"value": 7}
    assert asked_on and asked_on[0] != threading.main_thread().name


def test_the_posted_call_records_its_value_or_its_error():
    box, done = {}, threading.Event()
    bridge._run_the_posted(lambda: 3, box, done)
    assert box == {"value": 3} and done.is_set()
    box, done = {}, threading.Event()
    bridge._run_the_posted(lambda: 1 / 0, box, done)
    assert isinstance(box["error"], ZeroDivisionError) and done.is_set()
