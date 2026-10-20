"""Retiring live work at exit, driven in this process instead of a child.

``tests/qt/test_quitting_during_a_slow_job.py`` proves the whole exit path
end to end, but in subprocesses, because the failure it guards against is
an abort during interpreter teardown. That leaves the guards inside
:func:`spacr.qt.bridge.retire_live_work` unmeasured: what happens when a
drain throws, when a loose thread answers with ``RuntimeError`` because its
C++ half is already gone, and when a thread will not stop in the budget.
Each is pinned here as what the exit does next:

* a stubborn thread is taken off its parent and parked, once, never
  terminated, and a thread that stops in time is not parked;
* a thread whose C++ object has gone is skipped, not raised out of an
  ``atexit`` hook;
* a registry or runner that throws does not stop the rest of the drain;
* without ``shiboken6`` there is no way to tell our threads from Qt's, so
  no loose thread is touched at all.
"""
from __future__ import annotations

import logging
import sys
import threading
import weakref

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QThread                                   # noqa: E402

from spacr.qt import bridge, job_runner                              # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _own_parking_lot(monkeypatch):
    """Give each test an empty parking list, put back afterwards."""
    monkeypatch.setattr(bridge, "_PARKED_THREADS", [])


class _FakeThread:
    """A QThread stand-in whose answers each test chooses."""

    def __init__(self, *, running=True, waits=True, gone_on_ask=False,
                 gone_on_wait=False):
        self.running = running
        self.waits = waits
        self.gone_on_ask = gone_on_ask
        self.gone_on_wait = gone_on_wait
        self.asked = []
        self.parent_cleared = False

    def isRunning(self):
        return self.running

    def requestInterruption(self):
        if self.gone_on_ask:
            raise RuntimeError("Internal C++ object already deleted.")
        self.asked.append("interrupt")

    def quit(self):
        self.asked.append("quit")

    def wait(self, _ms):
        if self.gone_on_wait:
            raise RuntimeError("Internal C++ object already deleted.")
        if self.waits:
            self.running = False
        return self.waits

    def setParent(self, parent):
        assert parent is None
        self.parent_cleared = True


class _Spinner(QThread):
    """A real thread that runs until it is asked to stop."""

    def __init__(self):
        super().__init__()
        self.started_flag = threading.Event()

    def run(self):
        self.started_flag.set()
        while not self.isInterruptionRequested():
            self.msleep(5)


class _Deleted(QThread):
    """A Python QThread whose questions raise, as a deleted one's do."""

    def isRunning(self):
        raise RuntimeError("Internal C++ object already deleted.")


def test_a_running_thread_is_parked_once_and_a_stopped_one_not_at_all():
    running = _FakeThread(running=True)
    stopped = _FakeThread(running=False)

    assert bridge._park_still_running([(running, "worker"),
                                       (stopped, None),
                                       (None, None)]) == 1
    assert bridge._park_still_running([(running, "worker")]) == 0
    assert bridge._PARKED_THREADS == [(running, "worker")]
    assert bridge.parked_thread_count() == 1


def test_a_thread_that_will_not_stop_is_unparented_and_parked(monkeypatch):
    stubborn = _FakeThread(waits=False)
    obliging = _FakeThread(waits=True)
    deleted_on_ask = _FakeThread(gone_on_ask=True, gone_on_wait=True)
    monkeypatch.setattr(bridge, "_loose_qthreads",
                        lambda: [deleted_on_ask, stubborn, obliging])

    assert bridge._stop_loose_qthreads(50) == 1

    assert stubborn.asked == ["interrupt", "quit"]
    assert obliging.asked == ["interrupt", "quit"]
    assert stubborn.parent_cleared, (
        "a parked thread kept its parent, which takes it down at exit")
    assert not obliging.parent_cleared
    assert [pair[0] for pair in bridge._PARKED_THREADS] == [stubborn]
    assert deleted_on_ask not in [pair[0] for pair in bridge._PARKED_THREADS]


def test_a_real_thread_asked_to_stop_stops_and_is_not_parked(
        qtbot, monkeypatch):
    spinner = _Spinner()
    spinner.start()
    assert spinner.started_flag.wait(5)
    try:
        assert any(thread is spinner for thread in bridge._loose_qthreads())
        monkeypatch.setattr(bridge, "_loose_qthreads", lambda: [spinner])
        assert bridge._stop_loose_qthreads(5000) == 0
        assert spinner.isFinished()
        assert bridge._PARKED_THREADS == []
    finally:
        spinner.requestInterruption()
        spinner.wait(5000)


def test_a_thread_whose_cpp_half_is_gone_is_not_loose():
    gone = _Deleted()
    try:
        assert not any(thread is gone for thread in bridge._loose_qthreads())
    finally:
        del gone


def test_without_shiboken_no_loose_thread_is_touched(monkeypatch):
    spinner = _Spinner()
    spinner.start()
    assert spinner.started_flag.wait(5)
    try:
        monkeypatch.setitem(sys.modules, "shiboken6", None)
        assert bridge._loose_qthreads() == []
        assert spinner.isRunning()
    finally:
        spinner.requestInterruption()
        spinner.wait(5000)


class _ThrowingRegistry:
    """A run registry that cannot cancel, as one torn down mid-exit."""

    def active(self):
        return []

    def cancel_all(self, **_kwargs):
        raise RuntimeError("the registry is gone")


def test_a_registry_that_throws_still_lets_the_loose_threads_be_stopped(
        monkeypatch, caplog):
    budgets = []
    monkeypatch.setattr(bridge, "_REGISTRY", _ThrowingRegistry())
    monkeypatch.setattr(bridge, "_stop_loose_qthreads",
                        lambda ms: budgets.append(ms) or 0)
    monkeypatch.setattr(job_runner, "shutdown_all",
                        lambda ms: (_ for _ in ()).throw(
                            RuntimeError("no runners")))

    with caplog.at_level(logging.DEBUG, logger=bridge.LOG.name):
        assert bridge.retire_live_work(1000) == 0

    assert "could not shut down the job runners" in caplog.text
    assert "could not cancel the registered runs" in caplog.text
    assert len(budgets) == 1 and 0 <= budgets[0] <= 1000


def test_the_exit_hook_survives_a_failing_retire_and_waits_for_the_parked(
        monkeypatch, caplog):
    drained = []
    monkeypatch.setattr(bridge, "_drain_parked_threads_at_exit",
                        lambda: drained.append("drained"))
    monkeypatch.setattr(bridge, "retire_live_work",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    with caplog.at_level(logging.DEBUG, logger=bridge.LOG.name):
        bridge._retire_live_work_at_exit()
    assert "could not retire the live work at exit" in caplog.text
    assert drained == [], "nothing was parked, so nothing should be waited on"

    monkeypatch.setattr(bridge, "retire_live_work", lambda: 1)
    bridge._PARKED_THREADS.append((_FakeThread(), None))
    bridge._retire_live_work_at_exit()
    assert drained == ["drained"]


class _Runner:
    """A job runner stand-in that records what it was asked."""

    def __init__(self, *, cancel_fails=False, shutdown_fails=False):
        self.cancel_fails = cancel_fails
        self.shutdown_fails = shutdown_fails
        self.shut_down_with = None

    def cancel(self):
        if self.cancel_fails:
            raise RuntimeError("cannot cancel")

    def shutdown(self, remaining):
        if self.shutdown_fails:
            raise RuntimeError("cannot shut down")
        self.shut_down_with = remaining


def test_a_runner_that_cannot_cancel_is_still_shut_down(monkeypatch, caplog):
    stuck = _Runner(cancel_fails=True)
    broken = _Runner(shutdown_fails=True)
    live = weakref.WeakSet([stuck, broken])
    monkeypatch.setattr(job_runner, "_LIVE_RUNNERS", live)

    with caplog.at_level(logging.DEBUG, logger=job_runner.LOG.name):
        asked = job_runner.shutdown_all(500)

    assert asked == 1
    assert stuck.shut_down_with is not None and stuck.shut_down_with <= 500
    assert "could not cancel a job runner" in caplog.text
    assert "could not shut down a job runner" in caplog.text
