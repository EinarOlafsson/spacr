"""JobRunner: work off the GUI thread, results back on it, and every job retired.

The runner replaced eleven hand-rolled copies that disagreed in their failure
paths.  These walk those paths as case tables: a job or handler that raises
(with and without a message), crash-report bookkeeping that itself fails, a
cancel while work is in flight, a runner whose C++ half is already gone, and
the quit-time sweep across every live runner.  Threaded jobs go through real
QThreads from ``make_thread``, because the original bug only exists once a
real thread really finishes; inline mode (``threaded=False``) must emit the
same signals synchronously.
"""
from __future__ import annotations

import logging
import sys
import threading
import weakref

import pytest

pytest.importorskip("PySide6")

import shiboken6  # noqa: E402
from PySide6.QtCore import QObject, QThread  # noqa: E402

from spacr.qt import job_runner  # noqa: E402
from spacr.qt.job_runner import JobRunner, shutdown_all  # noqa: E402

pytestmark = pytest.mark.qt


class _Wordless(Exception):
    """An exception with no message, as some libraries raise."""


def _raise(exc):
    def _do(*_args):
        raise exc
    return _do


def _watch(runner):
    """Record every public signal of ``runner`` in one ordered list."""
    events = []
    runner.job_finished.connect(lambda ok: events.append(("finished", ok)))
    runner.job_failed.connect(lambda msg: events.append(("failed", msg)))
    runner.busy_changed.connect(lambda busy: events.append(("busy", busy)))
    return events


def test_threaded_jobs_report_each_outcome_and_always_retire(qapp, qtbot,
                                                             subtests):
    """Each threaded outcome emits the exact signals, then leaves the runner idle."""
    gui = threading.get_ident()
    seen_threads = {}

    def work():
        seen_threads["work"] = threading.get_ident()
        return 21 * 2

    def keep(value):
        seen_threads["handler"] = threading.get_ident()
        delivered.append(value)

    # case, job, handler, finished ok, failure messages, delivered
    cases = [
        ("success", work, keep, True, [], [42]),
        ("no handler", work, None, True, [], []),
        ("handler raises", lambda: 1,
         _raise(ValueError("the handler could not use the result")), False,
         ["the handler could not use the result"], []),
        ("handler raises without a message", lambda: 1, _raise(_Wordless()),
         False, ["_Wordless"], []),
        ("job raises", _raise(RuntimeError("the table is not there")), keep,
         False, ["RuntimeError: the table is not there"], []),
    ]
    for case, job, handler, ok, failures, expected in cases:
        with subtests.test(msg=case):
            runner = JobRunner(app_key="testing")
            delivered = []
            seen_threads.clear()
            events = _watch(runner)
            try:
                with qtbot.waitSignal(runner.job_finished,
                                      timeout=10000) as caught:
                    assert runner.submit(job, handler) is True
                    assert runner.is_busy() is True
                    assert runner.pending_jobs() == 1
                assert caught.args == [ok]
                assert delivered == expected
                assert [m for kind, m in events if kind == "failed"] == \
                    failures
                assert [e for e in events if e[0] != "failed"] == [
                    ("busy", True), ("busy", False), ("finished", ok)]
                assert runner.pending_jobs() == 0
                assert runner.is_busy() is False
                if case == "success":
                    assert seen_threads["work"] != gui
                    assert seen_threads["handler"] == gui
                qtbot.waitUntil(lambda: runner.active_jobs() == 0,
                                timeout=10000)
            finally:
                runner.shutdown(2000)

    with subtests.test(msg="crash-report bookkeeping fails, job error kept"):
        from spacr import crashreport

        original = crashreport._new_error_occurrence
        crashreport._new_error_occurrence = _raise(
            OSError("reporting metadata unavailable"))
        runner = JobRunner(app_key="testing")
        try:
            events = _watch(runner)
            delivered = []
            with qtbot.waitSignal(runner.job_finished,
                                  timeout=10000) as failed:
                assert runner.submit(
                    lambda: 42,
                    _raise(ValueError("the loaded measurement is invalid")))
            assert failed.args == [False]
            assert runner._spacr_error_occurrence == ""
            with qtbot.waitSignal(runner.job_finished,
                                  timeout=10000) as recovered:
                assert runner.submit(lambda: 99, delivered.append)
            assert recovered.args == [True] and delivered == [99]
            assert [m for kind, m in events if kind == "failed"] == [
                "the loaded measurement is invalid"]
            assert not runner.is_busy() and runner.pending_jobs() == 0
        finally:
            crashreport._new_error_occurrence = original
            runner.shutdown(2000)


def test_an_unthreaded_runner_keeps_the_same_contract_inline(qapp, subtests):
    """Inline jobs return their outcome and emit failure then finished, never busy."""
    # case, job, handler, returned, delivered, events
    cases = [
        ("success with a handler", lambda: 7, "keep", True, [7],
         [("finished", True)]),
        ("success without a handler", lambda: "result", None, True, [],
         [("finished", True)]),
        ("handler raises", lambda: 7, _raise(ValueError("no good")), False,
         [], [("failed", "no good"), ("finished", False)]),
        ("job raises, handler never called", _raise(OSError("disk gone")),
         "keep", False, [],
         [("failed", "disk gone"), ("finished", False)]),
        ("job raises without a message", _raise(_Wordless()), "keep", False,
         [], [("failed", "_Wordless"), ("finished", False)]),
    ]
    for case, job, handler, returned, delivered, expected in cases:
        with subtests.test(msg=case):
            runner = JobRunner(threaded=False)
            events = _watch(runner)
            got = []
            on_done = got.append if handler == "keep" else handler
            assert runner.submit(job, on_done) is returned
            assert got == delivered
            assert events == expected
            assert (runner.is_busy(), runner.pending_jobs(),
                    runner.active_jobs()) == (False, 0, 0)


def test_cancel_retires_work_without_delivering_it(qapp, qtbot, subtests):
    """A cancelled result never reaches its handler, and a dead thread is skipped."""
    with subtests.test(msg="a result arriving after cancel"):
        runner = JobRunner(app_key="testing")
        try:
            events = _watch(runner)
            delivered = []
            started, release = threading.Event(), threading.Event()

            def work():
                started.set()
                release.wait(5)
                return "stale"

            assert runner.submit(work, delivered.append) is True
            assert started.wait(5)
            runner.cancel()
            assert (runner.is_busy(), runner.pending_jobs()) == (False, 0)
            assert runner.active_jobs() == 1
            release.set()
            qtbot.waitUntil(lambda: runner.active_jobs() == 0, timeout=10000)
            qapp.processEvents()
            assert delivered == []
            assert events == [("busy", True), ("busy", False)]
            assert runner.is_busy() is False
        finally:
            runner.shutdown(2000)

    with subtests.test(msg="a thread whose C++ half has gone"):
        runner = JobRunner(app_key="testing")
        try:
            dead = QThread()
            runner._jobs[999] = (dead, QObject())
            runner._set_busy(True)
            shiboken6.delete(dead)
            runner.cancel()
            assert runner.is_busy() is False
            assert runner._generation == 1
        finally:
            runner._jobs.clear()
            runner.shutdown(2000)


def test_a_completion_after_the_runner_is_gone_is_silent(qapp):
    """A relay into a deleted runner clears its busy flag and raises nothing."""
    doomed = JobRunner(app_key="testing")
    doomed._pending[1] = ({}, None, 0)
    shiboken6.delete(doomed)
    # The destroyed hook already forgot the deliveries.
    assert doomed._pending == {} and doomed.is_busy() is False

    doomed._pending[1] = ({}, None, 0)
    doomed._pending[2] = ({}, None, 0)
    doomed._set_busy(True)
    assert doomed.is_busy() is True
    doomed._relay(1, True)
    assert doomed.is_busy() is True and list(doomed._pending) == [2]
    doomed._relay(2, True)
    assert doomed.is_busy() is False and doomed._pending == {}

    errors = []
    old = sys.excepthook
    sys.excepthook = lambda kind, value, trace: errors.append(value)
    try:
        qapp.processEvents()
    finally:
        sys.excepthook = old
    assert errors == []


class _Runner:
    """A live-runner stand-in that records what the sweep asked of it."""

    def __init__(self, name, log, *, cancel_fails=False, shutdown_fails=False):
        self.name, self.log = name, log
        self.cancel_fails, self.shutdown_fails = cancel_fails, shutdown_fails

    def cancel(self):
        self.log.append(("cancel", self.name))
        if self.cancel_fails:
            raise RuntimeError("cannot cancel")

    def shutdown(self, remaining):
        self.log.append(("shutdown", self.name, remaining))
        if self.shutdown_fails:
            raise RuntimeError("this runner will not stop")


def test_shutdown_all_cancels_everything_then_shares_one_budget(
        qapp, monkeypatch, caplog, subtests):
    """Every runner is cancelled before any wait; failures are logged, not fatal."""
    # case, runner specs (name, cancel fails, shutdown fails), budget,
    # completed shutdowns, log lines
    cases = [
        ("nothing live", [], 25, 0, []),
        ("one willing runner", [("a", False, False)], 25, 1, []),
        ("a stubborn runner does not strand the others",
         [("stuck", False, True), ("fine", False, False)], 25, 1,
         ["could not shut down a job runner"]),
        ("a runner that cannot cancel is still shut down",
         [("stuck", True, False), ("broken", False, True)], 500, 1,
         ["could not cancel a job runner",
          "could not shut down a job runner"]),
        ("a negative budget waits not at all", [("a", False, False)], -5, 1,
         []),
    ]
    for case, specs, budget, completed, logged in cases:
        with subtests.test(msg=case):
            log = []
            runners = [_Runner(name, log, cancel_fails=c, shutdown_fails=s)
                       for name, c, s in specs]
            live = weakref.WeakSet(runners)
            monkeypatch.setattr(job_runner, "_LIVE_RUNNERS", live)
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger=job_runner.LOG.name):
                assert shutdown_all(budget) == completed
            order = list(live)
            assert [entry[:2] for entry in log] == (
                [("cancel", r.name) for r in order]
                + [("shutdown", r.name) for r in order])
            for entry in log:
                if entry[0] == "shutdown":
                    assert 0 <= entry[2] <= max(0, budget)
            assert [r.getMessage() for r in caplog.records
                    if r.name == job_runner.LOG.name] == logged

    with subtests.test(msg="a real runner is reached and shut down"):
        asked = []

        class Recording(JobRunner):
            def shutdown(self, timeout_ms=3000):
                asked.append(timeout_ms)
                super().shutdown(timeout_ms)

        monkeypatch.setattr(job_runner, "_LIVE_RUNNERS", weakref.WeakSet())
        live_runner = Recording(app_key="testing")
        assert list(job_runner._LIVE_RUNNERS) == [live_runner]
        assert shutdown_all(25) == 1
        assert len(asked) == 1 and 0 <= asked[0] <= 25
        assert job_runner.SHUTDOWN_BUDGET_MS == 2000
        del live_runner


def test_retiring_releases_only_the_jobs_whose_thread_has_stopped(
        qapp, monkeypatch):
    """The sweep drops stopped jobs and their finished ids and keeps the ones still running."""
    runner = JobRunner()
    stopped, running = object(), object()
    runner._jobs.clear()
    runner._jobs.update({1: (stopped, "w1"), 2: (running, "w2"),
                         3: (stopped, "w3")})
    runner._finished_ids.clear()
    runner._finished_ids.update({1, 2, 3})
    monkeypatch.setattr(job_runner, "thread_has_stopped",
                        lambda thread: thread is stopped)

    runner._retire_finished_jobs()

    assert runner._jobs == {2: (running, "w2")}
    assert runner._finished_ids == {2}
