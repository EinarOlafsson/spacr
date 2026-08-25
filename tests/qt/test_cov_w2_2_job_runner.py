"""What a background job does when it, or the screen behind it, goes wrong.

`JobRunner` exists because eleven hand-rolled copies of it disagreed, and the
disagreements were all in the failure paths: a handler that raised, a worker
that raised, a screen destroyed while a thread was still winding down. Those
are the paths driven here, with real QThreads through `make_thread` rather
than an inline stand-in, because the bug this class was written for --
`thread.finished` connected to a closure -- only exists once a real thread
really finishes.

The two branches that guard against a runner's C++ half being taken away are
reached the way they happen: by deleting it.
"""

import threading

import pytest
import shiboken6
from PySide6.QtCore import QObject, QThread

from spacr.qt.job_runner import JobRunner


@pytest.fixture
def runner(qapp, qtbot):
    """A threaded runner that is shut down however the test ends."""
    made = JobRunner(app_key="testing")
    yield made
    made.shutdown(2000)


# ---------------------------------------------------------------------------
# the ordinary round trip
# ---------------------------------------------------------------------------

def test_a_job_runs_off_the_gui_thread_and_lands_back_on_it(runner, qtbot):
    """The callable runs on a worker thread; its handler runs on the GUI one.

    Both halves are asserted by thread identity rather than by trusting the
    wiring, because the whole class exists to keep the two apart: a handler
    that ran on the worker thread would be touching widgets from there.
    """
    gui_thread = threading.get_ident()
    ran_on = {}
    delivered = []

    def work():
        ran_on["worker"] = threading.get_ident()
        return 21 * 2

    def done(value):
        ran_on["handler"] = threading.get_ident()
        delivered.append(value)

    with qtbot.waitSignal(runner.job_finished, timeout=10000) as caught:
        assert runner.submit(work, done) is True

    assert caught.args == [True]
    assert delivered == [42]
    assert ran_on["worker"] != gui_thread, "the job ran on the GUI thread"
    assert ran_on["handler"] == gui_thread
    assert runner.pending_jobs() == 0
    assert runner.is_busy() is False


def test_the_runner_says_it_is_busy_only_while_work_is_outstanding(runner,
                                                                   qtbot):
    """`busy_changed` goes true on submit and false again on delivery."""
    seen = []
    runner.busy_changed.connect(seen.append)

    with qtbot.waitSignal(runner.job_finished, timeout=10000):
        runner.submit(lambda: "done", lambda _v: None)

    assert seen == [True, False]


# ---------------------------------------------------------------------------
# a handler that raises
# ---------------------------------------------------------------------------

def test_a_handler_that_raises_still_retires_its_job(runner, qtbot):
    """The failure is reported, and the runner does not stay busy.

    Bookkeeping happens for every job, including a failed one; a screen left
    permanently "busy" by an exception in a handler is the leak this
    guarantee exists to prevent.
    """
    failures = []
    runner.job_failed.connect(failures.append)

    def explode(_value):
        raise ValueError("the handler could not use the result")

    with qtbot.waitSignal(runner.job_finished, timeout=10000) as caught:
        runner.submit(lambda: 1, explode)

    assert caught.args == [False]
    assert failures == ["the handler could not use the result"]
    assert runner.pending_jobs() == 0
    assert runner.is_busy() is False


class _Wordless(Exception):
    """An exception that carries no message, which some libraries raise."""


def test_a_failure_with_no_message_is_reported_by_its_type(runner, qtbot):
    """An exception whose str() is empty still produces a status line.

    An empty status bar after a failed job is indistinguishable from a
    successful one.
    """
    failures = []
    runner.job_failed.connect(failures.append)

    def explode(_value):
        raise _Wordless()

    with qtbot.waitSignal(runner.job_finished, timeout=10000):
        runner.submit(lambda: 1, explode)

    assert failures == ["_Wordless"]


# ---------------------------------------------------------------------------
# a job that raises
# ---------------------------------------------------------------------------

def test_a_job_that_raises_reports_the_last_line_of_its_traceback(runner,
                                                                  qtbot):
    """The status bar gets the exception line, not the whole traceback.

    The worker emits a full traceback; a status bar has one line, and the
    useful one is the last.
    """
    failures = []
    runner.job_failed.connect(failures.append)

    def work():
        raise RuntimeError("the table is not there")

    with qtbot.waitSignal(runner.job_finished, timeout=10000) as caught:
        runner.submit(work, lambda _v: None)

    assert caught.args == [False]
    assert failures, "a raising job reported nothing"
    assert failures[-1] == "RuntimeError: the table is not there"


def test_an_error_with_nothing_in_it_still_says_something(runner, qtbot):
    """Blank error text becomes 'unknown error' rather than an empty line."""
    failures = []
    runner.job_failed.connect(failures.append)
    with qtbot.waitSignal(runner.job_failed, timeout=5000):
        runner._on_worker_error_text("   \n\n  \n")
    assert failures == ["unknown error"]


# ---------------------------------------------------------------------------
# running inline
# ---------------------------------------------------------------------------

def test_an_unthreaded_runner_emits_the_same_signals_in_the_same_order(qapp):
    """`threaded=False` is the same contract, run synchronously."""
    runner = JobRunner(threaded=False)
    order = []
    runner.job_finished.connect(lambda ok: order.append(("finished", ok)))
    runner.job_failed.connect(lambda msg: order.append(("failed", msg)))

    delivered = []
    assert runner.submit(lambda: 7, delivered.append) is True
    assert delivered == [7]
    assert order == [("finished", True)]


def test_an_unthreaded_handler_that_raises_reports_and_returns_false(qapp):
    """A handler exception inline is reported exactly as it is when threaded.

    The point of the inline mode is that a test driving a screen synchronously
    sees the same behaviour; a mode that swallowed handler errors would make
    the test greener than the application.
    """
    runner = JobRunner(threaded=False)
    failures = []
    finished = []
    runner.job_failed.connect(failures.append)
    runner.job_finished.connect(finished.append)

    def explode(_value):
        raise ValueError("no good")

    assert runner.submit(lambda: 7, explode) is False
    assert failures == ["no good"]
    assert finished == [False]


def test_an_unthreaded_job_that_raises_never_calls_the_handler(qapp):
    """The result handler is not reached when the work itself failed."""
    runner = JobRunner(threaded=False)
    failures = []
    called = []
    runner.job_failed.connect(failures.append)

    def work():
        raise OSError("disk gone")

    assert runner.submit(work, called.append) is False
    assert called == []
    assert failures == ["disk gone"]


# ---------------------------------------------------------------------------
# cancelling
# ---------------------------------------------------------------------------

def test_a_cancelled_result_is_retired_but_never_delivered(runner, qtbot):
    """After `cancel` the handler is not called, and the runner is not busy.

    Both halves matter: skipping the bookkeeping to avoid the handler is what
    left the old copies permanently busy.
    """
    delivered = []
    started = threading.Event()
    release = threading.Event()

    def work():
        started.set()
        release.wait(5)
        return "stale"

    runner.submit(work, delivered.append)
    assert started.wait(5)
    runner.cancel()
    assert runner.is_busy() is False
    assert runner.pending_jobs() == 0

    release.set()
    qtbot.waitUntil(lambda: runner.active_jobs() == 0, timeout=10000)

    assert delivered == [], "a cancelled result reached its handler"
    assert runner.is_busy() is False


def test_cancelling_survives_a_thread_whose_c_half_is_already_gone(runner):
    """A QThread PySide6 has destroyed is skipped, not raised over.

    Retiring and deleting a finished thread is wired off the same signal, so
    a runner asked to cancel can legitimately be holding a reference to an
    object that no longer exists.
    """
    dead = QThread()
    runner._jobs[999] = (dead, QObject())
    shiboken6.delete(dead)

    runner.cancel()          # must not raise
    assert runner.is_busy() is False


def test_shutdown_survives_a_drain_that_raises(runner, monkeypatch):
    """A drain that throws does not stop the remaining threads being released.

    `shutdown` runs from `closeEvent`; an exception escaping it leaves the
    widget half-closed with a live QThread, which is the abort this module is
    arranged around.
    """
    from spacr.qt import bridge

    def explode(_thread, _worker=None, timeout_ms=0):
        raise RuntimeError("Internal C++ object already deleted")

    runner._jobs[998] = (QThread(), QObject())
    monkeypatch.setattr(bridge, "drain_thread", explode)

    runner.shutdown(10)      # must not raise
    assert runner.active_jobs() == 0


# ---------------------------------------------------------------------------
# the relay across the thread boundary
# ---------------------------------------------------------------------------

def test_a_completion_arriving_after_the_runner_is_gone_is_silent(qapp):
    """A parked worker finishing after its runner died raises nothing.

    Unguarded, PySide6's "Signal source has been deleted" surfaces as an
    unhandled exception inside the Qt event loop and fails whichever test
    runs next -- which is how this was found.
    """
    doomed = JobRunner(app_key="testing")
    shiboken6.delete(doomed)

    doomed._relay(1, True)   # must not raise


# ---------------------------------------------------------------------------
# what travels between the threads
# ---------------------------------------------------------------------------

def test_a_jobs_return_value_travels_in_the_settings_dict():
    """`_capture` leaves the return value where the worker's signal cannot.

    `PipelineWorker.finished` carries only a success flag, so the result has
    to ride in the dict the worker was handed. Called directly because it
    only ever runs on a worker thread.
    """
    from spacr.qt.job_runner import _capture

    payload = {}
    _capture(lambda: {"rows": 3}, payload)
    assert payload == {"result": {"rows": 3}}


def test_a_job_returning_none_is_still_recorded_as_having_returned():
    """A None result is stored, not left absent.

    "the job returned None" and "the job never finished" are different, and
    the completion handler tells them apart by the key being there.
    """
    from spacr.qt.job_runner import _capture

    payload = {}
    _capture(lambda: None, payload)
    assert "result" in payload
    assert payload["result"] is None


# ---------------------------------------------------------------------------
# the quit-time sweep
# ---------------------------------------------------------------------------

def test_every_live_runner_is_asked_to_stop_at_quit(qapp):
    """`shutdown_all` reaches a runner whose widget is never closed.

    A runner shuts down in its widget's `closeEvent`, which does not happen
    when the application quits: the widget is destroyed, not closed. Qt aborts
    the process if a running QThread is destroyed, so the sweep is what keeps
    a quit with a job in flight from taking the process down without a word.
    """
    import gc

    from spacr.qt.job_runner import shutdown_all

    asked_of = []

    class Recording(JobRunner):
        def shutdown(self, timeout_ms=3000):
            asked_of.append(timeout_ms)
            super().shutdown(timeout_ms)

    live = Recording(app_key="testing")
    try:
        assert shutdown_all(25) >= 1
        assert asked_of == [25]
    finally:
        del live
        gc.collect()


def test_one_runner_that_will_not_stop_does_not_strand_the_others(qapp):
    """A shutdown that raises is logged and the sweep carries on.

    Ordering matters more than completeness on the way out; a single stuck
    runner must not leave the rest of them holding running threads.
    """
    import gc

    from spacr.qt.job_runner import shutdown_all

    reached = []

    class Stubborn(JobRunner):
        def shutdown(self, timeout_ms=3000):
            raise RuntimeError("this runner will not stop")

    class Willing(JobRunner):
        def shutdown(self, timeout_ms=3000):
            reached.append(timeout_ms)
            super().shutdown(timeout_ms)

    stuck = Stubborn(app_key="testing")
    fine = Willing(app_key="testing")
    try:
        asked = shutdown_all(25)      # must not raise
        assert reached == [25], "the sweep stopped at the stuck runner"
        assert asked >= 1
    finally:
        del stuck, fine
        gc.collect()
