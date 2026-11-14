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


# ---------------------------------------------------------------------------
# a handler that raises
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# a job that raises
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# cancelling
# ---------------------------------------------------------------------------


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


def test_a_runner_destroyed_mid_job_forgets_the_delivery_it_can_no_longer_make(
        qapp, qtbot):
    """Qt drops a dead runner's completion slot, so its pending record goes too.

    The Figure Queue's PDF render outlived its widget on a slow runner
    (Qt shard 2, 2026-10-20): the runner reported one pending, busy job
    forever, because the completion that would have cleared it was never
    delivered.
    """
    from spacr.qt.bridge import thread_has_stopped

    release = threading.Event()
    doomed = JobRunner(app_key="testing")
    assert doomed.submit(lambda: release.wait(10), lambda _result: None)
    threads = [pair[0] for pair in doomed._jobs.values()]
    assert doomed.pending_jobs() == 1 and doomed.is_busy()

    shiboken6.delete(doomed)
    assert doomed.pending_jobs() == 0
    assert not doomed.is_busy()

    release.set()
    qtbot.waitUntil(lambda: all(thread_has_stopped(t) for t in threads),
                    timeout=10000)
    qapp.processEvents()
    assert doomed.pending_jobs() == 0


def test_forgetting_the_deliveries_of_a_collected_runner_does_nothing():
    """The destroyed hook holds only a weak reference, which may be dead."""
    from spacr.qt.job_runner import _forget_deliveries

    assert _forget_deliveries(lambda: None) is None


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
