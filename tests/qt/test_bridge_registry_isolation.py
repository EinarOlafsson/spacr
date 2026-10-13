"""A job one test starts must not still be running as far as the next is told.

``bridge.make_thread`` puts a ``RunHandle`` in a PROCESS-WIDE registry and
takes it out again when the thread emits ``finished``. A test that stubs
``QThread.start``, or that builds a thread it never starts, gets no
``finished`` — so the handle stays registered for the rest of the process.

The cost is not the memory. ``registry().is_busy()`` answers True from then
on, the background activity spinner turns for a job nobody started, and a test
that asks whether spaCR is busy is told about a different file's work. That is
the order-dependent failure this suite has the most trouble with: it passes
alone and fails after somebody else, and the blame lands on whichever test
drew the short straw.

So the registry is cleaned at each test's setup, and only of handles with no
live thread. Both halves are pinned here — that a stale handle is gone, and
that a running one is left alone, since removing a live job's bookkeeping
would be a worse bug than the one being fixed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QThread

from spacr.qt import bridge


def _handles():
    return list(bridge.registry().active())


def test_a_job_whose_thread_never_started_is_left_registered(monkeypatch):
    """The leak itself, made on purpose so the next test can find it.

    Stubbing ``start`` is what several real tests here do -- they are about
    what happens BEFORE a thread runs -- and it is enough to strand the
    handle, because retirement is wired to ``finished``.
    """
    monkeypatch.setattr(QThread, "start", lambda self, *a, **k: None)

    thread, worker = bridge.make_thread(lambda settings: None, {},
                                        "a-job-that-never-runs",
                                        journal=False)
    thread.start()

    assert any(handle.app_key == "a-job-that-never-runs"
               for handle in _handles())
    assert bridge.registry().is_busy()


def test_the_next_test_is_not_told_that_job_is_running():
    """Runs straight after the leak above, which is the whole point.

    No fixture is requested and nothing is cleaned up here: if this passes,
    the cleaning happened between the two tests, which is where it has to
    happen for every other file in this suite to benefit from it.
    """
    assert not any(handle.app_key == "a-job-that-never-runs"
                   for handle in _handles())


def test_a_job_that_is_really_running_is_never_unregistered(qtbot):
    """The dangerous half. A live job's bookkeeping must survive.

    Cleaning up a handle whose thread is still working would leave the run
    with no record of it -- no spinner, no cancel path, and a QThread nothing
    holds a reference to, which is a process abort rather than an exception.
    """
    from spacr.qt import preferences  # noqa: F401  (import cost, not use)

    started = []

    def _slow(settings):
        started.append(True)
        QThread.msleep(400)

    thread, worker = bridge.make_thread(_slow, {}, "a-job-that-is-working",
                                        journal=False)
    thread.start()
    try:
        qtbot.waitUntil(lambda: bool(started), timeout=5000)

        # The cleaning rule, applied to the live job by hand rather than
        # waiting a whole test boundary for it: it must decline.
        assert not bridge.thread_has_stopped(thread)
        for handle in _handles():
            if bridge.thread_has_stopped(getattr(handle, "thread", None)):
                bridge.registry().unregister(handle)

        assert any(handle.app_key == "a-job-that-is-working"
                   for handle in _handles())
    finally:
        thread.quit()
        thread.wait(5000)
