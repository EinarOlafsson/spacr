"""The cyclic collector must not run destructors on a worker thread.

THE CRASH THIS IS ABOUT. Pressing Run in the live preview with ``cpsam`` gave::

    QObject::killTimer: Timers cannot be stopped from another thread
    QObject::~QObject: Timers cannot be stopped from another thread
    Segmentation fault (core dumped)

and the reproduction needs no spaCR code: a QObject owning a *running* QTimer,
dropped into a reference cycle so only the collector can free it, plus a
``gc.collect()`` on a worker thread. See :mod:`spacr.qt.gc_policy`.

WHY THE Qt HALF OF THAT IS NOT REPRODUCED HERE. It ends in a segmentation
fault, and a suite that deliberately corrupts its own process cannot then be
trusted about anything it reports afterwards -- instruction 288 is already
chasing a ``tests/qt`` segfault whose cause is an interaction between files.
So the load-bearing claim is proven with a plain Python object that records
which thread its destructor ran on, which is the same claim without the
crash: **the collecting thread runs the destructors.**
"""
from __future__ import annotations

import gc
import threading

import pytest

from spacr.qt import gc_policy


@pytest.fixture(autouse=True)
def _restore_policy():
    """Never leave automatic collection off for the rest of the suite."""
    yield
    gc_policy.uninstall()
    if not gc.isenabled():
        gc.enable()


class _RecordsItsDestroyingThread:
    """Stands in for a QObject: it notices where it was destroyed."""

    def __init__(self, log):
        self._log = log
        self.self_ref = self          # a CYCLE -- only the collector frees it

    def __del__(self):
        self._log.append(threading.current_thread().name)


def test_a_worker_thread_collection_runs_destructors_on_that_worker():
    """The mechanism behind the crash, stated as a test.

    This is the reason the defect exists at all. If destructors ran on the
    thread that CREATED the object, a worker-thread collection would be
    harmless and no policy would be needed.
    """
    destroyed_on = []
    for _ in range(20):
        _RecordsItsDestroyingThread(destroyed_on)

    def worker():
        gc.collect()

    thread = threading.Thread(target=worker, name="pretend-preview-worker")
    thread.start()
    thread.join()

    assert "pretend-preview-worker" in destroyed_on, (
        "the collector ran the destructor somewhere other than the collecting "
        "thread, so the premise of gc_policy no longer holds")


def test_the_policy_stops_a_worker_allocating_its_way_into_a_collection(qapp):
    """With the policy in force, a worker cannot trigger an automatic sweep.

    Allocating past the threshold is exactly what a Cellpose pass does, and it
    is what made the preview worker -- rather than the GUI thread -- the one
    that inherited the sweep.
    """
    assert gc_policy.install(qapp) is True
    assert gc.isenabled() is False, (
        "automatic collection is still on, so a worker can still be handed a "
        "sweep no matter what the timer does")

    destroyed_on = []
    for _ in range(50):
        _RecordsItsDestroyingThread(destroyed_on)

    def worker():
        # Far past gen-0's default threshold of 700: this WOULD collect here.
        junk = [{"n": i} for i in range(50_000)]
        del junk

    thread = threading.Thread(target=worker, name="pretend-preview-worker")
    thread.start()
    thread.join()

    assert "pretend-preview-worker" not in destroyed_on, (
        "a worker thread still ran destructors, which is the crash")


def test_the_gui_thread_still_collects_so_memory_is_not_abandoned():
    """Switching automatic collection off without replacing it would be worse
    than the defect: cycles would simply never be freed."""
    # BEFORE install, because the tick measures against the thresholds it
    # captured at install time -- setting them afterwards changes what the
    # interpreter would do and not what the tick does.
    gc.set_threshold(1, 1, 1)
    try:
        gc_policy.install(None)
        destroyed_on = []
        for _ in range(50):
            _RecordsItsDestroyingThread(destroyed_on)
        assert gc_policy.collect_once() >= 0, "nothing was collected"
    finally:
        gc.set_threshold(700, 10, 10)

    assert destroyed_on, "the tick collected nothing at all"
    assert set(destroyed_on) == {threading.current_thread().name}


def test_nothing_due_means_no_sweep():
    """The tick reproduces CPython's policy rather than sweeping every second.

    A full ``gc.collect()`` on every tick would walk every live numpy array in
    the process once a second, which is a performance defect traded for a
    correctness one.
    """
    gc.collect()
    gc.set_threshold(1_000_000, 1_000_000, 1_000_000)
    try:
        gc_policy.install(None)        # captures the raised thresholds
        assert gc_policy.collect_once() == -1
    finally:
        gc.set_threshold(700, 10, 10)


def test_uninstall_gives_the_interpreter_its_policy_back(qapp):
    gc_policy.install(qapp)
    assert gc_policy.is_installed() is True
    assert gc_policy.uninstall() is True
    assert gc.isenabled() is True
    assert gc_policy.is_installed() is False
    assert gc_policy.uninstall() is False, "a second uninstall is a no-op"


def test_the_policy_is_actually_installed_at_startup():
    """A source-level check, because every behavioural test above passes just
    as happily when nothing ever calls ``install`` in the real application."""
    from pathlib import Path

    import spacr.qt.app as app_module

    source = Path(app_module.__file__).read_text(encoding="utf-8")
    assert "from .gc_policy import install as _install_gc_policy" in source
    assert "_install_gc_policy(app)" in source, (
        "the policy exists but the application never installs it")
