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
import inspect
import subprocess
import sys
import textwrap
import threading

import pytest

from spacr.qt import gc_policy


@pytest.fixture(autouse=True)
def _restore_policy(_the_widget_tree_does_not_outgrow_the_session):
    """Exercise installation from a clean state after the suite's setup."""
    gc_policy.uninstall()
    if not gc.isenabled():
        gc.enable()
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
    # The explicit sweep must not touch Qt objects left by earlier tests.
    # A fresh interpreter keeps this deliberately unsafe counterexample
    # entirely Python-only, regardless of the suite's collection order.
    probe = 'import gc, sys, threading\n' + inspect.getsource(_RecordsItsDestroyingThread)
    probe += textwrap.dedent('''
        destroyed_on = []
        gc.disable()
        gc.collect()
        for _ in range(20):
            _RecordsItsDestroyingThread(destroyed_on)
        assert not destroyed_on
        thread = threading.Thread(target=gc.collect, name="pretend-preview-worker")
        thread.start()
        thread.join()
        assert len(destroyed_on) == 20
        assert set(destroyed_on) == {"pretend-preview-worker"}
        assert 'PySide6' not in sys.modules
    ''')
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


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


# -- failing to install must never be worse than the defect ------------------
#
# `install` promises it "never raises, because failing to install a mitigation
# must not be worse than the defect it mitigates". Every way it can fail is
# below, and each one asserts the SAME two things: it returns False, and
# automatic collection is left switched ON. A path that returned False with
# `gc.disable()` still in force would be the worst outcome of all -- cycles
# collected by nobody, in a process that thinks it has a policy.


def test_installing_twice_is_a_no_op_rather_than_a_second_timer(qapp):
    """Two timers would double the tick rate and leave one unstoppable.

    `uninstall` forgets all but the last timer it was given, so the first
    would go on collecting after a caller believed the policy was off.
    """
    assert gc_policy.install(qapp) is True
    assert gc_policy.install(qapp) is False
    assert gc_policy.is_installed() is True
    assert gc_policy.uninstall() is True
    assert gc_policy.is_installed() is False


def test_without_pyside_the_policy_declines_and_leaves_gc_alone(monkeypatch):
    """spaCR's headless CLI imports this module; there is no Qt there.

    The import is what fails, before anything is touched, so the interpreter
    keeps its own policy and nothing has to be put back.
    """
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "PySide6.QtCore":
            raise ImportError("No module named 'PySide6'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    assert gc_policy.install(None) is False
    assert gc_policy.is_installed() is False
    assert gc.isenabled() is True


def test_a_timer_that_cannot_be_started_puts_automatic_collection_back(
    monkeypatch, qapp,
):
    """THE PATH THAT MATTERS. gc is disabled BEFORE the timer is built.

    If constructing or starting the timer then fails, automatic collection
    has already been switched off -- and leaving it off would mean cyclic
    garbage is collected by nobody at all, which is strictly worse than the
    cross-thread destructor this module exists to prevent.
    """
    from PySide6 import QtCore

    class _RefusesToStart(QtCore.QTimer):
        def start(self, *args, **kwargs):
            raise RuntimeError("no event loop to attach to")

    monkeypatch.setattr(QtCore, "QTimer", _RefusesToStart)

    assert gc.isenabled() is True
    assert gc_policy.install(qapp) is False
    assert gc_policy.is_installed() is False
    assert gc.isenabled() is True, (
        "install failed with automatic collection still disabled: nothing "
        "would ever collect a cycle in this process")


def test_a_timer_that_cannot_be_stopped_still_uninstalls(monkeypatch, qapp):
    """Shutdown order can destroy the QApplication before this is called.

    Stopping the timer then raises, and the policy must still come off:
    otherwise `is_installed` stays true for ever and automatic collection is
    never restored.
    """
    assert gc_policy.install(qapp) is True

    class _Wedged:
        def stop(self):
            raise RuntimeError("Internal C++ object already deleted")

        def setParent(self, _parent):
            raise AssertionError("not reached; stop() raised first")

    with monkeypatch.context() as patch:
        patch.setattr(gc_policy, "_timer", _Wedged())
        assert gc_policy.uninstall() is True
        assert gc_policy.is_installed() is False
        assert gc.isenabled() is True
