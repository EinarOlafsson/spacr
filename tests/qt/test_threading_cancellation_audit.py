"""Stress Run/Stop/Close lifecycle behavior with real QThreads."""
from __future__ import annotations

import os
import threading
import time

import pytest
from PySide6.QtGui import QCloseEvent

from spacr.cancellation import checkpoint as cancellation_checkpoint
from spacr.qt.bridge import make_thread, registry


@pytest.fixture(autouse=True)
def _isolated_run_journal(monkeypatch, tmp_path):
    """Lifecycle stress must never write into the user's real run history."""
    from spacr import run_journal

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


def _console_text(console) -> str:
    from spacr.qt.widgets.console_panel import _StdoutBlock

    return "\n".join(
        block.text() for block in console.findChildren(_StdoutBlock))


def _cooperative_job(started: threading.Event):
    def run(_settings):
        started.set()
        while True:
            cancellation_checkpoint()
            time.sleep(0.002)

    return run


def _retired(thread) -> bool:
    """Has ``thread`` stopped running — including "it no longer exists"?

    ``make_thread`` wires ``thread.finished -> deleteLater``; the QThread is
    GUI-affine, so that deferred delete is flushed by the GUI loop (see the
    ownership essay in :func:`spacr.qt.bridge.make_thread`).  ``qtbot.wait``
    calls ``sendPostedEvents(None, QEvent.DeferredDelete)`` on every pass, so
    the single pump that lets the job finish also delivers ``finished`` *and*
    reaps the C++ QThread — and the next poll of the same Python wrapper
    raises ``RuntimeError: Internal C++ object (QThread) already deleted``
    rather than returning False.

    Measured, not theorised: a 50-cycle standalone harness over
    ``make_thread`` + ``qtbot.waitUntil(lambda: not thread.isRunning())``
    raised that RuntimeError on 20-22 cycles out of 50, every run.  That is
    why the naive poll cannot be used here.

    A reaped wrapper is not an error, it is the strongest possible evidence
    the thread retired: ``deleteLater`` is only ever posted from
    ``thread.finished``.  :meth:`spacr.qt.bridge.RunHandle.is_running`
    swallows the same RuntimeError for the same reason.
    """
    try:
        return not thread.isRunning()
    except RuntimeError:
        return True


def _join(qtbot, thread, timeout_ms: int = 3000) -> None:
    """Wait for ``thread`` to retire **while pumping the GUI event loop**.

    The pumping is the point, not an implementation detail.  Everything that
    tidies up after a run — ``thread.deleteLater``, ``RunHandle.retire``, the
    screens' own ``thread.finished`` slots — is queued to the GUI thread, so a
    join that never returns to the loop leaves fifty cycles' worth of teardown
    stacked up and flushed in one burst at the end.  That is the opposite of
    what these tests exist to stress: the crash this module guards against
    (two owners of one worker, gdb'd to ``QThread -> sendPostedEvents ->
    ~QObject``) needs a *new* job starting while a previous one's deferred
    deletes are being delivered.  A bare ``QThread.wait()`` runs no event loop
    and never produces that interleaving.
    """
    qtbot.waitUntil(lambda target=thread: _retired(target), timeout=timeout_ms)


def test_join_treats_a_thread_reaped_by_the_pump_as_retired(qtbot):
    """The tolerance in :func:`_retired` is load-bearing, so pin it.

    Without it every join in this module is a coin flip: the pump that
    retires the thread also runs its ``deleteLater``, and one poll in two
    lands after the reap.
    """
    class Reaped:
        def isRunning(self):
            raise RuntimeError(
                "libshiboken: Internal C++ object "
                "(PySide6.QtCore.QThread) already deleted.")

    class StillRunning:
        def isRunning(self):
            return True

    # The tolerance itself, asserted directly rather than inferred from the
    # absence of a timeout.
    assert _retired(Reaped()) is True

    # And the negative case, so the tolerance cannot silently widen into
    # "every thread is retired" -- which would make every join in this module
    # return immediately and stop stressing anything at all.
    assert _retired(StillRunning()) is False

    # _join must return promptly on a reaped wrapper. Timing it is what
    # distinguishes "recognised as retired" from "waited out the timeout and
    # happened not to raise": at 250 ms budget, a real detection is a small
    # fraction of that.
    start = time.monotonic()
    _join(qtbot, Reaped(), timeout_ms=250)
    assert time.monotonic() - start < 0.20

    # A thread that never retires must still raise, or `_join` would be a
    # no-op wearing a wait's clothes.
    with pytest.raises(Exception):
        _join(qtbot, StillRunning(), timeout_ms=150)


def test_rapid_repeated_start_cancel_retires_every_thread(
        qtbot, qt_theme_applied):
    """Fifty immediate Stop cycles must not leak or destroy a live QThread."""
    for index in range(50):
        started = threading.Event()
        thread, worker = make_thread(
            _cooperative_job(started),
            {},
            app_key=f"stress-{index}",
            journal=False,
        )
        thread.start()
        assert started.wait(10), f"cycle {index} never entered the worker"
        worker.request_cancel("rapid stop")
        thread.requestInterruption()
        _join(qtbot, thread)
        assert worker.was_cancelled
    qtbot.waitUntil(
        lambda: not any(
            handle.app_key.startswith("stress-")
            for handle in registry().active()
        ),
        timeout=3000,
    )


def _os_thread_count() -> int:
    """Threads the OS sees for this process, native ones included.

    ``threading.active_count()`` counts only threads Python created.  It is
    blind to Qt's own pools, to OpenMP, and to torch's intra-op workers —
    which is exactly the population a "the suite is running with 137 threads"
    report is looking at, and the reason such a report cannot be checked with
    the Python view alone.  Linux publishes the real number as one directory
    per thread under ``/proc/self/task``; elsewhere fall back to the Python
    view rather than skipping the measurement.
    """
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return threading.active_count()


#: A leak of the shape this module exists to catch — one helper thread per
#: run that nobody joins — adds one thread per cycle, i.e. 50 over the loop
#: below.  Qt and the pipeline imports do lazily start a small, *fixed*
#: number of native helpers the first time certain code paths run, and the
#: warm-up cycle cannot be relied on to have touched all of them.  Five is
#: therefore comfortably above the fixed noise and an order of magnitude
#: below the smallest per-cycle leak, so the tolerance cannot hide one.
_MAX_NEW_THREADS = 5


def test_fifty_run_cycles_leave_the_thread_census_flat(
        qtbot, qt_theme_applied):
    """Run fifty jobs to completion and prove the thread count does not grow.

    ``test_rapid_repeated_start_cancel_retires_every_thread`` above proves
    each individual thread retires; it says nothing about what the *process*
    accumulates, and "every join returned" is compatible with every cycle
    leaving a helper thread behind.  This is the census that distinguishes
    them, and it is the measurement to reach for the next time a Qt shard is
    reported spinning at 100% CPU with a three-digit thread count: if the
    number here is flat, the threads are ambient — Qt, OpenMP and torch
    between them hold 110-135 OS threads open in a *healthy* offscreen qt
    run — and the spin is somewhere else.

    Measured, both directions.  On this tree the fifty cycles move the
    census from 1 to 1 Python threads and 50 to 50 OS threads.  Restoring
    the per-worker "idle-flush pump" daemon that ``make_thread``'s ownership
    essay records as removed takes the same fifty cycles to 2 -> 52 Python
    and 51 -> 101 OS threads, i.e. both assertions below fire on it.
    """
    def _cycle(index: int):
        """One complete run: start, let the body finish, join.

        Deliberately *not* the immediate-Stop shape of the test above.
        ``PipelineWorker.run`` returns before it touches anything when the
        token is already cancelled — no stream router, no matplotlib
        interception — so a census built on cancelled runs would measure
        four lines of a hundred-line method.  Letting the body run is what
        puts the per-run machinery on the scale.
        """
        ran = threading.Event()
        thread, worker = make_thread(
            lambda _settings: ran.set(),
            {},
            app_key=f"census-{index}",
            journal=False,
        )
        thread.start()
        _join(qtbot, thread)
        assert ran.is_set(), f"cycle {index} never entered the pipeline body"
        return worker

    def _drained() -> bool:
        return not any(
            handle.app_key.startswith("census-")
            for handle in registry().active()
        )

    # One warm-up cycle before the baseline: the first job in a process
    # imports matplotlib and opens the reproducibility journal, and those
    # imports start native threads of their own.  Billing that one-off cost
    # to the loop would leave enough slack to hide a real per-cycle leak.
    _cycle(-1)
    qtbot.waitUntil(_drained, timeout=3000)
    baseline_python = threading.active_count()
    baseline_os = _os_thread_count()

    for index in range(50):
        assert not _cycle(index).was_cancelled

    qtbot.waitUntil(_drained, timeout=5000)
    grown_python = threading.active_count() - baseline_python
    grown_os = _os_thread_count() - baseline_os
    assert grown_python <= 0, (
        f"{grown_python} Python thread(s) survived 50 run cycles "
        f"({baseline_python} -> {threading.active_count()})"
    )
    assert grown_os <= _MAX_NEW_THREADS, (
        f"{grown_os} OS thread(s) survived 50 run cycles "
        f"({baseline_os} -> {_os_thread_count()}); a per-cycle leak would "
        f"show up as roughly 50"
    )


def test_registry_cancel_all_joins_cooperative_workers(
        qtbot, qt_theme_applied):
    started = threading.Event()
    thread, worker = make_thread(
        _cooperative_job(started), {}, app_key="registry-audit",
        journal=False,
    )
    thread.start()
    assert started.wait(2)

    remaining = registry().cancel_all(timeout_ms=3000, reason="test shutdown")

    assert remaining == []
    assert not thread.isRunning()
    assert worker.was_cancelled


def test_cancelled_worker_has_distinct_manifest_status(
        qtbot, qt_theme_applied, _isolated_run_journal):
    import json

    started = threading.Event()
    thread, worker = make_thread(
        _cooperative_job(started),
        {},
        app_key="manifest-cancel-audit",
    )
    errors = []
    worker.error.connect(errors.append)
    thread.start()
    assert started.wait(3)
    worker.request_cancel("manifest test")
    # The manifest is closed in ``PipelineWorker.run``'s finally block, before
    # ``finished`` fires and therefore before the thread's loop quits, so the
    # file on disk is complete by the time this join returns.
    _join(qtbot, thread)

    manifest_path = next(_isolated_run_journal.glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    assert errors == []


def _pin_stop_answer(monkeypatch, choice):
    """Answer the Stop prompt with ``choice`` and record how it was asked.

    ``AppScreen._on_stop`` does ``from ..shutdown import ask_how_to_quit``
    at call time, so patching the attribute on the module is what reaches
    it. Unstubbed, ``QMessageBox.exec`` trips the headless guard in
    ``tests/qt/conftest.py`` — a modal has nobody to answer it here.
    """
    import spacr.qt.shutdown as shutdown

    asked: list[dict] = []

    def _answer(*_args, **kwargs):
        asked.append(kwargs)
        return choice

    monkeypatch.setattr(shutdown, "ask_how_to_quit", _answer)
    return asked


def test_app_screen_stop_asks_then_cooperates_and_stays_live_to_escalate(
        qtbot, qt_theme_applied, monkeypatch):
    """A cooperative Stop cancels the run, says so, and does NOT disable Stop.

    Until 2026-08-06 Stop requested cancellation immediately and then
    disabled itself. Commit 5cb219f1 ("stop: ask whether to wait or to
    kill, instead of asking once and hoping") changed both halves: it asks
    through ``shutdown.ask_how_to_quit`` first, and the button deliberately
    stays enabled, because a cooperative stop that never lands used to
    leave the user with no way to escalate. The old
    ``assert not screen._btn_stop.isEnabled()`` pinned exactly the
    misfeature that commit removed, so it is INVERTED here rather than
    dropped -- re-disabling the button must fail this test.

    ``tests/qt/test_stop_soft_and_hard.py`` covers the three dialog answers
    against fake threads; this is the end-to-end case, on a real QThread
    running a real ``cancellation_checkpoint``.
    """
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.shutdown import GRACEFUL

    asked = _pin_stop_answer(monkeypatch, GRACEFUL)
    started = threading.Event()
    monkeypatch.setattr(
        "spacr.qt.screens.app_screen.resolve_pipeline_entry",
        lambda _key: _cooperative_job(started),
    )
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen.show()
    screen._on_run()
    assert started.wait(2)

    screen._on_stop()

    assert asked, "Stop cancelled the run without asking first"
    # A button labelled Stop that opens a dialog headed "Quit" reads as the
    # wrong dialog, and a user who thinks they mis-clicked cancels out of
    # the thing they wanted -- hence the `verb` argument.
    assert asked[0].get("verb") == "Stop"
    assert screen._btn_stop.isEnabled(), (
        "Stop disabled itself again; a cooperative stop that does not land "
        "then leaves the user with no way to escalate")
    assert screen._btn_stop.property("buttonActionBusy") is True
    qtbot.waitUntil(lambda: screen._btn_run.isEnabled(), timeout=5000)
    qtbot.waitUntil(lambda: screen._thread is None, timeout=5000)

    text = _console_text(screen._console)
    assert "Requesting stop" in text
    assert "Stopped safely" in text
    assert "Failed — see traceback" not in text


def test_app_screen_stop_cancelled_at_the_prompt_leaves_a_real_run_alone(
        qtbot, qt_theme_applied, monkeypatch):
    """Answering Cancel must not touch the run, on a live thread.

    The companion of the test above, and the more important half: since
    2026-08-06 Stop opens a prompt, so a mis-click is now recoverable --
    but only if Cancel really is inert. ``test_stop_soft_and_hard.py``
    asserts that against a fake thread; this asserts it against a worker
    that is genuinely running and genuinely checking for cancellation, so
    "nothing happened" is measured on the object that would have died.
    """
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.shutdown import CANCEL, GRACEFUL

    _pin_stop_answer(monkeypatch, CANCEL)
    started = threading.Event()
    monkeypatch.setattr(
        "spacr.qt.screens.app_screen.resolve_pipeline_entry",
        lambda _key: _cooperative_job(started),
    )
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen.show()
    screen._on_run()
    assert started.wait(2)
    thread, worker = screen._thread, screen._worker

    try:
        screen._on_stop()

        assert screen._thread is thread
        assert thread.isRunning()
        assert worker.was_cancelled is False
        assert not thread.isInterruptionRequested()
        assert screen._btn_stop.property("buttonActionBusy") is not True
        assert "Requesting stop" not in _console_text(screen._console)
    finally:
        # The job loops forever; it only ends because something cancels it.
        _pin_stop_answer(monkeypatch, GRACEFUL)
        screen._on_stop()
        qtbot.waitUntil(lambda: screen._thread is None, timeout=5000)


def test_screen_close_refuses_to_drop_a_live_stubborn_worker(
        qtbot, qt_theme_applied):
    from spacr.qt.screens.app_screen import AppScreen

    class StubbornWorker:
        def request_cancel(self, _reason):
            return True

    class StubbornThread:
        def requestInterruption(self):
            pass

        def quit(self):
            pass

        def wait(self, _timeout):
            return False

        def isRunning(self):
            return True

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    thread = StubbornThread()
    worker = StubbornWorker()
    screen._thread = thread
    screen._worker = worker
    event = QCloseEvent()

    screen.closeEvent(event)

    assert not event.isAccepted()
    assert screen._thread is thread
    assert screen._worker is worker
    assert "Close deferred" in _console_text(screen._console)
    screen._thread = None
    screen._worker = None


def test_main_window_refuses_shutdown_while_analysis_is_live(
        qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt.app import MainWindow
    import spacr.qt.bridge as bridge

    class LiveHandle:
        app_key = "measure"

    warnings = []
    monkeypatch.setattr(
        bridge.registry(),
        "cancel_all",
        lambda **_kwargs: [LiveHandle()],
    )
    monkeypatch.setattr(
        "spacr.qt.app.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )
    window = MainWindow()
    qtbot.addWidget(window)
    event = QCloseEvent()

    window.closeEvent(event)

    assert not event.isAccepted()
    assert not window._closing
    assert warnings and "finishing the current field" in warnings[0][2]
