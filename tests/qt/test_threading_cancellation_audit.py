"""Stress Run/Stop/Close lifecycle behavior with real QThreads."""
from __future__ import annotations

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
        thread, worker = make_thread(
            lambda _settings: cancellation_checkpoint(),
            {},
            app_key=f"stress-{index}",
            journal=False,
        )
        thread.start()
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


def test_app_screen_stop_is_cooperative_and_reports_cancelled(
        qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt.screens.app_screen import AppScreen

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
    assert not screen._btn_stop.isEnabled()
    assert screen._btn_stop.property("buttonActionBusy") is True
    qtbot.waitUntil(lambda: screen._btn_run.isEnabled(), timeout=5000)
    qtbot.waitUntil(lambda: screen._thread is None, timeout=5000)

    text = _console_text(screen._console)
    assert "Requesting stop" in text
    assert "Stopped safely" in text
    assert "Failed — see traceback" not in text


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
