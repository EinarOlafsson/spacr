"""Run / Stop / finish / close: the ``AppScreen`` run state machine.

Run disables itself, enables Stop and shows progress while a real worker
thread runs the pipeline entry; both the success and the failure arm hand
the buttons back. Stop asks how to quit and then has to be honest about what
it managed; closing the screen has to end every background part even when
some of them refuse, because the one left running is a QThread and a running
QThread destroyed with its parent aborts the process.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QCloseEvent                              # noqa: E402

from tests.qt.test_app_screen_a_construction import (               # noqa: E402,F401
    _reset_hover_tooltip_around_each_test,
    built,
    console_text,
    forbid_modals,
    recording_message_box,
)

pytestmark = pytest.mark.qt

ENTRY = "spacr.qt.screens.app_screen.resolve_pipeline_entry"


@pytest.mark.usefixtures("console_info_enabled")
def test_run_walks_idle_running_finished_for_every_pipeline_outcome(
        qtbot, subtests):
    """For a pipeline that succeeds (stopped mid-run), one that raises, and
    one whose console run-context decoration raises: Run/Stop/progress go
    idle -> running -> finished, the worker gets the collected settings, the
    console reports the outcome, and thread refs are released afterwards.
    """
    from spacr.qt import shutdown

    def succeeding(scr, mp):
        gate = threading.Event()
        seen = {}

        def _fn(settings):
            seen["src"] = settings.get("src")
            print("worker-alive")
            gate.wait(10)
            print("worker-done")

        mp.setattr(ENTRY, lambda key: _fn)
        mp.setattr(shutdown, "ask_how_to_quit",
                   lambda *a, **k: shutdown.GRACEFUL)
        scr._settings_model._widgets["src"].setText("/data/state_machine")
        # A cold process: the run itself must prepare pyplot for figure
        # capture (as the first run after launch does), so that path is
        # exercised however earlier tests in this worker left matplotlib.
        import matplotlib
        mp.delitem(sys.modules, "matplotlib.pyplot", raising=False)
        mp.setattr(matplotlib, "pyplot", None, raising=False)
        try:
            scr._on_run()
            assert "matplotlib.pyplot" in sys.modules
            assert not scr._btn_run.isEnabled()
            assert scr._btn_stop.isEnabled()
            assert scr._progress.isVisibleTo(scr)
            assert scr._thread is not None and scr._worker is not None
            # The worker reaches the entry only after the run journal
            # snapshots the environment; cold and loaded that takes ~9 s.
            qtbot.waitUntil(lambda: "worker-alive" in console_text(
                scr._console), timeout=30000)
            assert seen["src"] == "/data/state_machine"
            # Parked on the gate for longer than one GUI housekeeping tick:
            # the screen must still read "running" throughout.
            qtbot.wait(1500)
            assert not scr._btn_run.isEnabled() and scr._btn_stop.isEnabled()
            assert scr._thread.isRunning()
            scr._on_stop()                      # advisory: asks, never kills
            assert "Requesting stop" in console_text(scr._console)
            assert scr._thread.isInterruptionRequested()
        finally:
            gate.set()
        return ["Starting mask", "worker-done", "✓ Finished"], []

    def failing(scr, mp):
        def _fn(settings):
            raise RuntimeError("pipeline exploded")

        mp.setattr(ENTRY, lambda key: _fn)
        scr._on_run()
        return (["✗ Failed — see traceback above", "pipeline exploded"],
                [lambda: "RuntimeError" in scr._last_error_text])

    def decoration_fails(scr, mp):
        mp.setattr(ENTRY, lambda key: (lambda s: print("ran-anyway")))
        scr._console.set_run_context = lambda *a, **k: (
            _ for _ in ()).throw(RuntimeError("console busy"))
        scr._on_run()
        return ["ran-anyway"], []

    for name, drive in (("success", succeeding), ("failure", failing),
                        ("run context fails", decoration_fails)):
        with subtests.test(case=name), pytest.MonkeyPatch.context() as mp:
            forbid_modals(mp)
            with built("mask") as scr:
                assert scr._btn_run.isEnabled()
                assert not scr._btn_stop.isEnabled()
                assert not scr._progress.isVisibleTo(scr)
                assert scr._thread is None
                expected, extra = drive(scr, mp)
                qtbot.waitUntil(lambda: scr._btn_run.isEnabled(),
                                timeout=30000)
                qtbot.waitUntil(lambda: all(
                    s in console_text(scr._console) for s in expected),
                    timeout=10000)
                assert not scr._btn_stop.isEnabled()
                assert not scr._progress.isVisibleTo(scr)
                assert all(check() for check in extra)
                qtbot.waitUntil(lambda: scr._thread is None, timeout=10000)
                assert scr._worker is None


def test_run_refuses_to_start_what_it_cannot_run(qtbot, subtests):
    """An interactive-only app explains itself ("Not runnable") and settings
    that cannot be collected warn ("Bad settings" + the message); in both
    cases nothing starts and Run stays enabled.
    """
    def boom():
        raise ValueError("channels must be a list")

    cases = [
        ("interactive-only app", "annotate", None,
         ("info", "Not runnable"), "'annotate' app is interactive-only"),
        ("unreadable settings", "mask", boom,
         ("warn", "Bad settings"), "channels must be a list"),
    ]
    for name, app, collect, (kind, title), body in cases:
        with subtests.test(case=name), pytest.MonkeyPatch.context() as mp:
            if collect is not None:
                mp.setattr(ENTRY, lambda key: (lambda s: None))
            seen = []
            recording_message_box(mp, seen)
            with built(app) as scr:
                if collect is not None:
                    scr._settings_model.collect = collect
                scr._on_run()
                assert len(seen) == 1
                assert seen[0][0] == kind and seen[0][2] == title
                assert body in seen[0][3]
                assert scr._btn_run.isEnabled()
                assert not scr._btn_stop.isEnabled()
                assert scr._thread is None


@pytest.mark.usefixtures("console_info_enabled")
def test_stop_does_what_the_quit_answer_asked_and_says_what_it_managed(
        qtbot, subtests):
    """Without a thread Stop is silent and asks nothing. With one, CANCEL
    leaves it alone, RESTART restarts (no force stop), GRACEFUL requests a
    stop even when the thread refuses the interruption; the cooperative
    request reaches the thread even if the worker is gone and passes the
    reason to a live worker; a force stop reports a parked thread as still
    running and a drained one as having left files half-written.
    """
    from spacr.qt import shutdown

    def running_thread(**extra):
        return types.SimpleNamespace(isRunning=lambda: True,
                                     requestInterruption=lambda: None, **extra)

    with built("mask") as scr:
        with subtests.test(case="no thread"), \
                pytest.MonkeyPatch.context() as mp:
            asked = []
            mp.setattr(shutdown, "ask_how_to_quit",
                       lambda *a, **k: asked.append(a))
            before = console_text(scr._console)
            scr._thread = None
            scr._on_stop()
            assert asked == [] and scr._thread is None
            assert console_text(scr._console) == before

        for answer, expect_restart in ((shutdown.CANCEL, False),
                                       (shutdown.RESTART, True)):
            with subtests.test(case=f"answer {answer}"), \
                    pytest.MonkeyPatch.context() as mp:
                restarted = []
                mp.setattr(scr, "_thread", running_thread())
                mp.setattr(shutdown, "ask_how_to_quit", lambda *a, **k: answer)
                mp.setattr(scr, "force_restart", lambda: restarted.append(True))
                mp.setattr(scr, "_force_stop",
                           lambda: pytest.fail("force stop was not asked for"))
                scr._on_stop()
                assert restarted == ([True] if expect_restart else [])

        with subtests.test(case="graceful, thread refuses interruption"), \
                pytest.MonkeyPatch.context() as mp:
            mp.setattr(shutdown, "ask_how_to_quit",
                       lambda *a, **k: shutdown.GRACEFUL)

            class _Stubborn:
                def isRunning(self):
                    return True

                def requestInterruption(self):
                    raise RuntimeError("no")

            mp.setattr(scr, "_thread", _Stubborn())
            scr._on_stop()
            assert "Requesting stop" in console_text(scr._console)

        with subtests.test(case="cooperative stop, dead worker"), \
                pytest.MonkeyPatch.context() as mp:
            interrupted = []

            class DeadWorker:
                def request_cancel(self, _reason):
                    raise RuntimeError("Internal C++ object already deleted.")

            mp.setattr(scr, "_worker", DeadWorker(), raising=False)
            mp.setattr(scr, "_thread", types.SimpleNamespace(
                requestInterruption=lambda: interrupted.append(True)))
            scr._request_cooperative_stop()
            assert interrupted == [True]

        with subtests.test(case="cooperative stop, live worker"), \
                pytest.MonkeyPatch.context() as mp:
            asked = []
            mp.setattr(scr, "_worker", types.SimpleNamespace(
                request_cancel=asked.append), raising=False)
            mp.setattr(scr, "_thread", None)
            scr._request_cooperative_stop()
            assert asked == ["stopped by the user"]

        for drained, wanted, unwanted in ((None, None, None),
                                          (False, "still finishing in the "
                                                  "background", "Stopped."),
                                          (True, "half-written", None)):
            with subtests.test(case=f"force stop drained={drained}"), \
                    pytest.MonkeyPatch.context() as mp:
                said = []
                mp.setattr(scr, "_console",
                           types.SimpleNamespace(append_notice=said.append))
                mp.setattr(scr, "_worker", None, raising=False)
                mp.setattr(scr, "_thread", None if drained is None else
                           types.SimpleNamespace(
                               requestInterruption=lambda: None))
                mp.setattr("spacr.qt.bridge.drain_thread",
                           lambda *a, _d=drained, **k: _d)
                scr._force_stop()
                if drained is None:
                    assert said == []
                    continue
                assert len(said) == 1 and wanted in said[0]
                if unwanted:
                    assert unwanted not in said[0]
        scr._thread = None
        scr._worker = None


@pytest.mark.usefixtures("console_info_enabled")
def test_finishing_a_run_resets_the_buttons_and_notifies(qtbot, subtests):
    """``_on_finished`` (also reachable from a signal replay without a start
    stamp) re-enables Run, disables Stop, hides progress, prints the outcome,
    tells the notifier (app, success/failed, elapsed) and survives a notifier
    that raises; thread refs and the console clear on request.
    """
    notified = []
    cases = [
        ("failed, no start stamp", False, None,
         lambda *a, **k: notified.append(a), "✗ Failed", "failed"),
        ("success, timed", True, 2.0,
         lambda app, status, elapsed: notified.append((app, status, elapsed)),
         "✓ Finished", "success"),
        ("notifier raises", True, None,
         lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no dbus")),
         "✓ Finished", None),
    ]
    for name, ok, started_ago, notifier, line, status in cases:
        with subtests.test(case=name), pytest.MonkeyPatch.context() as mp:
            mp.setattr("spacr.qt.notify.announce_pipeline_finished", notifier)
            notified.clear()
            with built("mask") as scr:
                scr._btn_run.setEnabled(False)
                scr._btn_stop.setEnabled(True)
                scr._progress.setVisible(True)
                if started_ago is not None:
                    scr._run_started_at = time.time() - started_ago
                scr._on_finished(ok)
                assert scr._btn_run.isEnabled()
                assert not scr._btn_stop.isEnabled()
                assert not scr._progress.isVisibleTo(scr)
                assert line in console_text(scr._console)
                if status is not None:
                    assert len(notified) == 1
                    assert notified[0][:2] == ("mask", status)
                if started_ago is not None:
                    assert 1.5 <= notified[0][2] <= 10.0

    with subtests.test(case="thread refs and console clear"), \
            built("mask") as scr:
        scr._thread, scr._worker = object(), object()
        scr._clear_thread_refs()
        assert scr._thread is None and scr._worker is None
        scr._console.append_stdout("noise\n")
        assert "noise" in console_text(scr._console)
        scr._btn_clear.click()
        assert console_text(scr._console) == ""


class _Refusing:
    """A part of the screen that raises on every way of shutting it down."""

    def _gone(self, *a, **k):
        raise RuntimeError("Internal C++ object already deleted.")

    shutdown = stop = clear = close = requestInterruption = _gone


def test_closing_the_screen_ends_every_part_even_when_some_refuse(
        qtbot, subtests, monkeypatch):
    """Close stops a running worker and drops its refs, clears the figure
    queue (absent or raising queues are fine), and when the usage timer, job
    runners, montage, workspace registry, figure queue and even the UMAP
    explorer raise, the remaining parts are still shut down and the close
    event is accepted.
    """
    from matplotlib.figure import Figure

    with subtests.test(case="running worker"), \
            pytest.MonkeyPatch.context() as mp:
        forbid_modals(mp)
        gate = threading.Event()
        mp.setattr(ENTRY, lambda key: (lambda s: gate.wait(10)))
        with built("mask") as scr:
            scr._on_run()
            assert scr._thread is not None
            gate.set()
            scr.close()
            assert scr._thread is None and scr._worker is None

    with subtests.test(case="figure queue cleared"), built("mask") as scr:
        scr._on_figure_ready(Figure(figsize=(1, 1)), "")
        assert scr._figure_queue.count() == 1
        scr.close()
        assert scr._figure_queue.count() == 0

    for name, thread, queue in (("no figure queue", None, None),
                                ("thread and queue misbehave", _Refusing(),
                                 _Refusing())):
        with subtests.test(case=name), built("mask") as scr:
            scr._thread = thread
            scr._figure_queue = queue
            scr.close()
            assert scr._thread is None

    with subtests.test(case="worker still running defers the close"), \
            built("mask") as scr:
        with pytest.MonkeyPatch.context() as mp:
            said, cancelled = [], []
            mp.setattr(scr, "_console",
                       types.SimpleNamespace(append_notice=said.append))
            mp.setattr(scr, "_worker", types.SimpleNamespace(
                request_cancel=cancelled.append), raising=False)
            thread = types.SimpleNamespace(
                requestInterruption=lambda: None, wait=lambda ms: False,
                isRunning=lambda: True)
            mp.setattr(scr, "_thread", thread)
            event = QCloseEvent()
            scr.closeEvent(event)
            assert not event.isAccepted()
            assert cancelled == ["screen closed"]
            assert len(said) == 1 and "Close deferred" in said[0]
            assert scr._thread is thread, "refs dropped under a live worker"

    with subtests.test(case="every part refuses"), built("regression") as scr, \
            pytest.MonkeyPatch.context() as mp:
        closed = []
        for attr in ("_usage_timer", "_usage_jobs", "_jobs", "_cell_montage",
                     "_figure_queue"):
            mp.setattr(scr, attr, _Refusing())
        mp.setattr(scr, "unregister_workspace", lambda: (
            _ for _ in ()).throw(RuntimeError("the registry is closed")))
        mp.setattr(scr, "_umap_explorer", types.SimpleNamespace(
            close=lambda: closed.append("explorer")))
        event = QCloseEvent()
        scr.closeEvent(event)
        assert closed == ["explorer"]
        assert event.isAccepted()

    with subtests.test(case="explorer refuses"), built("regression") as scr, \
            pytest.MonkeyPatch.context() as mp:
        mp.setattr(scr, "_umap_explorer", _Refusing())
        event = QCloseEvent()
        scr.closeEvent(event)
        assert event.isAccepted()
