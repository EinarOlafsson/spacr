"""Shutting a module screen down when its parts will not shut down.

Closing a screen has to end five separate things: the usage poll, two job
runners, the cell montage's own loader, the settings panel's background work,
and the figure queue's temporary directory. Any of them can already be gone --
navigation destroys child widgets without giving them a close event -- and a
part that raises on the way out must not stop the ones after it, because the
one left running is a QThread, and a running QThread destroyed with its parent
aborts the process.

Stop has the same shape from the other end: a worker inside a C call cannot be
cancelled, so the screen asks the user what to do and then has to be honest
about which of the two things it managed.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QCloseEvent                            # noqa: E402

from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


class Refusing:
    """A part of the screen that raises on every way of shutting it down."""

    def shutdown(self):
        raise RuntimeError("Internal C++ object already deleted.")

    def stop(self):
        raise RuntimeError("Internal C++ object already deleted.")

    def clear(self):
        raise RuntimeError("Internal C++ object already deleted.")

    def close(self):
        raise RuntimeError("Internal C++ object already deleted.")


class TestClosingAScreenWhoseWorkersHaveGone:

    def test_one_part_that_will_not_close_does_not_strand_the_others(
            self, screen, monkeypatch):
        """The parts are shut down in order and the order has to complete.

        The last of them owns a QThread; leaving it running while the screen
        is destroyed aborts the process, so a raise half way through the list
        is the failure this guards.
        """
        closed = []
        monkeypatch.setattr(screen, "_usage_timer", Refusing())
        monkeypatch.setattr(screen, "_usage_jobs", Refusing())
        monkeypatch.setattr(screen, "_jobs", Refusing())
        monkeypatch.setattr(screen, "_cell_montage", Refusing())
        monkeypatch.setattr(screen, "unregister_workspace",
                            lambda: (_ for _ in ()).throw(
                                RuntimeError("the registry is closed")))
        monkeypatch.setattr(screen, "_figure_queue", Refusing())
        monkeypatch.setattr(screen, "_umap_explorer", types.SimpleNamespace(
            close=lambda: closed.append("explorer")))

        event = QCloseEvent()
        screen.closeEvent(event)

        assert closed == ["explorer"], (
            "the shutdown stopped at the first part that refused")
        assert event.isAccepted()

    def test_an_explorer_that_will_not_close_still_lets_the_screen_go(
            self, screen, monkeypatch):
        """A detached window is not a reason to keep the screen alive."""
        monkeypatch.setattr(screen, "_umap_explorer", Refusing())

        event = QCloseEvent()
        screen.closeEvent(event)

        assert event.isAccepted()


class TestAskingTheWorkerToStop:

    def test_a_worker_that_will_not_take_the_request_is_not_a_failure(
            self, screen, monkeypatch):
        """The thread is interrupted either way.

        A worker whose C++ side has gone raises on the first call; letting
        that out would skip the interruption request, which is the half that
        reaches a worker still checking.
        """
        interrupted = []

        class DeadWorker:
            def request_cancel(self, _reason):
                raise RuntimeError("Internal C++ object already deleted.")

        monkeypatch.setattr(screen, "_worker", DeadWorker(), raising=False)
        monkeypatch.setattr(screen, "_thread", types.SimpleNamespace(
            requestInterruption=lambda: interrupted.append(True)))

        screen._request_cooperative_stop()

        assert interrupted == [True]

    def test_a_worker_that_takes_it_is_given_the_reason(self, screen,
                                                         monkeypatch):
        asked = []
        monkeypatch.setattr(screen, "_worker", types.SimpleNamespace(
            request_cancel=asked.append), raising=False)
        monkeypatch.setattr(screen, "_thread", None)

        screen._request_cooperative_stop()

        assert asked == ["stopped by the user"]


class TestForceStopping:

    def test_with_no_thread_there_is_nothing_to_drain(self, screen,
                                                       monkeypatch):
        """Pressed after the run already ended: nothing to say about it."""
        said = []
        screen._console = types.SimpleNamespace(append_notice=said.append)
        monkeypatch.setattr(screen, "_thread", None)

        screen._force_stop()

        assert said == []

    def test_a_step_that_will_not_interrupt_is_reported_as_still_running(
            self, screen, monkeypatch):
        """"Stopped" would be a lie the user catches when the file grows.

        The thread is PARKED rather than terminated -- terminate() on a
        thread holding the GIL freezes the whole process -- so the run is
        genuinely still out there and the sentence has to say so.
        """
        said = []
        screen._console = types.SimpleNamespace(append_notice=said.append)
        monkeypatch.setattr(screen, "_thread", types.SimpleNamespace(
            requestInterruption=lambda: None))
        monkeypatch.setattr(screen, "_worker", None, raising=False)
        monkeypatch.setattr("spacr.qt.bridge.drain_thread",
                            lambda *a, **k: False)

        screen._force_stop()

        assert len(said) == 1
        assert "still finishing in the background" in said[0]
        assert "Stopped." not in said[0]

    def test_a_step_that_did_stop_says_what_it_left_half_written(
            self, screen, monkeypatch):
        said = []
        screen._console = types.SimpleNamespace(append_notice=said.append)
        monkeypatch.setattr(screen, "_thread", types.SimpleNamespace(
            requestInterruption=lambda: None))
        monkeypatch.setattr(screen, "_worker", None, raising=False)
        monkeypatch.setattr("spacr.qt.bridge.drain_thread",
                            lambda *a, **k: True)

        screen._force_stop()

        assert "half-written" in said[0]


class TestTheStopDialogsThreeAnswers:

    def _arm(self, screen, monkeypatch, answer):
        from spacr.qt import shutdown

        monkeypatch.setattr(screen, "_thread", types.SimpleNamespace(
            isRunning=lambda: True, requestInterruption=lambda: None))
        monkeypatch.setattr(shutdown, "ask_how_to_quit",
                            lambda *a, **k: answer)

    def test_restart_restarts_instead_of_stopping(self, screen, monkeypatch):
        """The last resort, offered from the button somebody actually presses.

        A fit that will not stop leaves restarting as the only way out, and
        it must not also run the cooperative path on the way there.
        """
        from spacr.qt import shutdown

        self._arm(screen, monkeypatch, shutdown.RESTART)
        restarted = []
        monkeypatch.setattr(screen, "force_restart",
                            lambda: restarted.append(True))
        monkeypatch.setattr(screen, "_force_stop",
                            lambda: pytest.fail("force stop was not asked for"))

        screen._on_stop()

        assert restarted == [True]

    def test_cancel_leaves_the_run_alone(self, screen, monkeypatch):
        from spacr.qt import shutdown

        self._arm(screen, monkeypatch, shutdown.CANCEL)
        monkeypatch.setattr(screen, "force_restart",
                            lambda: pytest.fail("the run was restarted"))
        monkeypatch.setattr(screen, "_force_stop",
                            lambda: pytest.fail("the run was force-stopped"))

        screen._on_stop()

    def test_stop_pressed_with_no_run_does_nothing(self, screen, monkeypatch):
        from spacr.qt import shutdown

        questions = []
        monkeypatch.setattr(screen, "_thread", None)
        monkeypatch.setattr(shutdown, "ask_how_to_quit",
                            lambda *args, **kwargs:
                            questions.append((args, kwargs)))

        screen._on_stop()

        assert questions == []
        assert screen._thread is None
