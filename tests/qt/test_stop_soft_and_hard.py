"""Stop has to stop, and has to say what kind of stopping it is doing.

The report was "spaCR is really bad at stopping things — the stop button
doesn't seem to do much". The mechanism was already written down.
INVARIANTS 11: cooperative cancellation sets a flag, and a worker wedged in
a C extension never checks it. cellpose, torch and cv2 calls do not come
back to look, so the run continued — and the old Stop button DISABLED itself
after asking once, so there was no way to escalate. The user waited,
believing the run was ending, while it was not.

`spacr/qt/shutdown.py` had already solved this for quitting, and the Home
banner's quit button already used it. Stop now offers the same choice
through the same dialog.

Cooperative stays the default and force is never what a stray Return key
does, because a pipeline killed mid-write leaves a half-written .npy and
silent corruption found later is worse than waiting.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    from spacr.qt.screens.app_screen import AppScreen

    widget = AppScreen(app_key="mask")
    qtbot.addWidget(widget)
    return widget


class _FakeThread:
    def __init__(self, running=True):
        self.running = running
        self.interrupted = False
        self.terminated = False
        self.quit_requested = False

    def isRunning(self):
        return self.running

    def requestInterruption(self):
        self.interrupted = True

    def terminate(self):
        self.terminated = True
        self.running = False

    def quit(self):
        self.quit_requested = True

    def wait(self, _timeout):
        self.running = False
        return True


class _FakeWorker:
    def __init__(self):
        self.cancelled = []

    def request_cancel(self, reason):
        self.cancelled.append(reason)


def _arm(screen, monkeypatch, choice):
    """Put a fake run on the screen and pin the dialog's answer."""
    from spacr.qt.screens import app_screen as module

    thread, worker = _FakeThread(), _FakeWorker()
    screen._thread = thread
    screen._worker = worker
    monkeypatch.setattr(module, "APP_TITLES", {"mask": "Mask"}, raising=False)
    import spacr.qt.shutdown as shutdown
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda *a, **k: choice)
    return thread, worker


def test_cancel_leaves_the_run_alone(screen, monkeypatch):
    """The most important of the three. A mis-clicked Stop that killed a
    six-hour run would be unforgivable."""
    from spacr.qt.shutdown import CANCEL

    thread, worker = _arm(screen, monkeypatch, CANCEL)
    screen._on_stop()

    assert worker.cancelled == []
    assert thread.interrupted is False
    assert thread.terminated is False


def test_a_cooperative_stop_asks_but_does_not_kill(screen, monkeypatch):
    from spacr.qt.shutdown import GRACEFUL

    thread, worker = _arm(screen, monkeypatch, GRACEFUL)
    screen._on_stop()

    assert worker.cancelled, "the worker was never asked to stop"
    assert thread.interrupted is True
    assert thread.terminated is False, "cooperative must not terminate"


def test_the_button_stays_live_after_a_cooperative_stop(screen, monkeypatch):
    """THE regression. It used to disable itself, so a stop that never
    landed left the user with no way to ask again — which is exactly what
    "the stop button doesn't do much" felt like."""
    from spacr.qt.shutdown import GRACEFUL

    _arm(screen, monkeypatch, GRACEFUL)
    # The button is disabled until a run starts, and this fixture fakes the
    # run rather than starting one -- so enable it first and assert that
    # _on_stop LEAVES it enabled. Asserting the absolute state would pass
    # for the wrong reason.
    screen._btn_stop.setEnabled(True)
    screen._on_stop()

    assert screen._btn_stop.isEnabled() is True


def test_a_cooperative_stop_starts_a_watcher(screen, monkeypatch):
    """Cooperative cancellation can silently never land. Something has to
    come back and offer the escalation without being asked."""
    from spacr.qt.shutdown import GRACEFUL

    _arm(screen, monkeypatch, GRACEFUL)
    screen._on_stop()

    assert getattr(screen, "_stop_watcher", None) is not None


def test_force_drains_without_terminating_the_thread(screen, monkeypatch):
    from spacr.qt.shutdown import FORCE

    thread, worker = _arm(screen, monkeypatch, FORCE)
    screen._on_stop()

    assert thread.quit_requested is True
    assert thread.terminated is False
    # ...and the worker was asked first regardless, so one that IS still
    # checking gets the chance to stop on its own terms in the moment
    # before its thread is taken away.
    assert worker.cancelled, "force must still request cancellation first"


def test_force_drain_is_reached_through_the_watcher_too(screen, monkeypatch):
    """The escalation path, not just the first prompt."""
    from spacr.qt.shutdown import GRACEFUL

    thread, _worker = _arm(screen, monkeypatch, GRACEFUL)
    screen._on_stop()
    screen._force_stop()
    assert thread.quit_requested is True
    assert thread.terminated is False


def test_stop_with_nothing_running_does_nothing(screen, monkeypatch):
    import spacr.qt.shutdown as shutdown

    asked = []
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda *a, **k: asked.append(1))
    screen._thread = None
    screen._on_stop()
    assert asked == [], "no run, so no prompt"


def test_the_dialog_says_stop_rather_than_quit(qtbot, monkeypatch):
    """A button labelled Stop that opens a dialog headed "Quit" reads as the
    wrong dialog, and a user who thinks they mis-clicked cancels out of the
    thing they wanted."""
    from PySide6.QtWidgets import QMessageBox

    from spacr.qt import shutdown

    seen = {}

    class _Box(QMessageBox):
        def setWindowTitle(self, title):
            seen["title"] = title
            super().setWindowTitle(title)

        def setText(self, text):
            seen["text"] = text
            super().setText(text)

        def exec(self):
            return 0

        def clickedButton(self):
            return None

    monkeypatch.setattr(shutdown, "QMessageBox", _Box)
    shutdown.ask_how_to_quit(None, what="Mask", verb="Stop")

    assert seen["title"] == "Stop Mask"
    assert seen["text"] == "Stop Mask?"


def test_the_verb_still_defaults_to_quit(qtbot, monkeypatch):
    """Every existing caller passes no verb and must be unchanged."""
    from PySide6.QtWidgets import QMessageBox

    from spacr.qt import shutdown

    seen = {}

    class _Box(QMessageBox):
        def setText(self, text):
            seen["text"] = text
            super().setText(text)

        def exec(self):
            return 0

        def clickedButton(self):
            return None

    monkeypatch.setattr(shutdown, "QMessageBox", _Box)
    shutdown.ask_how_to_quit(None, what="spaCR")
    assert seen["text"] == "Quit spaCR?"
