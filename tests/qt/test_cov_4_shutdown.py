"""Quitting reports the button the user actually pressed.

Each answer means a different thing to the caller: finish the work, kill it
now, save and come back, or do nothing. Reading one as another is
unrecoverable in both directions -- a Cancel read as Force loses whatever was
mid-write, and a Force read as Cancel leaves the user pressing a button that
does nothing. The five-minute re-ask has the same stakes and one more: once
force is chosen it must stop asking, because a user already leaving must not
be asked again on the way out.
"""
from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from spacr.qt import shutdown
from spacr.qt.shutdown import (CANCEL, FORCE, GRACEFUL, RESTART,
                               GracefulQuitWatcher, ask_how_to_quit,
                               describe_active, restart_spacr)


def _click(label):
    """A stand-in exec that presses the button carrying ``label``."""
    def _exec(box):
        for button in box.buttons():
            if button.text() == label:
                button.click()
                return 0
        raise AssertionError(
            f"no {label!r} button; saw {[b.text() for b in box.buttons()]}")
    return _exec


def _answer(monkeypatch, label, **kwargs):
    monkeypatch.setattr(QMessageBox, "exec", _click(label), raising=False)
    return ask_how_to_quit(None, what="Regression", verb="Stop", **kwargs)


def test_finishing_current_work_is_the_graceful_answer(qapp, monkeypatch):
    """Cooperative cancellation is what lets an active write finish."""
    assert _answer(monkeypatch, "Finish current work") == GRACEFUL


def test_force_stop_is_the_force_answer(qapp, monkeypatch):
    """The destructive button must not be read as anything milder."""
    assert _answer(monkeypatch, "Force stop") == FORCE


def test_force_restart_is_its_own_answer(qapp, monkeypatch):
    """Restart saves and comes back; force merely kills."""
    assert _answer(monkeypatch, "Force restart", offer_restart=True,
                   restart_detail="one run will be interrupted") == RESTART


def test_cancel_is_the_answer_when_nothing_was_chosen(qapp, monkeypatch):
    """The escape route must never be read as a destructive choice."""
    assert _answer(monkeypatch, "Cancel") == CANCEL


def test_the_default_launcher_starts_a_detached_process(monkeypatch,
                                                        tmp_path):
    """The replacement must outlive the signal that takes this process down."""
    import subprocess

    started = {}

    def _record(command, **kwargs):
        started["command"] = list(command)
        started["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(subprocess, "Popen", _record)
    monkeypatch.setattr("spacr.restart_state.save",
                        lambda **kwargs: str(tmp_path / "state.json"))
    monkeypatch.setattr("spacr.restart_state.command",
                        lambda: ["python", "-m", "spacr.qt"])
    exited = []
    assert restart_spacr("regression", {"a": 1}, exiter=exited.append) is True
    assert started["command"] == ["python", "-m", "spacr.qt"]
    assert started["kwargs"]["start_new_session"] is True
    assert exited == [0]


def test_a_watcher_can_be_stopped_before_it_ever_asks(qapp):
    """Whoever started the graceful attempt owns the re-ask cycle."""
    watcher = GracefulQuitWatcher(None, lambda: True, what="Regression",
                                  interval_ms=10)
    watcher.start()
    assert watcher._timer.isActive()
    watcher.stop()
    assert not watcher._timer.isActive()


def test_choosing_force_on_the_re_ask_stops_asking_and_kills(qapp,
                                                             monkeypatch):
    """A user already leaving must not be asked again on the way out."""
    monkeypatch.setattr(QMessageBox, "exec", _click("Force quit"),
                        raising=False)
    killed = []
    watcher = GracefulQuitWatcher(None, lambda: True, what="Regression",
                                  on_force=lambda: killed.append(True),
                                  interval_ms=10)
    watcher.start()
    watcher._recheck()
    assert killed == [True]
    assert not watcher._timer.isActive()


def test_keeping_waiting_leaves_the_cycle_running(qapp, monkeypatch):
    """The other button must not be read as a force quit."""
    monkeypatch.setattr(QMessageBox, "exec", _click("Keep waiting"),
                        raising=False)
    killed = []
    watcher = GracefulQuitWatcher(None, lambda: True, what="Regression",
                                  on_force=lambda: killed.append(True),
                                  interval_ms=10)
    watcher.start()
    watcher._recheck()
    assert killed == []
    assert watcher._timer.isActive()
    watcher.stop()


def test_nothing_running_produces_no_summary_at_all():
    """An empty "Still running:" heading would be a heading over nothing."""
    assert describe_active([]) == ""


def test_a_running_job_is_named_with_how_long_it_has_been_going():
    """"Something is still running" is not enough to decide with."""
    class _Handle:
        app_key = "regression"

        def elapsed(self):
            return 125.0

    text = describe_active([_Handle()])
    assert "regression" in text
    assert "2 min" in text
