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

from spacr.qt.shutdown import GracefulQuitWatcher, describe_active


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
