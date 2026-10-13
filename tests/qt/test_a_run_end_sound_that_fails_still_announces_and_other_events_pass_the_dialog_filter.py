"""A run-end sound that fails does not stop the run being announced, and the
dialog filter lets every event other than Polish and Show straight through.

``announce_pipeline_finished`` asks the sound engine for its chime only when
sound is loaded; if the chime raises, the desktop notification still goes
out with the module, status and time. The application-wide dialog filter
sees every event in the process, so anything that is not a Polish or a Show
must return at once, unconsumed, and leave the dialog untouched.
"""
import sys
import types

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QDialog

from spacr.qt import notify as n
from spacr.qt.dialogs import FILTER_DETACHED, _DetachEveryDialog


def test_a_failing_run_sound_still_sends_the_notification(monkeypatch):
    sound = types.ModuleType(f"{n.__package__}.sound")

    def _broken(status):
        raise RuntimeError(f"no audio device for {status}")

    sound.announce_run_end = _broken
    monkeypatch.setitem(sys.modules, f"{n.__package__}.sound", sound)
    sent = []
    monkeypatch.setattr(n, "notify", lambda title, body: sent.append(
        (title, body)) or True)
    monkeypatch.setattr(n, "notify_tray", lambda *a: sent.append(a) or True)

    n.announce_pipeline_finished("mask", "failed", 3.0)

    assert sent == [("⚠ spaCR — mask failed", "Finished in 3.0s.")]


def test_an_event_that_is_not_polish_or_show_passes_untouched(qtbot):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    detacher = _DetachEveryDialog()

    consumed = detacher.eventFilter(dialog, QEvent(QEvent.Type.MouseMove))

    assert consumed is False
    assert dialog.property(FILTER_DETACHED) is None
