"""A button whose C++ side is gone is styled quietly, not loudly.

Every action button is wired to a lambda that holds a Python reference to
it, so the Python object outlives the widget. Touching the dead widget
raises ``libshiboken: Internal C++ object already deleted`` -- once per wired
button, on every launch, for buttons nobody pressed. These pin the liveness
check and the no-ops it guards, including the fallback used on a build where
shiboken cannot be imported.
"""
from __future__ import annotations

import builtins

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QPushButton, QWidget

from spacr.qt import button_roles as br


def _delete_cpp_side(widget):
    """Destroy the C++ half of ``widget``, leaving the Python wrapper."""
    from shiboken6 import delete
    delete(widget)


def test_no_button_at_all_is_not_alive():
    """``None`` answers False rather than raising on attribute access.

    Callers pass whatever the timer captured, and a slot that fires after the
    owning screen tore itself down can hand this a None.
    """
    assert br.alive(None) is False


def test_liveness_is_answered_without_shiboken(qapp, monkeypatch):
    """With ``shiboken6`` unavailable, a probe call decides liveness.

    shiboken is the direct answer; a PySide build that does not expose it
    must still distinguish a live button from a deleted one, because the
    alternative is the deleted-object traceback this check exists to stop.
    """
    real_import = builtins.__import__

    def _no_shiboken(name, *args, **kwargs):
        if name == "shiboken6":
            raise ImportError("no shiboken6 in this build")
        return real_import(name, *args, **kwargs)

    live = QPushButton("Run")
    dead = QPushButton("Stop")
    _delete_cpp_side(dead)

    monkeypatch.setattr(builtins, "__import__", _no_shiboken)
    try:
        assert br.alive(live) is True
        assert br.alive(dead) is False
    finally:
        monkeypatch.undo()
    live.deleteLater()


def test_repolishing_a_deleted_button_does_nothing(qapp):
    """A repolish of a dead button returns instead of touching its style.

    ``button.style()`` on a deleted widget is the exact call that raises;
    the visual state being refreshed belongs to a widget that is gone, so
    there is nothing to show and nothing is lost by returning.
    """
    dead = QPushButton("Cancel")
    _delete_cpp_side(dead)

    assert br._repolish(dead) is None


def test_repolishing_without_a_style_still_requests_the_repaint(monkeypatch):
    """A host adapter may expose no style; updating it remains harmless."""
    updated = []

    class UnstyledButton:
        def style(self):
            return None

        def update(self):
            updated.append(True)

    monkeypatch.setattr(br, "alive", lambda _button: True)

    br._repolish(UnstyledButton())

    assert updated == [True]


def test_marking_a_deleted_button_busy_does_nothing(qapp):
    """Setting the busy state on a dead button is a silent no-op.

    This is the call the pressed-lambda makes, and the lambda outlives the
    widget by construction.
    """
    dead = QPushButton("Run")
    _delete_cpp_side(dead)

    assert br.set_button_busy(dead, True) is None
    assert br.set_button_busy(dead, False) is None


def test_repeating_the_same_busy_state_is_a_noop(qapp, monkeypatch):
    """An unchanged state does not pay for another stylesheet repolish."""
    button = QPushButton("Run")
    button.setProperty("buttonActionBusy", True)
    repolished = []
    monkeypatch.setattr(br, "_repolish", repolished.append)

    br.set_button_busy(button, True)

    assert repolished == []
    button.deleteLater()


def test_classifying_a_deleted_button_does_nothing(qapp):
    """A teardown event for a dead button is ignored at the filter boundary.

    Qt 6.6 can deliver a queued polish or parent-change event after a dialog
    deleted the C++ button.  The signal connection still retains the Python
    wrapper, but even asking it for its icon raises ``RuntimeError``.
    """
    dead = QPushButton("Cancel")
    _delete_cpp_side(dead)

    assert br._SemanticButtonFilter().classify(dead) is None


def test_adopting_a_spinner_for_a_deleted_button_does_nothing(qapp):
    """The second half of a show event observes the same liveness boundary."""
    dead = QPushButton("Clear console")
    _delete_cpp_side(dead)

    assert br._adopt_activity_spinner(dead) is None


def test_a_spinner_that_cannot_attach_is_retried_later(
        qapp, monkeypatch):
    """A failed attempt does not set the success-only ownership latch."""
    from spacr.qt.widgets import activity_spinner

    host = QWidget()
    button = QPushButton("Clear console", host)
    host._btn_clear = button
    monkeypatch.setattr(activity_spinner, "attach_activity_spinner",
                        lambda _host: None)

    br._adopt_activity_spinner(button)

    assert button.property(br._SPINNER_PROPERTY) is None
    host.deleteLater()


def test_classification_swallows_only_a_reentrant_deletion(
        qapp, monkeypatch):
    """A real RuntimeError stays visible; deletion during a Qt call does not."""
    button = object()
    answers = iter((True, False, True, True))
    real_alive = br.alive

    def staged_alive(candidate):
        # The application-wide event filter keeps receiving ordinary Qt
        # traffic while this test runs.  Only script liveness for the
        # synthetic object under test; real widgets must continue through the
        # real predicate instead of consuming this four-step scenario.
        if candidate is button:
            return next(answers)
        return real_alive(candidate)

    monkeypatch.setattr(br, "alive", staged_alive)

    def refuse(_self, _button):
        raise RuntimeError("classification failed")

    monkeypatch.setattr(br._SemanticButtonFilter, "_classify_live", refuse)
    event_filter = br._SemanticButtonFilter()

    assert event_filter.classify(button) is None
    with pytest.raises(RuntimeError, match="classification failed"):
        event_filter.classify(button)


def test_a_button_named_primary_is_positive_whatever_it_says(qapp):
    """Object identity supplies the role when translated text cannot."""
    button = QPushButton("Proceed with the analysis")
    button.setObjectName("PrimaryButton")

    br._SemanticButtonFilter().classify(button)

    assert button.property("buttonActionRole") == "positive"
    button.deleteLater()


def test_settling_a_deleted_clicked_button_does_nothing(qapp):
    """A dialog may close before the zero-delay click callback is delivered."""
    dead = QPushButton("Close")
    _delete_cpp_side(dead)

    assert br._SemanticButtonFilter._settle_after_handler(dead) is None


def test_event_filter_rechecks_liveness_after_classification(qapp):
    """Qt 6.6 may delete a dialog button during its parent-change event."""
    class DeletesWhileClassifying(br._SemanticButtonFilter):
        def classify(self, button):
            _delete_cpp_side(button)

    button = QPushButton("Cancel")

    assert DeletesWhileClassifying().eventFilter(
        button, QEvent(QEvent.ParentChange)) is False


def test_a_danger_button_reads_as_negative_whatever_it_says(qapp):
    """``DangerButton`` is styled negative even when its text is neutral.

    The text classifier covers "Delete", "Cancel" and friends; a destructive
    button whose label is a noun ("Overwrite the database") has only its
    object name to say what it does, and in a non-English locale the text
    never matches at all.
    """
    button = QPushButton("Overwrite the database")
    button.setObjectName("DangerButton")

    br._SemanticButtonFilter().classify(button)

    assert button.property("buttonActionRole") == "negative"
    assert button.property("buttonActionBusy") is False
    button.deleteLater()


def test_installing_the_filter_without_an_application_does_nothing(
        monkeypatch):
    """With no QApplication there is nothing to install onto.

    The installer is called from module import paths that also run under a
    console entry point, where no GUI application has been created; raising
    there would turn a headless invocation into a crash.
    """
    monkeypatch.setattr(br.QApplication, "instance", staticmethod(lambda: None))

    assert br.install_button_roles() is None
