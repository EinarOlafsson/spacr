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

from PySide6.QtWidgets import QApplication, QPushButton

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


def test_marking_a_deleted_button_busy_does_nothing(qapp):
    """Setting the busy state on a dead button is a silent no-op.

    This is the call the pressed-lambda makes, and the lambda outlives the
    widget by construction.
    """
    dead = QPushButton("Run")
    _delete_cpp_side(dead)

    assert br.set_button_busy(dead, True) is None
    assert br.set_button_busy(dead, False) is None


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
