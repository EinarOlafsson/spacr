"""Instruction 114 point 1 — a fallback that says so, rather than pretending.

    "could you modify the paramiter search so it dosnt crash the computer"
    "the paramiter search needs to be verry safe so it dosnt crash spacr or
     the machine (it crashed vscode many tinmes)"

VS Code dying is diagnostic: the editor is not part of the sweep, so it was
killed ALONGSIDE it, which is what an OOM killer does.

ACCOUNTING IS NOT CONTAINMENT. Every earlier fix was an estimate of what a
trial would use; the kernel enforcing a ceiling is the only thing that has
held. So a machine where `systemd-run --user --scope` is unavailable — a
container, a bare SSH session with no user manager — runs uncontained, and the
one thing it must not do is let the user believe otherwise. A user who thinks
a cap exists and is wrong is in a WORSE position than one who knows there is
none: they will start the sweep that took the desktop down.
"""
from __future__ import annotations

import pytest

from spacr.parameter_sweep import (TRIAL_CPU_QUOTA, TRIAL_MEMORY_MAX,
                                   containment_available, containment_note)


def test_the_note_says_which_of_the_two_situations_this_is():
    note = containment_note()
    if containment_available():
        assert "kernel cap" in note
    else:
        assert "NO KERNEL CAP" in note


def test_when_there_is_a_cap_the_note_names_the_actual_limits(monkeypatch):
    """A cap whose size is not stated cannot be judged against a trial."""
    monkeypatch.setattr("spacr.parameter_sweep.containment_available",
                        lambda: True)
    note = containment_note()

    assert TRIAL_MEMORY_MAX in note
    assert TRIAL_CPU_QUOTA in note
    assert "no swap" in note
    # And what happens when a trial hits it, because a killed trial is a
    # RESULT and not a crash (114 point 2).
    assert "killed" in note
    assert "carries on" in note


def test_when_there_is_none_the_note_says_so_and_says_why(monkeypatch):
    monkeypatch.setattr("spacr.parameter_sweep.containment_available",
                        lambda: False)
    note = containment_note()

    assert "NO KERNEL CAP" in note
    # The CAUSE, so a user can fix it rather than only fear it.
    assert "systemd-run" in note
    assert "container" in note
    # The DISTINCTION this instruction turns on.
    assert "ACCOUNTING, not containment" in note
    # And what to do instead.
    assert "fewer workers" in note or "desktop session" in note


def test_the_note_never_claims_a_cap_that_is_not_there(monkeypatch):
    """The single property everything else here is in service of."""
    monkeypatch.setattr("spacr.parameter_sweep.containment_available",
                        lambda: False)
    note = containment_note().lower()
    assert "each trial runs under a kernel cap" not in note


@pytest.mark.qt
def test_the_sweep_panel_puts_it_on_screen(qtbot, monkeypatch):
    """A note in a docstring is not a note the user reads."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")

    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("ml_analyze_regression")
    qtbot.addWidget(screen)
    panel = getattr(screen, "_sweep", None)
    assert panel is not None, "the screen built no parameter-sweep card"
    assert hasattr(panel, "containment"), (
        "the sweep card says nothing about whether the cap is real")

    assert panel.containment.text()
    assert panel.containment.wordWrap() is True


@pytest.mark.qt
def test_an_uncontained_machine_is_marked_as_a_warning(qtbot, monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    monkeypatch.setattr("spacr.parameter_sweep.containment_available",
                        lambda: False)

    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("ml_analyze_regression")
    qtbot.addWidget(screen)
    panel = getattr(screen, "_sweep", None)
    assert panel is not None, "the screen built no parameter-sweep card"
    assert hasattr(panel, "containment"), (
        "the sweep card says nothing about whether the cap is real")

    # RED, because it changes what the user should do next -- not a note among
    # notes.
    assert panel.containment.objectName() == "DangerLabel"
    assert "NO KERNEL CAP" in panel.containment.text()
