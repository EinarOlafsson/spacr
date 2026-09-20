"""Item 452: a module's live preview belongs above its console, not under it.

Reported 2026-09-20: "the plaque modular live [preview] being under the console".

THE CAUSE WAS ONE HELPER USED FOR TWO JOBS. Every registry-mounted preview card went
through ``_insert_above_actions``, which puts a widget in the runtime panel just above
the Run row -- and the figures/console splitter is added to that same panel BEFORE the
actions row. "Above the Run button" is therefore below the console.

Mask never showed it, which is why it survived: its screen builds the live preview into
the splitter itself and never takes the registry path. So the same card sat in two
different places depending on which screen mounted it, and only one of them was right.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import preview_registry as pr
from spacr.qt.screens.app_screen import AppScreen


def _positions(screen, card):
    splitter = getattr(screen, "_runtime_splitter", None)
    console = getattr(screen, "_console_wrap", None)
    assert splitter is not None and console is not None
    return splitter.indexOf(card), splitter.indexOf(console)


def test_the_plaque_preview_is_above_the_console(qtbot, qt_theme_applied):
    screen = AppScreen("analyze_plaques")
    qtbot.addWidget(screen)
    host = pr.install(screen)
    assert host is not None, "the plaque module should mount a live preview"
    card, console = _positions(screen, host.card)
    assert card >= 0, "the preview must be IN the splitter, not beneath it"
    assert card < console, (
        f"the preview is at {card} and the console at {console}; that is the report")


def test_the_mask_preview_is_still_above_its_console(qtbot, qt_theme_applied):
    """Mask builds its own into the splitter and must not have moved."""
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    card = getattr(screen, "_live_preview_card", None)
    assert card is not None
    at, console = _positions(screen, card)
    assert 0 <= at < console


def test_a_screen_with_no_splitter_still_gets_its_preview():
    """The fallback matters: no preview at all is worse than one above the Run row."""
    class Bare:
        _runtime_splitter = None
        _console_wrap = None

    calls = []
    original = pr._insert_above_actions
    pr._insert_above_actions = lambda s, w: (calls.append(w), True)[1]
    try:
        assert pr._insert_above_console(Bare(), object()) is True
    finally:
        pr._insert_above_actions = original
    assert len(calls) == 1


def test_a_console_that_is_not_in_the_splitter_falls_back():
    from PySide6.QtWidgets import QSplitter, QWidget

    class Odd:
        pass

    screen = Odd()
    screen._runtime_splitter = QSplitter()
    screen._console_wrap = QWidget()

    calls = []
    original = pr._insert_above_actions
    pr._insert_above_actions = lambda s, w: (calls.append(w), True)[1]
    try:
        assert pr._insert_above_console(screen, QWidget()) is True
    finally:
        pr._insert_above_actions = original
    assert len(calls) == 1, "indexOf returned -1, so it must fall back"
