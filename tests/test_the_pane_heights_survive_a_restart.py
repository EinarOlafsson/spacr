"""A screen's pane heights are the user's, and they last. Instruction 162.

REPORTED 2026-08-18: "the elements in measure need to be modifiable in height".

They already WERE draggable: the Measure branch builds a vertical QSplitter with
three panes and `setChildrenCollapsible(False)`. What did not happen is that the
sizes were KEPT -- `self._runtime_splitter` was assigned in seven branches and
read in NONE, so every launch went back to the hard-coded [420, 360, 300]. A
layout that snaps back is indistinguishable from one that will not move.

DRIVEN THROUGH THE HELPER, NOT THROUGH TWO AppScreens. Building two full
screens in one process segfaults -- the leaked-widget hazard HANDOFF 3d
describes -- so the restart is simulated by calling the same helper twice with
a fresh splitter, which is what a relaunch actually does.
"""

import spacr

assert "/codex/repo/spacr/" in spacr.__file__, spacr.__file__

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QWidget

from spacr.qt.screens.app_screen import AppScreen


class _Screen:
    """The two attributes the helper reads, and nothing else."""

    app_key = "measure"
    RUNTIME_SPLIT_SUFFIX = AppScreen.RUNTIME_SPLIT_SUFFIX
    _remember_runtime_splitter = AppScreen._remember_runtime_splitter


def _splitter(qtbot, sizes=(420, 360, 300)):
    split = QSplitter(Qt.Vertical)
    qtbot.addWidget(split)
    for _ in sizes:
        split.addWidget(QWidget())
    split.setChildrenCollapsible(False)
    split.setSizes(list(sizes))
    return split


def test_the_helper_is_wired_into_every_branch():
    """Seven branches assigned the splitter and none read it."""
    import inspect

    branches = inspect.getsource(AppScreen._build_runtime_panel)
    assert branches.count("_remember_runtime_splitter(splitter)") >= 7
    # The assignment survives in the HELPER and must not survive in a branch:
    # a branch that sets it directly is one whose sizes are silently not kept,
    # which is the bug this item is about.
    assert "self._runtime_splitter = splitter" not in branches


def test_a_drag_is_saved_and_a_relaunch_restores_it(qtbot):
    """The whole of the report: the arrangement outlives the screen."""
    from spacr.qt.widgets.console_panel import get_split_state

    screen = _Screen()
    first = _splitter(qtbot)
    screen._remember_runtime_splitter(first)

    first.setSizes([600, 200, 280])
    # `splitterMoved` is what a real drag emits; emitting it tests the
    # CONNECTION rather than `set_split_state` on its own.
    first.splitterMoved.emit(0, 0)

    key = f"measure{AppScreen.RUNTIME_SPLIT_SUFFIX}"
    assert get_split_state(key) is not None

    # A relaunch: a brand-new splitter on its defaults, then the helper.
    second = _splitter(qtbot)
    assert second.sizes() != first.sizes()
    _Screen()._remember_runtime_splitter(second)
    assert second.sizes() == first.sizes()


def test_the_runtime_key_is_not_the_console_key():
    """Two splitters on one screen must not restore each other's blob.

    `restoreState` on a mismatched blob silently does nothing, which is a
    layout that ignores the user with no message.
    """
    assert AppScreen.RUNTIME_SPLIT_SUFFIX
    assert AppScreen.RUNTIME_SPLIT_SUFFIX.strip() != ""
    assert "measure" != f"measure{AppScreen.RUNTIME_SPLIT_SUFFIX}"


def test_an_unusable_stored_state_falls_back_to_the_default(qtbot, monkeypatch):
    """A blob from an older layout restores nothing rather than raising."""
    from spacr.qt.widgets import console_panel

    monkeypatch.setattr(console_panel, "get_split_state",
                        lambda key: b"not a real splitter state")
    split = _splitter(qtbot)
    before = split.sizes()
    _Screen()._remember_runtime_splitter(split)
    # It must not raise, and it must not leave a mangled layout. The exact
    # numbers are not asserted: an unshown splitter normalises its own sizes,
    # so pinning them here would test Qt's layout rather than the fallback.
    assert split.count() == 3
    assert split.sizes() == before


def test_a_missing_splitter_is_not_a_crash():
    """Not every screen builds one."""
    screen = _Screen()
    screen._remember_runtime_splitter(None)
    assert screen._runtime_splitter is None
