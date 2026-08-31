"""A decorative backdrop must never hold up the module the user clicked.

WHAT THIS IS ABOUT. Opening a module soon after launch froze the GUI
thread for 3148 ms -- long enough for the compositor to offer to
force-quit spaCR, which is what the user saw as
"io.github.olafssonlab.spacr is not responding" every time they opened a
module. Sampling the thread through the block said 83% of it was one
frame: the GPU backdrop's constructor, waiting on HEAVY_IMPORT_LOCK,
which the startup preloader holds for a whole module import.

The rule that came out of it: the module is what somebody is waiting
for, the backdrop is not, so the backdrop is what waits.

These tests assert the DECISION -- does the install go ahead, or come
back later -- rather than a duration, because a wall-clock assertion on
a shared machine measures the neighbours.
"""
from __future__ import annotations

import threading

import pytest

from spacr.qt.screens.app_screen import AppScreen


def _make_screen(qtbot, app_key: str = "mask") -> AppScreen:
    scr = AppScreen(app_key)
    qtbot.addWidget(scr)
    return scr


class TestThePeekAtTheLock:
    """`_heavy_lock_is_free` is a peek, not a reservation."""

    def test_a_free_lock_reads_as_free(self, qtbot):
        scr = _make_screen(qtbot)
        assert scr._heavy_lock_is_free() is True

    def test_a_held_lock_reads_as_busy(self, qtbot):
        """The preloader holding it is the whole reason this exists."""
        from spacr.qt.app import HEAVY_IMPORT_LOCK

        scr = _make_screen(qtbot)
        with HEAVY_IMPORT_LOCK:
            assert scr._heavy_lock_is_free() is False
        assert scr._heavy_lock_is_free() is True

    def test_the_peek_does_not_keep_the_lock(self, qtbot):
        """A peek that forgot to release would deadlock the preloader.

        This is the failure the name is guarding against: `acquire` then
        no `release` reads as "free" exactly once and hangs the next
        import for ever.
        """
        from spacr.qt.app import HEAVY_IMPORT_LOCK

        scr = _make_screen(qtbot)
        assert scr._heavy_lock_is_free() is True
        got = HEAVY_IMPORT_LOCK.acquire(blocking=False)
        try:
            assert got is True, "the peek kept the lock it was only borrowing"
        finally:
            if got:
                HEAVY_IMPORT_LOCK.release()

    def test_a_tree_with_no_lock_to_ask_says_free(self, qtbot, monkeypatch):
        """A machine without the backdrop module behaves as it always did."""
        import spacr.qt.widgets.fractal_travel as ft

        scr = _make_screen(qtbot)
        monkeypatch.setattr(ft, "_heavy_import_lock", lambda: None)
        assert scr._heavy_lock_is_free() is True

    def test_a_lock_that_cannot_be_reached_says_free(self, qtbot,
                                                    monkeypatch):
        """An import failure must not stop a screen from being decorated."""
        import spacr.qt.widgets.fractal_travel as ft

        def explode():
            raise RuntimeError("no GL on this machine")

        scr = _make_screen(qtbot)
        monkeypatch.setattr(ft, "_heavy_import_lock", explode)
        assert scr._heavy_lock_is_free() is True


class TestTheInstallWaitsRatherThanTheUser:
    """What `_install_ambient` does when the lock is busy."""

    def test_a_busy_lock_defers_instead_of_blocking(self, qtbot,
                                                   monkeypatch):
        """The screen opens undecorated; the backdrop comes back later."""
        scr = _make_screen(qtbot)
        scr._ambient = None
        scr._backdrops_ready = True
        monkeypatch.setattr(scr, "_heavy_lock_is_free", lambda: False)

        scheduled = []
        from PySide6.QtCore import QTimer

        monkeypatch.setattr(
            QTimer, "singleShot",
            staticmethod(lambda ms, fn: scheduled.append((ms, fn))))
        scr._install_ambient()

        assert scr._ambient is None, "it built the backdrop anyway"
        assert scheduled, "it neither built the backdrop nor came back for it"
        delay, retry = scheduled[0]
        assert delay > 0
        assert retry == scr._install_ambient

    def test_a_free_lock_installs_now(self, qtbot, monkeypatch):
        """The ordinary path is unchanged: free lock, install immediately."""
        scr = _make_screen(qtbot)
        scr._ambient = None
        scr._backdrops_ready = True
        monkeypatch.setattr(scr, "_heavy_lock_is_free", lambda: True)

        scheduled = []
        from PySide6.QtCore import QTimer

        monkeypatch.setattr(
            QTimer, "singleShot",
            staticmethod(lambda ms, fn: scheduled.append((ms, fn))))
        scr._install_ambient()
        assert not scheduled, "it deferred an install it could have done"

    def test_a_screen_that_already_has_one_neither_defers_nor_rebuilds(
            self, qtbot, monkeypatch):
        """The early return comes first, so a busy lock is never consulted."""
        scr = _make_screen(qtbot)
        scr._backdrops_ready = True
        scr._ambient = object()
        asked = []
        monkeypatch.setattr(scr, "_heavy_lock_is_free",
                            lambda: asked.append(1) or True)

        scheduled = []
        from PySide6.QtCore import QTimer

        monkeypatch.setattr(
            QTimer, "singleShot",
            staticmethod(lambda ms, fn: scheduled.append((ms, fn))))
        scr._install_ambient()
        assert not scheduled
        assert not asked, "it peeked at the lock for a screen already done"

    def test_a_screen_not_ready_for_backdrops_does_not_defer(self, qtbot,
                                                            monkeypatch):
        """`_backdrops_ready` is a harder no than a busy lock."""
        scr = _make_screen(qtbot)
        scr._backdrops_ready = False
        scr._ambient = None
        monkeypatch.setattr(scr, "_heavy_lock_is_free", lambda: False)

        scheduled = []
        from PySide6.QtCore import QTimer

        monkeypatch.setattr(
            QTimer, "singleShot",
            staticmethod(lambda ms, fn: scheduled.append((ms, fn))))
        scr._install_ambient()
        assert not scheduled, "it queued a retry for a screen that wants none"
