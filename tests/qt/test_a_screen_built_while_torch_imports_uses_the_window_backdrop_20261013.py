"""A screen opened while the preloader holds the import lock defers to the window's field.

CI failed tests/qt/test_mask_screen_field_frame_costs_near_home_20261012.py
with one preference-store read per field frame on Make Masks, in every run
on 2026-10-13 and never on the workstation. The read was
:meth:`spacr.qt.screens.app_screen.AppScreen.page_fill`, which resolves the
theme from the store and fills the whole screen on every paint unless the
screen is marked as deferring to the window's backdrop.

A screen is marked when the window takes it
(``MainWindow._drop_a_redundant_screen_backdrop``), but only if it has built
a backdrop of its own by then. On a slow machine the pipeline preloader is
still importing torch when Make Masks opens, so the screen's own backdrop
waits for the heavy-import lock on a timer; the window found no backdrop to
retire and left the screen unmarked, and when the timer came back the
screen saw the window's field, built nothing and stayed unmarked. Every
field frame then painted a flat page under the field it could not see and
read the preference store to choose its colour.
"""
from __future__ import annotations

import traceback

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

pytestmark = pytest.mark.qt


@pytest.fixture
def lock_held_while_mask_opens(monkeypatch):
    """The heavy-import lock reads as taken until the test lets it go."""
    from spacr.qt.screens.app_screen import AppScreen

    held = [True]
    monkeypatch.setattr(AppScreen, "_heavy_lock_is_free",
                        lambda self: not held[0])
    return held


def test_the_screen_defers_to_the_window_field_once_the_lock_is_free(
        qtbot, monkeypatch, lock_held_while_mask_opens):
    from spacr.qt import preferences
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1280, 800)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window.window_backdrop() is not None,
                    timeout=20000)
    backdrop = window.window_backdrop()
    if not hasattr(backdrop, "_timer"):
        pytest.skip("the window backdrop is not the field in this setup")
    window._on_nav_selected("mask")
    qtbot.wait(300)
    screen = window._screens["mask"]
    assert screen._ambient is None

    lock_held_while_mask_opens[0] = False
    qtbot.wait(400)
    assert screen._ambient is None, "the screen built a second field"
    assert screen.page_fill() is None
    assert "AppScreen { background-color" not in screen.styleSheet(), (
        screen.styleSheet())
    assert not screen.autoFillBackground()

    backdrop._timer.stop()
    backdrop.repaint()
    reads = []
    real = preferences._settings

    def counting():
        reads.append("".join(traceback.format_stack(limit=14)))
        return real()

    monkeypatch.setattr(preferences, "_settings", counting)
    for _frame in range(5):
        backdrop.repaint()
    assert reads == [], (f"{len(reads) / 5:.1f} preference reads per field "
                         f"frame; the first from:\n{reads[0]}")



def test_a_screen_that_paints_its_own_page_reads_no_store_per_frame(
        qtbot, monkeypatch):
    """The page a screen fills under everything takes the paint-time theme.

    A screen that is not deferring to the window's field (none was there
    to defer to yet, or it has not heard of it) fills its page on every
    paint, and every frame of the field repaints it. The colour came from
    the preference store each time, one read a frame; it now comes from
    :func:`spacr.qt.theme._painting_theme`, as every other paint hook's.
    """
    from spacr.qt import preferences
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1280, 800)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window.window_backdrop() is not None,
                    timeout=20000)
    backdrop = window.window_backdrop()
    if not hasattr(backdrop, "_timer"):
        pytest.skip("the window backdrop is not the field in this setup")
    window._on_nav_selected("mask")
    qtbot.wait(300)
    screen = window._screens["mask"]
    screen._uses_window_backdrop = False
    assert screen._ambient is None and screen.page_fill() is not None

    backdrop._timer.stop()
    backdrop.repaint()
    reads = []
    real = preferences._settings

    def counting():
        reads.append("".join(traceback.format_stack(limit=14)))
        return real()

    monkeypatch.setattr(preferences, "_settings", counting)
    for _frame in range(5):
        backdrop.repaint()
    assert reads == [], (f"{len(reads) / 5:.1f} preference reads per field "
                         f"frame; the first from:\n{reads[0]}")
