"""An Annotate page is what fits, and it never scrolls (item 512, part 1).

The maintainer's words: "images should not be visable by scrolling they
should ocupy the screen relestate the container and settings gives to them
... if i start the console i get the ability to scroll, i dont want that".

So the properties asserted here are about the SCREEN, measured after the
layout has settled (HANDOFF, the measurement lessons of 2026-09-02: one
``processEvents`` after ``show`` still reports pre-layout widths):

* at several window sizes, with the console closed and open, the grid's
  scroll range is zero in both directions -- nothing is reachable by
  scrolling -- and every tile sits inside the viewport;
* the page holds exactly the crops the room fits, so opening the console
  shrinks the page and the page counter's total grows, and closing it
  gives the room back.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")

N_CROPS = 400


@pytest.fixture
def many_crops(tmp_path: Path) -> Path:
    """An experiment folder with more crops than any window here fits."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data").mkdir(parents=True)
    rng = np.random.default_rng(3)
    paths = []
    for i in range(N_CROPS):
        arr = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
        path = src / "data" / f"crop_{i:03d}.png"
        Image.fromarray(arr).save(path)
        paths.append(str(path))
    with sqlite3.connect(src / "measurements" / "measurements.db") as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" (png_path) VALUES (?)',
                         [(p,) for p in paths])
    return src


def _signature(screen):
    """Everything that moves while the layout is still settling."""
    scroll = screen._grid_scroll
    return (screen.size().width(), screen.size().height(),
            scroll.viewport().width(), scroll.viewport().height(),
            screen._settings.grid_rows, screen._settings.grid_cols,
            len(screen._thumbs), len(screen._page_paths),
            screen._resize_timer.isActive(), screen._page_worker is None)


def _settle(screen, qtbot, *, rounds: int = 8, timeout_ms: int = 8000):
    """Pump until nothing moves for ``rounds`` consecutive looks."""
    still = 0
    last = None
    waited = 0
    while waited < timeout_ms:
        qtbot.wait(40)
        waited += 40
        now = _signature(screen)
        idle = not screen._resize_timer.isActive() and \
            screen._page_worker is None
        if now == last and idle:
            still += 1
            if still >= rounds:
                return
        else:
            still = 0
        last = now
    raise AssertionError(f"the layout never settled: {last}")


@pytest.fixture
def screen(qtbot, qt_theme_applied, many_crops: Path):
    """A shown AnnotateScreen with the source open and the grid free."""
    from spacr.qt.screens.annotate import AnnotateScreen

    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.image_size = (48, 48)
    widget.resize(1000, 720)
    widget.show()
    qtbot.waitExposed(widget)
    widget._open_source(str(many_crops))
    qtbot.waitUntil(lambda: bool(widget._page_paths), timeout=10000)
    _settle(widget, qtbot)
    yield widget
    if widget._worker is not None:
        widget._worker.stop(wait=True)


def _set_console(widget, qtbot, on: bool) -> None:
    """Open or close the console the way the switch does."""
    widget._console_switch.setChecked(on)
    _settle(widget, qtbot)


def _assert_nothing_scrolls(widget) -> None:
    """Zero scroll range both ways, and every tile inside the viewport."""
    scroll = widget._grid_scroll
    vbar = scroll.verticalScrollBar()
    hbar = scroll.horizontalScrollBar()
    assert vbar.maximum() == 0, (
        f"the grid scrolls vertically by {vbar.maximum()} px at "
        f"{widget.width()}x{widget.height()} "
        f"(viewport {scroll.viewport().height()} px, grid "
        f"{widget._settings.grid_rows} rows)")
    assert hbar.maximum() == 0, (
        f"the grid scrolls horizontally by {hbar.maximum()} px at "
        f"{widget.width()}x{widget.height()}")
    viewport = scroll.viewport()
    room = viewport.rect()
    for thumb in widget._thumbs:
        corner = thumb.mapTo(viewport, thumb.rect().bottomRight())
        assert room.contains(corner), (
            f"tile {thumb.slot} ends at {corner.x()},{corner.y()}, outside "
            f"the {room.width()}x{room.height()} viewport")


def test_the_arithmetic_fits_tiles_gaps_and_margins():
    """The page size is arithmetic, so it is asserted without a window."""
    from spacr.qt.screens.annotate import grid_that_fits

    assert grid_that_fits(100, 100, 30, 30) == (3, 3)
    assert grid_that_fits(100, 100, 30, 30, gap=5) == (3, 3), (
        "three tiles and two gaps are 100 px exactly")
    assert grid_that_fits(99, 100, 30, 30, gap=5) == (3, 2)
    assert grid_that_fits(100, 100, 30, 30, gap=5, margin=1) == (2, 2)
    assert grid_that_fits(10, 10, 30, 30) == (1, 1), (
        "a room smaller than a tile still shows one crop")


@pytest.mark.parametrize("size", [(1366, 768), (1000, 720), (760, 560)])
def test_no_window_size_or_console_state_scrolls(screen, qtbot, size):
    """Closed and open, at three sizes: the range is zero and tiles fit."""
    screen.resize(*size)
    _settle(screen, qtbot)
    for console in (False, True, False):
        _set_console(screen, qtbot, console)
        _assert_nothing_scrolls(screen)
        rows, cols = screen._settings.grid_rows, screen._settings.grid_cols
        assert len(screen._thumbs) == rows * cols
        assert len(screen._page_paths) == rows * cols, (
            "the page must hold exactly the crops that fit: "
            f"{len(screen._page_paths)} loaded for a {rows}x{cols} grid")


def test_the_page_follows_the_console(screen, qtbot):
    """Opening the console pushes crops to the next page, and closing it
    gives them back -- and the page counter follows both ways.

    A bare screen with no main window around it GROWS when the console
    opens (the console has a minimum height), so the window is put back to
    the size it had before the closed page is compared: the property is
    "the same room gives the same page", not "a window never grows".
    """
    screen.resize(1366, 768)
    _settle(screen, qtbot)
    closed = screen._settings.page_size
    closed_text = screen._page_counter_text()

    _set_console(screen, qtbot, True)
    opened = screen._settings.page_size
    assert opened < closed, (
        f"the console took room but the page still holds {opened} crops "
        f"(it held {closed} with the console closed)")
    assert len(screen._page_paths) == opened
    assert screen._page_counter_text() != closed_text
    pages_open = -(-N_CROPS // opened)
    assert f"of {pages_open}" in screen._page_counter_text()

    _set_console(screen, qtbot, False)
    screen.resize(1366, 768)
    _settle(screen, qtbot)
    assert screen._settings.page_size == closed
    assert len(screen._page_paths) == closed


def test_the_first_crop_stays_when_the_page_resizes(screen, qtbot):
    """A refit changes the page's length, not where it starts."""
    screen.resize(1366, 768)
    _settle(screen, qtbot)
    screen._on_next()
    _settle(screen, qtbot)
    first = screen._page_paths[0][0]
    offset = screen._offset
    _set_console(screen, qtbot, True)
    assert screen._offset == offset
    assert screen._page_paths[0][0] == first


def test_the_console_opens_inside_a_768_pixel_window(screen, qtbot):
    """The window does not grow when the console opens at 1366 x 768.

    The crop pane used to inherit the empty state's 321 px minimum, and the
    console's own minimum on top of it made the screen taller than a 768 px
    display: the bottom of the window went off the screen instead of the
    crops going to the next page. With the judgement bar showing too, the
    screen must still open its console inside the window it has.
    """
    screen._judge_bar.show()
    screen.resize(1366, 768)
    _settle(screen, qtbot)
    _set_console(screen, qtbot, True)
    screen.resize(1366, 768)
    _settle(screen, qtbot)
    assert screen.height() == 768, (
        f"opening the console grew the window to {screen.height()} px; "
        f"its minimum is {screen.minimumSizeHint().height()} px")
    _assert_nothing_scrolls(screen)
    assert len(screen._page_paths) == screen._settings.page_size >= 1
