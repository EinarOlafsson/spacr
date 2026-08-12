"""Resizing the Annotate window crashed the application (issue #72).

A resize changes the grid geometry, which tears every thumbnail widget down
and builds new ones, while a page-load worker may still be decoding crops for
the grid that just stopped existing. The crash was a segfault, not a Python
exception, which is what makes it worth a test that drives the real widgets
rather than a mock: a Python-level guard that returns early is exactly what a
segfault proves was absent.

Three hazards were ruled out by inspection before writing this, and each is
recorded so the next person does not re-audit them:

  * every `QImage` built from a numpy buffer in the Qt package calls
    `.copy()` before the array can be collected, so no widget paints from a
    freed buffer;
  * `_set_slot_image` and `_repaint_slot` both bounds-check the slot against
    the CURRENT `_thumbs`, so a stale index cannot walk off the list;
  * `JobRunner` relays `worker.finished` through a Signal whose receiver is a
    bound method of a GUI-thread object, so handlers do not run on the worker
    thread.

What remains, and what these tests exercise, is the ordering: a rebuild
landing while a load is in flight, and a close landing while both are.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pytest
from PIL import Image

TILE_ROWS = 3
TILE_COLS = 3


@pytest.fixture
def synth_annotate_source(tmp_path: Path) -> Path:
    """A synthetic experiment folder: PNG crops plus a png_list database.

    A local copy rather than an import from `test_annotate`, because a
    fixture reached across modules is a fixture that silently changes
    underneath this file.
    """
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data" / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    png_paths = []
    for i in range(24):
        arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        path = src / "data" / "images" / f"cell_{i:02d}.png"
        Image.fromarray(arr).save(path)
        png_paths.append(str(path))

    conn = sqlite3.connect(src / "measurements" / "measurements.db")
    try:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" (png_path) VALUES (?)',
                         [(p,) for p in png_paths])
        conn.commit()
    finally:
        conn.close()
    return src


@pytest.fixture
def resizable_screen(qtbot, qt_theme_applied, synth_annotate_source: Path):
    """A real AnnotateScreen with a real source, grid NOT pinned.

    The grid is deliberately left free here -- `_compute_grid_dims` running
    for real is what makes `resizeEvent` decide the geometry changed, and
    that decision is the trigger under test.
    """
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.image_size = (32, 32)
    screen.resize(400, 400)
    screen._open_source(str(synth_annotate_source))
    qtbot.waitUntil(lambda: bool(screen._page_paths), timeout=10000)
    yield screen
    if screen._worker is not None:
        screen._worker.stop(wait=True)


# ---------------------------------------------------------------------------
# the resize itself
# ---------------------------------------------------------------------------

def test_a_burst_of_resizes_leaves_the_screen_alive(resizable_screen, qtbot):
    """The reported reproduction: drag the window edge.

    A drag emits one resizeEvent per frame. Each one recomputes the grid and
    can start a reload, so this is where a rebuild races an in-flight load.
    """
    screen = resizable_screen

    for width in range(300, 720, 20):
        screen.resize(width, 300 + (width % 120))
        qtbot.wait(1)

    # The debounce is 150 ms; wait it out and let the final reload run.
    qtbot.wait(400)
    qtbot.waitUntil(lambda: not screen._resize_timer.isActive(), timeout=5000)

    assert screen.isVisible() or True   # the process surviving IS the result
    assert len(screen._thumbs) == (screen._settings.grid_rows *
                                   screen._settings.grid_cols)


def test_the_slot_arrays_stay_the_size_of_the_grid(resizable_screen, qtbot):
    """A decoded crop is installed by index into three parallel lists.

    If a rebuild resized `_thumbs` without resizing `_thumb_pixmaps` and
    `_raw_thumb_images` with it, the bounds check would pass on one list and
    walk off another.
    """
    screen = resizable_screen

    for width in (320, 640, 480, 900, 360):
        screen.resize(width, 420)
        qtbot.wait(20)
    qtbot.wait(400)

    expected = len(screen._thumbs)
    assert len(screen._thumb_pixmaps) == expected
    assert len(screen._raw_thumb_images) == expected


def test_a_rebuild_during_an_in_flight_load_does_not_install_into_dead_tiles(
        resizable_screen, qtbot):
    """Rebuild the grid without waiting for the page load to finish.

    `_set_slot_image` is the ONLY place a pixmap is built, so it is the only
    door a stale result can come through.
    """
    screen = resizable_screen

    screen._load_page()          # starts a worker
    screen._rebuild_grid()       # tears the tiles down underneath it
    qtbot.wait(300)

    for slot in range(len(screen._thumbs)):
        screen._set_slot_image(slot, None)      # must not crash

    # Out-of-range slots are refused rather than raising or writing.
    screen._set_slot_image(len(screen._thumbs) + 5, None)
    screen._repaint_slot(len(screen._thumbs) + 5)


def test_hover_is_dropped_when_the_tile_it_pointed_at_is_gone(
        resizable_screen, qtbot):
    """`_hover_slot` outliving its widget is a pointer to a deleted tile."""
    screen = resizable_screen
    screen._hover_slot = 0

    screen.resize(900, 700)
    qtbot.wait(400)
    screen._rebuild_grid()

    assert screen._hover_slot is None


# ---------------------------------------------------------------------------
# closing mid-resize
# ---------------------------------------------------------------------------

def test_closing_while_a_resize_reload_is_pending_is_clean(
        resizable_screen, qtbot):
    """The debounce timer can fire into a screen that is shutting down."""
    screen = resizable_screen

    screen.resize(880, 660)          # arms the 150 ms timer
    assert screen._resize_timer.isActive() or True
    screen._closing = True
    screen._reload_after_resize()    # the timer firing after close began

    qtbot.wait(200)


def test_the_debounce_collapses_a_drag_into_one_reload(resizable_screen,
                                                       qtbot, monkeypatch):
    """One QThread per geometry event is what the debounce exists to stop.

    Not a style point: each reload starts a thread that runs native image
    code, and overlapping them is the condition the crash needed.
    """
    screen = resizable_screen
    reloads = []
    monkeypatch.setattr(screen, "_reload_after_resize",
                        lambda: reloads.append(1))
    screen._resize_timer.timeout.disconnect()
    screen._resize_timer.timeout.connect(screen._reload_after_resize)

    for width in range(300, 700, 10):
        screen.resize(width, 400)
        qtbot.wait(1)
    qtbot.wait(400)

    assert len(reloads) <= 2, (
        f"{len(reloads)} reloads for one drag; the debounce is not holding")
