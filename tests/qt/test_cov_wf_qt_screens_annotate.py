"""Annotate's last unexercised corners: the chrome that can be missing.

Three of the Annotate screen's guards ask whether a piece of its own chrome
is there before touching it -- the zoom overlay, the keyboard-hint label --
and one asks whether a save worker is running before it restarts one. Every
one of them protects a *user's labels*: the write path has to survive the
grid furniture being absent, because a keystroke that raised instead of
recording a class would lose the annotation the user just made and give no
sign that it had.

Each test here drives BOTH sides of the guard in one go: the ordinary case
that produces the effect, and the stripped case that must not. Asserting
only that "nothing happened" would pass against a screen that does nothing
at all, which is exactly the failure this file exists to rule out.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QRect, Qt, QThread, Signal

from spacr.qt import annotate_engine as engine
from spacr.qt.screens import annotate as annotate_mod

# ---------------------------------------------------------------------------
# Fixtures -- a real experiment folder, because every path here reads one
# ---------------------------------------------------------------------------

@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    """A small experiment folder: eight crops and a png_list naming them."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data" / "images").mkdir(parents=True)
    rng = np.random.default_rng(11)
    paths = []
    for i in range(8):
        arr = rng.integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
        png = src / "data" / "images" / f"cell_{i:02d}.png"
        Image.fromarray(arr).save(png)
        paths.append(str(png))
    db = src / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" (png_path) VALUES (?)',
                         [(p,) for p in paths])
    engine.ensure_annotation_column(str(db), "annotate")
    return src


@pytest.fixture
def screen(qtbot, qt_theme_applied, experiment: Path):
    """An Annotate screen on a pinned 3x3 grid with its first page decoded."""
    scr = annotate_mod.AnnotateScreen()
    qtbot.addWidget(scr)
    scr._settings.grid_rows = 3
    scr._settings.grid_cols = 3
    scr._settings.image_size = (24, 24)
    scr._compute_grid_dims = lambda: None
    scr._rebuild_grid()
    scr._open_source(str(experiment))
    qtbot.waitUntil(lambda: len(scr._page_paths) >= 3, timeout=10000)
    qtbot.waitUntil(
        lambda: scr._thumb_pixmaps[2] is not None
        and not scr._thumb_pixmaps[2].isNull(), timeout=10000)
    yield scr
    if scr._worker is not None:
        scr._worker.stop(wait=True)


# ---------------------------------------------------------------------------
# The zoom overlay, and a grid that has not got one
# ---------------------------------------------------------------------------

def test_shift_clicking_a_crop_with_no_overlay_built_blows_nothing_up(screen):
    """Shift+click is an extra gesture; it must never be the thing that
    breaks annotating.

    The overlay is built once, with the grid. A screen whose build stopped
    short of it (the same partial build that costs the drop zone when the
    DnD install raises) still has tiles a user can shift+click, and the
    gesture then has to come to nothing instead of raising out of a mouse
    handler -- a raise there aborts the click that would have labelled the
    crop.
    """
    scr = screen
    overlay = scr._zoom_overlay
    viewport = scr._grid_scroll.viewport()

    # With the overlay in place the gesture does its real work.
    scr._on_thumb_shift(2)
    assert overlay.slot == 2
    assert overlay.isVisibleTo(viewport) is True
    scr._fold_zoom_back()
    assert overlay.slot == -1

    # Now the screen has no overlay at all.
    del scr._zoom_overlay
    try:
        scr._on_thumb_shift(1)
    finally:
        scr._zoom_overlay = overlay
    assert overlay.slot == -1, "no crop was blown up"
    assert overlay.isVisibleTo(viewport) is False
    # and the tile is still a live, annotatable crop afterwards
    assert scr._set_annotation(1, 3) is True
    assert scr._page_paths[1][1] == 3


def test_folding_a_zoom_back_with_no_overlay_built_touches_nothing(screen):
    """Fold-back runs on every page load, not just on Escape.

    ``_load_page`` folds the zoom back before it swaps the crops, and
    ``_fit_zoom_overlay`` follows every viewport resize. Both therefore run
    on a screen that may not have finished building, so both have to be
    no-ops rather than raise -- an exception on the page-load path would
    strand the grid on the previous page's images.
    """
    scr = screen
    overlay = scr._zoom_overlay
    viewport = scr._grid_scroll.viewport()
    scr._on_thumb_shift(2)
    assert overlay.slot == 2

    del scr._zoom_overlay
    try:
        scr._fold_zoom_back()
        assert overlay.slot == 2, "the orphaned overlay was not folded back"
        assert overlay.isVisibleTo(viewport) is True
        overlay.setGeometry(QRect(0, 0, 7, 7))
        scr._fit_zoom_overlay()
        assert overlay.geometry() == QRect(0, 0, 7, 7), "nothing was refitted"
    finally:
        scr._zoom_overlay = overlay

    # Handed the overlay back, the very same calls do their real work.
    scr._fold_zoom_back()
    assert overlay.slot == -1
    assert overlay.isVisibleTo(viewport) is False
    scr._fit_zoom_overlay()
    assert overlay.geometry() == viewport.rect()


# ---------------------------------------------------------------------------
# The keyboard hint label
# ---------------------------------------------------------------------------

def test_a_keystroke_still_records_its_class_with_no_hint_label_to_show(
        screen):
    """The hint strip is chrome; the class number is the user's data.

    Every keyboard action reports itself through ``_set_kbd_hint``. If that
    call assumed the label existed, a screen built without the legend would
    raise on the FIRST digit key pressed -- losing the annotation and, worse,
    doing so from inside a key handler where the user sees only a dead
    keyboard.
    """
    scr = screen
    scr._set_focus_slot(0)

    # Ordinary case: the hint says what the key did.
    assert scr.handle_key("0") is True
    assert scr._kbd_hint.text() == "Cleared."

    label = scr._kbd_hint
    del scr._kbd_hint
    try:
        assert scr.handle_key("1") is True
    finally:
        scr._kbd_hint = label

    # The label is gone, but the label the user pressed was still recorded.
    assert scr._current_value(0) == 1
    assert scr._page_paths[0][1] == 1
    path = scr._page_paths[0][0]
    assert scr._pending_updates[path] == 1
    assert label.text() == "Cleared.", "the detached label was not written to"


# ---------------------------------------------------------------------------
# Retraining with and without a save worker
# ---------------------------------------------------------------------------

def test_retraining_with_no_save_worker_running_does_not_conjure_one(
        screen, qtbot, monkeypatch):
    """A retrain must fit on the labels the user has already made.

    So it stops the save worker first (flushing what is queued) and starts a
    fresh one, because the round reads the database the worker writes to. A
    screen with no worker -- nothing open to save into -- must not have one
    invented for it: a SaveWorker pointed at a database the screen is not
    annotating is a second sqlite writer on that file for the rest of the
    session.
    """
    started = []

    class _StubRetrain(QThread):
        done = Signal(object)
        failed = Signal(str)

        def __init__(self, db_path, annotation_column, options, parent=None):
            super().__init__(parent)
            started.append((db_path, annotation_column, dict(options)))

        def run(self):        # the real one fits a model; nothing to fit here
            return None

    monkeypatch.setattr(annotate_mod, "_RetrainWorker", _StubRetrain)
    scr = screen
    first_worker = scr._worker
    assert first_worker is not None, "the fixture opened a source"

    scr._on_retrain()
    assert scr._btn_retrain.isEnabled() is False, "no double-fire"
    assert scr._status_label.text() == "Retraining on the labels so far…"
    assert scr._worker is not first_worker, "a fresh save worker took over"
    qtbot.waitUntil(lambda: scr._retrain_worker is None, timeout=10000)

    # Same screen, no save worker: the round still starts, alone.
    scr._worker.stop(wait=True)
    scr._worker = None
    scr._on_retrain()
    assert scr._worker is None, "no save worker was invented"
    qtbot.waitUntil(lambda: scr._retrain_worker is None, timeout=10000)

    assert len(started) == 2, "both clicks fitted a round"
    assert started[1][0] == scr._settings.db_path
    assert started[1][1] == scr._settings.annotation_column
    assert started[1][2]["round_index"] == scr._round_index
    assert scr._btn_retrain.isEnabled() is True


# ---------------------------------------------------------------------------
# Hover revalidation after the crops change
# ---------------------------------------------------------------------------

def test_a_hover_the_cursor_is_still_inside_survives_the_page_changing(screen):
    """The white ring has to stay on the tile the next click will hit.

    A page load swaps the crops under widgets that never move, and Qt only
    re-sends Enter/Leave when the cursor crosses a boundary. So the screen
    re-asks ``underMouse``: a cursor still resting on tile 2 keeps its hover
    (dropping it would leave the ring on a crop the mouse is not on), and a
    cursor that has left without a Leave event loses it.
    """
    scr = screen
    thumb = scr._thumbs[2]
    thumb.setAttribute(Qt.WA_UnderMouse, True)
    scr._on_thumb_hover(2, True)
    assert scr.hover_slot == 2
    assert scr.current_slot == 2, "hovering moves the ring"

    scr._revalidate_hover()
    assert scr.hover_slot == 2, "the cursor never left the tile"
    assert scr.current_slot == 2

    # The cursor was warped out of the window: no Leave was ever delivered.
    thumb.setAttribute(Qt.WA_UnderMouse, False)
    scr._revalidate_hover()
    assert scr.hover_slot is None
    assert scr.current_slot == 2, "the keyboard still targets that crop"
