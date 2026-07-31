"""Tests for the Qt annotate screen + its pure-Python engine.

Uses a synthetic on-disk experiment: a folder with 8 crops as PNGs and
a `measurements/measurements.db` whose `png_list` table references them.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spacr.qt import annotate_engine as engine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_annotate_source(tmp_path: Path) -> Path:
    """Build a synthetic experiment folder with png_list DB + PNGs."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data" / "images").mkdir(parents=True)
    png_paths = []
    rng = np.random.default_rng(0)
    for i in range(8):
        arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        p = src / "data" / "images" / f"cell_{i:02d}.png"
        Image.fromarray(arr).save(p)
        png_paths.append(str(p))
    # Build DB
    db = src / "measurements" / "measurements.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            'CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)'
        )
        conn.executemany(
            'INSERT INTO "png_list" (png_path) VALUES (?)',
            [(p,) for p in png_paths],
        )
        conn.commit()
    finally:
        conn.close()
    return src


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

def test_label_to_hex():
    assert engine.label_to_hex(None) is None
    assert engine.label_to_hex(0) is None
    assert engine.label_to_hex("abc") is None
    assert engine.label_to_hex(1).startswith("#") and len(engine.label_to_hex(1)) == 7
    assert engine.label_to_hex(1) != engine.label_to_hex(2)
    assert engine.label_to_hex(5).startswith("#")


def test_ensure_annotation_column_adds_missing(synth_annotate_source: Path):
    db = str(synth_annotate_source / "measurements" / "measurements.db")
    engine.ensure_annotation_column(db, "my_col")
    with sqlite3.connect(db) as conn:
        cols = {r[1] for r in conn.execute('PRAGMA table_info("png_list")')}
    assert "my_col" in cols


def test_count_rows_and_fetch_page(synth_annotate_source: Path):
    db = str(synth_annotate_source / "measurements" / "measurements.db")
    engine.ensure_annotation_column(db, "annotate")
    assert engine.count_rows(db) == 8
    page = engine.fetch_page(db, "annotate", offset=0, page_size=5)
    assert len(page) == 5
    # Every row is (png_path, None) since we haven't annotated anything
    for path, val in page:
        assert os.path.isfile(path)
        assert val is None


def test_save_worker_persists_and_null_clears(synth_annotate_source: Path):
    db = str(synth_annotate_source / "measurements" / "measurements.db")
    engine.ensure_annotation_column(db, "annotate")
    page = engine.fetch_page(db, "annotate", 0, 8)
    paths = [p for p, _ in page]
    worker = engine.SaveWorker(db, "annotate")
    worker.start()
    try:
        worker.submit({paths[0]: 1, paths[1]: 2, paths[2]: 3})
        # Wait for save
        for _ in range(50):
            if not worker.busy and worker.last_save_ts is not None:
                break
            time.sleep(0.05)
        # Now null one of them
        worker.submit({paths[0]: None})
        for _ in range(50):
            if worker.last_save_ts and (time.time() - worker.last_save_ts) < 5:
                time.sleep(0.05)  # spin one more tick to let it commit
            break
        time.sleep(0.3)
    finally:
        worker.stop()
    with sqlite3.connect(db) as conn:
        rows = dict(conn.execute('SELECT png_path, annotate FROM "png_list"').fetchall())
    assert rows[paths[0]] is None
    assert rows[paths[1]] == 2
    assert rows[paths[2]] == 3


def test_save_worker_rolls_back_and_reports_a_failed_commit(tmp_path):
    """A daemon-thread database error must never look like a saved batch."""
    db = str(tmp_path / "measurements.db")
    with sqlite3.connect(db) as conn:
        conn.execute(
            'CREATE TABLE "png_list" ('
            'png_path TEXT PRIMARY KEY, '
            'annotate INTEGER CHECK (annotate BETWEEN 0 AND 2))')
        conn.executemany(
            'INSERT INTO "png_list" VALUES (?, NULL)',
            [("one.png",), ("two.png",)],
        )
    worker = engine.SaveWorker(db, "annotate")
    worker.start()
    try:
        worker.submit({"one.png": 1, "two.png": 9})
        for _ in range(100):
            if worker.last_error:
                break
            time.sleep(0.02)
        assert worker.last_error
        assert "IntegrityError" in worker.last_error
        assert worker.last_save_ts is None
        assert worker.pending_batches == 1
    finally:
        worker.stop()
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            'SELECT annotate FROM "png_list" ORDER BY png_path'
        ).fetchall() == [(None,), (None,)]


def test_class_counts_after_save(synth_annotate_source: Path):
    db = str(synth_annotate_source / "measurements" / "measurements.db")
    engine.ensure_annotation_column(db, "annotate")
    page = engine.fetch_page(db, "annotate", 0, 8)
    paths = [p for p, _ in page]
    worker = engine.SaveWorker(db, "annotate")
    worker.start()
    try:
        worker.submit({paths[0]: 1, paths[1]: 1, paths[2]: 2})
        for _ in range(50):
            if worker.last_save_ts:
                time.sleep(0.1)
                break
            time.sleep(0.05)
    finally:
        worker.stop()
    counts = engine.class_counts(db, "annotate")
    assert (1, 2) in counts
    assert (2, 1) in counts


def test_find_last_annotated_offset(synth_annotate_source: Path):
    db = str(synth_annotate_source / "measurements" / "measurements.db")
    engine.ensure_annotation_column(db, "annotate")
    with sqlite3.connect(db) as conn:
        # Annotate the 6th row (0-indexed 5)
        row_paths = [r[0] for r in conn.execute('SELECT png_path FROM "png_list"')]
        conn.execute('UPDATE "png_list" SET annotate = 1 WHERE png_path = ?',
                     (row_paths[5],))
        conn.commit()
    offset = engine.find_last_annotated_offset(db, "annotate", page_size=3)
    # 5 // 3 * 3 == 3
    assert offset == 3


def test_normalize_pil_no_channels_returns_input():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    out = engine.normalize_pil(img)
    assert isinstance(out, Image.Image)
    assert out.size == (10, 10)


def test_add_colored_border_grows_image():
    img = Image.new("RGB", (16, 16), color=(50, 50, 50))
    out = engine.add_colored_border(img, 3, "#ff0000")
    assert out.size == (22, 22)


# ---------------------------------------------------------------------------
# Widget tests
# ---------------------------------------------------------------------------

def test_annotate_screen_constructs(qtbot, qt_theme_applied):
    from spacr.qt.screens.annotate import AnnotateScreen
    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    # Grid holder has the expected number of empty thumbnails
    assert len(screen._thumbs) == screen._settings.grid_rows * screen._settings.grid_cols
    # Status label is present
    assert "Ready" in screen._status_label.text() or screen._status_label.text() == ""


def test_annotate_settings_expose_rgb_as_the_default_stored_order(
        qtbot, qt_theme_applied):
    from spacr.qt.annotate_engine import AnnotateSettings
    from spacr.qt.screens.annotate import _SettingsDialog

    settings = AnnotateSettings()
    dialog = _SettingsDialog(settings)
    qtbot.addWidget(dialog)
    assert dialog._stored_channel_order.currentData() == "rgb"
    for widget in (
            dialog._src_edit, dialog._ann_col, dialog._img_size,
            dialog._stored_channel_order, dialog._queue_measure,
            dialog._queue_limit):
        assert widget.toolTip() == ""
        label = widget._spacr_setting_label
        assert "href=" in label.toolTip()
        assert getattr(label, "_spacr_api_dot", None) is not None
    dialog._stored_channel_order.setCurrentIndex(
        dialog._stored_channel_order.findData("legacy_bgr"))
    assert dialog.collect().stored_channel_order == "legacy_bgr"


def test_annotate_screen_open_source_loads_page(qtbot, qt_theme_applied,
                                                  synth_annotate_source: Path):
    from spacr.qt.screens.annotate import AnnotateScreen
    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.grid_rows = 2
    screen._settings.grid_cols = 2
    screen._rebuild_grid()
    screen._open_source(str(synth_annotate_source))
    # Page load is deferred (QTimer) + processed on a worker thread now.
    qtbot.waitUntil(lambda: len(screen._page_paths) == 4, timeout=5000)
    # Total rows detected
    assert screen._total == 8
    for i, (path, _) in enumerate(screen._page_paths):
        assert os.path.isfile(path)
    # Thumb pixmaps should populate once the worker finishes.
    qtbot.waitUntil(
        lambda: screen._thumbs[0].pixmap() is not None
        and not screen._thumbs[0].pixmap().isNull(), timeout=5000)
    # Cleanup worker
    if screen._worker:
        screen._worker.stop(wait=True)


def test_annotate_screen_left_click_marks_class_1(qtbot, qt_theme_applied,
                                                    synth_annotate_source: Path):
    from spacr.qt.screens.annotate import AnnotateScreen
    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.grid_rows = 2
    screen._settings.grid_cols = 2
    screen._rebuild_grid()
    screen._open_source(str(synth_annotate_source))
    qtbot.waitUntil(lambda: len(screen._page_paths) >= 1, timeout=5000)
    screen._on_thumb_left(0)
    assert screen._page_paths[0][1] == 1
    # A second left-click clears
    screen._on_thumb_left(0)
    assert screen._page_paths[0][1] is None
    if screen._worker:
        screen._worker.stop(wait=True)


def test_annotate_screen_next_prev(qtbot, qt_theme_applied,
                                     synth_annotate_source: Path):
    from spacr.qt.screens.annotate import AnnotateScreen
    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.grid_rows = 2
    screen._settings.grid_cols = 2
    screen._rebuild_grid()
    screen._open_source(str(synth_annotate_source))
    start = screen._offset
    screen._on_next()
    assert screen._offset > start
    screen._on_prev()
    assert screen._offset == start
    if screen._worker:
        screen._worker.stop(wait=True)


def test_reanchor_png_path_resolves_moved_dataset(tmp_path):
    """A stored absolute png_path from a different/old root should re-anchor to
    the real file beside the opened database (fixes the grey-boxes bug)."""
    import os
    from spacr.qt.screens.annotate import _reanchor_png_path
    root = tmp_path / "moved_here"
    img_dir = root / "data" / "single_cell" / "plate1_A01" / "cell_png"
    img_dir.mkdir(parents=True)
    img = img_dir / "plate1_A01_f1_obj1.png"
    img.write_bytes(b"\x89PNG\r\n")   # content irrelevant; isfile is what matters
    db_path = str(root / "measurements" / "measurements.db")
    os.makedirs(os.path.dirname(db_path))
    open(db_path, "w").close()
    # Stored path points at an OLD absolute location that no longer exists.
    stored = "/old/gone/data/single_cell/plate1_A01/cell_png/plate1_A01_f1_obj1.png"
    resolved = _reanchor_png_path(stored, db_path)
    assert resolved == str(img)
    assert os.path.isfile(resolved)


def test_reanchor_keeps_valid_path(tmp_path):
    from spacr.qt.screens.annotate import _reanchor_png_path
    real = tmp_path / "x.png"; real.write_bytes(b"x")
    assert _reanchor_png_path(str(real), "") == str(real)


# ---------------------------------------------------------------------------
# Tile chrome — rounded square, resting gray ring, class ring, current ring
#
# Everything here asserts STATE the widget reports (`border_color()`,
# `ring_color()`, `is_current()`), never rendered pixels — the one
# exception is the corner-rounding test, where geometry is the claim and
# there is nothing else to read it off.
#
# The composition rule under test: the state ring (gray or class colour)
# and the white current ring live in two different bands, so hovering a
# classified crop shows BOTH, and moving the cursor away leaves the class
# colour exactly where it was.
# ---------------------------------------------------------------------------

TILE_ROWS, TILE_COLS = 3, 3          # 9 cells, 8 crops → slot 8 is empty


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Fail loudly if anything under test opens a modal dialog.

    A QMessageBox in a headless run blocks forever; this turns that into
    an assertion instead of a hang.
    """
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError("a modal dialog was opened")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("exec", "exec_", "open", "show"):
        monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


@pytest.fixture
def tile_screen(qtbot, qt_theme_applied, synth_annotate_source: Path):
    """An AnnotateScreen with a pinned 3x3 grid and page 0 fully decoded.

    ``_compute_grid_dims`` refits the grid to the viewport, which is
    non-deterministic offscreen, so it is pinned to keep slot indices
    meaningful.
    """
    from spacr.qt.screens.annotate import AnnotateScreen
    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.grid_rows = TILE_ROWS
    screen._settings.grid_cols = TILE_COLS
    screen._settings.image_size = (32, 32)
    screen._compute_grid_dims = lambda: None
    screen._rebuild_grid()
    screen._open_source(str(synth_annotate_source))
    qtbot.waitUntil(lambda: len(screen._page_paths) == 8, timeout=5000)
    qtbot.waitUntil(lambda: screen._raw_thumb_images[0] is not None,
                    timeout=5000)
    yield screen
    if screen._worker is not None:
        screen._worker.stop(wait=True)


def _resting(screen) -> str:
    from spacr.qt.screens.annotate import resting_border_color
    return resting_border_color()


# -- resting state ----------------------------------------------------------

def test_fresh_tile_reports_the_resting_gray_border(tile_screen):
    """Every unlabelled crop wears the thin gray ring and nothing else."""
    screen = tile_screen
    # Slot 0 is where the keyboard parks on load, so read a different one.
    tile = screen._thumbs[1]
    assert tile.border_color() == _resting(screen)
    assert tile.ring_color() is None
    assert tile.outline_color() == _resting(screen)
    assert tile.is_current() is False
    assert tile.is_occupied() is True


def test_resting_gray_comes_from_the_theme_not_a_literal(monkeypatch):
    """The gray is the shared palette's border colour on BOTH themes —
    a hard-coded one would be invisible on one of them."""
    from spacr.qt import preferences
    from spacr.qt.screens import annotate as annotate_mod
    from spacr.qt.theme import palette_for

    for theme in ("dark", "light"):
        monkeypatch.setattr(preferences, "resolve_effective_theme",
                            lambda t=theme: t)
        P = palette_for(theme)
        assert annotate_mod.resting_border_color() == P["border"]
        assert annotate_mod.current_ring_color() == P["fg"]
        # ...and it is not the colour of the canvas it sits on.
        assert annotate_mod.resting_border_color() != P["surface_alt"]
        assert annotate_mod.current_ring_color() != P["surface_alt"]


def test_current_ring_is_white_on_the_default_dark_theme(monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens import annotate as annotate_mod
    monkeypatch.setattr(preferences, "resolve_effective_theme",
                        lambda: "dark")
    assert annotate_mod.current_ring_color().lower() == "#ffffff"


def test_current_ring_can_never_collide_with_a_class_colour(monkeypatch):
    """`label_to_hex` must not be able to produce the ring colour on
    either theme, or "current" and "class N" would look the same."""
    from spacr.qt import preferences
    from spacr.qt.screens import annotate as annotate_mod
    classes = {(engine.label_to_hex(v) or "").lower() for v in range(1, 60)}
    for theme in ("dark", "light"):
        monkeypatch.setattr(preferences, "resolve_effective_theme",
                            lambda t=theme: t)
        assert annotate_mod.current_ring_color().lower() not in classes


def test_tile_is_a_square_with_room_for_both_rings(tile_screen):
    from spacr.qt.screens.annotate import (BORDER_WIDTH, HOVER_RING_WIDTH,
                                           IMAGE_RADIUS, TILE_INSET)
    screen = tile_screen
    w, h = screen._settings.image_size
    assert TILE_INSET == BORDER_WIDTH + HOVER_RING_WIDTH
    assert IMAGE_RADIUS > 0            # the image itself has round corners
    size = screen._thumbs[0].size()
    assert (size.width(), size.height()) == (w + 2 * TILE_INSET,
                                             h + 2 * TILE_INSET)
    assert size.width() == size.height()


def test_tile_palette_falls_back_to_dark_when_prefs_are_unreadable(monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens import annotate as annotate_mod
    from spacr.qt.theme import palette_for

    def _boom():
        raise RuntimeError("no QSettings here")

    monkeypatch.setattr(preferences, "resolve_effective_theme", _boom)
    assert annotate_mod.tile_palette() == palette_for("dark")
    assert annotate_mod.resting_border_color() == palette_for("dark")["border"]


def test_both_bands_are_painted_on_a_hovered_classified_tile(qtbot,
                                                             qt_theme_applied):
    """The one claim only pixels can settle: hovering a classified crop
    really does draw BOTH rings, in their own bands, at once."""
    from PIL.ImageQt import ImageQt
    from PySide6.QtGui import QImage, QPixmap
    from spacr.qt.screens.annotate import (BORDER_WIDTH, HOVER_RING_WIDTH,
                                           _Thumbnail)

    tile = _Thumbnail(0, border_color="#00ff00", ring_color="#ffffff")
    qtbot.addWidget(tile)
    tile.setFixedSize(48, 48)
    tile.set_occupied(True)
    flat = Image.new("RGB", (38, 38), (255, 0, 0))
    tile.setPixmap(QPixmap.fromImage(QImage(ImageQt(flat))))
    tile.set_current(True)
    shot = tile.grab().toImage()
    mid = 24

    def _rgb(x):
        c = shot.pixelColor(x, mid)
        return (c.red(), c.green(), c.blue())

    # Outer band = the current ring, inner band = the class colour, then
    # the crop. Neither has painted over the other.
    assert _rgb(HOVER_RING_WIDTH // 2) == (255, 255, 255)
    assert _rgb(HOVER_RING_WIDTH + BORDER_WIDTH // 2) == (0, 255, 0)
    assert _rgb(HOVER_RING_WIDTH + BORDER_WIDTH + 2) == (255, 0, 0)


def test_real_mouse_clicks_reach_the_annotation_write_path(qtbot, tile_screen):
    """Left = class 1, right = class 2, anything else falls through."""
    from PySide6.QtCore import Qt as _Qt
    screen = tile_screen
    qtbot.mouseClick(screen._thumbs[0], _Qt.LeftButton)
    assert screen._page_paths[0][1] == 1
    qtbot.mouseClick(screen._thumbs[1], _Qt.RightButton)
    assert screen._page_paths[1][1] == 2
    qtbot.mouseClick(screen._thumbs[2], _Qt.MiddleButton)
    assert screen._page_paths[2][1] is None


def test_mouse_click_logs_path_and_annotation_as_one_line(qtbot, tile_screen):
    """Each click leaves one searchable path/annotation console record."""
    from PySide6.QtCore import Qt as _Qt

    screen = tile_screen
    path = str(screen._page_paths[0][0]).replace("\r", r"\r").replace(
        "\n", r"\n"
    )

    qtbot.mouseClick(screen._thumbs[0], _Qt.LeftButton)
    output = screen._console._current_stdout.toPlainText()
    assert output.splitlines()[-1] == f"path={path} | annotation=1"

    # Clicking the active class again clears it and reports the resulting
    # annotation, rather than the class that was requested.
    qtbot.mouseClick(screen._thumbs[0], _Qt.LeftButton)
    output = screen._console._current_stdout.toPlainText()
    assert output.splitlines()[-1] == f"path={path} | annotation=None"


def test_corner_is_actually_rounded_not_a_frame_over_a_square(qtbot,
                                                              qt_theme_applied):
    """The crop is clipped to the rounded rect, so the corner of the tile
    is background — not the image showing through under a rounded frame."""
    from PIL.ImageQt import ImageQt
    from PySide6.QtGui import QImage, QPixmap
    from spacr.qt.screens.annotate import _Thumbnail

    tile = _Thumbnail(0)
    qtbot.addWidget(tile)
    tile.setFixedSize(48, 48)
    tile.set_occupied(True)
    flat = Image.new("RGB", (38, 38), (255, 0, 0))
    tile.setPixmap(QPixmap.fromImage(QImage(ImageQt(flat))))
    shot = tile.grab().toImage()
    centre = shot.pixelColor(24, 24)
    corner = shot.pixelColor(0, 0)
    assert (centre.red(), centre.green(), centre.blue()) == (255, 0, 0)
    assert (corner.red(), corner.green(), corner.blue()) != (255, 0, 0)


# -- class colours ----------------------------------------------------------

def test_assigning_a_class_recolours_the_border_to_the_class_colour(tile_screen):
    """The border uses the app's one class→colour map, not a private one."""
    screen = tile_screen
    screen._on_thumb_left(1)                     # class 1
    assert screen._thumbs[1].border_color() == engine.label_to_hex(1)
    screen._on_thumb_right(2)                    # class 2
    assert screen._thumbs[2].border_color() == engine.label_to_hex(2)
    # The same function the Class-counts dialog prints.
    assert screen._border_color_for(1) == engine.label_to_hex(1)


def test_keyboard_classes_use_the_same_colours(tile_screen):
    screen = tile_screen
    screen._set_focus_slot(4)
    screen.handle_key("7")
    assert screen._thumbs[4].border_color() == engine.label_to_hex(7)


def test_clearing_the_class_returns_the_border_to_gray(tile_screen):
    screen = tile_screen
    screen._on_thumb_left(1)
    assert screen._thumbs[1].border_color() == engine.label_to_hex(1)
    screen._on_thumb_left(1)                     # same class again = clear
    assert screen._thumbs[1].border_color() == _resting(screen)
    assert screen._thumbs[1].ring_color() is None


# -- hover ------------------------------------------------------------------

def test_hover_enter_gives_the_tile_the_white_ring(tile_screen):
    from spacr.qt.screens.annotate import current_ring_color
    screen = tile_screen
    screen._on_thumb_hover(3, True)
    assert screen.hover_slot == 3
    assert screen._thumbs[3].is_current()
    assert screen._thumbs[3].ring_color() == current_ring_color()
    assert screen._thumbs[3].outline_color() == current_ring_color()


def test_hover_leave_restores_the_previous_state_not_gray(tile_screen):
    """Leaving must not repaint a classified crop as unclassified."""
    screen = tile_screen
    screen._on_thumb_left(3)                     # class 1
    class_color = engine.label_to_hex(1)
    screen._on_thumb_hover(3, True)
    assert screen._thumbs[3].border_color() == class_color
    screen._on_thumb_hover(3, False)             # cursor left the tile
    assert screen.hover_slot is None
    assert screen._thumbs[3].border_color() == class_color


def test_hovering_a_classified_crop_shows_both_states(tile_screen):
    """The regression that would otherwise ship: hover must not hide the
    class colour, and the class colour must not hide the hover."""
    from spacr.qt.screens.annotate import current_ring_color
    screen = tile_screen
    screen._on_thumb_left(5)
    class_color = engine.label_to_hex(1)
    screen._on_thumb_hover(5, True)
    tile = screen._thumbs[5]
    # Inner band = class, outer band = current. Both readable at once.
    assert tile.border_color() == class_color
    assert tile.ring_color() == current_ring_color()
    assert tile.border_color() != tile.ring_color()
    # Move the current tile away: the class colour stays put.
    screen._on_thumb_hover(5, False)
    screen._on_thumb_hover(6, True)
    assert tile.is_current() is False
    assert tile.ring_color() is None
    assert tile.border_color() == class_color, \
        "leaving a classified crop reset it to gray"


def test_classifying_the_hovered_crop_keeps_the_ring(tile_screen):
    """Labelling the crop under the cursor must not drop the hover."""
    from spacr.qt.screens.annotate import current_ring_color
    screen = tile_screen
    screen._on_thumb_hover(2, True)
    screen._on_thumb_left(2)
    assert screen._thumbs[2].border_color() == engine.label_to_hex(1)
    assert screen._thumbs[2].ring_color() == current_ring_color()


def test_only_one_tile_is_current_at_a_time(tile_screen):
    screen = tile_screen
    screen._on_thumb_hover(1, True)
    assert [i for i, t in enumerate(screen._thumbs) if t.is_current()] == [1]
    # Moving between tiles: Leave then Enter, as Qt delivers them.
    screen._on_thumb_hover(1, False)
    screen._on_thumb_hover(4, True)
    assert [i for i, t in enumerate(screen._thumbs) if t.is_current()] == [4]
    # And even a stray Enter without the matching Leave cannot double up.
    screen._on_thumb_hover(7, True)
    assert [i for i, t in enumerate(screen._thumbs) if t.is_current()] == [7]


def test_a_stale_leave_from_another_tile_is_ignored(tile_screen):
    """Qt can deliver Leave for the old tile AFTER Enter for the new one."""
    screen = tile_screen
    screen._on_thumb_hover(4, True)
    screen._on_thumb_hover(1, True)              # cursor is now on 1
    screen._on_thumb_hover(4, False)             # late Leave for 4
    assert screen.hover_slot == 1
    assert screen._thumbs[1].is_current()


def test_hover_ignores_cells_that_hold_no_crop(tile_screen):
    """9 cells, 8 crops — the empty cell is not something to be "on"."""
    screen = tile_screen
    assert screen._thumbs[8].is_occupied() is False
    screen._on_thumb_hover(8, True)
    assert screen.hover_slot is None
    assert screen._thumbs[8].is_current() is False


def test_mouse_leaving_the_grid_clears_the_hover(tile_screen):
    from PySide6.QtCore import QEvent
    screen = tile_screen
    screen._on_thumb_hover(3, True)
    assert screen.hover_slot == 3
    # The cursor quits the grid without a per-tile Leave.
    screen.eventFilter(screen._grid_scroll.viewport(),
                       QEvent(QEvent.Type.Leave))
    assert screen.hover_slot is None
    # The ring stays on the crop the keyboard still targets — it does not
    # vanish, leaving the user with no idea what the next key will hit.
    assert screen.focus_slot == 3
    assert screen._thumbs[3].is_current()


def test_real_enter_and_leave_events_drive_the_hover(tile_screen):
    """Proves the widget signals are actually wired, not just the slots."""
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QEnterEvent
    from PySide6.QtWidgets import QApplication
    screen = tile_screen
    tile = screen._thumbs[2]
    pos = QPointF(5, 5)
    QApplication.sendEvent(tile, QEnterEvent(pos, pos, pos))
    assert screen.hover_slot == 2
    assert tile.is_current()
    QApplication.sendEvent(tile, QEvent(QEvent.Type.Leave))
    assert screen.hover_slot is None


# -- hover vs. keyboard -----------------------------------------------------

def test_mouse_hover_and_keyboard_move_the_same_highlight(tile_screen):
    """One current tile, two ways to move it — they must not diverge."""
    screen = tile_screen
    screen._on_thumb_hover(4, True)
    assert screen.focus_slot == 4 == screen.current_slot
    # A keyboard move takes the ring off the hovered tile: the cursor has
    # not moved, but it is no longer on the crop the next key will hit.
    screen.handle_key("left")
    assert screen.focus_slot == 3
    assert screen.hover_slot is None
    assert [i for i, t in enumerate(screen._thumbs) if t.is_current()] == [3]


def test_hover_then_keyboard_assign_labels_the_hovered_crop(tile_screen):
    """The white ring is a promise about what the next keystroke hits."""
    screen = tile_screen
    screen._on_thumb_hover(6, True)
    screen.handle_key("2")
    assert screen._page_paths[6][1] == 2
    assert screen._thumbs[6].border_color() == engine.label_to_hex(2)


def test_hover_slot_never_disagrees_with_the_focus_slot(tile_screen):
    """The invariant that makes "two current tiles" impossible."""
    screen = tile_screen
    for action in (lambda: screen._on_thumb_hover(2, True),
                   lambda: screen.handle_key("right"),
                   lambda: screen._on_thumb_hover(7, True),
                   lambda: screen.handle_key("up"),
                   lambda: screen._on_thumb_hover(7, False),
                   lambda: screen.handle_key("down")):
        action()
        assert screen.hover_slot is None \
            or screen.hover_slot == screen.focus_slot


# -- paging -----------------------------------------------------------------

def test_paging_does_not_leave_a_stale_hover(qtbot, tile_screen):
    """New crops under the same widgets — the recorded hover is only kept
    if the cursor is genuinely still inside that widget."""
    screen = tile_screen
    screen._on_thumb_hover(3, True)
    assert screen.hover_slot == 3
    screen._offset = 4
    screen._load_page()
    qtbot.waitUntil(lambda: screen._raw_thumb_images[0] is not None,
                    timeout=5000)
    assert screen.hover_slot is None


def test_reloading_the_grid_does_not_leave_a_stale_hover(qtbot, tile_screen):
    """Same offset, fresh load (e.g. a settings change) — same rule."""
    screen = tile_screen
    screen._on_thumb_hover(5, True)
    screen._load_page()
    qtbot.waitUntil(lambda: screen._raw_thumb_images[0] is not None,
                    timeout=5000)
    assert screen.hover_slot is None


def test_paging_resets_borders_for_the_new_crops(qtbot, tile_screen):
    screen = tile_screen
    screen._on_thumb_left(0)
    assert screen._thumbs[0].border_color() == engine.label_to_hex(1)
    screen._offset = 4
    screen._load_page()
    qtbot.waitUntil(lambda: screen._raw_thumb_images[0] is not None,
                    timeout=5000)
    # Slot 0 now holds a different, unlabelled crop — it must not still be
    # wearing the previous page's class colour.
    assert screen._page_paths[0][1] is None
    assert screen._thumbs[0].border_color() == _resting(screen)
    # ...and the cells past the end of the short page hold nothing.
    assert screen._thumbs[8].is_occupied() is False


def test_rebuilding_the_grid_drops_the_hover(tile_screen):
    """The hovered widget is destroyed by a rebuild; the index must go too."""
    screen = tile_screen
    screen._on_thumb_hover(2, True)
    screen._rebuild_grid()
    assert screen.hover_slot is None
    assert all(not t.is_current() or i == screen.focus_slot
               for i, t in enumerate(screen._thumbs))


# -- cost of a mouse move ---------------------------------------------------

def test_hover_never_rebuilds_a_pixmap(tile_screen, monkeypatch):
    """Changing a border must not cost an image conversion."""
    screen = tile_screen
    before = [screen._thumb_pixmaps[i] for i in range(len(screen._thumbs))]

    def _boom(*_a, **_k):
        raise AssertionError("hover rebuilt a pixmap")

    monkeypatch.setattr(screen, "_image_to_pixmap", _boom)
    screen._on_thumb_left(2)                     # class change too
    for slot in range(8):
        screen._on_thumb_hover(slot, True)
        screen._on_thumb_hover(slot, False)
    after = [screen._thumb_pixmaps[i] for i in range(len(screen._thumbs))]
    assert all(a is b for a, b in zip(before, after)), \
        "the pixmaps were replaced, not just repainted"


def test_moving_between_tiles_repaints_only_those_two(tile_screen,
                                                      monkeypatch):
    """The grid holds hundreds of crops; a mouse move must touch two."""
    from spacr.qt.screens.annotate import _Thumbnail
    painted = []
    monkeypatch.setattr(_Thumbnail, "update",
                        lambda self, *a, **k: painted.append(self.slot))
    screen = tile_screen
    screen._on_thumb_hover(2, True)
    painted.clear()
    screen._on_thumb_hover(2, False)             # Leave costs nothing:
    assert painted == []                          # the ring has not moved
    screen._on_thumb_hover(3, True)
    assert set(painted) == {2, 3}, painted


def test_re_entering_the_same_tile_costs_nothing(tile_screen, monkeypatch):
    """Mouse jitter inside one tile must not repaint anything."""
    from spacr.qt.screens.annotate import _Thumbnail
    painted = []
    monkeypatch.setattr(_Thumbnail, "update",
                        lambda self, *a, **k: painted.append(self.slot))
    screen = tile_screen
    screen._on_thumb_hover(4, True)
    painted.clear()
    for _ in range(20):
        screen._on_thumb_hover(4, True)
    assert painted == []


def test_cover_rect_fills_the_box_at_the_pixmaps_aspect(qtbot,
                                                        qt_theme_applied):
    """Crops are drawn to fill the rounded box, never letterboxed."""
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QPixmap
    from spacr.qt.screens.annotate import _cover_rect

    box = QRectF(5, 5, 40, 40)
    wide = QPixmap(80, 40)
    out = _cover_rect(wide, box)
    assert out.width() >= box.width() and out.height() >= box.height()
    assert round(out.width() / out.height(), 3) == 2.0     # aspect kept
    assert round(out.center().x(), 3) == round(box.center().x(), 3)
    # A null pixmap has no aspect to preserve — fall back to the box.
    assert _cover_rect(QPixmap(), box) == box


def test_a_tile_too_small_for_its_chrome_still_paints(qtbot, qt_theme_applied):
    """Defensive geometry: no crash when the cell is smaller than the rings."""
    from spacr.qt.screens.annotate import _Thumbnail
    tile = _Thumbnail(0)
    qtbot.addWidget(tile)
    tile.setFixedSize(4, 4)
    tile.grab()                       # unoccupied → paints nothing
    tile.set_occupied(True)
    shot = tile.grab()                # occupied but no room for the image
    assert not shot.isNull()


def test_moving_the_ring_without_a_scroll_area_is_harmless(tile_screen):
    """`ensureWidgetVisible` is best-effort — losing it must not break focus."""
    screen = tile_screen
    screen._grid_scroll = None
    screen._set_focus_slot(5)
    assert screen.focus_slot == 5
    assert screen._thumbs[5].is_current()


def test_event_filter_ignores_objects_that_are_not_events(tile_screen):
    class _NotAnEvent:
        def type(self):
            raise RuntimeError("not a Qt event")

    assert tile_screen.eventFilter(tile_screen, _NotAnEvent()) is False


def test_out_of_range_slots_are_ignored(tile_screen):
    """A page worker can hand back more rows than the grid has cells."""
    screen = tile_screen
    before = list(screen._thumb_pixmaps)
    screen._set_slot_image(999, None)
    screen._repaint_slot(999)
    screen._set_slot_image(-1, None)
    screen._repaint_slot(-1)
    assert screen._thumb_pixmaps == before


def test_focus_set_before_the_grid_exists_is_remembered(qtbot,
                                                        qt_theme_applied):
    """No widgets to repaint yet — the index is still recorded."""
    from spacr.qt.screens.annotate import AnnotateScreen
    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._thumbs.clear()
    screen._set_focus_slot(3)
    assert screen.focus_slot == 3
    assert screen.current_slot == 3


def test_setters_report_whether_a_repaint_was_needed(qtbot, qt_theme_applied):
    """The no-op guard that makes redundant `_repaint_slot` calls free."""
    from spacr.qt.screens.annotate import _Thumbnail, resting_border_color
    tile = _Thumbnail(0)
    qtbot.addWidget(tile)
    assert tile.set_border_color("#123456") is True
    assert tile.set_border_color("#123456") is False
    assert tile.set_border_color(None) is True          # back to resting
    assert tile.border_color() == resting_border_color()
    assert tile.set_current(True) is True
    assert tile.set_current(True) is False
    assert tile.set_occupied(True) is True
    assert tile.set_occupied(True) is False
