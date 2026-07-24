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


def test_annotate_screen_open_source_loads_page(qtbot, qt_theme_applied,
                                                  synth_annotate_source: Path):
    from spacr.qt.screens.annotate import AnnotateScreen
    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.grid_rows = 2
    screen._settings.grid_cols = 2
    screen._rebuild_grid()
    screen._open_source(str(synth_annotate_source))
    # Total rows detected
    assert screen._total == 8
    # First-page thumbnails populated (4 thumbs, 4 rows)
    assert len(screen._page_paths) == 4
    for i, (path, _) in enumerate(screen._page_paths):
        assert os.path.isfile(path)
        # Thumb pixmap should have been set
        assert screen._thumbs[i].pixmap() is not None
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
    import os
    from spacr.qt.screens.annotate import _reanchor_png_path
    real = tmp_path / "x.png"; real.write_bytes(b"x")
    assert _reanchor_png_path(str(real), "") == str(real)
