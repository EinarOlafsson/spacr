"""End-to-end regression test for the annotate screen.

Flow: build a real on-disk experiment (measurements.db + PNG crops)
→ open it in AnnotateScreen → simulate left/right clicks on
thumbnails → wait for the SaveWorker to persist → close the source
→ re-open a fresh AnnotateScreen against the same DB → assert the
annotations we made are still there.

This exercises the AnnotateScreen ↔ SaveWorker ↔ SQLite round-trip
in one shot, which is the workflow that matters to end users.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Synthetic experiment
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_annotate_experiment(tmp_path: Path) -> Path:
    """Create a folder tree that looks like a spacr `preprocess_generate_masks`
    output — with a `measurements/measurements.db` (png_list table) and
    a few small PNG crops referenced by the DB."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data").mkdir(parents=True)
    rng = np.random.default_rng(0)
    png_paths: List[str] = []
    for i in range(6):
        arr = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
        p = src / "data" / f"crop_{i:02d}.png"
        Image.fromarray(arr).save(p)
        png_paths.append(str(p))
    db = src / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany(
            'INSERT INTO "png_list" (png_path) VALUES (?)',
            [(p,) for p in png_paths],
        )
    return src


def _wait_worker_flush(screen, qtbot, timeout_ms: int = 4000) -> None:
    """Poll until the SaveWorker's pending batch is committed."""
    ticks = timeout_ms // 20
    for _ in range(ticks):
        qtbot.wait(20)
        w = screen._worker
        if w is not None and not w.busy and w.pending_batches == 0:
            return


def _read_annotations(db_path: Path, column: str) -> dict:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f'SELECT png_path, "{column}" FROM "png_list"'
        ).fetchall()
    return {p: a for p, a in rows}


# ---------------------------------------------------------------------------
# The E2E test
# ---------------------------------------------------------------------------

def test_annotate_click_save_reload_persists(
    qtbot, qt_theme_applied, synth_annotate_experiment: Path,
):
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    # Small grid so we know exactly which thumbnails map to which
    # database rows for the first page.
    screen._settings.grid_rows = 2
    screen._settings.grid_cols = 3
    screen._rebuild_grid()

    # 1. Open the synthetic experiment (page load is deferred + threaded now)
    screen._open_source(str(synth_annotate_experiment))
    qtbot.waitUntil(lambda: len(screen._page_paths) >= 2, timeout=5000)
    assert screen._total == 6
    # Grid rows/cols get recomputed from the (offscreen) viewport size,
    # so we only assert the page has SOME thumbnails and two are
    # available to click. All 6 DB rows exist regardless of pagination.
    assert len(screen._page_paths) >= 2

    # 2. Left-click slot 0 (class 1), right-click slot 1 (class 2)
    screen._on_thumb_left(0)
    screen._on_thumb_right(1)
    assert screen._page_paths[0][1] == 1
    assert screen._page_paths[1][1] == 2

    # 3. Trigger the pending flush by navigating (Next flushes)
    screen._flush_pending()
    _wait_worker_flush(screen, qtbot)

    # 4. Close this session (stops the worker, closes any connections)
    if screen._worker is not None:
        screen._worker.stop(wait=True)

    # 5. Read the DB back directly and check the two annotations landed
    db_path = synth_annotate_experiment / "measurements" / "measurements.db"
    stored = _read_annotations(db_path, "annotate")
    p0, p1 = screen._page_paths[0][0], screen._page_paths[1][0]
    assert stored[p0] == 1, "class-1 annotation didn't persist"
    assert stored[p1] == 2, "class-2 annotation didn't persist"
    # Every OTHER row (page or not) stays None
    for path, val in stored.items():
        if path not in (p0, p1):
            assert val is None, f"unexpected annotation on {path}"

    # 6. Open a FRESH AnnotateScreen and confirm the reload reflects
    #    the previous session's annotations.
    fresh = AnnotateScreen()
    qtbot.addWidget(fresh)
    fresh._settings.grid_rows = 2
    fresh._settings.grid_cols = 3
    fresh._rebuild_grid()
    fresh._open_source(str(synth_annotate_experiment))
    qtbot.waitUntil(lambda: len(fresh._page_paths) >= 2, timeout=5000)
    reload_map = {p: v for p, v in fresh._page_paths}
    # Both annotated paths should appear on page 1 with our values.
    # If pagination changed, at minimum the DB reflects them.
    if p0 in reload_map:
        assert reload_map[p0] == 1
    if p1 in reload_map:
        assert reload_map[p1] == 2
    if fresh._worker is not None:
        fresh._worker.stop(wait=True)


def test_annotate_toggle_off_second_click_clears(
    qtbot, qt_theme_applied, synth_annotate_experiment: Path,
):
    """Re-clicking the same class on the same slot should clear it
    (three-state toggle: unlabelled → 1 → unlabelled)."""
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.grid_rows = 2
    screen._settings.grid_cols = 3
    screen._rebuild_grid()
    screen._open_source(str(synth_annotate_experiment))
    qtbot.waitUntil(lambda: len(screen._page_paths) >= 1, timeout=5000)

    screen._on_thumb_left(0)
    assert screen._page_paths[0][1] == 1
    screen._on_thumb_left(0)      # same class again → cleared
    assert screen._page_paths[0][1] is None

    screen._flush_pending()
    _wait_worker_flush(screen, qtbot)
    if screen._worker is not None:
        screen._worker.stop(wait=True)
    stored = _read_annotations(
        synth_annotate_experiment / "measurements" / "measurements.db",
        "annotate",
    )
    assert stored[screen._page_paths[0][0]] is None
