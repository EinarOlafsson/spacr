"""Per-field timing log (``curate_timing.csv``) for timed curation sessions."""
from __future__ import annotations

import csv

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr import cli_make_masks
from spacr.curation_queue import build_queue
from spacr import curation_queue as ct
from spacr.qt.screens.make_masks import MakeMasksScreen


def _lab(*boxes):
    out = np.zeros((40, 40), np.uint16)
    for i, (r0, r1, c0, c1) in enumerate(boxes, 1):
        out[r0:r1, c0:c1] = i
    return out


def test_identical_masks_have_no_changes():
    m = _lab((2, 10, 2, 10), (20, 30, 20, 30))
    assert ct._timing_object_changes(m, m.copy()) == (2, 2, 0, 0, 0)


def test_added_removed_modified():
    before = _lab((2, 10, 2, 10), (20, 30, 20, 30), (32, 38, 2, 8))
    after = np.zeros_like(before)
    after[2:10, 2:10] = 1            # unchanged
    after[20:30, 20:26] = 2          # shrunk: IoU 0.6 -> modified
    after[0:2, 30:40] = 3            # new object; third removed
    assert ct._timing_object_changes(before, after) == (3, 3, 1, 1, 1)


def test_relabelled_ids_and_empty_masks():
    a = _lab((2, 10, 2, 10))
    assert ct._timing_object_changes(a, a * 7) == (1, 1, 0, 0, 0)
    z = np.zeros_like(a)
    assert ct._timing_object_changes(z, a) == (0, 1, 1, 0, 0)
    assert ct._timing_object_changes(a, z) == (1, 0, 0, 1, 0)
    assert ct._timing_object_changes(a, np.zeros((3, 3))) == (1, 0, 0, 1, 0)


def test_idle_gaps_are_not_active_time():
    t = ct._FieldTimer("/x/f.tif", _lab((2, 10, 2, 10)))
    base = t._last
    t.touch(base + 10)
    t.touch(base + 200)          # idle gap, dropped
    t.touch(base + 205)
    assert t.active_seconds(base + 205) == pytest.approx(15)


def test_row_is_appended_with_one_header(tmp_path, monkeypatch):
    monkeypatch.setenv(ct._TIMING_SESSION_TAG_ENV, "armA")
    pre = _lab((2, 10, 2, 10))
    t = ct._FieldTimer(str(tmp_path / "f_01.tif"), pre, n_log_open=1)
    ct._log_field_visit(tmp_path, t, "save", _lab((2, 10, 2, 10), (20, 30, 20, 30)), 4)
    ct._log_field_visit(tmp_path, t, "skip", pre, 1)
    with open(tmp_path / ct._TIMING_FILE, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["decision"] for r in rows] == ["save", "skip"]
    assert tuple(rows[0]) == ct._TIMING_COLUMNS
    assert rows[0]["stem"] == "f_01" and rows[0]["session_tag"] == "armA"
    assert rows[0]["n_edits"] == "3" and rows[0]["n_added"] == "1"
    assert rows[0]["n_objects_premask"] == "1" and rows[0]["n_objects_final"] == "2"
    assert rows[1]["n_edits"] == "0"


def test_unwritable_folder_never_raises(tmp_path):
    t = ct._FieldTimer("f.tif", _lab((2, 10, 2, 10)))
    assert ct._log_field_visit(tmp_path / "missing" / "dir", t, "save", None, 0) is None
    assert ct._log_field_visit(None, t, "save", None, 0) is None


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    cli_make_masks.hand_over(None)
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    cli_make_masks.hand_over(None)


def test_screen_skip_and_save_write_rows(qtbot, screen, tmp_path):
    folder = tmp_path / "q"
    (folder / "masks").mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i in range(3):
        imageio.imwrite(folder / f"f_{i:02d}.tif",
                        rng.integers(0, 4000, (40, 40), dtype=np.uint16))
    imageio.imwrite(folder / "masks" / "f_00.tif", _lab((2, 10, 2, 10)))
    queue = build_queue(folder, order="name")
    assert screen.open_queue(queue) is True
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)
    assert screen._field_timer is not None
    screen._on_skip()
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)
    screen._save_mask()
    with open(folder / ct._TIMING_FILE, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["decision"] for r in rows] == ["skip", "save"]
    assert rows[0]["stem"] == "f_00" and rows[0]["n_objects_premask"] == "1"
    assert rows[1]["stem"] == "f_01"
