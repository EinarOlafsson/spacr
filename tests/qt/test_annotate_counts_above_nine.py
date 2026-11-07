"""A count column in Annotate stores counts of ten and more as entered.

With ``annotation_counts`` on, + and - step the focused crop's count and #
types any number. The tests read the database after the save worker drains,
so the stored values are the outcome, and check that 11 is not mistaken for a
suggested 1 on the way.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt.screens import annotate as annotate_mod             # noqa: E402

N_CROPS = 4


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A measurements.db with four tiny crops."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data").mkdir(parents=True)
    rng = np.random.default_rng(5)
    paths = []
    for i in range(N_CROPS):
        p = src / "data" / f"crop_{i}.png"
        Image.fromarray(rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)).save(p)
        paths.append((str(p),))
    con = sqlite3.connect(src / "measurements" / "measurements.db", timeout=30)
    try:
        con.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        con.executemany('INSERT INTO "png_list" (png_path) VALUES (?)', paths)
        con.commit()
    finally:
        con.close()
    return src


def _open(qtbot, src: Path, counts: bool = True):
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.grid_rows = 1
    screen._settings.grid_cols = N_CROPS
    screen._settings.image_size = (24, 24)
    screen._settings.annotation_column = "pv_count"
    screen._settings.annotation_counts = counts
    screen._compute_grid_dims = lambda: None
    screen._rebuild_grid()
    screen._open_source(str(src))
    qtbot.waitUntil(lambda: len(screen._page_paths) == N_CROPS, timeout=5000)
    return screen


def _stored(screen, qtbot, src: Path) -> list:
    screen._flush_pending()
    worker = screen._worker
    qtbot.waitUntil(lambda: not worker.busy and worker.pending_batches == 0,
                    timeout=5000)
    con = sqlite3.connect(src / "measurements" / "measurements.db", timeout=30)
    try:
        rows = dict(con.execute('SELECT png_path, pv_count FROM png_list'))
    finally:
        con.close()
    return [rows[str(src / "data" / f"crop_{i}.png")] for i in range(N_CROPS)]


def test_typed_and_stepped_counts_above_nine_are_stored(qtbot, source):
    screen = _open(qtbot, source)
    screen._ask_count = lambda current: 14
    screen._set_focus_slot(0)

    assert screen.handle_key("#") is True
    assert screen.focus_slot == 1
    for _ in range(11):
        screen.handle_key("+")
    screen.handle_key("-")
    screen._set_focus_slot(2)
    screen.handle_key("9")

    assert _stored(screen, qtbot, source) == [14, 10, 9, None]
    assert screen._thumbs[0].count_text() == "14"
    assert not screen._is_suggested_slot(0)
    assert screen._displayed_class(0) == 14
    screen._worker.stop(wait=True)


def test_stepping_down_to_zero_clears_and_undo_puts_it_back(qtbot, source):
    screen = _open(qtbot, source)
    screen._set_focus_slot(2)
    screen.handle_key("+")
    screen.handle_key("-")

    assert screen._current_value(2) is None
    assert screen._thumbs[2].count_text() == ""
    screen.handle_key("u")
    assert screen._current_value(2) == 1
    screen._worker.stop(wait=True)


def test_a_dismissed_or_zero_typed_count(qtbot, source):
    screen = _open(qtbot, source)
    screen._set_focus_slot(0)
    screen._ask_count = lambda current: None
    screen.handle_key("#")
    assert screen._current_value(0) is None and screen.focus_slot == 0

    screen.handle_key("+")
    screen._ask_count = lambda current: 0
    screen.handle_key("#")
    assert screen._current_value(0) is None and screen.focus_slot == 0
    screen._worker.stop(wait=True)


def test_off_a_count_column_the_keys_are_left_alone(qtbot, source):
    screen = _open(qtbot, source, counts=False)
    screen._set_focus_slot(0)

    assert screen.handle_key("+") is False
    assert screen.handle_key("#") is False
    assert screen._current_value(0) is None
    screen._set_annotation(0, 11)
    assert screen._is_suggested_slot(0)
    assert screen._thumbs[0].count_text() == ""
    screen._worker.stop(wait=True)


def test_suggest_is_refused_on_a_count_column(qtbot, source):
    screen = _open(qtbot, source)

    screen._start_suggest(this_page=True)

    assert screen._suggest_worker is None
    assert "count column" in screen._status_label.text()
    screen._worker.stop(wait=True)


def test_the_settings_dialog_carries_the_switch(qtbot):
    from spacr.qt.annotate_engine import AnnotateSettings

    dialog = annotate_mod._SettingsDialog(AnnotateSettings())
    qtbot.addWidget(dialog)
    assert dialog._counts.isChecked() is False
    dialog._counts.setChecked(True)

    assert dialog.collect().annotation_counts is True


def test_the_setting_is_registered_with_its_default():
    from spacr.settings import expected_types, set_annotate_default_settings, tooltips

    assert set_annotate_default_settings({})["annotation_counts"] is False
    assert expected_types["annotation_counts"] is bool
    assert tooltips["annotation_counts"].endswith("Default False.")
