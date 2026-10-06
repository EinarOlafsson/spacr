"""Annotate's Like this button: alpha-gated, and it pins the grid to matches.

``AnnotateFindSimilar`` is registered in ``spacr.settings.ALPHA_FEATURES``,
so it is hidden until Preferences -> "Show alpha features" is on. Hiding is
display only: the search behind it still runs while the button is hidden.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("PySide6")


@pytest.fixture
def alpha(monkeypatch):
    from spacr.qt import preferences

    state = {"on": False}
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: state["on"])
    return state


@pytest.fixture
def plate(tmp_path) -> Path:
    """Twenty measured crops in two kinds, told apart by size and stain."""
    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    (src / "data").mkdir()
    rng = np.random.default_rng(2)
    cells, crops = [], []
    for obj in range(1, 21):
        kind = obj % 2
        prcf = "plate1_r1_c1_f1"
        path = src / "data" / f"{prcf}_o{obj}.png"
        Image.fromarray(rng.integers(0, 255, (12, 12, 3),
                                     dtype=np.uint8)).save(path)
        cells.append({"plateID": "plate1", "rowID": "r1", "columnID": "c1",
                      "fieldID": "f1", "object_label": obj, "prcf": prcf,
                      "prc": "plate1_r1_c1",
                      "cell_area": 500.0 + 400.0 * kind + rng.normal(0, 5),
                      "cell_perimeter": 80.0 + 30.0 * kind + rng.normal(0, 2),
                      "cell_channel_1_mean_intensity":
                          0.2 + 0.3 * kind + rng.normal(0, 0.02),
                      "cell_noise": rng.normal(0, 0.1)})
        crops.append((str(path), f"{prcf}_o{obj}", kind))
    with sqlite3.connect(src / "measurements" / "measurements.db") as con:
        pd.DataFrame(cells).to_sql("cell", con, index=False)
        con.execute('CREATE TABLE png_list (png_path TEXT PRIMARY KEY, '
                    'prcfo TEXT, annotate INTEGER)')
        con.executemany('INSERT INTO png_list (png_path, prcfo) VALUES (?, ?)',
                        [c[:2] for c in crops])
    return src


@pytest.fixture
def annotate(qtbot, qt_theme_applied, plate, alpha):
    from spacr.qt.screens.annotate import AnnotateScreen

    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.image_size = (32, 32)
    widget.resize(900, 700)
    widget._open_source(str(plate))
    qtbot.waitUntil(lambda: bool(widget._page_paths), timeout=10000)
    yield widget
    if widget._worker is not None:
        widget._worker.stop(wait=True)


def test_the_button_follows_the_alpha_switch(annotate, alpha):
    from spacr.qt.preferences import _apply_alpha_widgets

    from spacr.settings import ALPHA_FEATURES

    assert "AnnotateFindSimilar" in ALPHA_FEATURES[565]["widgets"]
    button = annotate._btn_similar
    assert button.isHidden()
    alpha["on"] = True
    _apply_alpha_widgets(annotate)
    assert not button.isHidden()
    alpha["on"] = False
    _apply_alpha_widgets(annotate)
    assert button.isHidden()


def test_a_search_while_hidden_still_pins_the_grid_to_the_matches(
        qtbot, annotate, alpha):
    assert annotate._btn_similar.isHidden()
    annotate._set_focus_slot(0)
    query = annotate._page_paths[0][0]
    annotate._on_find_similar()
    qtbot.waitUntil(lambda: annotate._similar_worker is None
                    and annotate._object_request is not None, timeout=20000)
    rows = [row[0] for row in annotate._filtered_rows]
    assert rows[0] == query
    assert len(rows) == 20
    area = {p: int(p.split("_o")[-1].split(".")[0]) % 2 for p in rows}
    first_matches = rows[1:10]
    assert all(area[p] == area[query] for p in first_matches)
    assert annotate._object_request.source == "similarity"
    assert annotate._similar_cache is not None
    annotate._on_find_similar()
    qtbot.waitUntil(lambda: annotate._similar_worker is None, timeout=20000)
    cached = annotate._similar_cache[2]
    annotate._on_find_similar()
    qtbot.waitUntil(lambda: annotate._similar_worker is None, timeout=20000)
    assert annotate._similar_cache[2] is cached


# ---------------------------------------------------------------------------
# What the search refuses, and answers that arrive late (coverage, 288)
# ---------------------------------------------------------------------------

def test_a_search_needs_a_source_a_free_worker_and_a_selected_crop(
        qtbot, qt_theme_applied, alpha, annotate, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from spacr.qt.screens.annotate import AnnotateScreen

    told = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: told.append(a[1])))
    empty = AnnotateScreen()
    qtbot.addWidget(empty)
    empty._on_find_similar()
    assert told == ["Open a source first"]

    annotate._similar_worker = object()
    annotate._on_find_similar()
    assert annotate._status_label.text() == "A search is already running."
    annotate._similar_worker = None
    annotate._set_focus_slot(10_000)
    annotate._focus_slot = 10_000
    assert annotate._similar_query_key() is None
    annotate._on_find_similar()
    assert annotate._status_label.text() == "No crop is selected to match."


def test_a_late_answer_for_another_source_is_ignored_and_failures_are_said(
        annotate):
    before = annotate._similar_cache
    annotate._on_similar_done({"db_path": "/elsewhere/measurements.db"})
    assert annotate._similar_cache is before
    annotate._on_similar_failed("no measurements table")
    assert "no measurements table" in annotate._status_label.text()
    annotate._similar_worker = None
    annotate._on_similar_finished()
    assert annotate._btn_similar.isEnabled()


def test_a_gui_scale_change_refits_the_grid_unless_the_screen_is_closing(
        annotate, monkeypatch):
    refits = []
    monkeypatch.setattr(annotate, "_refit_grid", lambda: refits.append(1))
    annotate._on_gui_scale_changed(1.25)
    annotate._closing = True
    try:
        annotate._on_gui_scale_changed(1.5)
    finally:
        annotate._closing = False
    assert refits == [1]
