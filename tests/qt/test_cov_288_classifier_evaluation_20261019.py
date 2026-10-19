"""Item 288: classifier-evaluation edges that the ratchet found unexercised.

The searchable text of a table with no columns, the prediction grid with no
model or outside its bounds, a loaded bundle without a prediction table, a
view destroyed while a filter pass runs, the search text built once per
bundle, and a screen closed after its filter timer is gone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.classifier_evaluation import (
    evaluate_predictions,
    write_evaluation_bundle,
)
from spacr.qt.screens import classifier_evaluation as CE
from spacr.qt.screens.classifier_evaluation import ClassifierEvaluationScreen

ROWS = 40


@pytest.fixture
def bundle_root(tmp_path):
    paths = [f"plate1_r{1 + i % 4}_c{1 + i % 3}_{i}.png" for i in range(ROWS)]
    positive = np.linspace(0.05, 0.95, ROWS)
    evaluation = evaluate_predictions(
        np.arange(ROWS) % 2,
        np.stack([1.0 - positive, positive], axis=1),
        paths,
        classes=["a", "b"],
        fold_ids=[1 + i % 3 for i in range(ROWS)],
        calibration_method="none",
        calibration_bins=5,
    )
    return write_evaluation_bundle(
        tmp_path / "model" / "evaluation", evaluation).parents[1]


@pytest.fixture
def screen(qtbot, qt_theme_applied, bundle_root):
    widget = ClassifierEvaluationScreen(None, threaded=False)
    qtbot.addWidget(widget)
    widget._source.setText(str(bundle_root))
    widget.scan()
    assert widget.bundle is not None
    return widget


def test_row_search_text_of_a_table_without_columns_is_empty_strings():
    frame = pd.DataFrame(index=[5, 6, 7])
    text = CE._row_search_text(frame)
    assert list(text.index) == [5, 6, 7]
    assert list(text) == ["", "", ""]


def test_prediction_grid_without_a_model_answers_none(qapp):
    table = CE._PredictionTable()
    assert table.rowCount() == 0
    assert table.columnCount() == 0
    assert table.item(0, 0) is None
    assert table.horizontalHeaderItem(0) is None


def test_prediction_grid_outside_its_bounds_answers_none(screen):
    table = screen._predictions
    assert table.rowCount() > 0
    assert table.item(table.rowCount() + 5, 0) is None
    assert table.item(0, table.columnCount() + 5) is None
    assert table.horizontalHeaderItem(-1) is None
    assert table.horizontalHeaderItem(table.columnCount()) is None
    assert table.item(0, 0) is not None
    assert table.horizontalHeaderItem(0).text() != ""


def test_a_bundle_without_a_prediction_frame_keeps_no_search_text(
        screen, monkeypatch):
    real = CE.load_evaluation_bundle(screen._bundle_choice.currentData())
    real["predictions"] = None
    monkeypatch.setattr(CE, "load_evaluation_bundle", lambda _m: real)
    monkeypatch.setattr(screen, "_finish_load", lambda ok: None)
    screen._load_selected_bundle()
    assert screen._pending_error == ""
    assert screen._pending_bundle is real
    assert screen._pending_search is None


def test_a_view_destroyed_mid_pass_ends_the_pass(screen, monkeypatch):
    shown = []
    monkeypatch.setattr(screen, "_show_filtered_predictions",
                        lambda: shown.append(1))
    real = CE._alive
    calls = {"n": 0}

    def alive(obj):
        if obj is screen._predictions:
            calls["n"] += 1
            if calls["n"] >= 2:
                return False
        return real(obj)

    monkeypatch.setattr(CE, "_alive", alive)
    screen._render_predictions()
    assert shown == []
    assert screen._rendering_predictions is False


def test_a_pass_without_a_filter_timer_still_filters(screen, monkeypatch):
    shown = []
    screen._filter_timer = None
    monkeypatch.setattr(screen, "_show_filtered_predictions",
                        lambda: shown.append(1))
    screen._render_predictions()
    assert shown == [1]


def test_setting_the_frame_on_a_destroyed_grid_returns_quietly(
        screen, monkeypatch):
    real = CE._alive
    monkeypatch.setattr(
        CE, "_alive",
        lambda obj: False if obj is screen._predictions else real(obj))
    screen._set_prediction_frame(pd.DataFrame())
    assert screen._predictions.rowCount() == 0


def test_search_text_is_built_once_per_frame(screen):
    frame = pd.DataFrame({"a": ["X", "y"], "b": ["Z", "w"]})
    screen._search_text = None
    first = screen._prediction_search_text(frame)
    assert list(first) == ["x z", "y w"]
    assert screen._search_text[0] is frame
    assert screen._prediction_search_text(frame) is first
    other = frame.copy()
    assert screen._prediction_search_text(other) is not first
    assert screen._search_text[0] is other


def test_closing_after_the_filter_timer_is_gone(screen):
    screen._filter_timer = None
    screen.close()
    assert screen._jobs == [] or not screen._jobs
