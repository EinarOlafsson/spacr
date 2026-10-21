"""Cached object pages keep their ordering when scores disappear or no run exists."""
import pytest

from spacr.qt.screens import annotate


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    widget = annotate.AnnotateScreen()
    qtbot.addWidget(widget)
    return widget


def test_cached_object_order_survives_missing_score_database_and_retains_all_crops(screen):
    rows = [("second", 2), ("unlabelled", None), ("first", 1)]
    screen._object_rows = list(rows)
    screen._settings.db_path = ""
    screen._sort_mode = "probability_high"
    completed = []
    screen._refresh_total(then=lambda: completed.append(True))
    assert screen._filtered_rows == rows
    assert screen._total == 3 and completed == [True]
    screen._sort_mode = "class_last"
    screen._refresh_total()
    assert screen._filtered_rows == [rows[2], rows[0], rows[1]]
    screen._settings.grid_rows = 1
    screen._settings.grid_cols = 2
    assert screen._frozen_pages() == [["first", "second"], ["unlabelled"]]
    assert screen._open_suggest_dialog() is None
    assert screen._synthetic_negatives_needed() is None


def test_model_controls_explain_absent_stale_and_running_rounds(screen, qtbot):
    assert screen._open_suggest_model() is None
    assert "Run Suggest first" in screen._status_label.text()
    dialog = annotate._SuggestModelDialog()
    qtbot.addWidget(dialog)
    screen._suggest_worker = object()
    screen._show_suggest_liveness(dialog)
    assert not dialog._threshold_box.isEnabled()
    assert "run is going" in dialog._live_note.text()
    screen._suggest_worker = None
    screen._suggest_cache = {"live": False}
    screen._show_suggest_liveness(dialog)
    assert not dialog._threshold_box.isEnabled()
    assert "kept or thrown away" in dialog._live_note.text()
    screen._suggest_cache = {"live": True}
    screen._show_suggest_liveness(dialog)
    assert dialog._threshold_box.isEnabled()
