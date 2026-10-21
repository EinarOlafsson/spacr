"""Annotation ordering and model displays retain scientific and user decisions."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

import spacr.qt.screens.annotate as annotate
from spacr.qt.annotate_engine import AnnotateSettings
from spacr.suggest import SUGGESTION_OFFSET


def test_sorting_keeps_human_labels_before_suggestions_and_unscored_crops_last():
    rows = [("unknown", None), ("suggested2", SUGGESTION_OFFSET + 2),
            ("label1", 1), ("label2", 2), ("invalid", "bad"), ("zero", 0)]
    names = lambda ordered: [row[0] for row in ordered]
    assert names(annotate._sort_rows(rows, "class_last")) == [
        "label1", "label2", "suggested2", "unknown", "invalid", "zero"]
    assert names(annotate._sort_rows(rows, "class_first")) == [
        "unknown", "invalid", "zero", "label1", "label2", "suggested2"]
    scores = {"suggested2": 0.9, "label1": 0.9, "label2": 0.4}
    assert names(annotate._sort_rows(rows, "probability_high", scores)) == [
        "suggested2", "label1", "label2", "unknown", "invalid", "zero"]
    assert names(annotate._sort_rows(rows, "probability_low", scores)) == [
        "label2", "suggested2", "label1", "unknown", "invalid", "zero"]
    assert annotate._sort_rows(rows, "original") == rows
    assert annotate._sort_rows(rows, "unsupported") == rows
    assert annotate._stored_suggestion(None) is None
    assert annotate._stored_suggestion(2) == SUGGESTION_OFFSET + 2
    assert annotate._replaceable(None)
    assert annotate._replaceable(SUGGESTION_OFFSET + 2)
    assert not annotate._replaceable(1)
    assert not annotate._replaceable("not a class")


def test_round_classes_and_blinding_do_not_guess_missing_or_invalid_classes():
    assert annotate._round_classes(None) is None
    assert annotate._round_classes(SimpleNamespace(classes=["1", "2.0"])) == [1, 2]
    assert annotate._round_classes(SimpleNamespace(classes=["1", "unknown"])) is None
    rows = [("new-a", None), ("known", 1), ("new-b", 2)]
    forward = annotate._blind_order(rows, {"known": 0})
    backward = annotate._blind_order(list(reversed(rows)), {"known": 0})
    assert forward == backward
    assert forward[0] == ("known", 1)
    assert set(forward[1:]) == {("new-a", None), ("new-b", 2)}


@pytest.fixture
def crop_scores(tmp_path):
    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE png_list (png_path TEXT, annotate INTEGER, "
                   "al_prob_0 REAL, al_prob_1 REAL)")
        db.executemany("INSERT INTO png_list VALUES (?,?,?,?)", [
            ("high", 2, 0.1, 0.9), ("low", 1, 0.6, 0.4),
            ("missing", None, None, 0.8)])
        db.execute("CREATE TABLE no_scores (png_path TEXT)")
        db.execute("CREATE TABLE empty_scores (png_path TEXT, al_prob_0 REAL)")
        db.execute("CREATE TABLE invalid_scores (png_path TEXT, al_prob_0 REAL)")
        db.execute("INSERT INTO invalid_scores VALUES ('missing', NULL)")
    return path


def test_confidence_order_uses_real_complete_score_rows_and_tolerates_empty_tables(crop_scores):
    assert annotate._crop_confidences(str(crop_scores), "png_list") == {
        "high": pytest.approx(0.9), "low": pytest.approx(0.6)}
    for table in ("no_scores", "empty_scores", "invalid_scores", 'broken" table'):
        assert annotate._crop_confidences(str(crop_scores), table) == {}


def test_total_order_fetches_a_real_population_and_retains_rows_when_scores_fail(
        crop_scores, monkeypatch):
    settings = AnnotateSettings(db_path=str(crop_scores))
    outcome = {"total": 3, "note": "kept"}
    assert annotate._sorted_total(outcome, settings, "original") is outcome
    assert annotate._sorted_total(outcome, settings, "unknown") is outcome
    ordered = annotate._sorted_total(outcome, settings, "probability_high")
    assert [row[0] for row in ordered["filtered_rows"]] == ["high", "low", "missing"]
    assert ordered["total"] == 3 and ordered["note"] == "kept"
    assert annotate._blinded_total(outcome, settings, None) is outcome
    blinded = annotate._blinded_total(outcome, settings, {"low": 0, "high": 1, "missing": 2})
    assert [row[0] for row in blinded["filtered_rows"]] == ["low", "high", "missing"]
    assert blinded["note"] == "" and blinded["total"] == 3
    prior = {"filtered_rows": ordered["filtered_rows"], "total": 3}

    def broken_scores(*_args):
        raise sqlite3.DatabaseError("stored scores unavailable")

    monkeypatch.setattr(annotate, "_crop_confidences", broken_scores)
    fallback = annotate._sorted_total(prior, settings, "probability_high")
    assert fallback["filtered_rows"] == prior["filtered_rows"]
    assert annotate._sorted_total(prior, settings, "class_last")["total"] == 3


def test_real_histogram_renders_empty_bins_mixed_class_ids_and_clamped_lines(qapp, qtbot):
    histogram = annotate._ProbabilityHistogram()
    qtbot.addWidget(histogram)
    histogram.resize(320, 180)
    histogram.set_distribution([1, "other"], {"1": [0.05, 0.15, 0.99], "other": [0.55]},
                               "Class probability")
    histogram.set_lines([-0.2, 0.5, 1.2])
    image = histogram.grab().toImage()
    assert not image.isNull()
    assert sum(histogram.counts()["1"]) == 3
    assert sum(histogram.counts()["other"]) == 1
    assert histogram.lines() == [-0.2, 0.5, 1.2]
    histogram.set_distribution([], {}, "No scores")
    assert not histogram.grab().isNull()
    assert histogram.counts() == {}


def _scorecard(classes):
    n = len(classes)
    truth = [i for i in range(n) for _ in range(2)]
    return {"classes": classes, "folds": 2, "n": 2 * n,
            "accuracy": 1.0, "balanced_accuracy": 1.0,
            "baseline_class": classes[0], "baseline_accuracy": 1 / n,
            "baseline_balanced_accuracy": 1 / n, "baseline_f1_macro": 2 / (n * (n + 1)),
            "precision": [1.0] * n, "recall": [1.0] * n, "f1": [1.0] * n,
            "support": [2] * n, "f1_macro": 1.0,
            "confusion": [[2 if i == j else 0 for j in range(n)] for i in range(n)],
            "truth": truth, "oof": np.eye(n)[truth].tolist()}


def test_model_dialog_shows_real_scorecard_cells_and_per_class_thresholds(qapp, qtbot):
    dialog = annotate._SuggestModelDialog()
    qtbot.addWidget(dialog)
    classes = [1, 2, 3]
    insight = {"classes": classes, "flat": True, "scorecard": _scorecard(classes),
               "model": {"model_type": "gb", "balance": "downsample", "leaf": 2,
                         "synthetic": 4, "rejections": 1, "features": ["area", "intensity"],
                         "class_counts": {"1": 3, "2": 1, "3": 2},
                         "weights": {"1": 1.0, "2": 3.0, "3": 1.5}}}
    dialog.set_insight(insight, {"minimum": {"1": 0.2, "2": 0.3, "3": 0.3}, "margin": 0.1})
    assert dialog._classes_table.rowCount() == 3
    assert dialog._features.toPlainText() == "area\nintensity"
    assert dialog._scorecard.rowCount() == 5
    assert dialog._confusion.rowCount() == dialog._confusion.columnCount() == 3
    assert sorted(dialog._confusion.item(i, i).text() for i in range(3)) == ["2", "2", "2"]
    assert dialog._histogram.lines() == [0.2, 0.3]
    assert dialog.thresholds()["margin"] == pytest.approx(0.1)
    assert {k: sum(v) for k, v in dialog._histogram.counts().items()} == {"1": 2, "2": 2, "3": 2}
    dialog.set_counts({"1": 3, "withheld": 2})
    assert "3" in dialog._counts.text() and "2" in dialog._counts.text()
    dialog.set_counts(None)
    assert dialog._counts.text() == ""
    dialog.set_insight(insight)
    assert dialog._threshold_layout.rowCount() == 4
    dialog.set_live(False, "Previous model is stale")
    assert not dialog._threshold_box.isEnabled()
    assert dialog._live_note.text() == "Previous model is stale"


def test_binary_threshold_controls_rebuild_after_multiclass_and_emit_a_normalized_band(qapp, qtbot):
    dialog = annotate._SuggestModelDialog()
    qtbot.addWidget(dialog)
    dialog.set_insight({"classes": [1, 2, 3]})
    dialog.set_insight({"classes": [1, 2], "scorecard": _scorecard([1, 2]),
                        "model": {"model_type": "custom", "reweights": True}})
    assert dialog._minimum == {} and dialog._margin is None
    assert dialog._histogram.counts() == {"1": [2] + [0] * 9, "2": [0] * 9 + [2]}
    dialog._slider.setValue(85)
    dialog._band.setChecked(True)
    dialog._low.setValue(0.9)
    dialog._high.setValue(0.2)
    delivered = []
    dialog.thresholds_changed.connect(delivered.append)
    dialog.apply_now()
    assert len(delivered) == 1
    assert delivered[0]["threshold"] == pytest.approx(0.85)
    assert delivered[0]["low"] == pytest.approx(0.2)
    assert delivered[0]["high"] == pytest.approx(0.9)
    assert delivered[0]["band"] is True
    assert dialog._histogram.lines() == [0.2, 0.9]
    dialog._band.setChecked(False)
    dialog.apply_now()
    assert dialog._histogram.lines() == [0.85]
    dialog.set_insight({"classes": []})
    assert dialog.thresholds()["minimum"] == {}


@pytest.fixture
def example_dialog(qapp, qtbot, tmp_path, monkeypatch):
    dialog = annotate._SettingsDialog(AnnotateSettings())
    qtbot.addWidget(dialog)
    monkeypatch.setattr(dialog, "example_destination", lambda: tmp_path)
    applied = []
    monkeypatch.setattr(dialog, "_apply_example_settings", lambda path: applied.append(path))
    return dialog, tmp_path, applied


@pytest.mark.parametrize("streaming", [False, True])
def test_failed_or_successful_example_download_restores_the_real_button(
        example_dialog, streaming):
    dialog, folder, applied = example_dialog
    button = QPushButton("Before", dialog)
    setattr(dialog, "_stream_btn" if streaming else "_example_btn", button)
    callbacks = []

    def download(owner, destination, done):
        assert owner is dialog and destination == folder
        callbacks.append(done)

    load = dialog._load_the_streaming_example if streaming else dialog._load_the_example_data
    assert load(ask=download) == ""
    assert not button.isEnabled()
    callbacks.pop()(None, "network unavailable")
    assert button.isEnabled() and not applied
    assert load(ask=download) == ""
    dialog._ann_col.clear()
    callbacks.pop()(object(), "")
    assert button.isEnabled()
    assert dialog._src_edit.text() == str(folder)
    assert dialog._ann_col.text() == "infected"
    assert applied == [folder / "settings/annotate_settings.csv"]
    if streaming:
        merged = folder / "merged"
        merged.mkdir()
        np.save(merged / "field.npy", np.zeros((1, 1)))
        assert load(ask=download) == str(folder)
        assert callbacks == []
