"""Explanation workbench — running, rendering and refusing.

Two panels, and between them almost every path here ends in the status
line rather than in a dialog, because both run long jobs on a worker and
neither may put a traceback in front of the user.

What is asserted:

* the two Run buttons' preconditions, which are the only thing standing
  between a half-filled form and a job that fails minutes later;
* the render of a finished result, from real
  :class:`~spacr.surrogate.SurrogateResult` and
  :class:`~spacr.hit_attribution.HitAttributionResult` objects, so the
  columns the tables show are the ones those classes actually carry;
* the importance→distribution link, including the cases where there is
  nothing selected, no ``feature`` column to select on, and an empty cell;
* promotion and its Undo, which is the reversibility the panel promises;
* a backend that is not installed, which must be listed and greyed rather
  than hidden -- a missing option nobody can see is a bug report.

The jobs are run synchronously through ``JobRunner.submit``'s own
signature, so nothing here sleeps or waits on a thread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.hit_attribution import HitAttributionResult
from spacr.qt.screens import model_explanation as mx
from spacr.qt.screens.model_explanation import (
    ExplainCvPanel, InvestigateHitPanel, _fill_table, _read_only_item,
)
from spacr.surrogate import SurrogateResult


# ---------------------------------------------------------------------------
# Table cells
# ---------------------------------------------------------------------------

def test_a_missing_number_is_an_empty_cell_not_the_word_nan(qtbot):
    """"nan" in a results table reads as a value that was measured."""
    assert _read_only_item(None).text() == ""
    assert _read_only_item(float("nan")).text() == ""
    assert _read_only_item(np.nan).text() == ""


def test_a_number_is_shown_to_five_significant_figures(qtbot):
    assert _read_only_item(0.123456789).text() == "0.12346"
    assert _read_only_item(1234567.0).text() == "1.2346e+06"
    assert _read_only_item("cell_area").text() == "cell_area"


def test_a_table_cell_cannot_be_edited(qtbot):
    from PySide6.QtCore import Qt

    item = _read_only_item(1.0)
    assert not (item.flags() & Qt.ItemIsEditable)


def test_a_frame_with_a_named_index_shows_the_index_as_a_column(qtbot):
    from PySide6.QtWidgets import QTableWidget

    table = QTableWidget()
    qtbot.addWidget(table)
    frame = pd.DataFrame({"value": [1.0, 2.0]},
                         index=pd.Index(["a", "b"], name="feature"))

    _fill_table(table, frame)

    assert table.columnCount() == 2
    assert table.horizontalHeaderItem(0).text() == "feature"
    assert table.item(0, 0).text() == "a"


def test_a_long_frame_is_truncated_rather_than_freezing_the_window(qtbot):
    from PySide6.QtWidgets import QTableWidget

    table = QTableWidget()
    qtbot.addWidget(table)
    _fill_table(table, pd.DataFrame({"value": range(50)}), limit=10)
    assert table.rowCount() == 10


# ---------------------------------------------------------------------------
# ExplainCvPanel
# ---------------------------------------------------------------------------

def _surrogate(faithful=True):
    return SurrogateResult(
        fidelity=0.86 if faithful else 0.51, baseline=0.5,
        importance=pd.DataFrame({"feature": ["cell_area", "texture"],
                                 "permutation": [0.3, 0.1],
                                 "gain": [0.6, 0.4]}),
        n_objects=120, class_counts={0: 60, 1: 60},
        minimum_fidelity_improvement=0.05,
        class_metrics=pd.DataFrame({"precision": [0.8], "recall": [0.9]}),
        confusion=pd.DataFrame([[50, 10], [8, 52]]),
        feature_distributions=pd.DataFrame({
            "feature": ["cell_area", "cell_area", "texture"],
            "cv_class": [0, 1, 1], "n": [10, 12, 12],
            "mean": [1.0, 2.0, 3.0], "median": [1.0, 2.0, 3.0],
            "std": [0.1, 0.2, 0.3]}),
        held_out=pd.DataFrame({"prcfo": ["p1_r1_c1_f1_o1"],
                               "cv_prediction": [1],
                               "surrogate_prediction": [1]}),
    )


@pytest.fixture
def explain(qtbot):
    panel = ExplainCvPanel()
    qtbot.addWidget(panel)
    return panel


def test_a_backend_that_is_not_installed_is_greyed_with_the_reason(qtbot,
                                                                   monkeypatch):
    """Hidden would be worse: an option nobody can see is a bug report."""
    from PySide6.QtCore import Qt

    from spacr.surrogate import MODEL_FAMILIES

    first = next(iter(MODEL_FAMILIES))
    monkeypatch.setattr(
        mx, "available_backends",
        lambda: {first: {"available": False,
                         "reason": "pip install xgboost"}})

    panel = ExplainCvPanel()
    qtbot.addWidget(panel)

    index = panel.backend.findData(first)
    assert index >= 0
    assert panel.backend.model().item(index).isEnabled() is False
    assert panel.backend.itemData(index, Qt.ToolTipRole) == (
        "pip install xgboost")


def test_running_without_a_database_or_predictions_says_which(explain):
    explain.run()
    assert "Could not explain model" in explain.status.text()
    assert "measurements database and predictions CSV" in explain.status.text()
    assert explain.run_button.isEnabled() is True


def test_a_run_hands_the_form_to_the_analysis_and_renders_the_result(
        explain, tmp_path, monkeypatch):
    database = tmp_path / "measurements.db"
    database.touch()
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"path": ["a.png"], "pred": [1]}).to_csv(predictions,
                                                          index=False)
    explain.database.setText(str(database))
    explain.predictions.setText(str(predictions))
    explain.output.setText(str(tmp_path / "out"))

    seen = {}
    monkeypatch.setattr(mx, "explain_classifier",
                        lambda db, frame, **kw: seen.update(
                            db=db, rows=len(frame), **kw) or _surrogate())
    monkeypatch.setattr(mx, "write_surrogate_result",
                        lambda result, out: {"importance": out})
    monkeypatch.setattr(explain._jobs, "submit",
                        lambda work, done: done(work()) or True)

    explain.run()

    assert seen["db"] == str(database)
    assert seen["rows"] == 1
    assert seen["split_by"] == "well"
    assert "Held-out fidelity 0.860" in explain.status.text()
    assert "Saved 1 artifacts." in explain.status.text()
    assert explain.importance.rowCount() == 2
    assert explain.run_button.isEnabled() is True


def test_an_analysis_with_no_output_folder_writes_nothing(explain, tmp_path,
                                                          monkeypatch):
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"path": ["a.png"], "pred": [1]}).to_csv(predictions,
                                                          index=False)
    monkeypatch.setattr(mx, "explain_classifier",
                        lambda db, frame, **kw: _surrogate())
    monkeypatch.setattr(mx, "write_surrogate_result",
                        lambda *a: pytest.fail("nothing was asked to be saved"))

    result, paths = explain.run_analysis("db", str(predictions))

    assert paths == {}
    assert result.fidelity == pytest.approx(0.86)


def test_an_unfaithful_model_withholds_the_importance_and_says_so(explain):
    """An importance ranking off a model that does not fit ranks nothing."""
    explain._loaded((_surrogate(faithful=False), {}))

    assert explain.importance.rowCount() == 0
    assert explain.results.tabText(1) == "Importance (withheld)"


# -- the importance -> distribution link ------------------------------------

def test_nothing_selected_shows_every_features_distribution(explain):
    explain._loaded((_surrogate(), {}))
    explain.importance.clearSelection()

    explain._show_selected_feature_distribution()

    assert explain.distributions.rowCount() == 3


def test_selecting_a_feature_narrows_the_distribution_table(explain):
    explain._loaded((_surrogate(), {}))
    explain.importance.selectRow(0)

    assert explain.distributions.rowCount() == 2
    header = [explain.distributions.horizontalHeaderItem(i).text()
              for i in range(explain.distributions.columnCount())]
    feature = header.index("feature")
    for row in range(explain.distributions.rowCount()):
        assert explain.distributions.item(row, feature).text() == "cell_area"


def test_the_link_is_inert_before_a_result_arrives(explain):
    assert explain._distribution_frame.empty
    explain._show_selected_feature_distribution()
    assert explain.distributions.rowCount() == 0


def test_an_importance_table_with_no_feature_column_links_to_nothing(explain):
    """A withheld table still exists; selecting in it must not raise."""
    result = _surrogate()
    result.importance = pd.DataFrame({"permutation": [0.3], "gain": [0.6]})
    explain._loaded((result, {}))
    explain.distributions.setRowCount(0)

    explain.importance.selectRow(0)

    assert explain.distributions.rowCount() == 0


def test_a_selection_whose_feature_cell_is_empty_links_to_nothing(explain):
    """A selected row with no feature name in it: nothing to narrow to."""
    explain._loaded((_surrogate(), {}))
    explain.importance.selectRow(0)
    explain.importance.setItem(0, 0, None)
    explain.distributions.setRowCount(0)

    explain._show_selected_feature_distribution()

    assert explain.distributions.rowCount() == 0


# -- prediction columns -----------------------------------------------------

def test_a_prediction_path_that_is_not_a_file_leaves_the_dropdowns_alone(
        explain, tmp_path):
    before = [explain.path_column.itemText(i)
              for i in range(explain.path_column.count())]
    explain.predictions.setText(str(tmp_path / "never_written.csv"))

    explain._refresh_prediction_columns()

    assert [explain.path_column.itemText(i)
            for i in range(explain.path_column.count())] == before


def test_a_prediction_file_that_will_not_read_says_so(explain, tmp_path):
    broken = tmp_path / "predictions.csv"
    broken.write_text('path,pred\n"unterminated,1\n')
    explain.predictions.setText(str(broken))

    explain._refresh_prediction_columns()

    assert "Could not read prediction columns" in explain.status.text()


def test_a_column_already_chosen_survives_a_refresh(explain, tmp_path):
    path = tmp_path / "predictions.csv"
    pd.DataFrame({"crop_path": ["a.png"], "probability": [0.9],
                  "model_decision": [1]}).to_csv(path, index=False)
    explain.predictions.setText(str(path))
    explain._refresh_prediction_columns()
    explain.prediction_column.setCurrentText("probability")

    explain._refresh_prediction_columns()

    assert explain.prediction_column.currentText() == "probability"


# -- the other buttons ------------------------------------------------------

def test_opening_held_out_objects_before_a_run_does_nothing(explain):
    assert explain.result is None
    explain.open_held_out_objects()
    assert "Could not explain model" not in explain.status.text()


def test_an_object_opener_that_refuses_is_reported_in_the_status(explain,
                                                                 monkeypatch):
    explain._loaded((_surrogate(), {}))

    def explode(*_args, **_kwargs):
        raise RuntimeError("no annotate screen is open")

    monkeypatch.setattr(mx, "open_objects", explode)
    explain.open_held_out_objects()

    assert "no annotate screen is open" in explain.status.text()


def test_opening_held_out_objects_hands_over_the_held_out_frame(explain,
                                                                monkeypatch):
    explain._loaded((_surrogate(), {}))
    seen = {}
    monkeypatch.setattr(mx, "open_objects",
                        lambda frame, **kw: seen.update(rows=len(frame), **kw))

    explain.open_held_out_objects()

    assert seen["rows"] == 1
    assert seen["reason"] == "Held-out surrogate objects"


def test_the_activation_map_button_asks_the_host_for_that_screen(qtbot):
    class Host:
        def __init__(self):
            self.asked = []

        def _on_train_requested(self, key, settings):
            self.asked.append((key, settings))

    host = Host()
    panel = ExplainCvPanel(host=host)
    qtbot.addWidget(panel)

    panel.open_activation_maps()

    assert host.asked == [("activation_maps", {})]


def test_the_activation_map_button_is_inert_without_a_host(explain):
    explain.open_activation_maps()          # must not raise


def test_closing_the_panel_shuts_the_job_runner_down(explain):
    explain.close()
    assert explain._jobs.is_busy() is False


# ---------------------------------------------------------------------------
# InvestigateHitPanel
# ---------------------------------------------------------------------------

def _attribution():
    cells = pd.DataFrame({
        "prcfo": [f"p1_r1_c{i}_f1_o1" for i in range(1, 5)],
        "target_guide_fraction": [0.8, 0.7, 0.0, 0.0],
        "pred": [1, 1, 0, 0],
        "candidate_rank": [1, 2, 3, 4],
        "hit_like_probability": [0.9, 0.8, 0.2, 0.1],
        "hit_like_uncertainty": [0.05, 0.06, 0.04, 0.03],
        "hit_like_call": [True, True, False, False],
        "attribution_fold": [0, 1, 0, 1],
    })
    return HitAttributionResult(
        cells=cells,
        wells=pd.DataFrame({"plateID": ["p1"], "n": [4]}),
        guide_evidence=pd.DataFrame({"guide": ["EAF1_g1"], "n_wells": [4]}),
        threshold_sensitivity=pd.DataFrame({"threshold": [0.5], "n": [2]}),
        validation={"prevalence_difference": 0.3, "bootstrap_ci_low": 0.1,
                    "bootstrap_ci_high": 0.5, "permutation_p_value": 0.01,
                    "guide_fraction_refit_p_value": 0.2,
                    "well_label_refit_p_value": 0.3},
        feature_columns=["cell_area"], well_columns=["plateID"],
        object_columns=["prcfo"], target_gene="EAF1",
        target_guides=["EAF1_g1"], score_column="pred",
        direction="positive", threshold=0.5, split_level="plate",
        random_seed=0)


def _payload():
    return {"result": _attribution(),
            "embedding": pd.DataFrame({"umap1": [0.1], "umap2": [0.2]}),
            "gallery": pd.DataFrame({"blinded_id": ["A"], "path": ["a.png"]}),
            "gallery_key": {"A": "p1_r1_c1_f1_o1"},
            "paths": {"cells": "/tmp/cells.csv"},
            "attribution_run_id": "run-0001"}


@pytest.fixture
def investigate(qtbot):
    panel = InvestigateHitPanel()
    qtbot.addWidget(panel)
    return panel


def test_a_half_filled_form_says_what_is_missing(investigate):
    investigate.run()
    assert "Could not investigate hit" in investigate.status.text()
    assert "regression" in investigate.status.text()
    assert investigate.run_button.isEnabled() is True


def test_a_gene_with_no_guides_named_is_still_a_half_filled_form(
        investigate, tmp_path):
    for row in (investigate.database, investigate.predictions,
                investigate.fractions, investigate.regression_folder):
        row.setText(str(tmp_path))
    investigate.gene.setText("EAF1")
    investigate.guides.setText("   ,  ")

    investigate.run()

    assert "Could not investigate hit" in investigate.status.text()


def test_a_run_hands_the_whole_hit_context_to_the_investigation(investigate,
                                                                tmp_path,
                                                                monkeypatch):
    for row in (investigate.database, investigate.predictions,
                investigate.fractions, investigate.regression_folder):
        row.setText(str(tmp_path))
    investigate.configure_hit(folder=str(tmp_path), gene="EAF1", effect=-1.2,
                              guides=["EAF1_g1", "EAF1_g2"], fdr=0.01,
                              phenotype="recruitment", guide_agreement=0.75,
                              n_guides=2, well_support=8)
    investigate.features.setText("cell_area, nucleus_area")

    seen = {}
    monkeypatch.setattr(mx, "investigate_hit",
                        lambda settings: seen.update(settings) or _payload())
    monkeypatch.setattr(investigate._jobs, "submit",
                        lambda work, done: done(work()) or True)

    investigate.run()

    assert seen["target_gene"] == "EAF1"
    assert seen["target_guides"] == ["EAF1_g1", "EAF1_g2"]
    assert seen["hit_direction"] == "negative"
    assert seen["hit_effect"] == pytest.approx(-1.2)
    assert seen["hit_fdr"] == pytest.approx(0.01)
    assert seen["hit_guide_agreement"] == pytest.approx(0.75)
    assert seen["hit_n_guides"] == 2
    assert seen["hit_well_support"] == 8
    assert seen["hit_feature_columns"] == ["cell_area", "nucleus_area"]
    assert seen["score_column"] == "recruitment"


def test_a_hit_with_no_recorded_fdr_passes_a_nan_rather_than_a_zero(
        investigate, tmp_path, monkeypatch):
    """A missing FDR recorded as 0 would read as the strongest hit there is."""
    for row in (investigate.database, investigate.predictions,
                investigate.fractions, investigate.regression_folder):
        row.setText(str(tmp_path))
    investigate.gene.setText("EAF1")
    investigate.guides.setText("EAF1_g1")

    seen = {}
    monkeypatch.setattr(mx, "investigate_hit",
                        lambda settings: seen.update(settings) or _payload())
    monkeypatch.setattr(investigate._jobs, "submit",
                        lambda work, done: done(work()) or True)

    investigate.run()

    assert np.isnan(seen["hit_fdr"])
    assert np.isnan(seen["hit_guide_agreement"])


def test_a_finished_investigation_fills_every_tab_and_ranks_the_cells(
        investigate):
    investigate._loaded(_payload())

    assert investigate.attribution_run_id == "run-0001"
    assert "Hit attribution: EAF1" in investigate.summary.toPlainText()
    assert "permutation_p_value: 0.01" in investigate.summary.toPlainText()
    assert investigate.well_table.rowCount() == 1
    assert investigate.guide_table.rowCount() == 1
    assert investigate.threshold_table.rowCount() == 1
    assert investigate.embedding_table.rowCount() == 1
    assert investigate.gallery_table.rowCount() == 1
    assert investigate.cell_table.rowCount() == 4
    # Ranked by probability, highest first.
    probability = investigate.cell_table.horizontalHeaderItem
    header = [probability(i).text()
              for i in range(investigate.cell_table.columnCount())]
    column = header.index("hit_like_probability")
    assert investigate.cell_table.item(0, column).text() == "0.9"
    assert investigate.promote_button.isEnabled() is True
    assert investigate.umap_button.isEnabled() is False
    assert "Promotion remains an explicit reversible step." in \
        investigate.status.text()


def test_the_score_dropdown_is_filled_from_the_prediction_file(investigate,
                                                               tmp_path):
    path = tmp_path / "predictions.csv"
    pd.DataFrame({"crop_path": ["a.png"], "prcfo": ["p_r_c_f_o"],
                  "score": [0.9], "prediction": [1]}).to_csv(path,
                                                             index=False)
    investigate.predictions.setText(str(path))

    investigate._refresh_prediction_columns()

    offered = [investigate.score.itemText(i)
               for i in range(investigate.score.count())]
    assert offered == ["score", "prediction"]
    assert investigate.score.currentText() == "prediction"


def test_a_prediction_file_that_will_not_read_is_reported(investigate,
                                                          tmp_path):
    broken = tmp_path / "predictions.csv"
    broken.write_text('prediction,path\n"unterminated,1\n')
    investigate.predictions.setText(str(broken))

    investigate._refresh_prediction_columns()

    assert "could not read prediction columns" in investigate.status.text()


def test_a_prediction_path_that_is_not_a_file_leaves_the_score_box_alone(
        investigate, tmp_path):
    before = investigate.score.currentText()
    investigate.predictions.setText(str(tmp_path / "gone.csv"))
    investigate._refresh_prediction_columns()
    assert investigate.score.currentText() == before


# -- candidates, promotion, undo --------------------------------------------

def test_opening_candidates_before_a_run_does_nothing(investigate):
    investigate.open_candidates()
    assert "Could not investigate hit" not in investigate.status.text()


def test_opening_candidates_carries_their_probabilities(investigate,
                                                        monkeypatch):
    investigate._loaded(_payload())
    seen = {}
    monkeypatch.setattr(mx, "open_objects",
                        lambda frame, **kw: seen.update(rows=len(frame), **kw))

    investigate.open_candidates()

    assert seen["rows"] == 4
    assert "EAF1" in seen["reason"]
    assert seen["context"]["scores"]["p1_r1_c1_f1_o1"] == pytest.approx(0.9)


def test_an_opener_that_refuses_is_reported_in_the_status(investigate,
                                                          monkeypatch):
    investigate._loaded(_payload())
    monkeypatch.setattr(mx, "open_objects",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("nowhere to open them")))

    investigate.open_candidates()

    assert "nowhere to open them" in investigate.status.text()


def test_promotion_needs_a_run_and_an_annotation_column(investigate):
    investigate.promote()
    assert investigate.promotion_id == ""

    investigate._loaded(_payload())
    investigate.annotation.setText("   ")
    investigate.promote()
    assert investigate.promotion_id == ""


def test_a_promotion_that_refuses_is_reported_and_leaves_undo_off(
        investigate, monkeypatch):
    investigate._loaded(_payload())
    monkeypatch.setattr(mx, "promote_hit_calls",
                        lambda *a, **k: (_ for _ in ()).throw(
                            ValueError("that column is not writable")))

    investigate.promote()

    assert "that column is not writable" in investigate.status.text()
    assert investigate.undo_button.isEnabled() is False


def test_a_promotion_records_its_id_and_arms_undo(investigate, monkeypatch):
    investigate._loaded(_payload())
    seen = {}
    monkeypatch.setattr(
        mx, "promote_hit_calls",
        lambda db, result, run_id, annotation_column: seen.update(
            db=db, run_id=run_id, column=annotation_column) or "promo-1")

    investigate.promote()

    assert investigate.promotion_id == "promo-1"
    assert seen["run_id"] == "run-0001"
    assert seen["column"] == "hit_like"
    assert investigate.undo_button.isEnabled() is True
    assert investigate.umap_button.isEnabled() is True
    assert "retained for Undo" in investigate.status.text()


def test_undo_before_a_promotion_does_nothing(investigate):
    investigate.undo()
    assert "Restored" not in investigate.status.text()


def test_an_undo_that_refuses_leaves_the_buttons_where_they_were(investigate,
                                                                 monkeypatch):
    investigate.promotion_id = "promo-1"
    investigate.undo_button.setEnabled(True)
    monkeypatch.setattr(mx, "undo_hit_promotion",
                        lambda *a: (_ for _ in ()).throw(
                            OSError("the database is locked")))

    investigate.undo()

    assert "the database is locked" in investigate.status.text()
    assert investigate.undo_button.isEnabled() is True


def test_an_undo_says_how_many_values_it_put_back(investigate, monkeypatch):
    investigate.promotion_id = "promo-1"
    investigate.undo_button.setEnabled(True)
    investigate.umap_button.setEnabled(True)
    monkeypatch.setattr(mx, "undo_hit_promotion", lambda db, promo: 1234)

    investigate.undo()

    assert "Restored 1,234 previous annotation values." == \
        investigate.status.text()
    assert investigate.undo_button.isEnabled() is False
    assert investigate.umap_button.isEnabled() is False


def test_the_umap_button_is_inert_without_a_host(investigate):
    investigate.open_umap()          # must not raise


def test_the_umap_button_asks_the_host_for_the_databases_folder(qtbot,
                                                                tmp_path):
    class Host:
        def __init__(self):
            self.asked = []

        def _on_train_requested(self, key, settings):
            self.asked.append((key, settings))

    host = Host()
    panel = InvestigateHitPanel(host=host)
    qtbot.addWidget(panel)
    panel.database.setText(str(tmp_path / "measurements.db"))
    panel.annotation.setText("hit_like")

    panel.open_umap()

    assert host.asked == [("umap", {"src": str(tmp_path),
                                    "color_by": "hit_like"})]


def test_closing_the_investigation_panel_shuts_its_job_runner_down(
        investigate):
    investigate.close()
    assert investigate._jobs.is_busy() is False
