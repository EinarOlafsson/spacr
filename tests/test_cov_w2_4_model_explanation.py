"""Explanation workbench — running, rendering and refusing.

Two panels, and between them almost every path here ends in the status
line rather than in a dialog, because both run long jobs on a worker and
neither may put a traceback in front of the user.

What is asserted:

* the explanation panel's Run precondition, which is the only thing standing
  between a half-filled form and a job that fails minutes later;
* the render of a finished result, from real
  :class:`~spacr.surrogate.SurrogateResult` and
  :class:`~spacr.hit_attribution.HitAttributionResult` objects, so the
  columns the tables show are the ones those classes actually carry;
* the hand-offs to the host screen (activation maps, Image UMAP), and that
  closing the investigation panel shuts its job runner down.

The cell rendering, backend greying, importance->distribution link,
prediction-column refresh, promotion and Undo are covered by
``tests/test_model_explanation_workbench_behaviour.py``.

The jobs are run synchronously through ``JobRunner.submit``'s own
signature, so nothing here sleeps or waits on a thread.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.hit_attribution import HitAttributionResult
from spacr.qt.screens import model_explanation as mx
from spacr.qt.screens.model_explanation import (
    ExplainCvPanel, InvestigateHitPanel, _fill_table,
)
from spacr.surrogate import SurrogateResult


# ---------------------------------------------------------------------------
# Table cells
# ---------------------------------------------------------------------------


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


# -- the importance -> distribution link ------------------------------------


# -- prediction columns -----------------------------------------------------


# -- the other buttons ------------------------------------------------------


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
    """No host is no screen to navigate to, so the button does nothing.

    The absent host has to be the whole reason: the same panel, given one,
    asks it immediately -- so this is a guard on the host and not a route
    that has been quietly disabled.
    """
    assert explain.host is None
    explain.open_activation_maps()

    class Host:
        def __init__(self):
            self.asked = []

        def _on_train_requested(self, key, settings):
            self.asked.append((key, settings))

    explain.host = Host()
    explain.open_activation_maps()
    assert explain.host.asked == [("activation_maps", {})]


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


# -- candidates, promotion, undo --------------------------------------------


def test_the_umap_button_is_inert_without_a_host(investigate, tmp_path):
    """Without a host there is nowhere to open the UMAP, so nothing opens.

    And nothing is read off the form either: the panel is left exactly as
    the user left it, and the same call with a host attached goes through.
    """
    assert investigate.host is None
    investigate.database.setText(str(tmp_path / "measurements.db"))
    investigate.open_umap()
    assert investigate.database.text() == str(tmp_path / "measurements.db")

    class Host:
        def __init__(self):
            self.asked = []

        def _on_train_requested(self, key, settings):
            self.asked.append((key, settings))

    investigate.host = Host()
    investigate.open_umap()
    assert investigate.host.asked == [
        ("umap", {"src": str(tmp_path),
                  "color_by": investigate.annotation.text()})]


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
