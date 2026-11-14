"""Model-explanation workbench: what the two panels show, refuse and hand over.

Both panels (:class:`ExplainCvPanel` and :class:`InvestigateHitPanel`) end
almost every path in their status line rather than a dialog, so the status
text, the table contents, the button states and the arguments handed to the
analysis functions are what is asserted here, exactly.

Results are built from real :class:`~spacr.surrogate.SurrogateResult` and
:class:`~spacr.hit_attribution.HitAttributionResult` objects; jobs run
synchronously through ``JobRunner.submit``'s own signature.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QComboBox

from spacr.hit_attribution import HitAttributionResult
from spacr.qt.screens import model_explanation as mx
from spacr.qt.screens.model_explanation import (
    ExplainCvPanel, InvestigateHitPanel, _read_only_item,
)
from spacr.surrogate import SurrogateResult


# ---------------------------------------------------------------------------
# Builders
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


def _column(table, name):
    """The texts of one named column of a QTableWidget, top to bottom."""
    header = [table.horizontalHeaderItem(i).text()
              for i in range(table.columnCount())]
    index = header.index(name)
    return [table.item(row, index).text() for row in range(table.rowCount())]


def _raises(exc):
    def _raise(*_args, **_kwargs):
        raise exc
    return _raise


class _ItemlessCombo(QComboBox):
    """A dropdown whose Python-side ``model()`` holds no item for any row.

    ``QComboBox.model()`` is not a C++ virtual, so Qt keeps using the real
    model; only the panel's ``model().item(index)`` lookup sees this empty
    one and gets ``None`` back.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._itemless = QStandardItemModel(self)

    def model(self):
        return self._itemless


# ---------------------------------------------------------------------------
# Table cells
# ---------------------------------------------------------------------------

def test_a_result_cell_is_read_only_text_with_blank_missing_values(
        qtbot, subtests):
    """None/NaN are blank, floats show 5 significant figures, nothing is editable."""
    cases = [
        ("none", None, ""),
        ("float nan", float("nan"), ""),
        ("numpy nan", np.nan, ""),
        ("small float", 0.123456789, "0.12346"),
        ("large float", 1234567.0, "1.2346e+06"),
        ("whole float", 1.0, "1"),
        ("int", 3, "3"),
        ("string", "cell_area", "cell_area"),
    ]
    for case_id, value, text in cases:
        with subtests.test(msg=case_id):
            item = _read_only_item(value)
            assert item.text() == text
            assert not (item.flags() & Qt.ItemIsEditable)
            assert item.flags() & Qt.ItemIsEnabled


# ---------------------------------------------------------------------------
# ExplainCvPanel
# ---------------------------------------------------------------------------

def _all_but_xgboost():
    return {key: {"available": key != "xgboost", "version": "1.0",
                  "reason": "" if key != "xgboost"
                            else "install xgboost to enable this backend"}
            for key in mx.MODEL_FAMILIES}


def test_explain_lists_every_backend_and_greys_the_unavailable_with_its_reason(
        qtbot, monkeypatch, subtests):
    """Each family stays listed; the unusable ones are disabled and say why."""
    first = next(iter(mx.MODEL_FAMILIES))
    reason = "install xgboost to enable this backend"
    cases = [
        ("every family usable",
         lambda: {k: {"available": True} for k in mx.MODEL_FAMILIES}, None,
         {"random_forest": (True, None), "hist_gradient_boosting": (True, None),
          "xgboost": (True, None)}),
        ("xgboost unusable", _all_but_xgboost, None,
         {"random_forest": (True, None), "hist_gradient_boosting": (True, None),
          "xgboost": (False, reason)}),
        ("families missing from the report fall back to Unavailable",
         lambda: {first: {"available": False,
                          "reason": "pip install xgboost"}}, None,
         {"random_forest": (False, "pip install xgboost"),
          "hist_gradient_boosting": (False, "Unavailable"),
          "xgboost": (False, "Unavailable")}),
        ("no model item to grey still keeps the reason", _all_but_xgboost,
         _ItemlessCombo,
         {"random_forest": (None, None), "hist_gradient_boosting": (None, None),
          "xgboost": (None, reason)}),
    ]
    for case_id, backends, combo, expected in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            m.setattr(mx, "available_backends", backends)
            if combo is not None:
                m.setattr(mx, "QComboBox", combo)
            panel = ExplainCvPanel()
            qtbot.addWidget(panel)
            keys = [panel.backend.itemData(i)
                    for i in range(panel.backend.count())]
            assert keys == list(mx.MODEL_FAMILIES)
            assert [panel.backend.itemText(i)
                    for i in range(panel.backend.count())] == list(
                        mx.MODEL_FAMILIES.values())
            seen = {}
            for i, key in enumerate(keys):
                item = panel.backend.model().item(i)
                seen[key] = (None if item is None else item.isEnabled(),
                             panel.backend.itemData(i, Qt.ToolTipRole))
            assert seen == expected


def test_explain_offers_the_prediction_files_columns_and_keeps_a_choice(
        qtbot, tmp_path, subtests):
    """Path-like columns go to the path box, the rest to the prediction box."""
    def panel_for(text):
        panel = ExplainCvPanel()
        qtbot.addWidget(panel)
        path = tmp_path / f"{len(list(tmp_path.iterdir()))}.csv"
        if text is not None:
            path.write_text(text, encoding="utf-8")
        panel.predictions.setText(str(path))
        return panel

    def boxes(panel):
        return ([panel.path_column.itemText(i)
                 for i in range(panel.path_column.count())],
                panel.path_column.currentText(),
                [panel.prediction_column.itemText(i)
                 for i in range(panel.prediction_column.count())],
                panel.prediction_column.currentText())

    defaults = (["path", "png_path", "image_path"], "path",
                ["pred", "prediction", "class"], "pred")
    initial_status = ("Choose predictions already produced by the CV model. "
                      "The model is not rerun here.")

    with subtests.test(msg="a path that is not a file changes nothing"):
        panel = panel_for(None)
        panel._refresh_prediction_columns()
        assert boxes(panel) == defaults
        assert panel.status.text() == initial_status

    with subtests.test(msg="an unreadable file says so and changes nothing"):
        panel = panel_for('path,pred\n"unterminated,1\n')
        panel._refresh_prediction_columns()
        assert panel.status.text().startswith(
            "Could not read prediction columns: ")
        assert boxes(panel) == defaults

    with subtests.test(msg="columns are split and the first fitting one is picked"):
        panel = panel_for("crop_path,probability,model_decision\na.png,0.9,1\n")
        panel._refresh_prediction_columns()
        assert boxes(panel) == (["crop_path"], "crop_path",
                                ["probability", "model_decision"],
                                "probability")
        panel.prediction_column.setCurrentText("model_decision")
        panel._refresh_prediction_columns()
        assert boxes(panel)[3] == "model_decision"

    with subtests.test(msg="known names are preferred over the first column"):
        panel = panel_for("zz,image_path,class,pred\n1,a.png,0,1\n")
        panel.path_column.setCurrentText("png_path")
        panel._refresh_prediction_columns()
        assert boxes(panel) == (["image_path"], "image_path",
                                ["zz", "class", "pred"], "pred")

    with subtests.test(msg="no path-like column offers every column to both"):
        panel = panel_for("a,b\n1,2\n")
        panel._refresh_prediction_columns()
        assert boxes(panel) == (["a", "b"], "a", ["a", "b"], "a")


def test_explain_links_an_importance_selection_to_its_distribution(
        qtbot, subtests):
    """Selecting a feature narrows the distribution table; odd states link nothing."""
    def loaded(result=None):
        panel = ExplainCvPanel()
        qtbot.addWidget(panel)
        panel._loaded((result or _surrogate(), {}))
        return panel

    with subtests.test(msg="before a result the link is inert"):
        panel = ExplainCvPanel()
        qtbot.addWidget(panel)
        assert panel._distribution_frame.empty
        panel._show_selected_feature_distribution()
        assert panel.distributions.rowCount() == 0
        assert panel.result is None

    with subtests.test(msg="a faithful result renders importance"):
        panel = loaded()
        assert panel.results.tabText(1) == "Importance"
        assert _column(panel.importance, "feature") == ["cell_area", "texture"]
        assert panel.status.text() == (
            "Held-out fidelity 0.860; majority baseline 0.500.")
        assert panel.run_button.isEnabled() is True

    with subtests.test(msg="an unfaithful result withholds the importance"):
        panel = loaded(_surrogate(faithful=False))
        assert panel.importance.rowCount() == 0
        assert panel.results.tabText(1) == "Importance (withheld)"
        assert panel.status.text() == (
            "Held-out fidelity 0.510; majority baseline 0.500.")

    with subtests.test(msg="nothing selected shows every feature"):
        panel = loaded()
        panel.importance.clearSelection()
        panel.distributions.setRowCount(0)
        panel._show_selected_feature_distribution()
        assert _column(panel.distributions, "feature") == [
            "cell_area", "cell_area", "texture"]

    for row, feature, means in ((0, "cell_area", ["1", "2"]),
                                (1, "texture", ["3"])):
        with subtests.test(msg=f"selecting row {row} narrows to {feature}"):
            panel = loaded()
            panel.importance.selectRow(row)
            assert _column(panel.distributions, "feature") == (
                [feature] * len(means))
            assert _column(panel.distributions, "mean") == means

    with subtests.test(msg="a table with no feature column links to nothing"):
        result = _surrogate()
        result.importance = pd.DataFrame({"permutation": [0.3], "gain": [0.6]})
        panel = loaded(result)
        panel.distributions.setRowCount(0)
        panel.importance.selectRow(0)
        assert panel.distributions.rowCount() == 0

    with subtests.test(msg="a selected row with an empty feature cell links to nothing"):
        panel = loaded()
        panel.importance.selectRow(0)
        panel.importance.setItem(0, 0, None)
        panel.distributions.setRowCount(0)
        panel._show_selected_feature_distribution()
        assert panel.distributions.rowCount() == 0


def test_explain_hands_held_out_objects_on_and_shuts_down_on_close(
        qtbot, monkeypatch, subtests):
    """The held-out frame reaches the opener; refusals land in the status."""
    def panel_for():
        panel = ExplainCvPanel()
        qtbot.addWidget(panel)
        return panel

    with subtests.test(msg="before a run nothing is opened"), \
            monkeypatch.context() as m:
        calls = []
        m.setattr(mx, "open_objects", lambda *a, **k: calls.append((a, k)))
        panel = panel_for()
        before = panel.status.text()
        panel.open_held_out_objects()
        assert calls == []
        assert panel.status.text() == before

    with subtests.test(msg="an empty held-out frame opens nothing"), \
            monkeypatch.context() as m:
        calls = []
        m.setattr(mx, "open_objects", lambda *a, **k: calls.append((a, k)))
        result = _surrogate()
        result.held_out = result.held_out.iloc[0:0]
        panel = panel_for()
        panel._loaded((result, {}))
        panel.open_held_out_objects()
        assert calls == []

    with subtests.test(msg="the held-out frame is handed over"), \
            monkeypatch.context() as m:
        seen = {}
        m.setattr(mx, "open_objects",
                  lambda frame, **kw: seen.update(
                      prcfo=list(frame["prcfo"]), **kw))
        panel = panel_for()
        panel._loaded((_surrogate(), {}))
        panel.open_held_out_objects()
        assert seen == {"prcfo": ["p1_r1_c1_f1_o1"],
                        "reason": "Held-out surrogate objects",
                        "source": "explain_cv"}

    with subtests.test(msg="an opener that refuses is reported"), \
            monkeypatch.context() as m:
        m.setattr(mx, "open_objects",
                  _raises(RuntimeError("no annotate screen is open")))
        panel = panel_for()
        panel._loaded((_surrogate(), {}))
        panel.run_button.setEnabled(False)
        panel.open_held_out_objects()
        assert panel.status.text() == (
            "Could not explain model: no annotate screen is open")
        assert panel.run_button.isEnabled() is True

    with subtests.test(msg="closing shuts the job runner down"):
        panel = panel_for()
        calls = []
        original = panel._jobs.shutdown
        monkeypatch.setattr(panel._jobs, "shutdown",
                            lambda *a, **k: calls.append(a) or original(*a, **k))
        panel.close()
        assert calls == [()]
        assert panel._jobs.is_busy() is False


# ---------------------------------------------------------------------------
# InvestigateHitPanel
# ---------------------------------------------------------------------------

_REFUSAL = ("Could not investigate hit: Choose database, prediction/fraction "
            "tables, regression folder, gene, guides and score column.")


def _investigate(qtbot, host=None):
    panel = InvestigateHitPanel(host=host)
    qtbot.addWidget(panel)
    return panel


def _fill_paths(panel, tmp_path):
    for row in (panel.database, panel.predictions, panel.fractions,
                panel.regression_folder):
        row.setText(str(tmp_path))


def test_investigate_refuses_a_half_filled_form_and_hands_over_the_whole_hit(
        qtbot, tmp_path, monkeypatch, subtests):
    """Run needs every field and a guide; a full form passes the hit context on."""
    base = {"db_path": str(tmp_path), "predictions_file": str(tmp_path),
            "guide_fractions_file": str(tmp_path),
            "results_folder": str(tmp_path), "hit_store_database": True,
            "verbose": False}

    with subtests.test(msg="an empty form is refused"):
        panel = _investigate(qtbot)
        panel.run()
        assert panel.status.text() == _REFUSAL
        assert panel.run_button.isEnabled() is True

    with subtests.test(msg="guides that are only commas are no guides"):
        panel = _investigate(qtbot)
        _fill_paths(panel, tmp_path)
        panel.gene.setText("EAF1")
        panel.guides.setText("   ,  ")
        panel.run()
        assert panel.status.text() == _REFUSAL

    with subtests.test(msg="a blank gene is refused"):
        panel = _investigate(qtbot)
        _fill_paths(panel, tmp_path)
        panel.gene.setText("   ")
        panel.guides.setText("EAF1_g1")
        panel.run()
        assert panel.status.text() == _REFUSAL

    with subtests.test(msg="a configured hit is passed on whole"), \
            monkeypatch.context() as m:
        panel = _investigate(qtbot)
        _fill_paths(panel, tmp_path)
        panel.configure_hit(folder=str(tmp_path), gene="EAF1", effect=-1.2,
                            guides=["EAF1_g1", "EAF1_g2"], fdr=0.01,
                            phenotype="recruitment", guide_agreement=0.75,
                            n_guides=2, well_support=8)
        panel.features.setText("cell_area, , nucleus_area")
        seen = {}
        m.setattr(mx, "investigate_hit",
                  lambda settings: seen.update(settings) or _payload())
        m.setattr(panel._jobs, "submit",
                  lambda work, done: done(work()) or True)
        panel.run()
        assert seen == {**base, "target_gene": "EAF1",
                        "target_guides": ["EAF1_g1", "EAF1_g2"],
                        "score_column": "recruitment",
                        "hit_phenotype": "recruitment",
                        "hit_effect": -1.2, "hit_fdr": 0.01,
                        "hit_guide_agreement": 0.75, "hit_n_guides": 2,
                        "hit_well_support": 8, "hit_direction": "negative",
                        "hit_feature_columns": ["cell_area", "nucleus_area"]}
        assert panel.status.text() == (
            "Stored 4 versioned cell probabilities. "
            "Promotion remains an explicit reversible step.")
        assert panel.run_button.isEnabled() is True
        assert panel.attribution_run_id == "run-0001"

    with subtests.test(msg="a hit with nothing recorded passes NaN, not zero"), \
            monkeypatch.context() as m:
        panel = _investigate(qtbot)
        _fill_paths(panel, tmp_path)
        panel.gene.setText("EAF1")
        panel.guides.setText("EAF1_g1")
        seen = {}
        m.setattr(mx, "investigate_hit",
                  lambda settings: seen.update(settings) or _payload())
        m.setattr(panel._jobs, "submit",
                  lambda work, done: done(work()) or True)
        panel.run()
        assert math.isnan(seen.pop("hit_fdr"))
        assert math.isnan(seen.pop("hit_guide_agreement"))
        assert seen == {**base, "target_gene": "EAF1",
                        "target_guides": ["EAF1_g1"],
                        "score_column": "prediction",
                        "hit_phenotype": "prediction", "hit_effect": 0.0,
                        "hit_n_guides": 0, "hit_well_support": 0,
                        "hit_direction": "positive",
                        "hit_feature_columns": []}


def test_investigate_offers_only_score_columns_from_the_prediction_file(
        qtbot, tmp_path, subtests):
    """Path-like and prcfo columns are dropped; a fitting previous choice is kept."""
    def scores(panel):
        return ([panel.score.itemText(i) for i in range(panel.score.count())],
                panel.score.currentText())

    defaults = (["prediction", "pred", "class"], "prediction")
    cases = [
        ("a path that is not a file changes nothing", None, None, defaults),
        ("the previous choice is kept",
         "crop_path,prcfo,score,prediction\na.png,p,0.9,1\n", None,
         (["score", "prediction"], "prediction")),
        ("a known name is preferred",
         "png_path,score,class\na.png,0.9,1\n", None,
         (["score", "class"], "class")),
        ("score is the last known name", "zeta,score\n1,2\n", None,
         (["zeta", "score"], "score")),
        ("otherwise the first column", "prcfo,zeta,eta\np,1,2\n", None,
         (["zeta", "eta"], "zeta")),
        ("only identifying columns leave an empty box",
         "prcfo,image_path\np,a.png\n", None, ([], "")),
    ]
    for index, (case_id, text, _unused, expected) in enumerate(cases):
        with subtests.test(msg=case_id):
            panel = _investigate(qtbot)
            path = tmp_path / f"predictions_{index}.csv"
            if text is not None:
                path.write_text(text, encoding="utf-8")
            panel.predictions.setText(str(path))
            before = panel.status.text()
            panel._refresh_prediction_columns()
            assert scores(panel) == expected
            assert panel.status.text() == before

    with subtests.test(msg="an unreadable file is reported"):
        panel = _investigate(qtbot)
        broken = tmp_path / "broken.csv"
        broken.write_text('prediction,path\n"unterminated,1\n',
                          encoding="utf-8")
        panel.predictions.setText(str(broken))
        panel._refresh_prediction_columns()
        assert panel.status.text().startswith(
            "Could not investigate hit: could not read prediction columns: ")
        assert scores(panel) == defaults


def test_investigate_opens_candidates_promotes_and_undoes_reversibly(
        qtbot, monkeypatch, subtests):
    """Candidates carry their scores; promotion arms Undo; refusals change nothing."""
    with subtests.test(msg="no result: candidates, promote and undo do nothing"), \
            monkeypatch.context() as m:
        calls = []
        for name in ("open_objects", "promote_hit_calls",
                     "undo_hit_promotion"):
            m.setattr(mx, name,
                      lambda *a, _n=name, **k: calls.append(_n))
        panel = _investigate(qtbot)
        before = panel.status.text()
        panel.open_candidates()
        panel.promote()
        panel.undo()
        assert calls == []
        assert panel.promotion_id == ""
        assert panel.status.text() == before

    with subtests.test(msg="candidates carry their probabilities, best first"), \
            monkeypatch.context() as m:
        seen = {}
        m.setattr(mx, "open_objects",
                  lambda frame, **kw: seen.update(
                      prcfo=list(frame["prcfo"]), **kw))
        panel = _investigate(qtbot)
        panel._loaded(_payload())
        panel.open_candidates()
        assert seen == {
            "prcfo": [f"p1_r1_c{i}_f1_o1" for i in range(1, 5)],
            "reason": "Top EAF1-hit-like candidates", "source": "explain_cv",
            "context": {"scores": {"p1_r1_c1_f1_o1": 0.9,
                                   "p1_r1_c2_f1_o1": 0.8,
                                   "p1_r1_c3_f1_o1": 0.2,
                                   "p1_r1_c4_f1_o1": 0.1}}}

    with subtests.test(msg="an opener that refuses is reported"), \
            monkeypatch.context() as m:
        m.setattr(mx, "open_objects",
                  _raises(RuntimeError("nowhere to open them")))
        panel = _investigate(qtbot)
        panel._loaded(_payload())
        panel.open_candidates()
        assert panel.status.text() == (
            "Could not investigate hit: nowhere to open them")

    with subtests.test(msg="a blank annotation column blocks promotion"), \
            monkeypatch.context() as m:
        calls = []
        m.setattr(mx, "promote_hit_calls", lambda *a, **k: calls.append(a))
        panel = _investigate(qtbot)
        panel._loaded(_payload())
        panel.annotation.setText("   ")
        panel.promote()
        assert calls == []
        assert panel.promotion_id == ""

    with subtests.test(msg="a refused promotion is reported, Undo stays off"), \
            monkeypatch.context() as m:
        m.setattr(mx, "promote_hit_calls",
                  _raises(ValueError("that column is not writable")))
        panel = _investigate(qtbot)
        panel._loaded(_payload())
        panel.promote()
        assert panel.status.text() == (
            "Could not investigate hit: that column is not writable")
        assert panel.promotion_id == ""
        assert panel.undo_button.isEnabled() is False
        assert panel.umap_button.isEnabled() is False

    with subtests.test(msg="a promotion records its id and arms Undo"), \
            monkeypatch.context() as m:
        seen = {}
        panel = _investigate(qtbot)
        panel._loaded(_payload())
        panel.database.setText("/data/measurements.db")
        panel.annotation.setText("  hit_like  ")
        m.setattr(mx, "promote_hit_calls",
                  lambda db, result, run_id, annotation_column: seen.update(
                      db=db, result=result, run_id=run_id,
                      column=annotation_column) or "promo-1")
        panel.promote()
        assert seen == {"db": "/data/measurements.db",
                        "result": panel.result, "run_id": "run-0001",
                        "column": "hit_like"}
        assert panel.promotion_id == "promo-1"
        assert panel.undo_button.isEnabled() is True
        assert panel.umap_button.isEnabled() is True
        assert panel.status.text() == (
            "Promoted calls to 'hit_like'; the previous values are "
            "retained for Undo.")

    with subtests.test(msg="a refused undo leaves the buttons as they were"), \
            monkeypatch.context() as m:
        panel = _investigate(qtbot)
        panel.promotion_id = "promo-1"
        panel.undo_button.setEnabled(True)
        panel.umap_button.setEnabled(True)
        m.setattr(mx, "undo_hit_promotion",
                  _raises(OSError("the database is locked")))
        panel.undo()
        assert panel.status.text() == (
            "Could not investigate hit: the database is locked")
        assert panel.undo_button.isEnabled() is True
        assert panel.umap_button.isEnabled() is True

    with subtests.test(msg="an undo says how many values it put back"), \
            monkeypatch.context() as m:
        seen = []
        panel = _investigate(qtbot)
        panel.database.setText("/data/measurements.db")
        panel.promotion_id = "promo-1"
        panel.undo_button.setEnabled(True)
        panel.umap_button.setEnabled(True)
        m.setattr(mx, "undo_hit_promotion",
                  lambda db, promo: seen.append((db, promo)) or 1234)
        panel.undo()
        assert seen == [("/data/measurements.db", "promo-1")]
        assert panel.status.text() == (
            "Restored 1,234 previous annotation values.")
        assert panel.undo_button.isEnabled() is False
        assert panel.umap_button.isEnabled() is False
