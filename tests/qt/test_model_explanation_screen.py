"""The explanation workbench exposes fidelity and hit attribution honestly."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QComboBox

from spacr.surrogate import MODEL_FAMILIES, SurrogateResult
from spacr.qt.screens.model_explanation import (
    ExplainCvPanel, InvestigateHitPanel, InvestigateHitScreen,
    ModelExplanationScreen, make_investigate_hit_screen,
    make_model_explanation_screen,
)


def _result(faithful=True):
    baseline = 0.5
    fidelity = 0.86 if faithful else 0.51
    return SurrogateResult(
        fidelity=fidelity, baseline=baseline,
        importance=pd.DataFrame({
            "feature": ["cell_area", "texture"],
            "permutation": [0.3, 0.1], "gain": [0.6, 0.4],
        }), n_objects=120, class_counts={0: 60, 1: 60},
        minimum_fidelity_improvement=0.05,
        class_metrics=pd.DataFrame({"precision": [0.8], "recall": [0.9]}),
        confusion=pd.DataFrame([[50, 10], [8, 52]]),
        feature_distributions=pd.DataFrame({
            "feature": ["cell_area", "cell_area", "texture"],
            "cv_class": [0, 1, 1], "n": [10, 12, 12],
            "mean": [1.0, 2.0, 3.0], "median": [1.0, 2.0, 3.0],
            "std": [0.1, 0.2, 0.3],
        }),
        held_out=pd.DataFrame({"prcfo": ["p_r_c_f_o"],
                               "cv_prediction": [1],
                               "surrogate_prediction": [1]}),
    )


def test_explain_screen_leads_with_the_fidelity_results(qtbot):
    screen = ModelExplanationScreen()
    qtbot.addWidget(screen)
    assert screen.explain.results.tabText(0) == "Fidelity"
    assert screen.explain.results.tabText(1) == "Importance"


def test_surrogate_family_is_a_fixed_dropdown(qtbot):
    panel = ExplainCvPanel()
    qtbot.addWidget(panel)
    assert isinstance(panel.backend, QComboBox)
    assert not panel.backend.isEditable()
    assert {panel.backend.itemData(i) for i in range(panel.backend.count())} == set(MODEL_FAMILIES)
    assert [panel.split.itemText(i) for i in range(panel.split.count())] == [
        "well", "plate"]


def test_prediction_columns_are_data_backed_dropdowns(qtbot, tmp_path):
    path = tmp_path / "predictions.csv"
    pd.DataFrame({
        "custom_crop_path": ["cell.png"],
        "model_decision": [1],
        "probability": [0.9],
    }).to_csv(path, index=False)
    panel = ExplainCvPanel()
    qtbot.addWidget(panel)
    panel.predictions.setText(str(path))
    panel._refresh_prediction_columns()
    assert [panel.path_column.itemText(i)
            for i in range(panel.path_column.count())] == ["custom_crop_path"]
    assert {panel.prediction_column.itemText(i)
            for i in range(panel.prediction_column.count())} == {
                "model_decision", "probability"}


def test_unfaithful_result_withholds_the_importance_table(qtbot):
    panel = ExplainCvPanel()
    qtbot.addWidget(panel)
    panel._loaded((_result(faithful=False), {}))
    assert panel.importance.rowCount() == 0
    assert "withheld" in panel.results.tabText(1).lower()
    assert "does NOT reproduce" in panel.summary.toPlainText()


def test_faithful_result_leads_with_fidelity_then_shows_importance(qtbot):
    panel = ExplainCvPanel()
    qtbot.addWidget(panel)
    panel._loaded((_result(faithful=True), {"manifest": "/tmp/manifest.json"}))
    assert panel.importance.rowCount() == 2
    assert "fidelity 0.860" in panel.status.text()
    assert panel.summary.toPlainText().index("baseline") < panel.summary.toPlainText().index("Top")


def test_importance_selection_filters_the_held_out_distribution(qtbot):
    panel = ExplainCvPanel()
    qtbot.addWidget(panel)
    panel._loaded((_result(faithful=True), {}))
    panel.importance.selectRow(0)
    panel._show_selected_feature_distribution()
    assert panel.distributions.rowCount() == 2
    headers = [panel.distributions.horizontalHeaderItem(i).text()
               for i in range(panel.distributions.columnCount())]
    assert panel.distributions.item(0, headers.index("feature")).text() == "cell_area"


def test_umap_handoff_uses_the_promoted_annotation(qtbot, tmp_path):
    calls = []
    host = type("Host", (), {"_on_train_requested": lambda self, key, seed:
                             calls.append((key, seed))})()
    panel = InvestigateHitPanel(host=host)
    qtbot.addWidget(panel)
    panel.database.setText(str(tmp_path / "measurements.db"))
    panel.annotation.setText("eaf1_hit_like")
    panel.open_umap()
    assert calls == [("umap", {"src": str(tmp_path),
                                "color_by": "eaf1_hit_like"})]


def test_registered_explain_app_uses_the_combined_factory(qtbot):
    from spacr.qt.app import registered_factory
    factory = registered_factory("explain_cv")
    screen = factory()
    qtbot.addWidget(screen)
    assert isinstance(screen, ModelExplanationScreen)


def test_registered_hit_app_uses_dedicated_reversible_workbench(qtbot):
    from spacr.qt.app import registered_factory
    factory = registered_factory("investigate_hit")
    screen = factory()
    qtbot.addWidget(screen)
    assert isinstance(screen, InvestigateHitScreen)


def test_hit_screen_accepts_exact_hit_list_seed(qtbot):
    screen = InvestigateHitScreen()
    qtbot.addWidget(screen)
    screen.apply_seed({
        "results_folder": "/results/exact-run",
        "target_gene": "EAF1",
        "target_guides": ["EAF1_1", "EAF1_2"],
        "hit_effect": -0.42,
        "hit_fdr": 0.01,
        "hit_guide_agreement": 0.8,
        "hit_n_guides": 3,
        "hit_well_support": 14,
        "hit_phenotype": "infection_score",
    })
    panel = screen.investigate
    assert panel.regression_folder.text() == "/results/exact-run"
    assert panel.gene.text() == "EAF1"
    assert panel.guides.text() == "EAF1_1, EAF1_2"
    assert panel.direction.currentText() == "negative"
    assert panel.score.currentText() == "infection_score"
    assert panel.gene.property("source_guide_agreement") == 0.8
    assert panel.gene.property("source_well_support") == 14


def test_hit_list_sends_the_exact_selected_result(qtbot):
    from spacr.hits import Hit, HitList
    from spacr.qt.screens.hit_list import HitListScreen
    screen = HitListScreen(threaded=False)
    qtbot.addWidget(screen)
    hit = Hit(gene="EAF1", effect=-0.42, q_value=0.01,
              agreeing_guides=("EAF1_1", "EAF1_2"), n_guides=3,
              agreement=2 / 3, n_obs=14, rank=1)
    listing = HitList(hits=(hit,), source="/results/exact-run")
    screen._shown = listing
    screen._fill_table(listing)
    screen._table.setCurrentItem(screen._table.topLevelItem(0))
    received = []
    screen.investigate_requested.connect(received.append)
    screen._on_investigate_selected()
    assert received == [{
        "folder": "/results/exact-run", "gene": "EAF1", "effect": -0.42,
        "guides": ("EAF1_1", "EAF1_2"), "fdr": 0.01,
        "guide_agreement": 2 / 3, "n_guides": 3, "well_support": 14,
        "phenotype": "",
    }]
