"""Reducer-specific Image UMAP settings stay explicit, typed and preserved."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from spacr.hyperparam import UMAP_METRICS
from spacr.qt.screens.settings_model import SettingsWidgets


def _model(qtbot):
    model = SettingsWidgets("umap")
    model.build_sections()
    qtbot.addWidget(model._widgets["reduction_method"])
    return model


def test_reducer_and_metric_are_closed_dropdowns(qtbot):
    model = _model(qtbot)
    reducer = model._widgets["reduction_method"]
    metric = model._widgets["metric"]
    assert isinstance(reducer, QComboBox)
    assert isinstance(metric, QComboBox)
    assert [reducer.itemText(i) for i in range(reducer.count())] == [
        "umap", "tsne", "pca", "isomap", "spectral"]
    assert [metric.itemText(i) for i in range(metric.count())] == \
        list(UMAP_METRICS)
    assert metric.currentText() == "euclidean"


def test_switching_reducers_greys_only_inactive_families_and_keeps_values(
        qtbot):
    model = _model(qtbot)
    model.set_value_for_key("n_neighbors", 73)
    model.set_value_for_key("reduction_method", "tsne")
    assert not model._widgets["n_neighbors"].isEnabled()
    assert not model._widgets["min_dist"].isEnabled()
    assert model._widgets["tsne_perplexity"].isEnabled()
    assert not model._widgets["pca_whiten"].isEnabled()
    assert model._widgets["metric"].isEnabled()  # DBSCAN still reads it

    model.set_value_for_key("reduction_method", "pca")
    assert model._widgets["pca_whiten"].isEnabled()
    assert model._widgets["pca_svd_solver"].isEnabled()
    assert not model._widgets["tsne_perplexity"].isEnabled()
    model.set_value_for_key("reduction_method", "umap")
    assert model.collect()["n_neighbors"] == 73
    assert model._widgets["n_neighbors"].isEnabled()


def test_spectral_neighbor_setting_tracks_the_affinity_dropdown(qtbot):
    model = _model(qtbot)
    model.set_value_for_key("reduction_method", "spectral")
    assert model._widgets["spectral_n_neighbors"].isEnabled()
    model.set_value_for_key("spectral_affinity", "rbf")
    assert not model._widgets["spectral_n_neighbors"].isEnabled()


def test_gpu_is_one_hidden_but_collected_run_setting(qtbot):
    model = _model(qtbot)
    assert "gpu" not in model._widgets
    assert model.set_hidden_value("gpu", True)
    assert model.collect()["gpu"] is True
