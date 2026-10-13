"""The Embeddings screen's labels, scorecard and Use-for picker are alpha.

Hidden until Preferences -> "Show alpha features" is on; labels chosen and a
use picked while hidden still drive the scorecard and the run.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.qt.screens import embeddings as em

pytestmark = pytest.mark.qt

_NAMES = ("EmbeddingsLabelsButton", "EmbeddingsUseForLabel",
          "EmbeddingsUseForPicker", "EmbeddingsUseForRun")


@pytest.fixture
def alpha(monkeypatch):
    from spacr.qt import preferences

    state = {"on": False}
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: state["on"])
    return state


@pytest.fixture
def screen(qtbot, alpha, monkeypatch):
    import spacr.embeddings as emb

    def fake(spec):
        def run(stack):
            return stack.mean(axis=(1, 2))
        run.in_channels = None
        return run

    monkeypatch.setattr(emb, "_foundation_encoder", fake)
    widget = em.EmbeddingsScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._foundation.setCurrentIndex(widget._foundation.findData("openphenom"))
    rng = np.random.default_rng(0)
    crops = np.concatenate([rng.random((10, 8, 8, 2)),
                            rng.random((10, 8, 8, 2)) + 3]).astype(np.float32)
    widget.set_crops(crops)
    widget.embed()
    return widget


def _labels(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame({"label": ["a"] * 10 + ["b"] * 10}).to_csv(path, index=False)
    return str(path)


def test_the_controls_follow_the_alpha_switch(screen, alpha):
    from PySide6.QtCore import QObject
    from spacr.qt.preferences import _apply_alpha_widgets
    from spacr.settings import ALPHA_FEATURES

    assert set(_NAMES) <= set(ALPHA_FEATURES[560]["widgets"])
    widgets = [screen.findChild(QObject, name) for name in _NAMES]
    assert all(w is not None and w.isHidden() for w in widgets)
    alpha["on"] = True
    _apply_alpha_widgets(screen)
    assert not any(w.isHidden() for w in widgets)


def test_labels_chosen_while_hidden_score_the_backbone(screen, tmp_path):
    assert screen._labels_button.isHidden()
    screen._choose_labels(_labels(tmp_path))
    card = screen._entry.metrics
    assert card["knn_accuracy"] == 1.0 and card["classes"] == 2
    assert "openphenom" in screen._entry.name
    assert "kNN accuracy 1.00" in screen._status.text()


def test_a_mismatched_label_table_is_refused(screen, tmp_path):
    path = tmp_path / "short.csv"
    pd.DataFrame({"label": ["a", "b"]}).to_csv(path, index=False)
    assert screen._choose_labels(str(path)) == ""
    assert "2 rows" in screen._status.text()


def test_the_classifier_runs_on_the_chosen_backbone(screen, tmp_path):
    screen._use_for.setCurrentIndex(screen._use_for.findData("classifier"))
    screen._use_embeddings()
    assert "labels first" in screen._status.text()
    screen._choose_labels(_labels(tmp_path))
    screen._use_embeddings()
    assert screen._status.text().startswith("Classifier on openphenom")


def test_the_image_umap_fills_the_preview(screen):
    pytest.importorskip("umap")
    screen._use_for.setCurrentIndex(screen._use_for.findData("umap"))
    screen._use_embeddings()
    assert list(screen._umap.columns) == ["umap_1", "umap_2"]
    assert len(screen._umap) == 20
    assert screen._table.columnCount() == 2
