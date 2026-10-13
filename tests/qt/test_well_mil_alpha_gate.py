"""The Embeddings screen's Learn from well labels button is alpha-gated.

``EmbeddingsWellMilButton`` is registered in ``spacr.settings.ALPHA_FEATURES``,
so it is hidden until Preferences -> "Show alpha features" is on. Hiding is
display only: the action still runs on a table given while it is hidden.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.qt.screens import embeddings as em

pytestmark = pytest.mark.qt


@pytest.fixture
def alpha(monkeypatch):
    from spacr.qt import preferences

    state = {"on": False}
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: state["on"])
    return state


@pytest.fixture
def screen(qtbot, alpha):
    widget = em.EmbeddingsScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_the_button_follows_the_alpha_switch(screen, alpha):
    from PySide6.QtWidgets import QPushButton

    from spacr.qt.preferences import _apply_alpha_widgets
    from spacr.settings import ALPHA_FEATURES

    assert ALPHA_FEATURES[562]["widgets"] == ("EmbeddingsWellMilButton",)
    button = screen.findChild(QPushButton, "EmbeddingsWellMilButton")
    assert button is not None and button.isHidden()
    alpha["on"] = True
    _apply_alpha_widgets(screen)
    assert not button.isHidden()
    alpha["on"] = False
    _apply_alpha_widgets(screen)
    assert button.isHidden()


def test_a_table_given_while_hidden_still_reaches_the_run(
        screen, monkeypatch, tmp_path):
    import spacr.embeddings as emb
    from spacr.tabular import read_table, write_table

    source = tmp_path / "cells.csv"
    write_table(__import__("pandas").DataFrame(
        {"well": ["A1", "A1", "B1", "B1"], "well_label": [0, 0, 1, 1],
         "emb_0": [0.1, 0.2, 0.9, 1.0]}), source)
    seen = []

    def fake(frame, **columns):
        seen.append((len(frame), columns["well_column"]))
        wells = frame.groupby("wellID", as_index=False)["well_label"].first()
        wells["mil_probability"] = wells["well_label"].astype(float)
        cells = frame.assign(mil_attention=1.0)
        return cells, wells, {"mil_auroc": 1.0, "mean_auroc": 0.5,
                              "mean_sd_auroc": 0.5, "wells": 2.0}

    monkeypatch.setattr(emb, "_mil_from_table", fake)
    assert screen._well_mil.isHidden()
    assert screen._learn_from_well_labels(
        str(source), columns={"well_column": "wellID",
                              "label_column": "well_label",
                              "feature_columns": ["emb_0"]}) == str(source)
    assert seen == [(4, "wellID")]
    assert screen._mil_card["mil_auroc"] == 1.0
    cells = read_table(tmp_path / "cells_mil_cells.csv", report=None)
    assert "mil_attention" in cells.columns
    assert (tmp_path / "cells_mil_wells.csv").exists()
    assert "AUROC 1.00" in screen._status.text()


def test_a_dismissed_table_dialog_starts_nothing(screen, monkeypatch):
    import spacr.embeddings as emb
    from PySide6.QtWidgets import QFileDialog

    called = []
    monkeypatch.setattr(emb, "_mil_from_table", called.append)
    asked = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: asked.append(a) or ("", "")))
    assert screen._learn_from_well_labels() == ""
    assert len(asked) == 1 and called == []


def test_the_form_defaults_to_the_usual_columns(screen):
    dialog, well, label, features = screen._mil_column_form(
        ["plate", "well", "well_label", "emb_0", "emb_1", "area"])
    assert dialog.objectName() == "EmbeddingsWellMilForm"
    assert well.currentText() == "well"
    assert label.currentText() == "well_label"
    assert sorted(i.text() for i in features.selectedItems()) == [
        "emb_0", "emb_1"]


def test_a_dismissed_form_starts_nothing(screen, monkeypatch, tmp_path):
    import pandas as pd

    import spacr.embeddings as emb
    from spacr.tabular import write_table

    source = tmp_path / "cells.csv"
    write_table(pd.DataFrame({"well": ["A1"], "well_label": [0],
                              "emb_0": [0.1]}), source)
    called = []
    monkeypatch.setattr(emb, "_mil_from_table",
                        lambda *a, **k: called.append(a))
    monkeypatch.setattr(screen, "_ask_mil_columns", lambda columns: None)
    assert screen._learn_from_well_labels(str(source)) == ""
    assert called == []


def test_chosen_columns_reach_the_model(screen, tmp_path):
    import numpy as np
    import pandas as pd
    from PySide6.QtGui import QColor, QImage

    from spacr.embeddings import _synthetic_mil_bags

    bags, labels = _synthetic_mil_bags(wells=8, cells=10, features=3)[:2]
    rows = []
    for w, (bag, lab) in enumerate(zip(bags, labels)):
        for c, v in enumerate(bag):
            crop = tmp_path / f"{w}_{c}.png"
            image = QImage(8, 8, QImage.Format_RGB32)
            image.fill(QColor(int(w * 20), 0, 0))
            image.save(str(crop))
            rows.append({"site": f"W{w}", "group": "hit" if lab else "ctl",
                         "f0": v[0], "f1": v[1], "f2": v[2],
                         "png_path": str(crop)})
    source = tmp_path / "cells.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    screen._learn_from_well_labels(str(source), columns={
        "well_column": "site", "label_column": "group",
        "feature_columns": ["f0", "f1", "f2"], "positive": "hit",
        "folds": 2})
    assert np.isfinite(screen._mil_card["mil_auroc"])
    assert screen._mil_result.findChild(
        object, "EmbeddingsWellMilScorecard").rowCount() == len(
            screen._mil_card)
    assert screen._mil_montage_count == 24
    assert not screen._mil_result.findChild(
        object, "EmbeddingsWellMilMontage").pixmap().isNull()


def test_a_database_reaches_the_model_through_its_stored_embeddings(
        screen, monkeypatch, tmp_path):
    import spacr.embeddings as emb
    from tests.test_multiple_instance_learning import _embedded_project
    from spacr.tabular import write_table

    db, labels = _embedded_project(tmp_path)
    table = tmp_path / "well_labels.csv"
    write_table(labels, table)
    seen = []

    def fake(frame, **columns):
        seen.append((len(frame), sorted(frame["well_label"].unique()),
                     "prcfo" in frame.columns))
        return frame.assign(mil_attention=1.0), frame.head(1), {
            "mil_auroc": 1.0, "mean_auroc": 0.5, "wells": 8.0}

    monkeypatch.setattr(emb, "_mil_from_table", fake)
    assert screen._well_mil.isHidden()
    assert screen._learn_from_well_labels(
        db, columns={"well_column": "wellID", "label_column": "well_label",
                     "feature_columns": None}, labels=str(table)) == db
    assert seen == [(96, [0, 1], True)]
    assert os.path.exists(os.path.splitext(db)[0] + "_mil_cells.csv")


def test_a_database_without_embeddings_says_so(screen, tmp_path):
    from tests.test_cov_active_learning_rounds import _make_project

    db = _make_project(tmp_path)["db"]
    assert screen._learn_from_well_labels(
        db, columns={}, labels=str(tmp_path / "unused.csv")) == ""
    assert "no stored crop embeddings" in screen._status.text()
