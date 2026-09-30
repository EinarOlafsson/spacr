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

    def fake(frame):
        seen.append(len(frame))
        wells = frame.groupby("wellID", as_index=False)["well_label"].first()
        wells["mil_probability"] = wells["well_label"].astype(float)
        cells = frame.assign(mil_attention=1.0)
        return cells, wells, {"mil_auroc": 1.0, "mean_auroc": 0.5,
                              "mean_sd_auroc": 0.5, "wells": 2.0}

    monkeypatch.setattr(emb, "_mil_from_table", fake)
    assert screen._well_mil.isHidden()
    assert screen._learn_from_well_labels(str(source)) == str(source)
    assert seen == [4]
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
