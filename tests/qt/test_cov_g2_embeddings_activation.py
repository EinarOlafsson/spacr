"""Embeddings and activation screens when the user cancels or the folder is empty."""
from __future__ import annotations

import types

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog, QWidget  # noqa: E402

from spacr.qt.screens import activation as act  # noqa: E402
from spacr.qt.screens import embeddings as em  # noqa: E402


@pytest.fixture
def screen(qtbot):
    widget = em.EmbeddingsScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_cancelling_the_label_table_changes_nothing(screen, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    assert screen._choose_labels() == ""


def test_using_embeddings_before_embedding_says_so(screen):
    screen._frame = None
    screen._use_embeddings()
    assert "Embed the crops first" in screen._status.text()


def test_labels_that_cannot_be_scored_say_why(screen, monkeypatch):
    import spacr.embeddings as embeddings

    monkeypatch.setattr(embeddings, "_scored_encoder_entry",
                        lambda spec, frame, labels: types.SimpleNamespace(
                            metrics={}))
    screen._labels = {"0": "a"}
    screen._show_scorecard()
    assert "cannot be scored" in screen._status.text()


def test_a_dismissed_column_form_returns_none(screen, monkeypatch):
    from PySide6.QtWidgets import QDialog

    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Rejected)
    assert screen._ask_mil_columns(["wellID", "well_label", "f1"]) is None
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Accepted)
    chosen = screen._ask_mil_columns(["wellID", "well_label", "f1"])
    assert chosen["well_column"] == "wellID"
    assert chosen["label_column"] == "well_label"


def test_cancelling_the_well_label_table_learns_nothing(screen, monkeypatch,
                                                         tmp_path):
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    db = tmp_path / "measurements.db"
    db.write_bytes(b"")
    assert screen._learn_from_well_labels(str(db)) == ""


def test_the_counterfactual_viewer_needs_a_header_to_hang_on(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    assert act._add_counterfactual_viewer_button(widget) is None


def test_an_empty_counterfactual_folder_says_so(qtbot, tmp_path):
    dialog = act._counterfactual_viewer(str(tmp_path))
    qtbot.addWidget(dialog)
    assert dialog.rows == []


def test_the_viewer_button_opens_the_chosen_folder(qtbot, monkeypatch,
                                                  tmp_path):
    class _Header(QWidget):
        def add_trailing(self, widget):
            self.trailing = widget

    screen = QWidget()
    screen._header = _Header(screen)
    qtbot.addWidget(screen)
    button = act._add_counterfactual_viewer_button(screen)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    button.click()
    assert getattr(screen, "_counterfactual_viewer", None) is None
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    button.click()
    assert screen._counterfactual_viewer.rows == []
    screen._counterfactual_viewer.close()


def test_the_viewer_ignores_sequences_it_has_no_frames_for(qtbot, tmp_path):
    import pandas as pd

    pd.DataFrame({"cell": ["a", "b"], "flipped": ["True", "False"]}).to_csv(
        tmp_path / "counterfactual_cells.csv", index=False)
    np.save(tmp_path / "counterfactual_frames.npy",
            np.zeros((1, 3, 4, 4, 3), np.uint8))
    dialog = act._counterfactual_viewer(str(tmp_path))
    qtbot.addWidget(dialog)
    assert len(dialog.rows) == 2
    shown = len(dialog.strip_labels)
    dialog.listing.setCurrentRow(1)
    dialog.listing.setCurrentRow(0)
    assert len(dialog.strip_labels) == shown
