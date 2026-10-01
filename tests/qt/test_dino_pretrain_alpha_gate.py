"""The Embeddings screen's Pretrain on these crops button is alpha-gated.

``EmbeddingsDinoPretrainButton`` is registered in
``spacr.settings.ALPHA_FEATURES``, so it is hidden until Preferences ->
"Show alpha features" is on. Hiding is display only: a checkpoint path given
while it is hidden still trains, and the result reaches the next Embed.
"""
from __future__ import annotations

import os

import numpy as np
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

    assert ALPHA_FEATURES[561]["widgets"] == ("EmbeddingsDinoPretrainButton",)
    button = screen.findChild(QPushButton, "EmbeddingsDinoPretrainButton")
    assert button is not None and button.isHidden()
    alpha["on"] = True
    _apply_alpha_widgets(screen)
    assert not button.isHidden()
    alpha["on"] = False
    _apply_alpha_widgets(screen)
    assert button.isHidden()


def test_a_run_started_while_hidden_reaches_the_next_embed(
        screen, monkeypatch, tmp_path):
    import spacr.embeddings as emb

    seen = []

    def fake(crops, path, **kwargs):
        seen.append((crops.shape, kwargs["channel_policy"], kwargs["epochs"]))
        return {"path": path, "epochs": kwargs["epochs"], "loss": [1.5, 1.2],
                "seconds": 1.0}

    monkeypatch.setattr(emb, "_dino_pretrain", fake)
    assert screen._pretrain_dino(str(tmp_path / "own.pt")) == ""
    screen.set_crops(np.zeros((3, 8, 8, 2), dtype=np.float32))
    assert screen._dino.isHidden()
    path = str(tmp_path / "own.pt")
    assert screen._pretrain_dino(path, epochs=2) == path
    assert seen == [((3, 8, 8, 2), "per_channel", 2)]
    assert screen.spec().backbone == emb._DINO_PREFIX + path
    assert "last loss 1.200" in screen._status.text()


def test_a_dismissed_save_dialog_starts_nothing_and_a_second_run_reuses_the_entry(
        screen, monkeypatch, tmp_path):
    """No path asks where to save; dismissing it starts nothing. Pretraining
    to the same file again chooses the entry already in the picker rather
    than adding a second one."""
    import spacr.embeddings as emb
    from PySide6.QtWidgets import QFileDialog

    runs = []
    monkeypatch.setattr(emb, "_dino_pretrain",
                        lambda crops, path, **kw: runs.append(path) or {
                            "path": path, "epochs": 1, "loss": [],
                            "seconds": 0.1})
    screen.set_crops(np.zeros((2, 8, 8, 1), dtype=np.float32))
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    assert screen._pretrain_dino() == ""
    assert runs == []
    path = str(tmp_path / "again.pt")
    assert screen._pretrain_dino(path, epochs=1) == path
    count = screen._foundation.count()
    assert screen._pretrain_dino(path, epochs=1) == path
    assert screen._foundation.count() == count
    assert screen.spec().backbone == emb._DINO_PREFIX + path
    assert runs == [path, path]
    assert "nan" in screen._status.text()
