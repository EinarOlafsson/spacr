"""The Embeddings screen's Foundation model picker is alpha-gated.

``EmbeddingsFoundationLabel`` and ``EmbeddingsFoundationPicker`` are
registered in ``spacr.settings.ALPHA_FEATURES``, so they are hidden until
Preferences -> "Show alpha features" is on. Hiding is display only: a model
chosen while the picker is hidden still names the backbone of the run.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.qt.screens import embeddings as em

pytestmark = pytest.mark.qt

_NAMES = ("EmbeddingsFoundationLabel", "EmbeddingsFoundationPicker")


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


def _widgets(screen):
    from PySide6.QtCore import QObject

    return [screen.findChild(QObject, name) for name in _NAMES]


def test_the_picker_follows_the_alpha_switch(screen, alpha):
    from spacr.qt.preferences import _apply_alpha_widgets
    from spacr.settings import ALPHA_FEATURES

    assert set(_NAMES) <= set(ALPHA_FEATURES[560]["widgets"])
    widgets = _widgets(screen)
    assert all(w is not None and w.isHidden() for w in widgets)
    alpha["on"] = True
    _apply_alpha_widgets(screen)
    assert not any(w.isHidden() for w in widgets)
    alpha["on"] = False
    _apply_alpha_widgets(screen)
    assert all(w.isHidden() for w in widgets)


def test_the_picker_offers_every_foundation_model_and_none_first(screen):
    from spacr.embeddings import _foundation_names

    picker = screen._foundation
    offered = [picker.itemData(i) for i in range(picker.count())]
    assert offered == [""] + list(_foundation_names())
    assert screen.spec().backbone == screen._backbone.currentText().strip()


def test_a_model_chosen_while_hidden_still_reaches_the_run(
        screen, monkeypatch):
    import spacr.embeddings as emb

    picker = screen._foundation
    assert picker.isHidden()
    picker.setCurrentIndex(picker.findData("openphenom"))
    assert screen.spec().backbone == "openphenom"

    seen = []

    def fake(spec):
        seen.append(spec.backbone)

        def run(stack):
            return stack.mean(axis=(1, 2))

        run.in_channels = None
        return run

    monkeypatch.setattr(emb, "_foundation_encoder", fake)
    screen.set_crops(np.random.default_rng(0).random((5, 8, 8, 2),
                                                     dtype=np.float32))
    screen.embed()
    assert seen == ["openphenom"]
    assert screen._result.values.shape == (5, 2)
