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


def _four_plane_screen(screen, alpha):
    from spacr.qt.preferences import _apply_alpha_widgets

    alpha["on"] = True
    _apply_alpha_widgets(screen)
    index = screen._foundation.findData("subcell_rybg")
    assert index >= 0
    screen._foundation.setCurrentIndex(index)
    return screen


def _mapping_form(screen):
    from PySide6.QtWidgets import QComboBox

    dialog = screen._subcell_channels_dialog()
    names = (
        "EmbeddingsSubCellMicrotubulesChannel",
        "EmbeddingsSubCellErChannel",
        "EmbeddingsSubCellDnaChannel",
        "EmbeddingsSubCellProteinChannel",
    )
    selectors = tuple(dialog.findChild(QComboBox, name) for name in names)
    assert all(selector is not None for selector in selectors)
    return dialog, selectors


def _save_mapping(screen, channels):
    from PySide6.QtWidgets import QDialog, QDialogButtonBox

    dialog, selectors = _mapping_form(screen)
    try:
        for selector, channel in zip(selectors, channels):
            selector.setCurrentIndex(selector.findData(channel))
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
        assert dialog.result() == QDialog.Accepted
    finally:
        dialog.deleteLater()


def test_four_plane_mapping_starts_without_a_guessed_stain(screen, alpha):
    from PySide6.QtWidgets import QDialogButtonBox, QLabel
    from spacr.embeddings import EmbeddingError
    from spacr.settings import ALPHA_FEATURES

    screen.set_crops(np.zeros((2, 16, 16, 5), dtype=np.float32))
    _four_plane_screen(screen, alpha)
    assert screen._subcell_button.isEnabled()
    assert not screen._subcell_button.isHidden()
    assert screen._policy.currentData() == "project"
    assert "four" in screen._policy.currentText().lower()
    assert not screen._policy.isEnabled()
    with pytest.raises(EmbeddingError, match="Choose four distinct"):
        screen.spec()

    dialog, selectors = _mapping_form(screen)
    try:
        assert dialog.objectName() in ALPHA_FEATURES[560]["widgets"]
        assert all(selector.objectName() in ALPHA_FEATURES[560]["widgets"]
                   for selector in selectors)
        assert [selector.currentData() for selector in selectors] == [None] * 4
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
        problem = dialog.findChild(QLabel, "EmbeddingsSubCellChannelsProblem")
        assert "choose a crop channel" in problem.text().lower()
        assert screen._subcell_channels is None
    finally:
        dialog.deleteLater()


def test_four_plane_mapping_routes_reordered_raw_planes_and_preserves_origin(
        screen, alpha, monkeypatch):
    import spacr.embeddings as emb
    from types import SimpleNamespace

    crops = np.zeros((2, 16, 16, 5), dtype=np.float32)
    for channel in range(5):
        crops[..., channel] = channel + 1
    screen.set_crops(crops)
    _four_plane_screen(screen, alpha)
    _save_mapping(screen, (3, 1, 4, 0))
    spec = screen.spec()
    assert spec.backbone == "subcell_rybg"
    assert spec.channels == (3, 1, 4, 0)
    assert spec.channel_policy == emb.CHANNEL_PROJECT
    assert spec.normalize is False
    original_fingerprint = spec.fingerprint()

    seen = []

    def fake_encoder(chosen):
        assert chosen.fingerprint() == original_fingerprint

        def run(stack):
            seen.append(stack.copy())
            return stack.mean(axis=(1, 2))

        run.in_channels = 4
        return run

    monkeypatch.setattr(emb, "_foundation_encoder", fake_encoder)
    screen.embed()
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], crops[..., [3, 1, 4, 0]])
    assert screen._result.spec.fingerprint() == original_fingerprint

    _save_mapping(screen, (1, 3, 4, 0))
    assert screen.spec().fingerprint() != original_fingerprint
    assert screen._result.spec.fingerprint() == original_fingerprint

    scored = []
    monkeypatch.setattr(emb, "_scored_encoder_entry",
                        lambda source, frame, labels: (
                            scored.append(source),
                            SimpleNamespace(metrics={}),
                        )[1])
    screen._labels = {"0": "one", "1": "two"}
    screen._show_scorecard()
    assert scored[0].fingerprint() == original_fingerprint
    screen._on_used(("classifier", {
        "accuracy": 0.5, "accuracy_sd": 0.1, "folds": 2,
        "chance": 0.5, "n": 2, "classes": 2,
    }))
    assert "subcell_rybg" in screen._status.text()


def test_duplicate_or_missing_mapping_never_submits_a_job(screen, alpha,
                                                          monkeypatch):
    from PySide6.QtWidgets import QDialogButtonBox, QLabel

    screen.set_crops(np.zeros((2, 16, 16, 4), dtype=np.float32))
    _four_plane_screen(screen, alpha)
    submitted = []
    monkeypatch.setattr(screen._jobs, "submit", lambda *a: submitted.append(a))
    screen.embed()
    assert "choose" in screen._status.text().lower()
    assert not submitted

    dialog, selectors = _mapping_form(screen)
    try:
        for selector, channel in zip(selectors, (0, 0, 1, 2)):
            selector.setCurrentIndex(selector.findData(channel))
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
        problem = dialog.findChild(QLabel, "EmbeddingsSubCellChannelsProblem")
        assert "four different" in problem.text().lower()
        assert screen._subcell_channels is None
    finally:
        dialog.deleteLater()
    screen.embed()
    assert not submitted
    screen._subcell_channels = (True, 1, 2, 3)
    screen.embed()
    assert not submitted
    assert "choose a crop channel" in screen._status.text().lower()


def test_reload_keeps_only_still_valid_mapping_and_alpha_hides_only_the_control(
        screen, alpha, monkeypatch):
    from spacr.qt.preferences import _apply_alpha_widgets

    screen.set_crops(np.zeros((2, 16, 16, 5), dtype=np.float32))
    _four_plane_screen(screen, alpha)
    _save_mapping(screen, (4, 2, 0, 1))
    fingerprint = screen.spec().fingerprint()
    screen.set_crops(np.zeros((2, 16, 16, 6), dtype=np.float32))
    assert screen.spec().fingerprint() == fingerprint

    alpha["on"] = False
    _apply_alpha_widgets(screen)
    assert screen._foundation.isHidden()
    assert screen._subcell_button.isHidden()
    assert screen.spec().fingerprint() == fingerprint
    alpha["on"] = True
    _apply_alpha_widgets(screen)
    assert not screen._foundation.isHidden()
    assert not screen._subcell_button.isHidden()

    screen.set_crops(np.zeros((2, 16, 16, 4), dtype=np.float32))
    assert screen._subcell_channels is None
    submitted = []
    monkeypatch.setattr(screen._jobs, "submit", lambda *a: submitted.append(a))
    screen.embed()
    assert not submitted
    assert "choose" in screen._status.text().lower()


def test_other_foundation_models_keep_their_policy_and_two_channel_inputs(
        screen, alpha, monkeypatch):
    import spacr.embeddings as emb

    screen.set_crops(np.zeros((2, 8, 8, 2), dtype=np.float32))
    _four_plane_screen(screen, alpha)
    assert not screen._subcell_button.isEnabled()
    screen.embed()
    assert "four" in screen._status.text().lower()

    screen._foundation.setCurrentIndex(screen._foundation.findData("subcell"))
    old_subcell = screen.spec()
    assert old_subcell.backbone == "subcell"
    assert old_subcell.channels is None
    assert old_subcell.normalize is True
    assert old_subcell.channel_policy == "per_channel"
    assert screen._policy.isEnabled()
    assert screen._subcell_button.isHidden()
    assert "three" in screen._policy.itemText(
        screen._policy.findData("project")).lower()

    screen._foundation.setCurrentIndex(
        screen._foundation.findData("openphenom"))
    assert screen.spec().channels is None
    assert screen.spec().normalize is True
    assert screen._subcell_button.isHidden()

    observed = []

    def variable_encoder(spec):
        assert spec.backbone == "openphenom"

        def encode(stack):
            observed.append(stack.shape[-1])
            return stack.mean(axis=(1, 2))

        encode.in_channels = None
        return encode

    monkeypatch.setattr(emb, "_foundation_encoder", variable_encoder)
    screen.set_crops(np.ones((2, 8, 8, 5), dtype=np.float32))
    screen.embed()
    assert observed == [1] * 5
    assert screen._result.values.shape == (2, 5)


def test_cancelled_mapping_disposes_its_dialog_without_changing_saved_indices(
        screen, alpha, monkeypatch):
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication, QComboBox
    from shiboken6 import isValid

    screen.set_crops(np.zeros((2, 16, 16, 4), dtype=np.float32))
    _four_plane_screen(screen, alpha)
    _save_mapping(screen, (0, 1, 2, 3))
    original = screen._subcell_channels_dialog
    created = []

    def cancel_after_open():
        dialog = original()
        created.append(dialog)
        return dialog

    def cancel_exec(dialog):
        selector = dialog.findChild(
            QComboBox, "EmbeddingsSubCellMicrotubulesChannel")
        selector.setCurrentIndex(selector.findData(3))
        dialog.reject()
        return dialog.result()

    monkeypatch.setattr(screen, "_subcell_channels_dialog", cancel_after_open)
    monkeypatch.setattr(em.QDialog, "exec", cancel_exec)
    screen._subcell_button.click()
    assert screen._subcell_channels == (0, 1, 2, 3)
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert len(created) == 1 and not isValid(created[0])
