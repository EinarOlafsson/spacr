"""Cell-DINO alpha configuration preserves explicit model-plane identity."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QLabel, QLineEdit

from spacr.qt.screens import embeddings as screen_module

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot, monkeypatch):
    from spacr.qt import preferences

    state = {"alpha": True}
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: state["alpha"])
    widget = screen_module.EmbeddingsScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_crops(np.stack([
        np.full((20, 20), channel + 1, dtype=np.float32)
        for channel in range(5)], axis=-1)[None].repeat(2, axis=0))
    widget._foundation.setCurrentIndex(
        widget._foundation.findData("cell_dino"))
    return widget, state


def _dialog_parts(widget):
    dialog = widget._cell_dino_dialog()
    factory = dialog.findChild(QComboBox, "EmbeddingsCellDinoFactory")
    planes = tuple(dialog.findChild(
        QComboBox, f"EmbeddingsCellDinoPlane{index}")
        for index in range(1, 6))
    path = dialog.findChild(QLineEdit, "EmbeddingsCellDinoCheckpointPath")
    digest = dialog.findChild(QLineEdit, "EmbeddingsCellDinoDigest")
    return dialog, factory, planes, path, digest


def _save(widget, checkpoint, *, factory="cell_dino_cp_vits8",
          channels=(4, 2, 0, 3, 1)):
    dialog, chosen_factory, planes, path, digest = _dialog_parts(widget)
    try:
        chosen_factory.setCurrentIndex(chosen_factory.findData(factory))
        path.setText(str(checkpoint))
        digest.setText("a" * 64)
        for selector, channel in zip(planes, channels):
            selector.setCurrentIndex(selector.findData(channel))
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
        assert dialog.result() == QDialog.Accepted
    finally:
        dialog.deleteLater()


def test_no_factory_or_plane_is_guessed_and_alpha_registration(screen):
    from spacr.embeddings import EmbeddingError
    from spacr.settings import ALPHA_FEATURES

    widget, state = screen
    names = set(ALPHA_FEATURES[560]["widgets"])
    assert widget._cell_dino_button.objectName() in names
    assert widget._cell_dino_button.isEnabled()
    assert not widget._cell_dino_button.isHidden()
    with pytest.raises(EmbeddingError, match="Choose an official"):
        widget.spec()
    widget.embed()
    assert "choose an official" in widget._status.text().lower()
    assert widget._result is None
    dialog, factory, planes, path, digest = _dialog_parts(widget)
    try:
        assert dialog.objectName() in names
        assert factory.currentData() == ""
        assert [plane.currentData() for plane in planes] == [None] * 5
        for control in (factory, path, digest, *planes):
            assert control.objectName() in names
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
        assert dialog.result() != QDialog.Accepted
        assert dialog.findChild(QLabel, "EmbeddingsCellDinoProblem").text()
    finally:
        dialog.deleteLater()
    state["alpha"] = False
    widget._sync_subcell_controls()
    assert widget._cell_dino_button.isHidden()


def test_declared_five_plane_order_reaches_encoder_and_persists(
        screen, tmp_path, monkeypatch):
    import spacr.embeddings as emb

    widget, state = screen
    checkpoint = tmp_path / "official.pth"
    checkpoint.write_bytes(b"fixture; actual checkpoint is not loaded")
    _save(widget, checkpoint)
    chosen = widget.spec()
    assert chosen.channels == (4, 2, 0, 3, 1)
    assert chosen.channel_policy == emb.CHANNEL_PROJECT
    assert chosen.cell_dino_factory == "cell_dino_cp_vits8"
    assert chosen.checkpoint_path == str(checkpoint)
    assert chosen.checkpoint_sha256 == "a" * 64
    reopened, reopened_factory, reopened_planes, reopened_path, reopened_digest = (
        _dialog_parts(widget))
    try:
        assert reopened_factory.currentData() == chosen.cell_dino_factory
        assert [plane.currentData() for plane in reopened_planes] == list(
            chosen.channels)
        assert reopened_path.text() == chosen.checkpoint_path
        assert reopened_digest.text() == chosen.checkpoint_sha256
    finally:
        reopened.deleteLater()
    seen = []

    def fake_encoder(spec):
        assert spec.cell_dino_factory == chosen.cell_dino_factory
        assert spec.channels == chosen.channels
        assert spec.checkpoint_sha256 == chosen.checkpoint_sha256
        assert spec.channel_scale == (5, 3, 1, 4, 2)

        def run(stack):
            seen.append(stack.copy())
            return stack.mean(axis=(1, 2))

        run.in_channels = 5
        return run

    monkeypatch.setattr(emb, "_foundation_encoder", fake_encoder)
    widget.embed()
    assert len(seen) == 1, widget._status.text()
    np.testing.assert_allclose(seen[0][0, 0, 0], [1, 1, 1, 1, 1])
    assert widget._result is not None
    state["alpha"] = False
    widget._sync_subcell_controls()
    assert widget.spec() == chosen


def test_new_crop_width_invalidates_mapping_without_losing_weights(
        screen, tmp_path):
    from spacr.embeddings import EmbeddingError

    widget, _ = screen
    checkpoint = tmp_path / "official.pth"
    checkpoint.write_bytes(b"fixture")
    _save(widget, checkpoint)
    widget.set_crops(np.ones((1, 20, 20, 4), dtype=np.float32))
    assert widget._cell_dino_channels is None
    assert widget._cell_dino_checkpoint == str(checkpoint)
    with pytest.raises(EmbeddingError, match="crop channel"):
        widget.spec()
    widget.embed()
    assert widget._result is None


def test_hpa_four_plane_choice_rejects_repeat_and_keeps_old_models(
        screen, tmp_path):
    widget, _ = screen
    checkpoint = tmp_path / "official.pth"
    checkpoint.write_bytes(b"fixture")
    dialog, factory, planes, path, digest = _dialog_parts(widget)
    try:
        factory.setCurrentIndex(factory.findData("cell_dino_hpa_vitl16"))
        assert planes[4].isHidden()
        path.setText(str(checkpoint))
        digest.setText("c" * 64)
        for selector, channel in zip(planes, (1, 1, 2, 3)):
            selector.setCurrentIndex(selector.findData(channel))
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
        assert dialog.result() != QDialog.Accepted
        assert "different" in dialog.findChild(
            QLabel, "EmbeddingsCellDinoProblem").text().lower()
        planes[1].setCurrentIndex(planes[1].findData(0))
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
        assert dialog.result() == QDialog.Accepted
    finally:
        dialog.deleteLater()
    assert widget.spec().channels == (1, 0, 2, 3)
    widget._foundation.setCurrentIndex(widget._foundation.findData("subcell"))
    assert widget.spec().backbone == "subcell"
    assert widget._policy.isEnabled()


def test_missing_file_bad_digest_and_stale_channel_refuse_before_job(
        screen, tmp_path):
    widget, _ = screen
    checkpoint = tmp_path / "official.pth"
    checkpoint.write_bytes(b"fixture")
    _save(widget, checkpoint, factory="cell_dino_hpa_vitl16",
          channels=(0, 1, 2, 3))
    widget._cell_dino_channels = (0, 1, 2, 8)
    widget.embed()
    assert "outside" in widget._status.text().lower()
    assert widget._result is None
    widget._cell_dino_channels = (0, 1, 2, 3)
    widget._cell_dino_checkpoint = str(tmp_path / "missing.pth")
    widget.embed()
    assert "local official" in widget._status.text().lower()
    widget._cell_dino_checkpoint = str(checkpoint)
    widget._cell_dino_digest = "not-a-digest"
    widget.embed()
    assert "64-character" in widget._status.text()
    assert widget._result is None


def test_browse_keeps_cancel_empty_and_dialog_disposes_after_close(
        screen, tmp_path, monkeypatch, qtbot):
    from PySide6.QtWidgets import QFileDialog, QPushButton
    import shiboken6

    widget, _ = screen
    checkpoint = tmp_path / "official.pth"
    checkpoint.write_bytes(b"fixture")
    dialog, _, _, path, _ = _dialog_parts(widget)
    browse = dialog.findChild(QPushButton, "EmbeddingsCellDinoBrowse")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a: ("", ""))
    browse.click()
    assert path.text() == ""
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a: (str(checkpoint), ""))
    browse.click()
    assert path.text() == str(checkpoint)
    dialog.deleteLater()

    closing = widget._cell_dino_dialog()
    monkeypatch.setattr(widget, "_cell_dino_dialog", lambda: closing)
    monkeypatch.setattr(closing, "exec", closing.reject)
    widget._choose_cell_dino()
    qtbot.waitUntil(lambda: not shiboken6.isValid(closing))
    assert widget._cell_dino_channels is None
