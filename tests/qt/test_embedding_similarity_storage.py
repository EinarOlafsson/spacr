"""Database crop identity survives control changes and asynchronous results."""

import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")

from spacr import active_learning as al
from spacr import embeddings as emb
from spacr.qt import preferences
from spacr.qt.screens.embeddings import EmbeddingsScreen


@pytest.fixture
def plate(tmp_path):
    database = tmp_path / "measurements" / "measurements.db"
    database.parent.mkdir()
    data = tmp_path / "data" / "cell_png"
    data.mkdir(parents=True)
    rows = []
    for index in range(4):
        path = data / f"cell{index}.png"
        Image.fromarray(np.full((16, 16, 3), index * 40, np.uint8)).save(path)
        rows.append((str(path), f"plate1_r1_c1_f1_o{index + 1}"))
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE png_list (png_path TEXT, prcfo TEXT)")
        db.executemany("INSERT INTO png_list VALUES (?, ?)", rows)
    return database, rows


@pytest.fixture
def screen(qtbot, monkeypatch):
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    monkeypatch.setattr(emb, "encoder_entry", lambda spec: SimpleNamespace(
        name=spec.backbone, sha256=""))

    def encode(crops, spec, *, record):
        values = crops.mean(axis=(1, 2)).astype(np.float32)
        return emb.EmbeddingResult(values, ("emb_0", "emb_1", "emb_2"),
                                   spec, 3, 3)

    monkeypatch.setattr(emb, "_embed_plate", encode)
    widget = EmbeddingsScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _load(screen, database):
    screen._path.setText(str(database))
    screen._on_path_changed()
    screen._load.click()
    assert getattr(screen, "_crops", None) is not None, screen._status.text()
    assert len(screen._crops) == 4


def test_real_load_embed_save_uses_original_database_keys_and_spec(screen, plate, tmp_path):
    database, rows = plate
    _load(screen, database)
    screen._run.click()
    spec = screen._result.spec
    assert screen._save_similar.isEnabled()
    screen._path.setText(str(tmp_path / "other.db"))
    screen._backbone.setCurrentText("a-different-backbone")
    screen._save_similar.click()
    stored = al._stored_embedding_frame(str(database))
    assert stored["prcfo"].tolist() == [row[1] for row in rows]
    assert set(stored["_embedding_fingerprint"]) == {spec.fingerprint()}
    assert json.loads(stored["_embedding_spec"].iloc[0])["backbone"] == spec.backbone
    joined = al._stored_embeddings(str(database))
    assert joined.index.tolist() == [row[0] for row in rows]
    assert joined.columns.tolist() == ["emb_0", "emb_1", "emb_2"]
    assert len(al._similarity_index(str(database), backend="numpy")) == 4
    assert "Saved 4" in screen._status.text()
    assert not (tmp_path / "other.db").exists()


def test_unsourced_crops_and_replaced_load_cannot_save_old_vectors(screen, plate):
    _load(screen, plate[0])
    screen.embed()
    screen.set_crops(np.zeros((2, 16, 16, 3), np.uint8))
    assert not screen._save_similar.isEnabled()
    screen._save_for_similarity()
    assert al._stored_embedding_frame(str(plate[0])) is None
    assert "Load database crops" in screen._status.text()


def test_late_embedding_from_previous_crop_selection_is_ignored(screen, plate, monkeypatch):
    _load(screen, plate[0])
    pending = []
    monkeypatch.setattr(screen._jobs, "submit", lambda work, done: pending.append((work, done)))
    screen.embed()
    work, done = pending[0]
    screen.set_crops(np.ones((2, 16, 16, 3), np.uint8))
    done(work())
    assert screen._result is None
    assert not screen._save_similar.isEnabled()


def test_save_failure_keeps_result_available_for_retry(screen, plate, monkeypatch):
    _load(screen, plate[0])
    screen.embed()
    original = al._store_crop_embeddings
    monkeypatch.setattr(al, "_store_crop_embeddings", lambda *_args: (_ for _ in ()).throw(
        OSError("database is read-only")))
    screen._save_similar.click()
    assert "read-only" in screen._status.text()
    assert screen._save_similar.isEnabled()
    monkeypatch.setattr(al, "_store_crop_embeddings", original)
    screen._save_similar.click()
    assert len(al._stored_embedding_frame(str(plate[0]))) == 4


def test_save_control_is_alpha_gated(screen, monkeypatch):
    from spacr.settings import ALPHA_FEATURES

    assert "EmbeddingsSaveForSimilarity" in ALPHA_FEATURES[565]["widgets"]
    assert not screen._save_similar.isHidden()
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: False)
    preferences._apply_alpha_widgets(screen)
    assert screen._save_similar.isHidden()


def test_same_width_different_encoder_replaces_instead_of_mixing_rows(plate):
    database, rows = plate

    def result(backbone):
        return emb.EmbeddingResult(np.array([[1, 2, 3]], np.float32),
                                   ("emb_0", "emb_1", "emb_2"),
                                   emb.EmbeddingSpec(backbone=backbone), 3, 3)

    first = result("resnet18")
    assert al._store_crop_embeddings(str(database), [rows[0][1]], first) == 1
    assert al._store_crop_embeddings(str(database), [rows[1][1]], first) == 2
    second = result("resnet50")
    assert al._store_crop_embeddings(str(database), [rows[2][1]], second) == 1
    assert al._stored_embedding_frame(str(database))["prcfo"].tolist() == [rows[2][1]]
    assert al._store_crop_embeddings(str(database), [rows[3][1]], second.values) == 1
    assert "_embedding_spec" not in al._stored_embedding_frame(str(database))
