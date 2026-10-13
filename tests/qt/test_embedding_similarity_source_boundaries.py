"""Database crop vectors stay attached to their actual loaded object keys."""

import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")

from spacr import active_learning as al
from spacr import embeddings as emb
from spacr.crop_loader import CropQuery
from spacr.qt import preferences
from spacr.qt.screens.embeddings import EmbeddingsScreen


def _plate(tmp_path, keys):
    database = tmp_path / "measurements.db"
    images = tmp_path / "cell_png"
    images.mkdir()
    rows = []
    for index, key in enumerate(keys):
        path = images / f"cell{index}.png"
        Image.fromarray(np.full((16, 16, 3), 30 + index * 20,
                                np.uint8)).save(path)
        rows.append((str(path), key))
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE png_list (png_path TEXT, prcfo TEXT)")
        connection.executemany("INSERT INTO png_list VALUES (?, ?)", rows)
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
    assert screen._crops is not None, screen._status.text()


def test_stopped_database_load_saves_only_the_completed_page(screen, tmp_path,
                                                              monkeypatch):
    keys = [f"plate1_r1_c1_f1_o{index}" for index in range(1, 5)]
    database, _rows = _plate(tmp_path, keys)
    query = CropQuery(source="database", path=str(database), page_size=2)
    monkeypatch.setattr(screen, "crop_query", lambda: query)
    screen.crops_progress.connect(lambda done, _total: screen.stop_loading()
                                  if done == 2 else None)
    _load(screen, database)
    assert len(screen._crops) == 2
    assert screen.crop_record()["stopped"] is True
    screen.embed()
    screen._save_similar.click()
    stored = al._stored_embedding_frame(str(database))
    assert stored["prcfo"].tolist() == keys[:2]


@pytest.mark.parametrize("bad_key", (None, "", " ", float("nan")))
def test_missing_database_object_key_cannot_enable_similarity_save(
    screen, tmp_path, bad_key
):
    database, _rows = _plate(
        tmp_path, ["plate1_r1_c1_f1_o1", bad_key])
    _load(screen, database)
    assert len(screen._crops) == 2
    screen.embed()
    assert not screen._save_similar.isEnabled()
    screen._save_for_similarity()
    assert al._stored_embedding_frame(str(database)) is None


def test_folder_crops_have_no_database_identity_to_save(screen, tmp_path,
                                                        monkeypatch):
    database, rows = _plate(tmp_path, ["plate1_r1_c1_f1_o1"])
    folder = str((tmp_path / "cell_png").resolve())
    monkeypatch.setattr(screen, "crop_query", lambda: CropQuery(
        source="folder", path=folder))
    _load(screen, folder)
    assert len(screen._crops) == len(rows)
    screen.embed()
    assert not screen._save_similar.isEnabled()
    screen._save_for_similarity()
    assert al._stored_embedding_frame(str(database)) is None


def test_duplicate_object_keys_cannot_silently_collapse_distinct_crops(
    screen, tmp_path
):
    database, _rows = _plate(tmp_path, ["plate1_r1_c1_f1_o1"] * 2)
    _load(screen, database)
    assert len(screen._crops) == 2
    screen.embed()
    assert not screen._save_similar.isEnabled()
    screen._save_for_similarity()
    assert al._stored_embedding_frame(str(database)) is None


@pytest.mark.parametrize("save_fails", (False, True))
def test_late_save_completion_keeps_new_crop_status_and_original_source(
    screen, tmp_path, monkeypatch, save_fails
):
    database, rows = _plate(tmp_path, ["plate1_r1_c1_f1_o1"])
    _load(screen, database)
    screen.embed()
    pending = []
    monkeypatch.setattr(screen._jobs, "submit",
                        lambda work, done: pending.append((work, done)) or True)
    if save_fails:
        monkeypatch.setattr(al, "_store_crop_embeddings", lambda *_args, **_kwargs: (
            _ for _ in ()).throw(OSError("old database is read-only")))
    screen._save_similar.click()
    work, done = pending.pop()
    screen.set_crops(np.ones((2, 16, 16, 3), np.uint8))
    screen._status.setText("New crop selection")
    try:
        reply = work()
    except OSError as error:
        screen._on_job_failed(str(error))
    else:
        done(reply)
    assert screen._status.text() == "New crop selection"
    assert not screen._save_similar.isEnabled()
    stored = al._stored_embedding_frame(str(database))
    if save_fails:
        assert stored is None
    else:
        assert stored["prcfo"].tolist() == [rows[0][1]]


def test_late_embedding_failure_does_not_overwrite_new_crop_status(
    screen, tmp_path, monkeypatch
):
    database, _rows = _plate(tmp_path, ["plate1_r1_c1_f1_o1"])
    _load(screen, database)
    pending = []
    monkeypatch.setattr(screen._jobs, "submit",
                        lambda work, done: pending.append((work, done)) or True)
    screen.embed()
    work, done = pending.pop()
    monkeypatch.setattr(emb, "_embed_plate", lambda *_args, **_kwargs: (
        _ for _ in ()).throw(ValueError("old crop shape failed")))
    screen.set_crops(np.ones((2, 16, 16, 3), np.uint8))
    screen._status.setText("New crop selection")
    try:
        reply = work()
    except ValueError as error:
        screen._on_job_failed(str(error))
    else:
        done(reply)
    assert screen._status.text() == "New crop selection"
    assert screen._result is None
    assert not screen._save_similar.isEnabled()


def test_current_embedding_error_is_visible_without_enabling_save(
    screen, tmp_path, monkeypatch
):
    database, _rows = _plate(tmp_path, ["plate1_r1_c1_f1_o1"])
    _load(screen, database)
    monkeypatch.setattr(emb, "_embed_plate", lambda *_args, **_kwargs: (
        _ for _ in ()).throw(ValueError("invalid crop planes")))
    screen.embed()
    assert "invalid crop planes" in screen._status.text()
    assert not screen._save_similar.isEnabled()


def test_missing_encoder_provenance_prevents_source_bound_save(
    screen, tmp_path, monkeypatch
):
    database, _rows = _plate(tmp_path, ["plate1_r1_c1_f1_o1"])
    _load(screen, database)
    monkeypatch.setattr(emb, "encoder_entry", lambda _spec: (
        _ for _ in ()).throw(OSError("checkpoint metadata unavailable")))
    screen.embed()
    assert "checkpoint metadata unavailable" in screen._status.text()
    assert screen._result is None
    assert not screen._save_similar.isEnabled()
    assert al._stored_embedding_frame(str(database)) is None


def test_old_embedding_error_cannot_replace_a_pending_load_status(
    screen, tmp_path, monkeypatch
):
    first = tmp_path / "first"
    first.mkdir()
    second = tmp_path / "second"
    second.mkdir()
    first_db, _rows = _plate(first, ["plate1_r1_c1_f1_o1"])
    second_db, _rows = _plate(second, ["plate2_r1_c1_f1_o1"])
    _load(screen, first_db)
    pending = []
    monkeypatch.setattr(screen._jobs, "submit",
                        lambda work, done: pending.append((work, done)) or True)
    screen.embed()
    embed_work, embed_done = pending.pop()
    screen._path.setText(str(second_db))
    screen._on_path_changed()
    screen._load.click()
    assert screen.is_loading()
    loading_status = screen._status.text()
    monkeypatch.setattr(emb, "_embed_plate", lambda *_args, **_kwargs: (
        _ for _ in ()).throw(ValueError("old crop planes failed")))
    embed_done(embed_work())
    assert screen._status.text() == loading_status
    assert not screen._save_similar.isEnabled()


def test_old_embedding_completion_cannot_enable_save_during_new_load(
    screen, tmp_path, monkeypatch
):
    first = tmp_path / "first"
    first.mkdir()
    second = tmp_path / "second"
    second.mkdir()
    first_db, _rows = _plate(first, ["plate1_r1_c1_f1_o1"])
    second_db, _rows = _plate(second, ["plate2_r1_c1_f1_o1"])
    _load(screen, first_db)
    pending = []
    monkeypatch.setattr(screen._jobs, "submit",
                        lambda work, done: pending.append((work, done)) or True)
    screen.embed()
    embed_work, embed_done = pending.pop()
    screen._path.setText(str(second_db))
    screen._on_path_changed()
    screen._load.click()
    assert screen.is_loading()
    assert not screen._save_similar.isEnabled()
    embed_done(embed_work())
    assert not screen._save_similar.isEnabled()
