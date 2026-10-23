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
    monkeypatch.setattr(al, "_store_crop_embeddings", lambda *_args, **_kwargs: (_ for _ in ()).throw(
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


def _well_labels(tmp_path):
    from spacr.tabular import write_table
    import pandas as pd
    path = tmp_path / "well_labels.csv"
    write_table(pd.DataFrame({"wellID": ["plate1_r1_c1"], "well_label": [1]}), path)
    return path


def test_current_result_reaches_mil_with_frozen_keys_wells_vectors_and_provenance(
        screen, plate, tmp_path, monkeypatch):
    from hashlib import sha256
    from spacr.tabular import read_table
    database, rows = plate
    _load(screen, database)
    screen.embed()
    result = screen._result
    before = sha256(database.read_bytes()).hexdigest()
    labels = _well_labels(tmp_path)
    screen._path.setText(str(tmp_path / "different.db"))
    screen._backbone.setCurrentText("another-backbone")
    seen = []
    completed = []

    def transport(frame, **columns):
        seen.append(frame.copy(deep=True))
        assert columns["feature_columns"] == ["emb_0", "emb_1", "emb_2"]
        return frame.assign(mil_attention=0.25), frame[["wellID", "well_label"]].drop_duplicates(), {"transport": True}

    monkeypatch.setattr(emb, "_mil_from_table", transport)
    monkeypatch.setattr(screen, "_on_well_mil_done", completed.append)
    assert screen._current_well_mil.isEnabled()
    assert screen._learn_from_current_embeddings(str(labels), columns={
        "well_column": "wellID", "label_column": "well_label",
        "feature_columns": ["emb_0", "emb_1", "emb_2"]}) == str(labels)
    [frame] = seen
    assert frame["prcfo"].tolist() == [row[1] for row in rows]
    assert frame["png_path"].tolist() == [row[0] for row in rows]
    assert frame["wellID"].tolist() == ["plate1_r1_c1"] * 4
    assert frame["well_label"].tolist() == [1] * 4
    np.testing.assert_array_equal(frame[list(result.columns)], result.values)
    assert frame["_embedding_fingerprint"].eq(result.spec.fingerprint()).all()
    assert frame["_embedding_spec"].map(json.loads).map(lambda value: value["backbone"]).eq(result.spec.backbone).all()
    assert len(completed) == 1
    output = read_table(tmp_path / "well_labels_current_embeddings_mil_cells.csv", report=None)
    assert output["prcfo"].tolist() == frame["prcfo"].tolist()
    assert sha256(database.read_bytes()).hexdigest() == before
    assert al._stored_embedding_frame(str(database)) is None
    assert not (tmp_path / "different.db").exists()


def test_current_mil_preserves_encoder_key_and_weights_without_storing_vectors(screen, plate, tmp_path):
    _load(screen, plate[0])
    screen.embed()
    frame = emb._mil_frame_from_result(*screen._result_identity, screen._result,
        str(_well_labels(tmp_path)), encoder_entry=SimpleNamespace(
            sha256="a" * 64, key="encoder:test", source="fixture"))
    assert frame["_embedding_weights_sha256"].eq("a" * 64).all()
    assert frame["_embedding_encoder_key"].eq("encoder:test").all()
    assert frame["_embedding_encoder_source"].eq("fixture").all()
    assert al._stored_embedding_frame(str(plate[0])) is None


def test_current_mil_is_disabled_for_unsourced_and_replaced_crops(screen, plate, tmp_path):
    assert not screen._current_well_mil.isEnabled()
    _load(screen, plate[0])
    screen.embed()
    assert screen._current_well_mil.isEnabled()
    screen.set_crops(np.zeros((2, 16, 16, 3), np.uint8))
    assert not screen._current_well_mil.isEnabled()
    assert screen._learn_from_current_embeddings(str(_well_labels(tmp_path))) == ""
    assert "Load database crops" in screen._status.text()
    assert not list(tmp_path.glob("*_mil_*.csv"))


def test_current_mil_cancelled_label_dialog_submits_nothing(screen, plate, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    _load(screen, plate[0])
    screen.embed()
    submitted = []
    monkeypatch.setattr(screen._jobs, "submit", lambda *args: submitted.append(args))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *args: ("", "")))
    assert screen._learn_from_current_embeddings() == ""
    assert submitted == []


def test_current_mil_frozen_snapshot_and_late_callback_do_not_replace_new_result(
        screen, plate, tmp_path, monkeypatch):
    _load(screen, plate[0])
    screen.embed()
    values = screen._result.values.copy()
    pending = []
    seen = []
    completed = []
    monkeypatch.setattr(screen._jobs, "submit", lambda work, done: pending.append((work, done)))
    monkeypatch.setattr(screen, "_on_well_mil_done", completed.append)
    def transport(frame, **columns):
        seen.append(frame.copy(deep=True))
        return frame, frame[["wellID"]].drop_duplicates(), {}
    monkeypatch.setattr(emb, "_mil_from_table", transport)
    screen._learn_from_current_embeddings(str(_well_labels(tmp_path)), columns={})
    screen._result.values[:] = -1
    screen.set_crops(np.ones((2, 16, 16, 3), np.uint8))
    [(work, done)] = pending
    done(work())
    np.testing.assert_array_equal(seen[0][["emb_0", "emb_1", "emb_2"]], values)
    assert completed == []
    assert not screen._current_well_mil.isEnabled()


def test_current_mil_refuses_missing_crop_metadata_without_training(screen, plate, tmp_path, monkeypatch):
    _load(screen, plate[0])
    screen.embed()
    with sqlite3.connect(plate[0]) as database:
        database.execute("DELETE FROM png_list WHERE prcfo = ?", (plate[1][0][1],))
    submitted = []
    monkeypatch.setattr(screen._jobs, "submit", lambda *args: submitted.append(args))
    assert screen._learn_from_current_embeddings(str(_well_labels(tmp_path)), columns={}) == ""
    assert "no longer have crop metadata" in screen._status.text()
    assert submitted == []


@pytest.mark.parametrize("keys", [("same", "same", "same", "same"), ("", "b", "c", "d"), ("one",)])
def test_current_result_rejects_invalid_object_cardinality(screen, plate, tmp_path, keys):
    _load(screen, plate[0])
    screen.embed()
    with pytest.raises(ValueError):
        emb._mil_frame_from_result(str(plate[0]), keys, screen._result, str(_well_labels(tmp_path)))


@pytest.mark.parametrize("language", ["sv", "de", "es", "fr", "pt", "is", "hi", "ko", "zh_CN"])
@pytest.mark.parametrize("failure,source", [
    ("keys", "The current embedding has no unique source object keys."),
    ("metadata", "Some current embedding objects no longer have crop metadata."),
])
def test_current_mil_refusal_uses_the_actual_selected_locale(
        plate, language, failure, source, monkeypatch):
    from hashlib import sha256
    from importlib import import_module
    monkeypatch.setenv("SPACR_LANGUAGE", language)
    catalog = import_module("spacr.qt.i18n_catalogs." + language)
    expected = catalog.UI[source]
    assert expected != source
    database, rows = plate
    before = sha256(database.read_bytes()).hexdigest()
    result = emb.EmbeddingResult(np.arange(12, dtype=np.float32).reshape(4, 3),
        ("emb_0", "emb_1", "emb_2"), emb.EmbeddingSpec(backbone="fixture"), 3, 3)
    keys = [row[1] for row in rows]
    if failure == "keys":
        keys[-1] = keys[0]
    else:
        keys[-1] = "plate1_r1_c1_f1_o999"
    with pytest.raises(ValueError) as error:
        emb._mil_frame_from_result(str(database), keys, result, None)
    assert str(error.value) == expected
    assert sha256(database.read_bytes()).hexdigest() == before
