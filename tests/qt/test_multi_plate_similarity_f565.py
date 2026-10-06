"""Cross-plate similarity opens each hit through its own annotation writer."""

import os
import shutil
import sqlite3
from types import SimpleNamespace

import numpy as np

from tests.qt.test_find_similar_alpha_gate import alpha, annotate, plate  # noqa: F401


def test_external_plate_results_switch_source_before_a_label_is_saved(
        qtbot, annotate, plate, tmp_path):  # noqa: F811
    other = tmp_path / "plate2"
    shutil.copytree(plate, other)
    other_db = other / "measurements" / "measurements.db"
    with sqlite3.connect(other_db) as db:
        rows = db.execute("SELECT png_path FROM png_list").fetchall()
        db.executemany("UPDATE png_list SET png_path=? WHERE png_path=?",
                       [(path.replace(str(plate), str(other)), path)
                        for (path,) in rows])
    assert annotate._on_add_similarity_plate(str(other_db))
    query_db = os.path.abspath(annotate._settings.db_path)
    assert annotate._similar_paths() == (query_db, str(other_db))
    annotate._set_focus_slot(0)
    annotate._similar_k.setValue(6)
    annotate._on_find_similar()
    qtbot.waitUntil(lambda: annotate._similar_worker is None and
                    annotate._similar_navigation is not None, timeout=20000)
    hits = annotate._similar_navigation["hits"]
    assert str(other_db) in set(hits.db_path)
    assert annotate._similar_result_plate.count() == 2
    annotate._similar_result_plate.setCurrentIndex(
        annotate._similar_result_plate.findData(str(other_db)))
    qtbot.waitUntil(lambda: annotate._object_request is not None and
                    annotate._object_request.context.get("source_db") == str(other_db)
                    and bool(annotate._page_paths), timeout=20000)
    assert annotate._settings.db_path == str(other_db)
    assert all(path.startswith(str(other)) for path, _ in annotate._page_paths)
    selected = annotate._page_paths[0][0]
    assert annotate._set_annotation(0, 1)
    assert selected in annotate._pending_updates
    annotate._similar_result_plate.setCurrentIndex(
        annotate._similar_result_plate.findData(query_db))
    qtbot.waitUntil(lambda: annotate._settings.db_path == query_db and
                    annotate._object_request is not None and
                    annotate._object_request.context.get("source_db") == query_db,
                    timeout=20000)
    with sqlite3.connect(other_db) as db:
        saved = db.execute("SELECT annotate FROM png_list WHERE png_path=?",
                           (selected,)).fetchone()
    assert saved == (1,)
    with sqlite3.connect(query_db) as db:
        old = db.execute("SELECT COUNT(*) FROM png_list WHERE annotate IS NOT NULL").fetchone()
    assert old == (0,)
    assert annotate._page_paths[0][0].startswith(str(plate))
    other_index = annotate._similar_result_plate.findData(str(other_db))
    annotate._similar_feature_kind.setCurrentIndex(
        annotate._similar_feature_kind.findData("embeddings"))
    annotate._on_similar_result_plate(other_index)
    assert annotate._settings.db_path == query_db
    assert "settings changed" in annotate._status_label.text()
    annotate._similar_feature_kind.setCurrentIndex(
        annotate._similar_feature_kind.findData("auto"))
    unavailable = other_db.with_suffix(".moved")
    other_db.rename(unavailable)
    try:
        annotate._on_similar_result_plate(other_index)
        assert annotate._settings.db_path == query_db
        assert "no longer available" in annotate._status_label.text()
    finally:
        unavailable.rename(other_db)


def test_unlabelled_filter_uses_each_source_labels_and_current_pending_edits(
        qtbot, plate, tmp_path):  # noqa: F811
    from spacr import active_learning as al
    from spacr.qt.screens.annotate import _SimilarityWorker

    other = tmp_path / "plate2"
    shutil.copytree(plate, other)
    first_db = str(plate / "measurements" / "measurements.db")
    second_db = str(other / "measurements" / "measurements.db")
    with sqlite3.connect(first_db) as db:
        keys = [row[0] for row in db.execute(
            "SELECT png_path FROM png_list ORDER BY png_path")]
    with sqlite3.connect(second_db) as db:
        db.execute("UPDATE png_list SET annotate=1 WHERE png_path=?", (keys[0],))
    index = al._multi_similarity_index([first_db, second_db],
                                       feature_kind="measurements")
    answers = []
    worker = _SimilarityWorker(first_db, None, keys[0], index=index, k=40,
                               db_paths=[first_db, second_db],
                               feature_kind="measurements",
                               unlabelled_only=True,
                               pending_labels={keys[1]: 2})
    worker.done.connect(answers.append)
    worker.run()
    found = set(zip(answers[0]["hits"].db_path, answers[0]["hits"].key))
    assert (first_db, keys[0]) not in found
    assert (first_db, keys[1]) not in found
    assert (second_db, keys[0]) not in found
    assert (second_db, keys[1]) in found


def test_worker_honours_an_explicit_measurement_choice_without_saved_vectors(
        plate):  # noqa: F811
    from spacr.qt.screens.annotate import _SimilarityWorker

    database = str(plate / "measurements" / "measurements.db")
    with sqlite3.connect(database) as db:
        query = db.execute("SELECT png_path FROM png_list LIMIT 1").fetchone()[0]
    answers = []
    measured = _SimilarityWorker(database, None, query, k=3,
                                 feature_kind="measurements")
    measured.done.connect(answers.append)
    measured.run()
    assert len(answers) == 1
    assert len(answers[0]["hits"]) == 3
    failures = []
    embedded = _SimilarityWorker(database, None, query,
                                 feature_kind="embeddings")
    embedded.failed.connect(failures.append)
    embedded.run()
    assert len(failures) == 1
    assert "No stored crop embeddings" in failures[0]


def test_an_unavailable_extra_plate_reports_failure_without_switching_source(
        qtbot, annotate, tmp_path):  # noqa: F811
    original = annotate._settings.db_path
    missing = tmp_path / "gone" / "measurements" / "measurements.db"
    assert annotate._on_add_similarity_plate(str(missing))
    annotate._set_focus_slot(0)
    annotate._on_find_similar()
    qtbot.waitUntil(lambda: annotate._similar_worker is None and
                    "does not exist" in annotate._status_label.text(),
                    timeout=10000)
    assert annotate._settings.db_path == original
    assert annotate._similar_navigation is None


def test_plate_picker_rejects_other_files_and_clear_restores_open_source(
        annotate, monkeypatch, tmp_path):  # noqa: F811
    from spacr.qt.screens import annotate as annotate_module

    current = annotate._settings.db_path
    monkeypatch.setattr(annotate_module.QFileDialog, "getOpenFileName",
                        lambda *_args: ("", ""))
    assert not annotate._on_add_similarity_plate(False)
    assert not annotate._on_add_similarity_plate(str(tmp_path / "unrelated.db"))
    assert "measurements.db" in annotate._status_label.text()
    other = tmp_path / "other" / "measurements" / "measurements.db"
    assert annotate._on_add_similarity_plate(str(other))
    assert annotate._similar_paths() == (current, str(other))
    assert annotate._on_add_similarity_plate(str(other))
    assert annotate._similar_paths() == (current, str(other))
    annotate._on_clear_similarity_plates()
    assert annotate._similar_paths() == (current,)
    assert annotate._similar_result_plate.count() == 0


def test_blind_mode_refuses_multi_plate_search_before_starting_a_worker(
        annotate, tmp_path):  # noqa: F811
    other = tmp_path / "other" / "measurements" / "measurements.db"
    assert annotate._on_add_similarity_plate(str(other))
    annotate._set_focus_slot(0)
    original = annotate._settings.db_path
    annotate._blind = {"codes": {}}
    try:
        annotate._on_find_similar()
        assert annotate._similar_worker is None
        assert "Leave blind mode" in annotate._status_label.text()
        assert annotate._settings.db_path == original
    finally:
        annotate._blind = None


def test_saving_new_vectors_invalidates_a_cached_similarity_search(
        qtbot, annotate):  # noqa: F811
    from spacr import active_learning as al
    from spacr.embeddings import EmbeddingSpec

    database = annotate._settings.db_path
    with sqlite3.connect(database) as db:
        rows = db.execute("SELECT png_path, prcfo FROM png_list ORDER BY png_path").fetchall()
    query = annotate._page_paths[0][0]
    keys = [object_key for _, object_key in rows]
    spec = EmbeddingSpec(backbone="resnet18")
    encoder = SimpleNamespace(sha256="a" * 64, key="test-encoder", source="local")
    vectors = np.asarray([[index + 1., index % 3 + 1.]
                          for index in range(len(rows))], np.float32)
    query_row = next(index for index, (path, _) in enumerate(rows) if path == query)
    vectors[query_row] = [100., 1.]

    def save(values):
        """Persist a complete crop result from the same encoder weights."""
        result = SimpleNamespace(values=values, columns=("emb_0", "emb_1"),
                                 spec=spec)
        al._store_crop_embeddings(database, keys, result, encoder_entry=encoder)

    save(vectors)
    annotate._similar_feature_kind.setCurrentIndex(
        annotate._similar_feature_kind.findData("embeddings"))
    annotate._set_focus_slot(0)
    annotate._on_find_similar()
    qtbot.waitUntil(lambda: annotate._similar_worker is None and
                    annotate._similar_cache is not None, timeout=20000)
    first = annotate._similar_cache[2]
    original_vector = first.vector(query).copy()
    replacement = vectors.copy()
    replacement[query_row] = [1., 100.]
    save(replacement)
    annotate._on_find_similar()
    qtbot.waitUntil(lambda: annotate._similar_worker is None and
                    annotate._similar_cache is not None and
                    annotate._similar_cache[2] is not first, timeout=20000)
    assert not np.allclose(annotate._similar_cache[2].vector(query),
                           original_vector)
