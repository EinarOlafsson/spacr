"""Cross-plate similarity opens each hit through its own annotation writer."""

import os
import shutil
import sqlite3

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
