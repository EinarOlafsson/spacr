"""Unified-store step S5: the append-only edit log and the ported writers.

Annotations, verdicts, suggestions, UMAP lasso labels, attributions, crop
formats, annotation samples and classifier scores all go through
:mod:`spacr.store`. These tests pin four things:

1. old vs new: on an older ``measurements.db`` every ported writer leaves
   the database exactly as the SQL it replaced did (the reference SQL is
   kept here, copied from the writers before the port);
2. a store project shows the same labels after the same clicks, through
   ``<table>_labeled``, without changing a Parquet part;
3. the journal survives a crash, undo appends, the last writer wins and the
   full log keeps both annotators;
4. the new public API is registered in its arrival contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from spacr import store  # noqa: E402

OFFSET = 10
PATHS = [f"/crops/plate1/r1/c1/f1/o{i}.png" for i in range(8)]
PRCFO = [f"plate1_r1_c1_f1_o{i}" for i in range(8)]


# fixtures


def _legacy(path, *, annotate=None, extra=()):
    """A ``measurements.db`` whose ``png_list`` holds eight crops."""
    path = str(path)
    with sqlite3.connect(path) as db:
        columns = ["png_path TEXT", "prcfo TEXT", "file_name TEXT"]
        columns += list(extra)
        if annotate is not None:
            columns.append("annotate INTEGER")
        db.execute(f"CREATE TABLE png_list ({', '.join(columns)})")
        for i, (p, k) in enumerate(zip(PATHS, PRCFO)):
            row = [p, k, os.path.basename(p)] + [None] * len(extra)
            if annotate is not None:
                row.append(annotate[i])
            marks = ",".join("?" * len(row))
            db.execute(f"INSERT INTO png_list VALUES ({marks})", row)
        db.execute("CREATE TABLE settings (setting_key TEXT PRIMARY KEY, "
                   "setting_value TEXT)")
    return path


def _dump(path):
    """Schema and rows of a SQLite file, for old-vs-new comparison."""
    with sqlite3.connect(path) as db:
        return list(db.iterdump())


def _twins(tmp_path, **kwargs):
    """Two identical legacy files: one for the old SQL, one for the new."""
    old = _legacy(tmp_path / "old.db", **kwargs)
    new = str(tmp_path / "new.db")
    shutil.copy(old, new)
    return old, new


def _project(root, *, annotate=None):
    """A store project holding the same eight crops as ``png_list`` and
    ten cells; a non-null ``annotate`` is imported as edits."""
    project = os.path.join(str(root), "measurements")
    png = pd.DataFrame({"png_path": PATHS, "prcfo": PRCFO,
                        "file_name": [os.path.basename(p) for p in PATHS]})
    store.write_table(project, "png_list", png, plate="plate1", well="r1c1",
                      run_id="s5", sort_by=("png_path",),
                      key_columns=("png_path",))
    cells = pd.DataFrame({"object_key": np.arange(10, dtype=np.int64),
                          "timeID": np.zeros(10, dtype=np.int64),
                          "area": np.arange(10, dtype=np.float64)})
    store.write_table(project, "cell", cells, plate="plate1", well="r1c1",
                      run_id="s5", key_columns=("object_key", "timeID"))
    store.register(project)
    if annotate is not None:
        rows = [{"key": p, "value": v} for p, v in zip(PATHS, annotate)
                if v is not None]
        store.edit(project, rows, kind="annotation", source="import",
                   table="png_list", column="annotate")
    return project


def _labels(project, column, table="png_list"):
    """``{png_path: value}`` of one label column of a store project."""
    with store.open(project) as reader:
        frame = reader.query(table=f"{table}_labeled")
    if column not in frame.columns:
        return {}
    return {p: (None if pd.isna(v) else int(v)) for p, v in
            zip(frame["png_path"], frame[column])}


def _legacy_labels(path, column):
    """``{png_path: value}`` of one column of a legacy ``png_list``."""
    with sqlite3.connect(path) as db:
        return dict(db.execute(f'SELECT png_path, "{column}" FROM png_list'))


def _part_hashes(project):
    """sha256 of every Parquet part under ``tables/``."""
    out = {}
    for folder, _dirs, files in os.walk(os.path.join(project, "tables")):
        for name in files:
            path = os.path.join(folder, name)
            with open(path, "rb") as handle:
                out[os.path.relpath(path, project)] = hashlib.sha256(
                    handle.read()).hexdigest()
    return out


# 1. old vs new on a legacy measurements.db


def test_annotate_writer_matches_the_old_sql(tmp_path):
    from spacr.qt import annotate_engine as engine

    old, new = _twins(tmp_path, annotate=[None] * 8)
    batches = [{PATHS[0]: 1, PATHS[1]: 2}, {PATHS[1]: None, PATHS[2]: 1},
               {PATHS[3]: 2}]
    with sqlite3.connect(old) as db:
        for batch in batches:
            db.executemany('UPDATE "png_list" SET "annotate" = NULL WHERE '
                           '"png_path" = ?',
                           [(p,) for p, v in batch.items() if v is None])
            db.executemany('UPDATE "png_list" SET "annotate" = ? WHERE '
                           '"png_path" = ?',
                           [(int(v), p) for p, v in batch.items()
                            if v is not None])
    worker = engine.SaveWorker(new, "annotate")
    worker.start()
    for batch in batches:
        worker.submit(batch)
    worker.stop(wait=True)
    assert worker.last_error is None
    assert _dump(new) == _dump(old)


def test_annotate_writer_puts_verdicts_in_their_own_column(tmp_path):
    from spacr.qt import annotate_engine as engine

    old, new = _twins(tmp_path, annotate=[None] * 8,
                      extra=["annotate_verdict INTEGER"])
    with sqlite3.connect(old) as db:
        db.execute('UPDATE png_list SET annotate = 1 WHERE png_path = ?',
                   (PATHS[0],))
        db.execute('UPDATE png_list SET annotate_verdict = -2 WHERE '
                   'png_path = ?', (PATHS[4],))
    worker = engine.SaveWorker(new, "annotate")
    worker.submit({PATHS[0]: 1})
    worker.submit({PATHS[4]: -2}, column="annotate_verdict")
    worker.start()
    worker.stop(wait=True)
    assert _dump(new) == _dump(old)


def test_ensure_and_clear_match_the_old_sql(tmp_path):
    from spacr.qt import annotate_engine as engine

    old, new = _twins(tmp_path, extra=["annotate_verdict INTEGER"])
    with sqlite3.connect(old) as db:
        db.execute('ALTER TABLE "png_list" ADD COLUMN "annotate" INTEGER')
        db.execute('CREATE INDEX IF NOT EXISTS "idx_png_path" ON "png_list" '
                   '("png_path")')
        db.execute('UPDATE "png_list" SET "annotate" = 3')
        db.execute('UPDATE "png_list" SET "annotate" = NULL')
        db.execute('UPDATE "png_list" SET "annotate_verdict" = NULL')
    engine.ensure_annotation_column(new, "annotate")
    engine.ensure_annotation_column(new, "annotate")
    with sqlite3.connect(new) as db:
        db.execute('UPDATE "png_list" SET "annotate" = 3')
    engine.clear_column(new, "annotate")
    assert _dump(new) == _dump(old)


def test_clearing_a_column_without_a_verdict_column_adds_none(tmp_path):
    from spacr.qt import annotate_engine as engine

    path = _legacy(tmp_path / "m.db", annotate=[1] * 8)
    engine.clear_column(path, "annotate")
    with sqlite3.connect(path) as db:
        names = [r[1] for r in db.execute("PRAGMA table_info(png_list)")]
    assert "annotate_verdict" not in names
    assert set(_legacy_labels(path, "annotate").values()) == {None}


def _old_write_suggestions(path, rows):
    with sqlite3.connect(path) as db:
        return sum(db.execute(
            'UPDATE "png_list" SET "annotate" = ? WHERE png_path = ? AND '
            '"annotate" IS NULL', (v, p)).rowcount for v, p in rows)


def test_suggest_writers_match_the_old_sql(tmp_path):
    from spacr import suggest

    start = [1, None, None, 2, None, 3, 2, None]
    old, new = _twins(tmp_path, annotate=start,
                      extra=["annotate_verdict INTEGER"])
    frame = pd.DataFrame({"png_path": PATHS[:5], "suggested": [1, 2, 1, 2, 1],
                          "stored": [11, 12, 11, 12, 11]})
    assert suggest.write_suggestions(new, "annotate", frame) == \
        _old_write_suggestions(old, [(11, PATHS[0]), (12, PATHS[1]),
                                     (11, PATHS[2]), (12, PATHS[3]),
                                     (11, PATHS[4])])
    assert _dump(new) == _dump(old)

    keep = [PATHS[1], PATHS[3]]
    with sqlite3.connect(old) as db:
        marks = ",".join("?" * len(keep))
        where = f'"annotate" > {OFFSET} AND png_path IN ({marks})'
        db.execute(f'UPDATE "png_list" SET "annotate_verdict" = "annotate" - '
                   f'{OFFSET} WHERE {where}', keep)
        kept = db.execute(f'UPDATE "png_list" SET "annotate" = "annotate" - '
                          f'{OFFSET} WHERE {where}', keep).rowcount
    assert suggest.resolve_suggestions(new, "annotate", keep=True,
                                       paths=keep) == kept
    assert _dump(new) == _dump(old)

    with sqlite3.connect(old) as db:
        thrown = db.execute(f'UPDATE "png_list" SET "annotate" = NULL WHERE '
                            f'"annotate" > {OFFSET}').rowcount
    assert suggest.resolve_suggestions(new, "annotate", keep=False) == thrown
    assert _dump(new) == _dump(old)

    snapshot = {PATHS[2]: 11, PATHS[0]: 12}
    assert suggest._restore_suggestions(new, "annotate", snapshot) == \
        _old_write_suggestions(old, [(11, PATHS[2]), (12, PATHS[0])])
    assert _dump(new) == _dump(old)

    changes = {PATHS[2]: None, PATHS[4]: 2, PATHS[0]: 1}
    with sqlite3.connect(old) as db:
        before = db.total_changes
        db.executemany(f'UPDATE "png_list" SET "annotate" = NULL WHERE '
                       f'png_path = ? AND "annotate" > {OFFSET}',
                       [(PATHS[2],)])
        db.executemany(f'UPDATE "png_list" SET "annotate" = ? WHERE '
                       f'png_path = ? AND ("annotate" IS NULL OR "annotate" > '
                       f'{OFFSET})', [(12, PATHS[4]), (11, PATHS[0])])
        moved = db.total_changes - before
    assert suggest._write_rethresholded(new, "annotate", changes) == moved
    assert _dump(new) == _dump(old)


def test_verdict_column_is_added_as_before(tmp_path):
    from spacr import suggest

    old, new = _twins(tmp_path, annotate=[None] * 8)
    with sqlite3.connect(old) as db:
        db.execute('ALTER TABLE "png_list" ADD COLUMN "annotate_verdict" '
                   'INTEGER')
    assert suggest.ensure_verdict_column(new, "annotate") is True
    assert suggest.ensure_verdict_column(new, "annotate") is True
    assert _dump(new) == _dump(old)


def test_umap_labels_match_the_old_sql(tmp_path):
    from spacr.umap_annotations import write_umap_annotations

    old, new = _twins(tmp_path)
    with sqlite3.connect(old) as db:
        db.execute('ALTER TABLE "png_list" ADD COLUMN "lasso" INTEGER')
        db.executemany('UPDATE "png_list" SET "lasso" = ? WHERE png_path = ?',
                       [(3, PATHS[1]), (4, PATHS[5]), (5, PATHS[1])])
    records = [{"db_path": new, "db_png_path": p}
               for p in (PATHS[1], PATHS[5], PATHS[1], "/not/here.png")]
    assert write_umap_annotations(records, [3, 4, 5, 6], "lasso") == (3, 1)
    assert _dump(new) == _dump(old)


def test_attribution_columns_match_the_old_sql(tmp_path):
    from spacr import attribution_columns as ac

    old, new = _twins(tmp_path)
    records = [{"prcfo": PRCFO[0], "grna_attributed": "g1",
                "grna_attributed_p": 0.9},
               {"prcfo": PRCFO[3], "grna_attributed": "ambiguous",
                "grna_attributed_p": 0.4},
               {"prcfo": "nowhere", "grna_attributed": "g2"},
               {"prcfo": None, "grna_attributed": "g3"}]
    with sqlite3.connect(old) as db:
        for name, kind in ac.COLUMNS.items():
            db.execute(f"ALTER TABLE png_list ADD COLUMN {name} {kind}")
        matched = sum(db.execute(
            "UPDATE png_list SET grna_attributed = ?, grna_attributed_p = ? "
            "WHERE prcfo = ?", (r.get("grna_attributed"),
                                r.get("grna_attributed_p"), r["prcfo"])
        ).rowcount for r in records if r["prcfo"] is not None)
        db.execute("INSERT OR REPLACE INTO settings (setting_key, "
                   "setting_value) VALUES (?, ?)",
                   (ac.NOTE_KEY, ac.ATTRIBUTION_NOTE))
    assert ac.write(new, records, confirmed=True) == {"matched": matched,
                                                      "added": 5}
    assert _dump(new) == _dump(old)


def test_crop_format_stamp_matches_the_old_sql(tmp_path):
    from spacr import crops

    old, new = _twins(tmp_path)
    with sqlite3.connect(old) as db:
        db.execute('ALTER TABLE "png_list" ADD COLUMN "crop_format" INTEGER')
        every = db.execute('UPDATE "png_list" SET "crop_format" = ?',
                           (crops.CROP_FORMAT_CURRENT,)).rowcount
        some = db.executemany('UPDATE "png_list" SET "crop_format" = ? WHERE '
                              'png_path = ?',
                              [(crops.CROP_FORMAT_LEGACY_BGR, PATHS[2])]
                              ).rowcount
    assert crops.stamp_crop_format_in_db(new) == every
    assert crops.stamp_crop_format_in_db(
        new, [PATHS[2]], fmt=crops.CROP_FORMAT_LEGACY_BGR) == some
    assert _dump(new) == _dump(old)


def test_annotation_sample_marks_match_the_old_sql(tmp_path):
    from spacr import ml

    src = tmp_path / "plate"
    (src / "measurements").mkdir(parents=True)
    new = _legacy(src / "measurements" / "measurements.db",
                  annotate=[None] * 8)
    old = str(tmp_path / "old.db")
    shutil.copy(new, old)
    with sqlite3.connect(new) as db:
        db.execute("ALTER TABLE png_list ADD COLUMN annotate_sample INTEGER")
        db.execute("UPDATE png_list SET annotate_sample = 9")
    shutil.copy(new, old)
    settings = {"src": str(src), "annotate_sample_n": 3,
                "annotate_sample_seed": 4, "ml_label_column": "annotate"}
    sample = ml.export_annotation_sample(settings)
    with sqlite3.connect(old) as db:
        db.execute("UPDATE png_list SET annotate_sample = NULL")
        db.executemany("UPDATE png_list SET annotate_sample = ? WHERE "
                       "png_path = ?", [(int(o), str(p)) for o, p in zip(
                           sample["annotate_sample"], sample["png_path"])])
    assert _dump(new) == _dump(old)


def test_gate_export_provenance_matches_the_old_sql(tmp_path):
    from spacr.qt.screens import gate_editor

    from spacr.qt.widgets.gate_spec import GateSet

    old, new = _twins(tmp_path)
    gate_editor._record_gate_export_provenance(
        new, "cell", GateSet(), "", [("gate_a", 3), ("gate_b", 1)], [])
    with sqlite3.connect(new) as db:
        stored = db.execute("SELECT exported_utc, source_table, gate_column, "
                            "receipt_json FROM filter_export_provenance"
                            ).fetchall()
    with sqlite3.connect(old) as db:
        db.execute("CREATE TABLE IF NOT EXISTS filter_export_provenance "
                   "(exported_utc TEXT NOT NULL, source_table TEXT NOT NULL, "
                   "gate_column TEXT NOT NULL, receipt_json TEXT NOT NULL)")
        db.executemany("INSERT INTO filter_export_provenance VALUES "
                       "(?, ?, ?, ?)", stored)
    assert [r[2] for r in stored] == ["gate_a", "gate_b"]
    assert _dump(new) == _dump(old)


def test_bookkeeping_rows_on_a_store_project_read_as_a_table(tmp_path):
    project = _project(tmp_path)
    ddl = "CREATE TABLE IF NOT EXISTS audit (a TEXT, b INTEGER)"
    assert store.append_rows(project, "audit", [{"a": "x", "b": 1}],
                             columns=("a", "b"), ddl=ddl) == 1
    assert store.append_rows(project, "audit", [{"a": "y", "b": 2}],
                             columns=("a", "b"), ddl=ddl) == 1
    with store.open(project) as reader:
        assert "audit" in reader.tables()
        frame = reader.sql("SELECT a, b FROM audit ORDER BY b")
    assert frame.values.tolist() == [["x", 1], ["y", 2]]
    assert store.open(project).edits(latest=False).empty


def test_a_legacy_rerun_is_byte_identical(tmp_path):
    from spacr import suggest

    def run(path):
        frame = pd.DataFrame({"png_path": PATHS[:4], "suggested": [1] * 4,
                              "stored": [11] * 4})
        suggest.ensure_verdict_column(path, "annotate")
        suggest.write_suggestions(path, "annotate", frame)
        suggest.resolve_suggestions(path, "annotate", keep=True,
                                    paths=PATHS[:2])
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    first, second = _twins(tmp_path, annotate=[None] * 8)
    assert run(first) == run(second)


# 2. a store project shows the same labels after the same clicks


def test_the_annotator_on_a_store_project_shows_the_legacy_labels(tmp_path):
    from spacr.qt import annotate_engine as engine

    legacy = _legacy(tmp_path / "m.db", annotate=[None] * 8)
    project = _project(tmp_path / "store")
    before = _part_hashes(project)
    batches = [{PATHS[0]: 1, PATHS[1]: 2}, {PATHS[1]: None, PATHS[2]: 1}]
    for target in (legacy, project):
        worker = engine.SaveWorker(target, "annotate")
        worker.start()
        for batch in batches:
            worker.submit(batch)
        worker.stop(wait=True)
        assert worker.last_error is None
    assert _labels(project, "annotate") == _legacy_labels(legacy, "annotate")
    assert _part_hashes(project) == before
    assert not store.open(project).edits()["source"].isin(["import"]).any()


def test_attribution_and_crop_format_on_a_store_project(tmp_path):
    from spacr import attribution_columns as ac
    from spacr import crops

    project = _project(tmp_path)
    records = [{"prcfo": PRCFO[2], "grna_attributed": "g1",
                "grna_attributed_p": 0.75}]
    assert ac.write(project, records, confirmed=True)["matched"] == 1
    assert store.edit(project, [{"value": 2}], kind="crop_format",
                      source="crops", table="png_list", column="crop_format",
                      every_row=True) == 1
    with store.open(project) as reader:
        frame = reader.query(table="png_list_labeled").set_index(
            "png_path")
    assert frame.loc[PATHS[2], "grna_attributed"] == "g1"
    assert frame.loc[PATHS[2], "grna_attributed_p"] == 0.75
    assert frame["grna_attributed"].isna().sum() == 7
    assert set(frame["crop_format"]) == {2}
    assert crops.CROP_FORMAT_DB_COLUMN == "crop_format"


def test_predictions_on_a_store_project_become_an_active_score_table(tmp_path):
    from spacr import predictions

    legacy = _legacy(tmp_path / "m.db")
    project = _project(tmp_path / "store")
    before = _part_hashes(project)
    results = pd.DataFrame({"prcfo": PRCFO[:5],
                            "pred": [0.1, 0.9, 0.4, 0.6, 0.5]})
    columns = {"pred": ("pred", "REAL")}
    first = predictions.merge_prediction_results(results, legacy, columns,
                                                 verbose=False)
    second = predictions.merge_prediction_results(results, project, columns,
                                                  verbose=False)
    assert second.matched_rows == first.matched_rows == 5
    with store.open(project) as reader:
        active = reader.active_outputs()
        frame = reader.query(table="png_list_labeled")
    assert list(active["kind"]) == ["prediction"]
    assert active["table"][0].startswith("score__pred__")
    assert {p: None if pd.isna(v) else float(v) for p, v in zip(
        frame["png_path"], frame["pred"])} == _legacy_labels(legacy, "pred")
    assert {k: v for k, v in _part_hashes(project).items()
            if not k.startswith(os.path.join("tables", "score__"))} == before


def test_the_same_scores_write_the_same_bytes(tmp_path):
    frame = pd.DataFrame({"object_key": np.arange(10, dtype=np.int64),
                          "timeID": 0, "score": np.linspace(0, 1, 10,
                                                            dtype="float32")})
    hashes = []
    for name in ("a", "b"):
        project = _project(tmp_path / name)
        table = store.write_scores(project, frame, model="m", run_id="r1",
                                   target="cell", plate="plate1", well="r1c1")
        hashes.append({k: v for k, v in _part_hashes(project).items()
                       if table in k})
    assert hashes[0] == hashes[1] and hashes[0]


def test_switching_the_active_model_changes_one_row(tmp_path):
    project = _project(tmp_path)
    for model, offset in (("m1", 0.0), ("m2", 1.0)):
        store.write_scores(project, pd.DataFrame({
            "object_key": np.arange(10, dtype=np.int64), "timeID": 0,
            "score": np.arange(10) + offset}), model=model, run_id="r",
            target="cell", plate="plate1", well="r1c1")
    with store.open(project) as reader:
        assert reader.active_outputs()["table"].tolist() == ["score__m2__r"]
        assert reader.query(table="cell_labeled")["score"].iloc[0] == 1.0
    store.activate(project, "classifier", "score__m1__r", "cell")
    with pytest.raises(store.StoreError):
        store.activate(project, "classifier", "score__none__r", "cell")
    with store.open(project) as reader:
        assert reader.query(table="cell_labeled")["score"].iloc[0] == 0.0


# 3. journal, undo, last writer wins, agreement


def test_a_click_is_one_fsynced_line_and_other_screens_see_it(tmp_path):
    project = _project(tmp_path)
    reader = store.open(project)
    with store.journal(project, kind="annotation", source="annotator",
                       author="ann") as journal:
        took = []
        for path in PATHS[:5]:
            start = time.perf_counter()
            journal.add(path, 1, column="annotate", table="png_list")
            took.append(time.perf_counter() - start)
        with open(journal.file, encoding="utf-8") as handle:
            assert len(handle.readlines()) == 5
        assert reader.query(table="png_list_labeled")[
            "annotate"].notna().sum() == 5
        assert sorted(took)[2] < 0.05
    assert not os.path.exists(journal.file)
    assert reader.query(table="png_list_labeled")[
        "annotate"].notna().sum() == 5


def test_a_journal_folds_every_n_edits(tmp_path):
    project = _project(tmp_path)
    journal = store.journal(project, kind="annotation", source="annotator",
                            fold_every=3)
    for path in PATHS[:7]:
        journal.add(path, 2, column="annotate", table="png_list")
    files, journals = store._edit_sources(project)
    assert len([f for f in files if "annotation" in f]) == 2
    assert len(journals) == 1
    journal.close()
    assert len(store.open(project).edits(kind="annotation")) == 7


def test_a_crashed_journal_loses_nothing_that_was_fsynced(tmp_path):
    project = _project(tmp_path)
    journal = store.journal(project, kind="annotation", source="annotator",
                            session="crash")
    journal.add(PATHS[0], 1, column="annotate", table="png_list")
    journal.add(PATHS[1], 2, column="annotate", table="png_list")
    with open(journal.file, "a", encoding="utf-8") as handle:
        handle.write('{"table": "png_list", "key": "torn')
    dead = journal.file.replace(f"-{os.getpid()}.jsonl", "-999999999.jsonl")
    os.replace(journal.file, dead)
    assert _labels(project, "annotate")[PATHS[1]] == 2
    result = store.compact(project)
    assert result["journals"] == 1 and not os.path.exists(dead)
    labels = _labels(project, "annotate")
    assert (labels[PATHS[0]], labels[PATHS[1]]) == (1, 2)
    assert sum(v is not None for v in labels.values()) == 2


def test_a_fold_interrupted_before_the_journal_empties_counts_once(tmp_path):
    project = _project(tmp_path)
    journal = store.journal(project, kind="annotation", source="annotator")
    journal.add(PATHS[0], 1, column="annotate", table="png_list")
    rows = store._read_journal(journal.file)
    store._publish_edit_file(project, "annotation", rows)
    assert len(store.open(project).edits(kind="annotation",
                                         latest=False)) == 1
    journal.close()
    assert len(store.open(project).edits(kind="annotation",
                                         latest=False)) == 1


def test_undo_appends_the_earlier_value_and_keeps_history(tmp_path):
    project = _project(tmp_path)
    with store.journal(project, kind="annotation", source="annotator",
                       author="ann") as journal:
        journal.add(PATHS[0], 1, column="annotate", table="png_list")
        journal.add(PATHS[0], 2, column="annotate", table="png_list")
        assert _labels(project, "annotate")[PATHS[0]] == 2
        assert journal.undo() == 1
        assert _labels(project, "annotate")[PATHS[0]] == 1
        assert journal.undo() == 1
        assert _labels(project, "annotate")[PATHS[0]] is None
        assert journal.undo() == 0
    history = store.open(project).edits(latest=False)
    assert list(history["op"]) == ["set", "set", "set", "clear"]
    stamp = tuple(history[["seq_ns", "seq_writer"]].iloc[0])
    store.edit(project, [{"key": PATHS[1], "value": 2}], kind="annotation",
               source="annotator", table="png_list", column="annotate")
    last = store.open(project).edits(latest=False).iloc[-1]
    assert store.undo(project, (last["seq_ns"], last["seq_writer"]),
                      kind="annotation", source="annotator") == 1
    assert _labels(project, "annotate")[PATHS[1]] is None
    assert store.undo(project, (1, 1), kind="annotation",
                      source="annotator") == 0
    assert stamp[0] > 0
    with pytest.raises(store.StoreError):
        store.undo(_legacy(tmp_path / "m.db"), stamp, kind="annotation",
                   source="annotator")


def test_two_annotators_last_writer_wins_and_the_log_keeps_both(tmp_path):
    from spacr import agreement

    project = _project(tmp_path)
    first = [1, 1, 2, 2, 1, 2, 1, 1]
    second = [1, 2, 2, 2, 1, 1, 1, 1]
    for author, labels in (("ann", first), ("ben", second)):
        with store.journal(project, kind="annotation", source="annotator",
                           author=author) as journal:
            journal.add_many("png_list", "annotate", dict(zip(PATHS, labels)))
    assert list(_labels(project, "annotate").values()) == second
    latest = store.open(project).edits(kind="annotation")
    assert set(latest["author"]) == {"ben"}
    wide = agreement.annotations_by_author(project, "annotate")
    assert list(wide.columns) == ["key", "ann", "ben"]
    by_key = wide.set_index("key")
    assert [by_key.loc[p, "ann"] for p in PATHS] == first
    assert [by_key.loc[p, "ben"] for p in PATHS] == second
    kappa = agreement.cohens_kappa(wide["ann"].tolist(), wide["ben"].tolist())
    assert -1.0 <= kappa < 1.0


def test_guards_keep_answers_and_a_key_column_matches_by_prcfo(tmp_path):
    project = _project(tmp_path, annotate=[2] + [None] * 7)
    written = store.edit(project, [{"key": PATHS[0], "value": 11},
                                   {"key": PATHS[1], "value": 11}],
                         kind="suggestion", source="suggest",
                         table="png_list", column="annotate", guard="unset")
    assert written == 1
    assert store.edit(project, [{"key": PRCFO[5], "value": 3},
                                {"key": "missing", "value": 3}],
                      kind="annotation", source="annotator",
                      table="png_list", column="annotate",
                      key_column="prcfo") == 1
    labels = _labels(project, "annotate")
    assert (labels[PATHS[0]], labels[PATHS[1]], labels[PATHS[5]]) == (2, 11, 3)
    with pytest.raises(store.StoreError):
        store.edit(project, [{"value": 1}], kind="annotation",
                   source="annotator", table="png_list", column="annotate")


def test_every_row_edits_then_keyed_edits_resolve_in_order(tmp_path):
    project = _project(tmp_path, annotate=[1] * 8)
    store.edit(project, [{"value": None}, {"key": PATHS[3], "value": 7}],
               kind="annotation_sample", source="ml", table="png_list",
               column="annotate", every_row=True)
    labels = _labels(project, "annotate")
    assert labels[PATHS[3]] == 7
    assert sum(v is not None for v in labels.values()) == 1
    latest = store.open(project).edits(column="annotate")
    assert list(latest["key"]) == [PATHS[3]]


def test_the_training_table_pulls_only_labelled_rows(tmp_path):
    project = _project(tmp_path)
    store.edit(project, [{"object_key": 2, "timeID": 0, "value": 1},
                         {"object_key": 7, "timeID": 0, "value": 2},
                         {"object_key": 99, "timeID": 0, "value": 2}],
               kind="annotation", source="annotator", table="cell",
               column="cls")
    path = store._training_path(project, "cell", "cls")
    assert os.path.isfile(path)
    with store.open(project) as reader:
        training = reader.training("cell", "cls", features=["area"])
    assert training.sort_values("object_key")[
        ["object_key", "area", "cls"]].values.tolist() == [[2, 2.0, 1],
                                                           [7, 7.0, 2]]
    with store.journal(project, kind="annotation", source="annotator") as j:
        j.add(column="cls", table="cell", object_key=4, timeID=0, value=1)
        with store.open(project) as reader:
            assert len(reader.training("cell", "cls")) == 3
    assert len(_require_parquet().read_table(path)) == 3


def _require_parquet():
    import pyarrow.parquet as pq

    return pq


def test_compact_keeps_every_edit_and_every_label(tmp_path):
    project = _project(tmp_path, annotate=[1, 2, None, None, 1, 2, 1, 2])
    for value in (3, 4):
        store.edit(project, [{"key": PATHS[2], "value": value}],
                   kind="annotation", source="annotator", table="png_list",
                   column="annotate")
    before = (_labels(project, "annotate"),
              len(store.open(project).edits(latest=False)))
    result = store.compact(project)
    files, _journals = store._edit_sources(project)
    assert result["files"] == 3 and len(files) == 1
    assert os.path.basename(files[0]).startswith("snapshot-")
    assert (_labels(project, "annotate"),
            len(store.open(project).edits(latest=False))) == before
    with pytest.raises(store.StoreError):
        store.compact(_legacy(tmp_path / "m.db"))


def test_declared_columns_appear_typed_before_any_edit(tmp_path):
    project = _project(tmp_path)
    assert store.ensure_columns(project, "png_list",
                                {"annotate": "INTEGER"}) == ["annotate"]
    assert store.ensure_columns(project, "png_list",
                                {"annotate": "INTEGER"}) == []
    with store.open(project) as reader:
        frame = reader.query(table="png_list_labeled")
    assert "annotate" in frame.columns and frame["annotate"].isna().all()


def test_legacy_edits_need_a_project_and_a_safe_kind(tmp_path):
    with pytest.raises(store.StoreError):
        store.edit("", [{"key": "a", "value": 1}], kind="annotation",
                   source="x", table="png_list", column="c")
    with pytest.raises(store.StoreError):
        store.edit(str(tmp_path), [{"key": "a", "value": 1}],
                   kind="annotation", source="x", table="png_list",
                   column="c")
    path = _legacy(tmp_path / "m.db")
    with pytest.raises(store.StoreError):
        store.edit(path, [{"key": "a", "value": 1}], kind="../x", source="x",
                   table="png_list", column="c")
    assert store.open(path).edits().empty


# 4. the arrival contract of the S5 public API (2026-11-23)

S5_BIRTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        "release_contracts",
                        "s5_store_edit_log_api_arrivals_2026-11-23.json")
S5_ARRIVALS = frozenset({
    "spacr.store.edit", "spacr.store.journal", "spacr.store.EditJournal",
    "spacr.store.EditJournal.__enter__", "spacr.store.EditJournal.__exit__",
    "spacr.store.EditJournal.add", "spacr.store.EditJournal.add_many",
    "spacr.store.EditJournal.add_batch", "spacr.store.EditJournal.undo",
    "spacr.store.EditJournal.fold", "spacr.store.EditJournal.close",
    "spacr.store.ensure_columns", "spacr.store.undo", "spacr.store.compact",
    "spacr.store.write_scores", "spacr.store.append_rows",
    "spacr.store.activate",
    "spacr.store.Store.edits",
    "spacr.store.Store.active_outputs", "spacr.store.Store.training",
    "spacr.agreement.annotations_by_author"})


def _s5_source_docstring(symbol):
    """The docstring of one arrival read from its source file, so no module is imported."""
    import ast
    import inspect

    parts = symbol.split(".")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "spacr", parts[1] + ".py")
    with open(path, encoding="utf-8") as handle:
        scope = ast.parse(handle.read())
    for name in parts[2:]:
        scope = next(node for node in scope.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                     and node.name == name)
    text = inspect.cleandoc(ast.get_docstring(scope, clean=False))
    if isinstance(scope, ast.ClassDef):
        init = next((node for node in scope.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "__init__"), None)
        if init is not None and ast.get_docstring(init, clean=False):
            text += "\n" + inspect.cleandoc(ast.get_docstring(init,
                                                              clean=False))
    return text


def _s5_birth_arrivals(existing=None):
    """The S5 public symbols, bound to their source docstrings."""
    with open(S5_BIRTH, encoding="utf-8") as handle:
        entries = json.load(handle)
    assert set(entries) == S5_ARRIVALS
    for symbol, body in entries.items():
        assert body == _s5_source_docstring(symbol), symbol
    if existing is not None:
        assert not entries.keys() & existing.keys()
    return entries


def _validated_s5_arrivals(callables):
    """Remove only the S5 exact new callables from the frozen boundary."""
    entries = _s5_birth_arrivals()
    found = {item.symbol: item for item in callables
             if item.symbol in entries}
    assert set(found) == {s for s in S5_ARRIVALS if ".__" not in s}
    for symbol, item in found.items():
        assert item.docstring == entries[symbol], symbol
    return [item for item in callables if item.symbol not in entries]


def test_the_s5_arrivals_are_exactly_the_new_public_api():
    assert len(_s5_birth_arrivals()) == len(S5_ARRIVALS)
