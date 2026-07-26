"""Classification results must land on the ``png_list`` row they belong to.

Both classifiers score a whole database: the CV one in :mod:`spacr.deep_spacr`
and the classical-ML one in :mod:`spacr.ml`. :mod:`spacr.predictions` is the one
merge path they share, and this module is its regression suite.

Every database here is built with the **real** writer,
:func:`spacr.utils.filepaths_to_database` --- the same call
:func:`spacr.measure.measure_crop` makes --- and the legacy databases with the
real legacy writer, :func:`spacr.utils.add_column_to_database`. That is
load-bearing, not tidiness: the merge this replaces was tested against a
hand-built ``png_list`` of ``(png_path, prcfo)``, and a real ``png_list`` has a
column called ``rowID``. SQLite identifiers are case-insensitive, so on a real
table the bare name ``rowid`` resolves to the *plate row* rather than to the row
id, and the old ``UPDATE ... WHERE rowid = ?`` wrote one object's score over
every crop in that plate row. A fixture that "happens to match" is exactly how
that survived.

Everything is CPU-only, offline and deterministic.
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# fixtures built with the real writers
# ---------------------------------------------------------------------------

WELLS = ("A1", "A2")
FIELDS = (1, 2)
OBJECTS = (1, 2, 3)


def crop_paths(src, plate, crop_mode="cell", wells=WELLS, fields=FIELDS,
               objects=OBJECTS):
    """The crop paths ``measure_crop`` would hand to ``filepaths_to_database``.

    Names follow :func:`spacr.utils._generate_names`:
    ``<plate>_<well>_<field>_<object>.png`` under ``data/.../<mode>_png/``.
    """
    folder = os.path.join(src, "data", "single_nucleus", "single_pathogen",
                          f"{plate}_{wells[0]}", f"{crop_mode}_png")
    os.makedirs(folder, exist_ok=True)
    return [os.path.join(folder, f"{plate}_{w}_{f}_{o}.png")
            for w in wells for f in fields for o in objects]


def write_png_list(src, plate, crop_mode="cell", **kwargs):
    """Populate ``<src>/measurements/measurements.db`` via the real writer."""
    from spacr.utils import filepaths_to_database

    os.makedirs(os.path.join(src, "measurements"), exist_ok=True)
    paths = crop_paths(src, plate, crop_mode, **kwargs)
    filepaths_to_database(paths, {"timelapse": False}, src, crop_mode)
    return paths


def db_of(src):
    return os.path.join(src, "measurements", "measurements.db")


def vision_results(members, preds, threshold=0.5):
    """The frame ``apply_model_to_tar`` returns, built by the real helper.

    The tar stores member names as bare basenames
    (:func:`spacr.utils.add_images_to_tar` uses ``arcname=basename``), which is
    the whole reason the old merge reached for a basename in the first place.
    """
    from spacr.utils import process_vision_results

    df = pd.DataFrame({"path": [os.path.basename(m) for m in members],
                       "pred": list(preds)})
    return process_vision_results(df, threshold)


def ml_results(prcfos, classes, probabilities):
    """The frame ``ml_analysis`` returns: prcfo + predictions + per-class probs."""
    return pd.DataFrame({
        "prcfo": list(prcfos),
        "predictions": list(classes),
        "prediction_probability_class_0": [1.0 - p for p in probabilities],
        "prediction_probability_class_1": list(probabilities),
    })


def read(db, columns, table="png_list"):
    con = sqlite3.connect(str(db))
    try:
        return pd.read_sql_query(f"SELECT {columns} FROM {table}", con)
    finally:
        con.close()


def column_names(db, table="png_list"):
    con = sqlite3.connect(str(db))
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 1. a CV run over a real two-plate database scores every row, correctly
# ---------------------------------------------------------------------------

def test_cv_merge_scores_every_row_of_a_two_plate_database(tmp_path):
    """Per-row correctness, not just a row count.

    Two plates in one database, every crop given a distinct score, every score
    checked back against the row it belongs to.

    This is the test the previous implementation fails. ``png_list`` carries a
    column called ``rowID``; ``SELECT rowid`` on that table returns ``'r1'`` /
    ``'r2'`` and ``UPDATE ... WHERE rowid = ?`` then matches every crop in the
    plate row, so all twelve crops of row 1 ended up with one object's score.
    """
    from spacr.predictions import merge_cv_predictions

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1") + write_png_list(src, "plate2")
    db = db_of(src)
    assert len(read(db, "png_path")) == len(paths) == 24

    preds = [i / len(paths) for i in range(len(paths))]
    df = vision_results(paths, preds)

    report = merge_cv_predictions(df, db)

    assert report.key == "prcfo", "prcfo is the canonical per-object identity"
    assert report.matched_rows == report.db_rows == 24
    assert report.unmatched_db_rows == 0
    assert report.unmatched_result_rows == 0
    assert report.ambiguous_keys == 0

    back = read(db, "file_name, pred, cv_predictions")
    got = dict(zip(back["file_name"], back["pred"]))
    want = dict(zip(df["path"], df["pred"]))
    assert got.keys() == want.keys()
    for name in want:
        assert got[name] == pytest.approx(want[name]), name
    classes = dict(zip(back["file_name"], back["cv_predictions"]))
    assert classes == dict(zip(df["path"], df["cv_predictions"]))


def test_cv_merge_does_not_smear_one_score_across_a_plate_row(tmp_path):
    """The ``rowID``/``rowid`` collision, isolated.

    Every crop in plate row 1 gets a different score; if the row id were being
    read from the ``rowID`` column they would all end up identical.
    """
    from spacr.predictions import merge_cv_predictions

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1")
    df = vision_results(paths, [i / 100 for i in range(len(paths))])

    merge_cv_predictions(df, db_of(src))

    scores = read(db_of(src), "pred")["pred"].tolist()
    assert len(set(scores)) == len(scores), (
        "every crop had a distinct score; identical values mean the UPDATE "
        "matched on the plate row instead of the row id")


# ---------------------------------------------------------------------------
# 2. two plates that share crop names must not cross-assign
# ---------------------------------------------------------------------------

def test_two_source_folders_with_the_same_plate_name_do_not_cross_assign(tmp_path):
    """The regression test for the reported bug.

    Two source folders whose plates are both called ``plate1`` -- what happens
    whenever the plate name comes from the source folder name -- produce crops
    with identical names. ``generate_dataset`` tars both under the same member
    name, so the results frame carries the same key twice with two different
    scores.

    The old merge built its lookup with ``lookup[key] = value``, so the second
    silently won and one plate was scored with the other plate's predictions.
    Here the collision is refused and counted: nothing is written, and the
    report says why.
    """
    from spacr.predictions import merge_cv_predictions

    src_a = str(tmp_path / "experiment_a" / "plate1")
    src_b = str(tmp_path / "experiment_b" / "plate1")
    paths_a = write_png_list(src_a, "plate1")
    paths_b = write_png_list(src_b, "plate1")
    assert ([os.path.basename(p) for p in paths_a]
            == [os.path.basename(p) for p in paths_b])

    df = vision_results(paths_a + paths_b,
                        [0.05] * len(paths_a) + [0.95] * len(paths_b))

    for src in (src_a, src_b):
        report = merge_cv_predictions(df, db_of(src))
        assert report.ambiguous_keys == len(paths_a)
        assert report.ambiguous_result_rows == len(df)
        assert report.matched_rows == 0
        scored = read(db_of(src), "pred")["pred"]
        assert scored.isna().all(), (
            "a crop that cannot be told apart from another plate's crop must "
            "be left unscored, never given the other plate's score")


def test_a_collision_is_reported_not_absorbed(tmp_path, capsys):
    """The collision is on stdout, with the reason."""
    from spacr.predictions import merge_cv_predictions

    src_a = str(tmp_path / "a" / "plate1")
    src_b = str(tmp_path / "b" / "plate1")
    paths_a = write_png_list(src_a, "plate1", wells=("A1",), fields=(1,))
    paths_b = write_png_list(src_b, "plate1", wells=("A1",), fields=(1,))
    df = vision_results(paths_a + paths_b, [0.1] * 3 + [0.9] * 3)

    merge_cv_predictions(df, db_of(src_a))

    out = capsys.readouterr().out
    assert "conflicting values and were NOT written" in out
    assert "same name" in out


def test_identical_duplicates_are_not_treated_as_a_collision(tmp_path):
    """The same key twice with the *same* value is not ambiguous."""
    from spacr.predictions import merge_cv_predictions

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    df = vision_results(paths + paths, [0.25] * 3 + [0.25] * 3)

    report = merge_cv_predictions(df, db_of(src))

    assert report.ambiguous_keys == 0
    assert report.matched_rows == 3
    assert read(db_of(src), "pred")["pred"].tolist() == pytest.approx([0.25] * 3)


def test_one_key_on_several_crop_modes_fans_out_and_says_so(tmp_path):
    """Cell and cytoplasm crops of one object share a name *and* a prcfo.

    ``_generate_names`` gives both ``<file>_<cell_id>.png``; only the folder
    differs. Two database rows for one key is not an ambiguity -- one value,
    two rows -- so both are written and the fan-out is counted.
    """
    from spacr.predictions import merge_cv_predictions

    src = str(tmp_path / "screen")
    cells = write_png_list(src, "plate1", "cell", wells=("A1",), fields=(1,))
    write_png_list(src, "plate1", "cytoplasm", wells=("A1",), fields=(1,))
    df = vision_results(cells, [0.11, 0.22, 0.33])

    report = merge_cv_predictions(df, db_of(src))

    assert report.db_rows == 6
    assert report.matched_rows == 6
    assert report.matched_keys == 3
    assert report.fanout_rows == 3
    assert report.ambiguous_keys == 0
    back = read(db_of(src), "png_path, pred")
    for _, row in back.iterrows():
        expected = {"1": 0.11, "2": 0.22, "3": 0.33}[
            os.path.basename(row["png_path"]).split("_")[-1].split(".")[0]]
        assert row["pred"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 3. CV and ML results coexist
# ---------------------------------------------------------------------------

def test_ml_results_merge_and_cv_afterwards_leaves_both_readable(tmp_path):
    """Running ML then CV leaves four columns, none overwritten."""
    from spacr.predictions import merge_cv_predictions, merge_ml_predictions

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1")
    db = db_of(src)
    prcfos = read(db, "prcfo")["prcfo"].tolist()

    ml = ml_results(prcfos,
                    [i % 2 for i in range(len(prcfos))],
                    [i / len(prcfos) for i in range(len(prcfos))])
    ml_report = merge_ml_predictions(ml, db)
    assert ml_report.key == "prcfo"
    assert ml_report.matched_rows == len(prcfos)

    cv = vision_results(paths, [1 - i / len(paths) for i in range(len(paths))])
    cv_report = merge_cv_predictions(cv, db)
    assert cv_report.matched_rows == len(paths)

    back = read(db, "prcfo, file_name, pred, cv_predictions, ml_pred, predictions")
    assert set(back.columns) == {"prcfo", "file_name", "pred", "cv_predictions",
                                 "ml_pred", "predictions"}
    assert back[["pred", "cv_predictions", "ml_pred", "predictions"]].notna().all().all()

    want_ml = dict(zip(ml["prcfo"], ml["prediction_probability_class_1"]))
    want_cv = dict(zip(cv["path"], cv["pred"]))
    for _, row in back.iterrows():
        assert row["ml_pred"] == pytest.approx(want_ml[row["prcfo"]])
        assert row["pred"] == pytest.approx(want_cv[row["file_name"]])
    assert (dict(zip(back["prcfo"], back["predictions"]))
            == dict(zip(ml["prcfo"], ml["predictions"])))


def test_cv_first_then_ml_also_keeps_both(tmp_path):
    """Order does not matter: the two stages own different columns."""
    from spacr.predictions import merge_cv_predictions, merge_ml_predictions

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    merge_cv_predictions(vision_results(paths, [0.9, 0.8, 0.7]), db)
    prcfos = read(db, "prcfo")["prcfo"].tolist()
    merge_ml_predictions(ml_results(prcfos, [1, 1, 0], [0.6, 0.5, 0.4]), db)

    back = read(db, "pred, cv_predictions, ml_pred, predictions")
    assert back["pred"].tolist() == pytest.approx([0.9, 0.8, 0.7])
    assert back["cv_predictions"].tolist() == [1, 1, 1]
    assert back["ml_pred"].tolist() == pytest.approx([0.6, 0.5, 0.4])
    assert back["predictions"].tolist() == [1, 1, 0]


def test_ml_frame_without_probabilities_still_merges_the_class(tmp_path):
    """A frame with no ``prediction_probability_class_1`` writes the class only."""
    from spacr.predictions import merge_ml_predictions

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    prcfos = read(db, "prcfo")["prcfo"].tolist()
    frame = pd.DataFrame({"prcfo": prcfos, "predictions": [0, 1, 1]})

    report = merge_ml_predictions(frame, db)

    assert report.columns == ("predictions",)
    assert "ml_pred" not in column_names(db)
    assert read(db, "predictions")["predictions"].tolist() == [0, 1, 1]


def test_ml_frame_with_no_prediction_columns_is_refused_loudly(tmp_path, capsys):
    from spacr.predictions import merge_ml_predictions

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    frame = pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1"]})

    assert merge_ml_predictions(frame, db_of(src)) is None
    assert "No prediction columns" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 4. re-running a stage updates in place
# ---------------------------------------------------------------------------

def test_rerunning_a_stage_updates_in_place(tmp_path):
    """No duplicated rows, no ``predictions_1`` sibling column."""
    from spacr.predictions import merge_cv_predictions, merge_ml_predictions

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    prcfos = read(db, "prcfo")["prcfo"].tolist()

    for score in (0.1, 0.9):
        merge_cv_predictions(vision_results(paths, [score] * 3), db)
        merge_ml_predictions(ml_results(prcfos, [0, 0, 0], [score] * 3), db)

    names = column_names(db)
    for expected in ("pred", "cv_predictions", "ml_pred", "predictions"):
        assert names.count(expected) == 1, f"{expected} duplicated on re-run"
    assert not [n for n in names if n.endswith(("_1", "_2"))]

    back = read(db, "pred, cv_predictions, ml_pred")
    assert len(back) == 3, "a re-run must not append rows"
    assert back["pred"].tolist() == pytest.approx([0.9] * 3)
    assert back["cv_predictions"].tolist() == [1, 1, 1]
    assert back["ml_pred"].tolist() == pytest.approx([0.9] * 3)


# ---------------------------------------------------------------------------
# 5. unmatched rows are counted; a zero-match merge is loud
# ---------------------------------------------------------------------------

def test_unmatched_rows_on_both_sides_are_counted(tmp_path):
    """A partial overlap is reported from both directions."""
    from spacr.predictions import merge_cv_predictions

    src = str(tmp_path / "screen")
    scored = write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    write_png_list(src, "plate1", wells=("A2",), fields=(1,))
    other = crop_paths(str(tmp_path / "elsewhere"), "plate9",
                       wells=("A1",), fields=(1,))
    db = db_of(src)

    report = merge_cv_predictions(vision_results(scored + other, [0.5] * 6), db)

    assert report.db_rows == 6
    assert report.result_rows == 6
    assert report.matched_rows == 3
    assert report.unmatched_db_rows == 3
    assert report.unmatched_result_rows == 3
    assert read(db, "pred")["pred"].isna().sum() == 3


def test_a_merge_that_matches_nothing_is_loud(tmp_path, capsys):
    """Zero matches must not look like a successful run."""
    from spacr.predictions import merge_cv_predictions

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    strangers = crop_paths(str(tmp_path / "elsewhere"), "plate7",
                           wells=("A1",), fields=(1,))
    db = db_of(src)

    report = merge_cv_predictions(vision_results(strangers, [0.5] * 3), db)

    assert report.matched_rows == 0
    out = capsys.readouterr().out
    assert "0/3 rows matched" in out
    assert "NOTHING MATCHED" in out
    assert read(db, "pred")["pred"].isna().all()


def test_an_unparseable_crop_name_is_counted_not_dropped(tmp_path):
    """A member name no key can be built from is reported, not ignored."""
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    df = vision_results(paths, [0.5] * 3)
    df = pd.concat([df, pd.DataFrame([{"path": "junk.png", "pred": 0.7,
                                       "cv_predictions": 1}])],
                   ignore_index=True)

    report = merge_prediction_results(
        df, db_of(src),
        {"pred": ("pred", "REAL"), "cv_predictions": ("cv_predictions", "INTEGER")},
        key="prcfo")

    assert report.unparsed_result_rows == 1
    assert report.matched_rows == 3


def test_a_missing_source_column_is_named(tmp_path):
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    frame = pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1"], "pred": [0.5]})

    with pytest.raises(KeyError, match="nope"):
        merge_prediction_results(frame, db_of(src),
                                 {"pred": ("pred", "REAL"),
                                  "cls": ("nope", "INTEGER")})


def test_a_frame_with_nothing_to_key_on_is_refused(tmp_path):
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    frame = pd.DataFrame({"pred": [0.5]})

    with pytest.raises(ValueError, match="No usable join key"):
        merge_prediction_results(frame, db_of(src), {"pred": ("pred", "REAL")})


def test_an_explicit_key_that_cannot_be_built_is_refused(tmp_path):
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    frame = pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1"], "pred": [0.5]})

    with pytest.raises(ValueError, match="cannot be built"):
        merge_prediction_results(frame, db_of(src), {"pred": ("pred", "REAL")},
                                 key="png_path")


def test_an_unknown_key_name_is_refused(tmp_path):
    from spacr.predictions import _db_keys, _result_keys, merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    frame = pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1"], "pred": [0.5]})

    with pytest.raises(ValueError, match="Unknown join key"):
        merge_prediction_results(frame, db_of(src), {"pred": ("pred", "REAL")},
                                 key="banana")
    # both sides refuse it, not only whichever happens to be asked first
    with pytest.raises(ValueError, match="Unknown join key"):
        _result_keys("banana", frame, False)
    with pytest.raises(ValueError, match="Unknown join key"):
        _db_keys("banana", frame)


def test_a_missing_database_is_reported_and_skipped(tmp_path, capsys):
    from spacr.predictions import merge_cv_predictions

    missing = tmp_path / "nowhere" / "measurements.db"
    frame = pd.DataFrame({"path": ["plate1_A1_1_1.png"], "pred": [0.5],
                          "cv_predictions": [1]})

    assert merge_cv_predictions(frame, str(missing)) is None
    assert "Database not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 6. column names that need quoting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    'weird "score"',      # embedded double quotes
    "group by",           # spaces and a keyword
    "select",             # a bare reserved word
    "col-with-dash",
])
def test_a_column_name_needing_quoting_does_not_break_the_sql(tmp_path, name):
    """Column names are quoted, both in ALTER TABLE and in UPDATE."""
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    df = vision_results(paths, [0.2, 0.4, 0.6])

    report = merge_prediction_results(df, db, {name: ("pred", "REAL")})

    assert report.matched_rows == 3
    assert name in column_names(db)
    con = sqlite3.connect(db)
    try:
        values = [r[0] for r in con.execute(
            f'SELECT "{name.replace(chr(34), chr(34) * 2)}" FROM png_list')]
    finally:
        con.close()
    assert values == pytest.approx([0.2, 0.4, 0.6])


def test_an_empty_identifier_is_refused(tmp_path):
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    frame = pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1"], "pred": [0.5]})

    with pytest.raises(ValueError, match="Invalid SQL identifier"):
        merge_prediction_results(frame, db_of(src), {"pred": ("pred", "REAL")},
                                 table="")


def test_a_table_declaring_every_rowid_spelling_is_refused(tmp_path):
    """No way to address a row by identity, so say so instead of guessing."""
    from spacr.predictions import _rowid_alias

    with pytest.raises(ValueError, match="rowid, oid and _rowid_"):
        _rowid_alias(["rowid", "OID", "_RowID_", "png_path"])
    assert _rowid_alias(["png_path", "rowID"]) == "_rowid_"
    assert _rowid_alias(["png_path"]) == "_rowid_"


# ---------------------------------------------------------------------------
# 7. an interrupted merge changes nothing
# ---------------------------------------------------------------------------

def test_an_interrupted_merge_leaves_the_table_unchanged(tmp_path, monkeypatch):
    """A failure halfway rolls back the added columns *and* the writes.

    SQLite's DDL is transactional but Python's driver only opens an implicit
    transaction for DML, so an ``ALTER TABLE`` autocommits unless the
    transaction is opened by hand -- the same trap ``rename_columns_in_db`` was
    fixed for.
    """
    import spacr.predictions as predictions

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1")
    db = db_of(src)
    before = column_names(db)
    df = vision_results(paths, [i / 100 for i in range(len(paths))])

    def half_then_fail(cursor, sql, updates):
        cursor.executemany(sql, updates[:len(updates) // 2])
        raise RuntimeError("interrupted mid-merge")

    monkeypatch.setattr(predictions, "_execute_updates", half_then_fail)

    with pytest.raises(RuntimeError, match="interrupted mid-merge"):
        predictions.merge_cv_predictions(df, db)

    assert column_names(db) == before, "the added columns must be rolled back"
    assert len(read(db, "png_path")) == len(paths)

    # and the database is still usable: a clean re-run scores everything
    monkeypatch.undo()
    report = predictions.merge_cv_predictions(df, db)
    assert report.matched_rows == len(paths)


def test_an_interrupted_migration_leaves_the_table_unchanged(tmp_path, monkeypatch):
    """Same guarantee for the legacy-encoding repair.

    Half a repaired column is a column in two encodings at once, which is
    worse than the encoding it started in.
    """
    import spacr.predictions as predictions

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    prcfos = read(db, "prcfo")["prcfo"].tolist()
    legacy_write(db, prcfos, [0, 1, 0])
    assert read(db, "predictions")["predictions"].tolist() == [2, 1, 2]

    def apply_then_fail(cursor, sql):
        cursor.execute(sql)
        raise RuntimeError("interrupted mid-migration")

    monkeypatch.setattr(predictions, "_execute", apply_then_fail)

    with pytest.raises(RuntimeError, match="interrupted mid-migration"):
        predictions.migrate_prediction_columns(db)

    assert read(db, "predictions")["predictions"].tolist() == [2, 1, 2]

    monkeypatch.undo()
    assert predictions.migrate_prediction_columns(db) == [
        ("png_list", "predictions", 2)]
    assert read(db, "predictions")["predictions"].tolist() == [0, 1, 0]


def test_a_missing_table_fails_loudly(tmp_path):
    from spacr.predictions import merge_prediction_results

    db = tmp_path / "measurements.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE something_else (x INTEGER)")
    con.commit()
    con.close()
    frame = pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1"], "pred": [0.5]})

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        merge_prediction_results(frame, str(db), {"pred": ("pred", "REAL")})


def test_a_database_without_png_list_is_left_alone_by_the_migration(tmp_path):
    from spacr.predictions import migrate_prediction_columns

    db = tmp_path / "measurements.db"
    sqlite3.connect(db).close()
    assert migrate_prediction_columns(str(db)) == []
    assert migrate_prediction_columns(str(tmp_path / "absent.db")) == []


# ---------------------------------------------------------------------------
# 8. an old database reads correctly with no manual action
# ---------------------------------------------------------------------------

def legacy_write(db, prcfos, classes, column="predictions"):
    """Write an ML result the way spaCR used to, with the real legacy writer.

    :func:`spacr.utils.add_column_to_database` reads a CSV, adds the column and
    -- because the Annotate app labels classes 1 and 2 -- replaces every 0 with
    a 2 on the way in.
    """
    from spacr.utils import add_column_to_database

    csv_path = os.path.splitext(str(db))[0] + f"_legacy_{column}.csv"
    pd.DataFrame({"prcfo": list(prcfos), column: list(classes)}).to_csv(
        csv_path, index=False)
    add_column_to_database({"csv_path": csv_path, "db_path": str(db),
                            "table_name": "png_list", "update_column": column,
                            "match_column": "prcfo"})
    return csv_path


def test_a_database_written_before_this_change_is_repaired_on_merge(tmp_path):
    """The legacy class encoding is undone on the way in, in place."""
    from spacr.predictions import merge_ml_predictions

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    prcfos = read(db, "prcfo")["prcfo"].tolist()
    legacy_write(db, prcfos, [0, 1, 1])

    stored = read(db, "predictions")["predictions"].tolist()
    assert stored == [2, 1, 1], "the legacy writer stored 0 as 2"

    report = merge_ml_predictions(
        ml_results(prcfos, [1, 0, 0], [0.7, 0.2, 0.1]), db)

    assert ("png_list", "predictions", 1) in report.repaired
    names = column_names(db)
    assert names.count("predictions") == 1, "one column, not a _1 sibling"
    # repaired first, then overwritten by this run
    assert read(db, "predictions")["predictions"].tolist() == [1, 0, 0]


def test_the_legacy_encoding_is_repaired_when_it_is_not_overwritten(tmp_path, capsys):
    """Migration alone restores 0/1 from the annotate-style 1/2 encoding."""
    from spacr.predictions import migrate_prediction_columns

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    prcfos = read(db, "prcfo")["prcfo"].tolist()
    legacy_write(db, prcfos, [0, 1, 0])
    assert read(db, "predictions")["predictions"].tolist() == [2, 1, 2]

    repaired = migrate_prediction_columns(db)

    assert repaired == [("png_list", "predictions", 2)]
    assert "Repaired 2 row(s)" in capsys.readouterr().out
    assert read(db, "predictions")["predictions"].tolist() == [0, 1, 0]
    # idempotent: a second pass finds nothing to do and changes nothing
    assert migrate_prediction_columns(db) == []
    assert read(db, "predictions")["predictions"].tolist() == [0, 1, 0]


def test_a_column_already_in_the_model_encoding_is_left_alone(tmp_path):
    """Nothing to repair once the values are 0/1, which is the normal state."""
    from spacr.predictions import merge_ml_predictions, migrate_prediction_columns

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    prcfos = read(db, "prcfo")["prcfo"].tolist()
    merge_ml_predictions(ml_results(prcfos, [0, 1, 1], [0.1, 0.8, 0.9]), db)

    assert migrate_prediction_columns(db) == []
    assert read(db, "predictions")["predictions"].tolist() == [0, 1, 1]


def test_an_all_class_one_column_is_left_alone(tmp_path):
    """1s with no 2s are already the model's own labels; nothing to undo."""
    from spacr.predictions import migrate_prediction_columns

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    prcfos = read(db, "prcfo")["prcfo"].tolist()
    legacy_write(db, prcfos, [1, 1, 1])

    assert migrate_prediction_columns(db) == []
    assert read(db, "predictions")["predictions"].tolist() == [1, 1, 1]


def test_a_multiclass_legacy_column_is_not_reinterpreted(tmp_path):
    """0 -> 2 is only reversible when the column holds nothing but 1s and 2s.

    A three-class model's genuine class 2 is indistinguishable from a mangled
    0, so the values are left exactly as they are rather than guessed at.
    """
    from spacr.predictions import migrate_prediction_columns

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    prcfos = read(db, "prcfo")["prcfo"].tolist()
    legacy_write(db, prcfos, [1, 2, 3])

    assert migrate_prediction_columns(db) == []
    assert read(db, "predictions")["predictions"].tolist() == [1, 2, 3]


def test_a_table_without_the_legacy_column_is_a_no_op(tmp_path):
    from spacr.predictions import migrate_prediction_columns

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    assert migrate_prediction_columns(db_of(src)) == []


def test_the_ml_entry_point_repairs_before_reading_png_list(tmp_path, monkeypatch):
    """``generate_ml_scores`` repairs an old database on the way in.

    The annotation-column branch reads ``png_list`` directly, so the repair has
    to happen before the read -- the same place ``rename_columns_in_db`` sits in
    ``_read_db``.
    """
    import spacr.io as io
    import spacr.ml as ml

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    prcfos = read(db, "prcfo")["prcfo"].tolist()
    legacy_write(db, prcfos, [0, 1, 1])
    assert read(db, "predictions")["predictions"].tolist() == [2, 1, 1]

    seen = {}

    def spy(path, tables):
        seen["values"] = read(path, "predictions")["predictions"].tolist()
        raise RuntimeError("stop here")

    # generate_ml_scores imports these inside the function body, so the
    # bindings that matter are the ones on spacr.io.
    monkeypatch.setattr(io, "_read_db", spy)
    monkeypatch.setattr(io, "_read_and_merge_data",
                        lambda *a, **k: (pd.DataFrame({"x": [1.0]}), None))
    with pytest.raises(RuntimeError, match="stop here"):
        ml.generate_ml_scores({"src": src, "annotation_column": "test",
                               "channel_of_interest": 1, "verbose": False})

    assert seen["values"] == [0, 1, 1]


# ---------------------------------------------------------------------------
# key selection
# ---------------------------------------------------------------------------

def test_prcfo_wins_a_tie_against_the_basename(tmp_path):
    """Both keys match everything; the canonical identity is the one used."""
    from spacr.predictions import merge_cv_predictions

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    report = merge_cv_predictions(vision_results(paths, [0.1, 0.2, 0.3]),
                                  db_of(src))
    assert report.key == "prcfo"


def test_a_full_path_is_used_when_the_names_are_not_spacr_crop_names(tmp_path):
    """Never a basename when a full path is on offer."""
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    frame = pd.DataFrame({"png_path": paths, "pred": [0.1, 0.2, 0.3]})
    # scrub the key the frame would otherwise win on
    con = sqlite3.connect(db_of(src))
    con.execute("UPDATE png_list SET prcfo = NULL")
    con.commit()
    con.close()

    report = merge_prediction_results(frame, db_of(src),
                                      {"pred": ("pred", "REAL")})

    assert report.key == "png_path"
    assert report.matched_rows == 3


def test_prcfo_is_rebuilt_from_metadata_when_the_column_is_gone(tmp_path):
    """``png_list`` carries what the key is made of, so it can be rebuilt."""
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE png_list DROP COLUMN prcfo")
    con.execute("ALTER TABLE png_list DROP COLUMN png_path")
    con.execute("ALTER TABLE png_list DROP COLUMN file_name")
    con.commit()
    con.close()

    frame = pd.DataFrame({
        "prcfo": [f"plate1_r1_c1_f1_o{o}" for o in (1, 2, 3)],
        "pred": [0.1, 0.2, 0.3]})
    report = merge_prediction_results(frame, db, {"pred": ("pred", "REAL")})

    assert report.key == "prcfo"
    assert report.matched_rows == 3
    assert read(db, "pred")["pred"].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_a_row_missing_a_metadata_component_gets_no_rebuilt_key(tmp_path):
    """An empty component would let two different objects share a key."""
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE png_list DROP COLUMN prcfo")
    con.execute("ALTER TABLE png_list DROP COLUMN png_path")
    con.execute("ALTER TABLE png_list DROP COLUMN file_name")
    con.execute("UPDATE png_list SET fieldID = NULL WHERE cell_id = 'o2'")
    con.commit()
    con.close()

    frame = pd.DataFrame({
        "prcfo": [f"plate1_r1_c1_f1_o{o}" for o in (1, 2, 3)],
        "pred": [0.1, 0.2, 0.3]})
    report = merge_prediction_results(frame, db, {"pred": ("pred", "REAL")})

    assert report.matched_rows == 2
    assert report.unmatched_db_rows == 1
    assert report.unmatched_result_rows == 1


def test_a_table_with_no_metadata_cannot_rebuild_prcfo(tmp_path):
    """No key at all is refused, not silently turned into an empty string."""
    from spacr.predictions import merge_prediction_results

    db = tmp_path / "measurements.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE png_list (something TEXT)")
    con.execute("INSERT INTO png_list VALUES ('x')")
    con.commit()
    con.close()
    frame = pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1"], "pred": [0.5]})

    with pytest.raises(ValueError, match="No usable join key"):
        merge_prediction_results(frame, str(db), {"pred": ("pred", "REAL")})


def test_metadata_without_an_object_id_column_cannot_rebuild_prcfo(tmp_path):
    """plate/row/column/field name a *field*, not an object."""
    from spacr.predictions import merge_prediction_results

    db = tmp_path / "measurements.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE png_list (plateID TEXT, rowID TEXT, "
                "columnID TEXT, fieldID TEXT)")
    con.execute("INSERT INTO png_list VALUES ('plate1','r1','c1','f1')")
    con.commit()
    con.close()
    frame = pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1"], "pred": [0.5]})

    with pytest.raises(ValueError, match="No usable join key"):
        merge_prediction_results(frame, str(db), {"pred": ("pred", "REAL")})


def test_a_non_scalar_key_cell_is_treated_as_no_key(tmp_path):
    """``pd.isna`` on a list raises; a list is not a key either way."""
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    frame = pd.DataFrame({
        "prcfo": ["plate1_r1_c1_f1_o1", ["not", "a", "key"]],
        "pred": [0.5, 0.6]})

    report = merge_prediction_results(frame, db, {"pred": ("pred", "REAL")},
                                      key="prcfo")

    assert report.unparsed_result_rows == 0  # str(list) is a string, just a bad one
    assert report.matched_rows == 1
    assert report.unmatched_result_rows == 1


def test_a_non_scalar_value_cell_is_stored_as_text(tmp_path):
    """``_sql_value`` must not blow up on something ``pd.isna`` chokes on."""
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    frame = pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1"],
                          "note": [["a", "b"]]})

    report = merge_prediction_results(frame, db, {"note": ("note", "TEXT")})

    assert report.matched_rows == 1
    assert read(db, "note")["note"][0] == "['a', 'b']"


def test_missing_and_numpy_valued_cells_survive_the_round_trip(tmp_path):
    """``None``, ``pd.NA`` and a numpy scalar all reach SQLite intact.

    An object-dtype column is what a frame assembled from several sources
    ends up with, and it can hold any of the three.
    """
    import numpy as np
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    frame = pd.DataFrame({
        "prcfo": [f"plate1_r1_c1_f1_o{o}" for o in (1, 2, 3)],
        "score": pd.Series([np.float32(0.25), None, pd.NA], dtype=object)})

    report = merge_prediction_results(frame, db, {"pred": ("score", "REAL")})

    assert report.matched_rows == 3
    stored = read(db, "pred")["pred"]
    assert stored[0] == pytest.approx(0.25)
    assert stored[1:].isna().all()


def test_a_missing_key_cell_is_not_a_key(tmp_path):
    """A NaN prcfo matches nothing rather than matching the NaN row."""
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    frame = pd.DataFrame({
        "prcfo": ["plate1_r1_c1_f1_o1", float("nan")],
        "pred": [0.5, 0.6]})

    report = merge_prediction_results(frame, db, {"pred": ("pred", "REAL")},
                                      key="prcfo")

    assert report.unparsed_result_rows == 1
    assert report.matched_rows == 1


def test_a_missing_path_cell_is_not_a_crop_name(tmp_path):
    """The same, one tier down: no name, no derived prcfo."""
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    df = vision_results(paths, [0.1, 0.2, 0.3])
    df = pd.concat([df, pd.DataFrame([{"path": None, "pred": 0.9,
                                       "cv_predictions": 1}])],
                   ignore_index=True)

    report = merge_prediction_results(
        df, db_of(src), {"pred": ("pred", "REAL")}, key="prcfo")

    assert report.unparsed_result_rows == 1
    assert report.matched_rows == 3


def test_file_name_is_derived_from_png_path_when_the_column_is_gone(tmp_path):
    """Last tier: a basename, computed only because nothing better is there."""
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE png_list DROP COLUMN file_name")
    con.execute("UPDATE png_list SET prcfo = NULL")
    con.commit()
    con.close()

    frame = pd.DataFrame({"file_name": [os.path.basename(p) for p in paths],
                          "pred": [0.1, 0.2, 0.3]})
    report = merge_prediction_results(frame, db, {"pred": ("pred", "REAL")})

    assert report.key == "file_name"
    assert report.matched_rows == 3
    assert read(db, "pred")["pred"].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_a_timelapse_prcfo_is_rebuilt_with_its_timepoint(tmp_path):
    """The rebuilt key must match the writer's, timepoint included."""
    from spacr.predictions import merge_prediction_results
    from spacr.utils import filepaths_to_database

    src = str(tmp_path / "screen")
    folder = os.path.join(src, "data", "plate1_A1", "cell_png")
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(src, "measurements"), exist_ok=True)
    paths = [os.path.join(folder, f"plate1_A1_1_{t}_{o}.png")
             for t in (1, 2) for o in (1, 2)]
    filepaths_to_database(paths, {"timelapse": True}, src, "cell")
    db = db_of(src)
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE png_list DROP COLUMN prcfo")
    con.execute("ALTER TABLE png_list DROP COLUMN png_path")
    con.execute("ALTER TABLE png_list DROP COLUMN file_name")
    con.commit()
    con.close()

    frame = pd.DataFrame({
        "prcfo": [f"plate1_r1_c1_f1_t{t}_o{o}" for t in (1, 2) for o in (1, 2)],
        "pred": [0.1, 0.2, 0.3, 0.4]})
    report = merge_prediction_results(frame, db, {"pred": ("pred", "REAL")})

    assert report.key == "prcfo"
    assert report.matched_rows == 4
    assert read(db, "pred")["pred"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_a_timelapse_database_keys_on_the_timepoint_too(tmp_path):
    """``prcfo`` carries the timepoint when the table does."""
    from spacr.predictions import merge_cv_predictions
    from spacr.utils import filepaths_to_database

    src = str(tmp_path / "screen")
    folder = os.path.join(src, "data", "plate1_A1", "cell_png")
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(src, "measurements"), exist_ok=True)
    paths = [os.path.join(folder, f"plate1_A1_1_{t}_{o}.png")
             for t in (1, 2) for o in (1, 2)]
    filepaths_to_database(paths, {"timelapse": True}, src, "cell")
    db = db_of(src)
    assert "timeID" in column_names(db)

    report = merge_cv_predictions(
        vision_results(paths, [0.1, 0.2, 0.3, 0.4]), db)

    assert report.key == "prcfo"
    assert report.matched_rows == 4
    back = read(db, "prcfo, pred")
    assert back["prcfo"].tolist() == ["plate1_r1_c1_f1_t1_o1",
                                      "plate1_r1_c1_f1_t1_o2",
                                      "plate1_r1_c1_f1_t2_o1",
                                      "plate1_r1_c1_f1_t2_o2"]
    assert back["pred"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_a_prcfo_indexed_frame_is_keyed_off_its_index(tmp_path):
    """``_read_and_merge_data`` hands back a frame indexed by prcfo."""
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    frame = pd.DataFrame({"pred": [0.4, 0.5, 0.6]},
                         index=pd.Index([f"plate1_r1_c1_f1_o{o}" for o in (1, 2, 3)],
                                        name="prcfo"))

    report = merge_prediction_results(frame, db, {"pred": ("pred", "REAL")})

    assert report.key == "prcfo"
    assert read(db, "pred")["pred"].tolist() == pytest.approx([0.4, 0.5, 0.6])


def test_nan_scores_are_stored_as_null(tmp_path):
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    frame = pd.DataFrame({
        "prcfo": [f"plate1_r1_c1_f1_o{o}" for o in (1, 2, 3)],
        "pred": [0.1, float("nan"), 0.3],
        "cls": [1, None, 0]})

    merge_prediction_results(frame, db, {"pred": ("pred", "REAL"),
                                         "cv_predictions": ("cls", "INTEGER")})

    back = read(db, "pred, cv_predictions")
    assert back["pred"].isna().tolist() == [False, True, False]
    assert back["cv_predictions"].tolist()[0] == 1
    assert pd.isna(back["cv_predictions"].tolist()[1])


def test_a_value_that_cannot_be_cast_becomes_null(tmp_path):
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    frame = pd.DataFrame({
        "prcfo": [f"plate1_r1_c1_f1_o{o}" for o in (1, 2, 3)],
        "score": ["nope", 0.2, 0.3],
        "cls": ["nope", 1, 0]})

    merge_prediction_results(frame, db, {"pred": ("score", "REAL"),
                                         "cv_predictions": ("cls", "INTEGER"),
                                         "note": ("score", "TEXT")})

    back = read(db, "pred, cv_predictions, note")
    assert pd.isna(back["pred"][0]) and pd.isna(back["cv_predictions"][0])
    assert back["note"][0] == "nope"


def test_an_empty_results_frame_adds_the_columns_and_matches_nothing(tmp_path):
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    frame = pd.DataFrame({"prcfo": [], "pred": []})

    report = merge_prediction_results(frame, db, {"pred": ("pred", "REAL")})

    assert report.matched_rows == 0
    assert report.result_rows == 0
    assert "pred" in column_names(db)


def test_a_mapping_value_may_be_a_bare_column_name(tmp_path):
    """``{db_col: 'src_col'}`` defaults the type to REAL."""
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    frame = pd.DataFrame({"prcfo": [f"plate1_r1_c1_f1_o{o}" for o in (1, 2, 3)],
                          "pred": [0.1, 0.2, 0.3]})

    merge_prediction_results(frame, db, {"pred": "pred"})

    con = sqlite3.connect(db)
    try:
        types = {r[1]: r[2] for r in con.execute("PRAGMA table_info(png_list)")}
    finally:
        con.close()
    assert types["pred"] == "REAL"


def test_a_non_dataframe_results_object_is_accepted(tmp_path):
    from spacr.predictions import merge_prediction_results

    src = str(tmp_path / "screen")
    write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    db = db_of(src)
    records = [{"prcfo": f"plate1_r1_c1_f1_o{o}", "pred": o / 10}
               for o in (1, 2, 3)]

    report = merge_prediction_results(records, db, {"pred": ("pred", "REAL")})

    assert report.matched_rows == 3


def test_the_report_prints_itself(tmp_path):
    from spacr.predictions import merge_cv_predictions

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    report = merge_cv_predictions(vision_results(paths, [0.1, 0.2, 0.3]),
                                  db_of(src), verbose=False)
    text = str(report)
    assert "3/3 rows matched" in text
    assert report.summary() == text


# ---------------------------------------------------------------------------
# deep_spacr's wrapper keeps its contract
# ---------------------------------------------------------------------------

def test_the_deep_spacr_wrapper_returns_the_matched_count(tmp_path):
    from spacr.deep_spacr import merge_predictions_into_db

    src = str(tmp_path / "screen")
    paths = write_png_list(src, "plate1", wells=("A1",), fields=(1,))
    df = vision_results(paths, [0.1, 0.2, 0.9])

    assert merge_predictions_into_db(df, db_of(src)) == 3
    assert merge_predictions_into_db(df, str(tmp_path / "no.db")) is None
    back = read(db_of(src), "pred, cv_predictions")
    assert back["pred"].tolist() == pytest.approx([0.1, 0.2, 0.9])
    assert back["cv_predictions"].tolist() == [0, 0, 1]


# ---------------------------------------------------------------------------
# the same key, one layer up: interperate_vision_model's scores join
# ---------------------------------------------------------------------------
#
# spacr.ml.interperate_vision_model merges the scores CSV onto the *object*
# tables rather than onto png_list, but it is answering the same question --
# which object is this crop? -- so it has to use the same key. It did not: it
# joined on plate/row/column/field/object with no timepoint, which was
# harmless only for as long as io._read_and_merge_data was collapsing the
# frames on the way in. It no longer is (commit 5a64981), so that join now
# multiplies a timelapse dataset by its frame count.

TL_TIMES = (1, 2, 3)
TL_OBJECTS = (1, 2)


def timelapse_src(tmp_path, name="timelapse"):
    """A real timelapse measurements.db: cell table + png_list, real writers."""
    import numpy as np
    from spacr.utils import _merge_and_save_to_database, filepaths_to_database

    src = str(tmp_path / name)
    os.makedirs(os.path.join(src, "measurements"), exist_ok=True)
    for t in TL_TIMES:
        file_name = f"plate1_A1_1_{t}"
        morph = pd.DataFrame({"label": list(TL_OBJECTS),
                              "cell_area": [100.0 * o + t for o in TL_OBJECTS]})
        intensity = pd.DataFrame({
            "label": list(TL_OBJECTS),
            "cell_channel_0_mean_intensity": [10.0 * o + t for o in TL_OBJECTS]})
        _merge_and_save_to_database(morph, intensity, "cell", src, file_name,
                                    "exp", timelapse=True)

    folder = os.path.join(src, "data", "plate1_A1", "cell_png")
    os.makedirs(folder, exist_ok=True)
    paths = [os.path.join(folder, f"plate1_A1_1_{t}_{o}.png")
             for t in TL_TIMES for o in TL_OBJECTS]
    filepaths_to_database(paths, {"timelapse": True}, src, "cell")
    return src, paths


def vision_settings(src, scores_csv, **over):
    settings = dict(src=src, scores=str(scores_csv), tables=["cell"],
                    score_column="cv_predictions", feature_importance=True,
                    permutation_importance=False, shap=False, top_features=3,
                    n_jobs=1, save=False, nuclei_limit=10, pathogen_limit=10)
    settings.update(over)
    return settings


def test_a_timelapse_scores_join_does_not_fan_out(tmp_path):
    """One row per object per frame in, one row per object per frame out.

    Against the timepoint-free join this returns 18 rows for 6 objects --
    every frame's object matched to every frame's score -- and the features
    of each object appear three times.
    """
    from spacr.ml import interperate_vision_model

    src, paths = timelapse_src(tmp_path)
    scores = vision_results(paths, [0.05, 0.15, 0.45, 0.55, 0.85, 0.95])
    scores_csv = tmp_path / "scores.csv"
    scores.to_csv(scores_csv, index=False)

    merged = interperate_vision_model(vision_settings(src, scores_csv))

    n_objects = len(TL_TIMES) * len(TL_OBJECTS)
    assert len(merged) == n_objects, "the join must not multiply by the frames"
    # ... and every object-frame carries its own score, not another frame's
    got = dict(zip(
        merged["timeID"].astype(str) + "_" + merged["object_label"].astype(str),
        merged["cv_predictions"]))
    want = {f"t{t}_{o}": int(p >= 0.5)
            for (t, o), p in zip([(t, o) for t in TL_TIMES for o in TL_OBJECTS],
                                 [0.05, 0.15, 0.45, 0.55, 0.85, 0.95])}
    assert got == want


def test_a_scores_file_with_no_recoverable_timepoint_is_refused(tmp_path):
    """Better a loud stop than a silently multiplied frame."""
    from spacr.io import TimelapseKeyMismatch
    from spacr.ml import interperate_vision_model

    src, paths = timelapse_src(tmp_path)
    # a hand-written scores file: metadata columns, no crop name to read the
    # timepoint off, so there is no timepoint on the scores side at all
    scores = pd.DataFrame({
        "plateID": ["plate1"] * 2, "rowID": ["r1"] * 2, "columnID": ["c1"] * 2,
        "fieldID": ["f1"] * 2, "object_label": ["1", "2"],
        "cv_predictions": [0, 1]})
    scores_csv = tmp_path / "scores_no_time.csv"
    scores.to_csv(scores_csv, index=False)

    with pytest.raises(TimelapseKeyMismatch, match="timepoint"):
        interperate_vision_model(vision_settings(src, scores_csv))


def test_a_doubled_scores_file_is_caught_as_a_fan_out(tmp_path):
    """Two score rows per object duplicate every feature; say so."""
    from spacr.io import JoinFanOut
    from spacr.ml import interperate_vision_model

    src, paths = timelapse_src(tmp_path)
    scores = vision_results(paths, [0.1, 0.2, 0.3, 0.4, 0.6, 0.7])
    doubled = pd.concat([scores, scores], ignore_index=True)
    scores_csv = tmp_path / "scores_doubled.csv"
    doubled.to_csv(scores_csv, index=False)

    with pytest.raises(JoinFanOut, match="duplicated"):
        interperate_vision_model(vision_settings(src, scores_csv))


def test_a_non_timelapse_scores_join_is_unchanged(tmp_path):
    """No timepoint anywhere: the key, and the result, are what they were."""
    import numpy as np
    from spacr.ml import interperate_vision_model
    from spacr.utils import _merge_and_save_to_database

    src = str(tmp_path / "flat")
    os.makedirs(os.path.join(src, "measurements"), exist_ok=True)
    for field in (1, 2):
        morph = pd.DataFrame({"label": [1, 2],
                              "cell_area": [100.0 * field, 200.0 * field]})
        intensity = pd.DataFrame({
            "label": [1, 2],
            "cell_channel_0_mean_intensity": [10.0 * field, 20.0 * field]})
        _merge_and_save_to_database(morph, intensity, "cell", src,
                                    f"plate1_A1_{field}", "exp")
    paths = [f"plate1_A1_{f}_{o}.png" for f in (1, 2) for o in (1, 2)]
    scores = vision_results(paths, [0.1, 0.2, 0.8, 0.9])
    scores_csv = tmp_path / "flat_scores.csv"
    scores.to_csv(scores_csv, index=False)

    merged = interperate_vision_model(vision_settings(src, scores_csv))

    assert len(merged) == 4
    assert merged["cv_predictions"].tolist() == [0, 0, 1, 1]


def test_a_timelapse_crops_object_id_comes_from_the_name_not_a_position(tmp_path):
    """The object id is the LAST token of a crop name, never the fourth.

    ``path.split('_')[3]`` is the object on ``plate_well_field_object`` and
    the *timepoint* on ``plate_well_field_time_object``.
    ``process_vision_results`` used to take ``[3]`` and so read the timepoint
    (MEASURED: ``['2', '3']`` for these two names, whose objects are 7 and 9);
    it splits from the right now, which is correct for both layouts. This
    assertion had been left pinning the old answer. Rebuilding the metadata
    with the writer's own parser is what makes the join land on the right
    row, and both derivations must agree.
    """
    from spacr.predictions import crop_name_metadata
    from spacr.utils import process_vision_results

    names = ["plate1_A1_1_2_7.png", "plate1_A1_1_3_9.png"]
    positional = process_vision_results(
        pd.DataFrame({"path": names, "pred": [0.1, 0.9]}), 0.5)
    assert positional["object"].tolist() == ["7", "9"], "not the timepoint"

    parsed = crop_name_metadata(names, timelapse=True)
    assert parsed["object_label"].tolist() == ["7", "9"]
    assert parsed["timeID"].tolist() == ["t2", "t3"]
    assert parsed["prcfo"].tolist() == ["plate1_r1_c1_f1_t2_o7",
                                        "plate1_r1_c1_f1_t3_o9"]
    # The two derivations of the object id are one answer now.
    assert positional["object"].tolist() == parsed["object_label"].tolist()


def test_crop_name_metadata_marks_a_name_it_cannot_parse(tmp_path):
    from spacr.predictions import crop_name_metadata

    parsed = crop_name_metadata(["plate1_A1_1_3.png", "junk.png", None])
    assert parsed["prcfo"].tolist()[0] == "plate1_r1_c1_f1_o3"
    assert parsed["object_label"].tolist()[0] == "3"
    assert parsed["prcfo"].isna().tolist() == [False, True, True]


def test_the_legacy_timepoint_spelling_on_a_scores_file_still_joins(tmp_path):
    """``time_id`` and ``timeID`` are one concept; either spells the key.

    A scores file written before the two were unified carries ``time_id``
    while the object table carries ``timeID``. Refusing that pair would send a
    user back to re-score a dataset over a column name.
    """
    from spacr.ml import interperate_vision_model

    src, paths = timelapse_src(tmp_path)
    scores = pd.DataFrame({
        "plateID": ["plate1"] * 6,
        "rowID": ["r1"] * 6,
        "columnID": ["c1"] * 6,
        "fieldID": ["f1"] * 6,
        "time_id": [f"t{t}" for t in TL_TIMES for _ in TL_OBJECTS],
        "object_label": [str(o) for _ in TL_TIMES for o in TL_OBJECTS],
        "cv_predictions": [0, 0, 0, 1, 1, 1]})
    scores_csv = tmp_path / "legacy_time_scores.csv"
    scores.to_csv(scores_csv, index=False)

    merged = interperate_vision_model(vision_settings(src, scores_csv))

    assert len(merged) == 6
    assert (dict(zip(merged["timeID"].astype(str) + "_"
                     + merged["object_label"].astype(str),
                     merged["cv_predictions"]))
            == {"t1_1": 0, "t1_2": 0, "t2_1": 0, "t2_2": 1,
                "t3_1": 1, "t3_2": 1})
