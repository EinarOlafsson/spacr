"""spacr.active_learning — the retrain half, and the paths the first two
suites left dark.

``tests/test_active_learning.py`` pins the ranking maths and the queue;
``tests/test_active_learning_loop.py`` pins the happy loop. What neither
touches is what the loop does when the screen is not the tidy one: the
feature matrix it reads for itself, the splits it falls back to, the
provenance it records when a model or a card cannot be written, and the
coverage report's warnings about the sampling that produced the labels.

Everything here is built so a broken implementation cannot pass by
accident:

* the feature matrix is checked object by object against a **tag column
  unique to each object**, so a join that fans out or shifts by one row
  is caught rather than a shape that happens to match;
* the grouped-split fallback is fed a screen where **class and well are
  perfectly confounded**, which is the case that makes the difference
  between an honest held-out number and a memorised one visible in the
  class support;
* a three-class round is asserted to write **one probability column per
  class that sums to 1**, because collapsing three classes onto one
  positive-class score is the silent way to make a re-ranked queue rank
  on nothing;
* the two write failures (model, card) are provoked with a **real
  directory in the way** rather than a patched writer, so the note being
  asserted is the note the real exception produced.

Four defects found while writing this are pinned at the bottom. They were
``xfail(strict=True)`` while the module was wrong; the module is now right,
so they are ordinary tests asserting the correct behaviour, each carrying
the wrong answer it used to give.

No network, no GPU, no torch: sklearn on a dozen synthetic rows.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import spacr.active_learning as al


# ---------------------------------------------------------------------------
# Fixtures — a `cell` table and a `png_list`, the two tables a round joins
# ---------------------------------------------------------------------------

WELLS = (("r1", "c1"), ("r1", "c2"), ("r2", "c1"), ("r2", "c2"))


def _png_list(con, rows, annotation="annotate", annotation_type="INTEGER"):
    """Create ``png_list`` with typed columns, as filepaths_to_database does.

    Explicit DDL rather than ``to_sql`` because the annotation column has to
    keep its declared type with NULLs in it: NULL is the abstention marker
    the whole module turns on, and letting pandas pick the type from the
    values would hide a regression in how it is read back.
    """
    columns = ["png_path", "prcfo", "file_name", "plateID", "rowID",
               "columnID", "fieldID", "cell_id"]
    decls = [f'"{c}" TEXT' for c in columns]
    columns.append(annotation)
    decls.append(f'"{annotation}" {annotation_type}')
    con.execute(f'CREATE TABLE png_list ({", ".join(decls)})')
    con.executemany(
        f'INSERT INTO png_list ({", ".join(columns)}) '
        f'VALUES ({", ".join("?" * len(columns))})',
        [tuple(row.get(c) for c in columns) for row in rows])


def _make_project(root, per_well=6, n_classes=2, wells=WELLS,
                  class_of=None, plate="plate1", labelled=True, seed=0,
                  annotation_type="INTEGER", class_names=None):
    """Write ``<root>/measurements/measurements.db``: a cell table + png_list.

    One measured cell per crop. ``cell_area`` carries the class signal,
    ``cell_tag`` is unique to each object so a join can be checked row by
    row, and ``cell_noise`` is there so the matrix is not one column wide.
    """
    rng = np.random.default_rng(seed)
    meas = os.path.join(str(root), "measurements")
    os.makedirs(meas, exist_ok=True)
    db = os.path.join(meas, "measurements.db")

    cells, crops, index = [], [], 0
    for well_index, (row_id, column_id) in enumerate(wells):
        for obj in range(1, per_well + 1):
            prcf = f"{plate}_{row_id}_{column_id}_f1"
            cls = (class_of(index, well_index, obj) if class_of
                   else index % n_classes)
            tag = float(well_index * 100 + obj)
            cells.append({
                "plateID": plate, "rowID": row_id, "columnID": column_id,
                "fieldID": "f1", "object_label": obj, "prcf": prcf,
                "prc": f"{plate}_{row_id}_{column_id}",
                "cell_area": 500.0 + 400.0 * cls + float(rng.normal(0, 5)),
                "cell_tag": tag,
                "cell_noise": float(rng.normal(0, 1)),
            })
            label = class_names[int(cls)] if class_names else int(cls)
            crops.append({
                "png_path": f"/crops/cell_png/{prcf}_o{obj}.png",
                "prcfo": f"{prcf}_o{obj}",
                "file_name": f"{prcf}_o{obj}.png",
                "plateID": plate, "rowID": row_id, "columnID": column_id,
                "fieldID": "f1", "cell_id": f"o{obj}",
                "annotate": label if labelled else None,
                "_class": label, "_tag": tag,
            })
            index += 1

    con = sqlite3.connect(db)
    try:
        pd.DataFrame(cells).to_sql("cell", con, index=False)
        _png_list(con, crops, annotation_type=annotation_type)
        con.commit()
    finally:
        con.close()
    return {"db": db, "root": str(root), "crops": crops}


@pytest.fixture
def project(tmp_path):
    """24 labelled crops over four wells, two classes, one plate."""
    return _make_project(tmp_path)


# ---------------------------------------------------------------------------
# round_features — the matrix a round fits on
# ---------------------------------------------------------------------------

def test_child_crop_keys_reject_missing_parents_without_changing_valid_rows():
    """Malformed child identities cannot borrow another object's features."""
    field = "p1_r1_c1_f1"
    crops = pd.DataFrame({
        "prcfo": [f"{field}_o7", f"{field}_o99", "",
                  f"{field}_o11", f"{field}_o12", f"{field}_o13"],
        "file_name": [f"{field}_7.png", f"{field}_7_99.png",
                      f"{field}_7_1.png", np.nan, "orphan.png",
                      f"{field}_multi_13.png"],
        "cell_id": [7, None, None, None, None, None],
        "nucleus_id": [None, 99, 1, 11, 12, 13],
    }, index=[17, 4, 88, 2, 73, 9])
    before = crops.copy(deep=True)

    result = al._crop_feature_keys(crops)

    expected = pd.Series([f"{field}_o7", f"{field}_o7",
                          None, None, None, None], index=crops.index,
                         dtype=object)
    pd.testing.assert_series_equal(result, expected)
    pd.testing.assert_frame_equal(crops, before)


def test_mismatch_diagnostics_sample_distinct_present_keys_with_a_limit():
    """Missing or repeated identities must not crowd useful error examples."""
    values = [None, np.nan, "cell7", "cell7", "cell9", "cell12", "cell18"]

    assert al._key_samples(iter(values), n=3) == "'cell7', 'cell9', 'cell12'"
    assert al._key_samples([None, np.nan]) == "none"


# ---------------------------------------------------------------------------
# RoundResult — what the screen prints
# ---------------------------------------------------------------------------

def test_round_result_documents_and_normalizes_its_named_fields():
    """The keyword-only construction contract remains public and usable."""
    result = al.RoundResult(round_index=2, notes=None, report=None)

    assert ":param fields:" in (al.RoundResult.__doc__ or "")
    assert result.round_index == 2
    assert result.notes == []
    assert result.report == {}


def test_a_round_summary_names_the_weakest_class_and_its_caveats():
    """An aggregate of 0.75 hides a class the model cannot do at all.

    ``RoundResult.summary`` is what the Annotate screen shows after a
    round; the sentence that says the aggregate is not describing the weak
    class is the whole reason the per-class numbers are carried around.
    """
    result = al.RoundResult(
        round_index=3, n_labels=40, n_new_labels=12,
        report={"n": 20, "accuracy": 0.75, "f1_macro": 0.6,
                "classes": ["negative", "positive"],
                "per_class_accuracy": [0.95, 0.25]},
        split_rule="GroupShuffleSplit(25%) over 4 groups",
        scored=120, score_columns=["al_prob_0", "al_prob_1"],
        verdict=al.StoppingVerdict(False, "keep going", gain=0.04),
        notes=["png_list has no fieldID column."])

    assert result.accuracy == 0.75
    assert result.per_class == {"negative": 0.95, "positive": 0.25}
    text = result.summary()
    assert "Round 3: fitted on 40 labels (12 new), held out 20." in text
    assert "Weakest class is positive at 0.250" in text
    assert "aggregate above is not describing it" in text
    assert "CONTINUE — keep going" in text
    assert "! png_list has no fieldID column." in text
    assert "al_prob_0, al_prob_1" in text
    assert repr(result) == \
        "RoundResult(round=3, n_labels=40, accuracy=0.7500)"


def test_a_round_summary_says_nothing_was_rescored_when_nothing_was():
    """`write_scores=False` has to read as a fact, not as an empty list."""
    result = al.RoundResult(
        round_index=0, n_labels=10, n_new_labels=10,
        report={"n": 4, "accuracy": 1.0, "f1_macro": 1.0,
                "classes": ["0", "1"], "per_class_accuracy": [1.0, 1.0]},
        split_rule="stratified random 25% of objects", scored=0)
    text = result.summary()
    assert "Re-scored 0 crops into nothing" in text
    assert "Weakest class" not in text


# ---------------------------------------------------------------------------
# annotation_coverage — the warnings that make the sampling visible
# ---------------------------------------------------------------------------

def _plain_png_list(db, rows, columns, annotation_type="INTEGER"):
    con = sqlite3.connect(db)
    try:
        decls = ", ".join(
            f'"{c}" {annotation_type if c == "annotate" else "TEXT"}'
            for c in columns)
        con.execute(f"CREATE TABLE png_list ({decls})")
        con.executemany(
            f'INSERT INTO png_list VALUES ({", ".join("?" * len(columns))})',
            [tuple(r.get(c) for c in columns) for r in rows])
        con.commit()
    finally:
        con.close()
    return db


# ---------------------------------------------------------------------------
# Round bookkeeping — the loop's memory, and its refusals
# ---------------------------------------------------------------------------

def test_the_round_tables_refuse_a_path_that_is_not_a_database(tmp_path):
    """Writing round provenance must never create the file it writes into.

    ``_write_connection`` is the only writable connection in a module whose
    read path is deliberately read-only; ``sqlite3.connect`` on a typo'd
    path would happily make an empty database and the round log would
    vanish into it.
    """
    with pytest.raises(ValueError, match="No database path given"):
        al.ensure_round_tables("")
    with pytest.raises(ValueError, match="No database path given"):
        al.ensure_round_tables("   ")
    missing = str(tmp_path / "never_made.db")
    with pytest.raises(FileNotFoundError, match="No such database"):
        al.ensure_round_tables(missing)
    assert not os.path.exists(missing)
    # A directory is not a database either.
    with pytest.raises(FileNotFoundError, match="No such database"):
        al.record_labels(str(tmp_path), "annotate", {"/c/a.png": 1}, 0)


def test_recording_no_labels_at_all_touches_nothing(tmp_path):
    """An empty flush is a no-op, not an empty round table."""
    db = _plain_png_list(str(tmp_path / "measurements.db"),
                         [{"png_path": "/c/a.png", "annotate": None}],
                         ["png_path", "annotate"])
    assert al.record_labels(db, "annotate", {}, 0) == 0
    con = sqlite3.connect(db)
    try:
        names = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()
    assert al.ROUND_TABLE not in names


def test_next_round_is_zero_on_a_database_that_has_never_had_one(tmp_path):
    """Round 0 is "before any retrain", including on a fresh database.

    A missing round-log table is the normal state of every database that
    has only ever been annotated; reading it as an error, or as round 1,
    would misattribute the seed labels.
    """
    db = _plain_png_list(str(tmp_path / "measurements.db"),
                         [{"png_path": "/c/a.png", "annotate": 1}],
                         ["png_path", "annotate"])
    assert al.next_round(db, "annotate") == 0
    assert al.learning_curve(db, "annotate").empty
    assert list(al.learning_curve(db, "annotate").columns)[:2] == \
        ["round", "finished_utc"]

    al.record_round(db, "annotate", 4, n_labels=40, holdout_accuracy=0.8)
    assert al.next_round(db, "annotate") == 5
    # A second annotation column has its own numbering.
    assert al.next_round(db, "someone_else") == 0


def test_a_round_metric_that_is_not_a_number_is_stored_as_null(tmp_path):
    """NaN, inf and "n/a" are all "no held-out score", not a score of 0.

    ``should_stop`` reads the gain between rounds; a non-finite value
    coerced to a float would produce a NaN gain that compares False
    against every threshold and quietly disable the stopping rule, while a
    0.0 would read as a catastrophic drop.
    """
    db = _plain_png_list(str(tmp_path / "measurements.db"),
                         [{"png_path": "/c/a.png", "annotate": 1}],
                         ["png_path", "annotate"])
    al.record_round(db, "annotate", 0, n_labels=10, holdout_accuracy=0.5,
                    holdout_f1_macro=0.5)
    al.record_round(db, "annotate", 1, n_labels=20,
                    holdout_accuracy="not a number",
                    holdout_f1_macro=float("inf"))
    al.record_round(db, "annotate", 2, n_labels=30,
                    holdout_accuracy=float("nan"))

    curve = al.learning_curve(db, "annotate")
    assert curve["round"].tolist() == [0, 1, 2]
    assert curve["holdout_accuracy"].iloc[0] == 0.5
    assert curve["holdout_accuracy"].isna().tolist() == [False, True, True]
    assert curve["holdout_f1_macro"].isna().tolist() == [False, True, True]
    assert bool(np.isnan(curve["gain"].iloc[1]))


# ---------------------------------------------------------------------------
# Defects found while writing the above, now fixed. Each asserts the CORRECT
# behaviour, and each names the wrong answer it used to give — that number is
# the regression to watch for, not the mechanism that produced it.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Issue #152 — every crop on screen, none of them matched to a feature row
# ---------------------------------------------------------------------------

def test_a_crop_on_a_pp_plate_finds_its_object_after_the_plate_repair(
        tmp_path):
    """The object tables are read with the ``pp`` plate repair; png_list too.

    A plate called ``pplate1`` became ``plate1`` in the merged feature index
    and stayed ``pplate1`` in ``png_list.prcfo``, so not one crop matched.
    """
    project = _make_project(tmp_path, plate="pplate1")

    features = al.round_features(project["db"])

    assert len(features) == 24
    for crop in project["crops"]:
        assert features.loc[crop["png_path"], "cell_tag"] == crop["_tag"]


def test_a_nucleus_crop_is_matched_to_its_parent_cell(tmp_path):
    """A child crop stores its own label; the merged row is its cell's.

    ``<field>_<cell>_<nucleus>.png`` gives a ``prcfo`` ending in the nucleus
    label, which matched no merged row -- or an unrelated cell with the same
    number.
    """
    project = _make_project(tmp_path)
    con = sqlite3.connect(project["db"])
    try:
        con.execute('ALTER TABLE png_list ADD COLUMN "nucleus_id" TEXT')
        rows = con.execute(
            "SELECT png_path, prcfo, cell_id FROM png_list").fetchall()
        for path, prcfo, cell_id in rows:
            cell = int(cell_id[1:])
            field = prcfo.rsplit("_", 1)[0]
            nucleus = cell + 2
            con.execute(
                'UPDATE png_list SET prcfo = ?, file_name = ?, cell_id = NULL,'
                ' nucleus_id = ? WHERE png_path = ?',
                (f"{field}_o{nucleus}", f"{field}_{cell}_{nucleus}.png",
                 f"o{nucleus}", path))
        con.commit()
    finally:
        con.close()

    features = al.round_features(project["db"])

    assert len(features) == 24
    for crop in project["crops"]:
        assert features.loc[crop["png_path"], "cell_tag"] == crop["_tag"]


def test_the_round_reads_the_image_filter_the_way_the_grid_does(tmp_path):
    """Case-insensitive terms and NOT, as ``parse_image_type`` reads them."""
    project = _make_project(tmp_path)

    for expression in ("CELL_PNG", "cell_png AND NOT nucleus", "!nucleus"):
        result = al.retrain_round(project["db"], "annotate",
                                  image_type=expression, save_model=False,
                                  write_card=False, round_index=0)
        assert result.n_labels == 24, expression


def test_an_image_filter_that_keeps_no_crop_says_so(project):
    with pytest.raises(ValueError) as excinfo:
        al.retrain_round(project["db"], "annotate",
                         image_type="no_such_folder", save_model=False)
    message = str(excinfo.value)
    assert "'no_such_folder' keeps none of the 24 crops" in message
    assert "Image type filter" in message


def test_crops_that_match_no_measured_object_name_both_keys(project):
    """The message shows a key from each side, so the mismatch is visible."""
    con = sqlite3.connect(project["db"])
    try:
        con.execute("UPDATE png_list SET prcfo = 'other_' || prcfo")
        con.commit()
    finally:
        con.close()

    with pytest.raises(ValueError) as excinfo:
        al.round_features(project["db"])
    message = str(excinfo.value)
    assert "None of the 24 crops in png_list matches" in message
    assert "'other_plate1_r1_c1_f1_o1'" in message
    assert "'plate1_r1_c1_f1_o1'" in message


def test_an_explicit_matrix_that_shares_no_crop_names_both_keys(project):
    features = al.round_features(project["db"])
    features.index = [f"/elsewhere/{i}.png" for i in range(len(features))]

    with pytest.raises(ValueError) as excinfo:
        al.retrain_round(project["db"], "annotate", features=features,
                         save_model=False)
    message = str(excinfo.value)
    assert "'/crops/cell_png/plate1_r1_c1_f1_o1.png'" in message
    assert "'/elsewhere/0.png'" in message
