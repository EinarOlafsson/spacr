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
from spacr import selection


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


def _label(db, values, column="annotate"):
    con = sqlite3.connect(db)
    try:
        con.executemany(f'UPDATE png_list SET "{column}"=? WHERE png_path=?',
                        [(v, p) for p, v in values.items()])
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# round_features — the matrix a round fits on
# ---------------------------------------------------------------------------

def test_round_features_gives_every_crop_its_own_objects_measurements(project):
    """Each crop's row must be *its* object's, not a neighbour's.

    The join goes through ``prcfo``, and a prcfo join that fans out or
    silently shifts produces a full-sized matrix fitted against the wrong
    labels — a model with a plausible score and no relationship to the
    annotation. ``cell_tag`` is unique per object, so a wrong row is a
    wrong number here, not a wrong shape.
    """
    features = al.round_features(project["db"])

    assert len(features) == 24
    assert features.index.name == "png_path"
    assert set(features.index) == {c["png_path"] for c in project["crops"]}
    for crop in project["crops"]:
        assert features.loc[crop["png_path"], "cell_tag"] == crop["_tag"]
    # prcfo was the join key, not a feature; leaving it in would put a
    # non-numeric column into the fit.
    assert "prcfo" not in features.columns
    assert all(np.issubdtype(dtype, np.number) for dtype in features.dtypes)


def test_round_features_drops_a_crop_with_no_measured_object(project):
    """A crop whose object was never measured has no features, so no row.

    Keeping it with NaNs would put a row of zeros (``nan_to_num``) into the
    fit and let the model learn "unmeasured" as if it were biology.
    """
    con = sqlite3.connect(project["db"])
    try:
        con.execute(
            'INSERT INTO png_list (png_path, prcfo, plateID, rowID, columnID,'
            ' fieldID, cell_id, annotate) VALUES (?,?,?,?,?,?,?,?)',
            ("/crops/cell_png/plate1_r9_c9_f1_o1.png",
             "plate1_r9_c9_f1_o1", "plate1", "r9", "c9", "f1", "o1", 1))
        con.commit()
    finally:
        con.close()

    features = al.round_features(project["db"])
    assert len(features) == 24
    assert "/crops/cell_png/plate1_r9_c9_f1_o1.png" not in features.index


def test_round_features_refuses_a_crop_table_with_no_prcfo(tmp_path):
    """Without prcfo there is no way to say which object a crop shows."""
    db = str(tmp_path / "measurements.db")
    con = sqlite3.connect(db)
    try:
        con.execute('CREATE TABLE png_list ("png_path" TEXT, "annotate" INT)')
        con.execute('INSERT INTO png_list VALUES ("/c/a.png", 1)')
        con.execute('CREATE TABLE cell ("prcf" TEXT)')
        con.commit()
    finally:
        con.close()

    with pytest.raises(ValueError, match="prcfo"):
        al.round_features(db)


def test_round_features_names_the_object_tables_it_could_not_find(tmp_path):
    """"Run Measure first" is the fix, so the error has to say which tables."""
    db = str(tmp_path / "measurements.db")
    con = sqlite3.connect(db)
    try:
        con.execute('CREATE TABLE png_list ("png_path" TEXT, "prcfo" TEXT)')
        con.execute('INSERT INTO png_list VALUES ("/c/a.png", "p_r1_c1_f1_o1")')
        con.commit()
    finally:
        con.close()

    with pytest.raises(ValueError) as excinfo:
        al.round_features(db)
    message = str(excinfo.value)
    assert "cell" in message and "nucleus" in message
    assert "Measure" in message


# ---------------------------------------------------------------------------
# retrain_round — what it refuses, and what it reads for itself
# ---------------------------------------------------------------------------

def test_a_round_reads_its_own_features_when_none_are_given(project):
    """``features=None`` must fit the same model as the explicit matrix.

    The Annotate screen calls this with no matrix at all; if the internal
    read differed from the documented one, every in-screen number would be
    from a different model than the one a script reproduces.
    """
    explicit = al.retrain_round(project["db"], "annotate",
                                features=al.round_features(project["db"]),
                                save_model=False, write_card=False,
                                round_index=0)
    implicit = al.retrain_round(project["db"], "annotate",
                                save_model=False, write_card=False,
                                round_index=0)

    assert implicit.n_labels == explicit.n_labels == 24
    assert implicit.accuracy == explicit.accuracy
    assert implicit.split_rule == explicit.split_rule
    assert implicit.report["confusion_matrix"] == \
        explicit.report["confusion_matrix"]


def test_a_round_refuses_a_column_the_crop_table_does_not_have(project):
    """Naming a column that is not there is a caller error, not zero labels."""
    with pytest.raises(ValueError, match="has no 'not_a_column' column"):
        al.retrain_round(project["db"], "not_a_column", save_model=False)


def test_a_round_refuses_a_feature_matrix_that_shares_no_crop(project):
    """Measure and the crop export must have run over the same objects.

    An empty intersection used to be the shape that fitted on nothing; the
    message has to name the two halves that disagree, because that is the
    only thing the user can act on.
    """
    features = al.round_features(project["db"])
    features.index = [f"/somewhere/else/{i}.png" for i in range(len(features))]

    with pytest.raises(ValueError) as excinfo:
        al.retrain_round(project["db"], "annotate", features=features,
                         save_model=False)
    assert "has a row in the feature matrix" in str(excinfo.value)
    assert "Measure" in str(excinfo.value)


def test_a_round_fits_only_the_crops_the_image_type_filter_keeps(tmp_path):
    """The Annotate screen's filter has to reach the model, not just the view.

    A round fitted on every crop mode in the database while the annotator
    was labelling one of them would report a held-out accuracy for a
    population the annotator never saw.
    """
    project = _make_project(tmp_path)
    con = sqlite3.connect(project["db"])
    try:
        # The same objects, exported a second time as nucleus crops.
        con.execute(
            'INSERT INTO png_list (png_path, prcfo, plateID, rowID, columnID,'
            ' fieldID, cell_id, annotate) '
            'SELECT replace(png_path, "cell_png", "nucleus_png"), prcfo, '
            'plateID, rowID, columnID, fieldID, cell_id, annotate '
            'FROM png_list')
        con.commit()
        assert con.execute("SELECT COUNT(*) FROM png_list").fetchone()[0] == 48
    finally:
        con.close()

    result = al.retrain_round(project["db"], "annotate",
                              image_type="cell_png", save_model=False,
                              write_card=False, round_index=0)
    assert result.n_labels == 24
    assert result.scored == 24
    # Both crop modes share a prcfo, so both reach the feature matrix: the
    # filter is doing the work, not the join.
    unfiltered = al.retrain_round(project["db"], "annotate",
                                  save_model=False, write_card=False,
                                  round_index=0)
    assert unfiltered.n_labels == 48


@pytest.mark.parametrize("model_type", ["random_forest", "gradient_boosting"])
def test_every_supported_model_type_fits_and_reports_the_same_shape(
        project, model_type):
    """The three estimators are interchangeable from the caller's side.

    A round that only worked for the default would make ``model_type`` a
    parameter that silently does nothing but change the string on the card.
    """
    result = al.retrain_round(project["db"], "annotate", model_type=model_type,
                              save_model=False, write_card=False,
                              round_index=0, seed=0)
    assert result.model_type == model_type
    assert result.report["classes"] == ["0", "1"]
    assert result.report["num_classes"] == 2
    assert sum(result.report["class_support"]) == result.report["n"]
    curve = al.learning_curve(project["db"], "annotate")
    assert curve["model_type"].iloc[-1] == model_type


def test_an_unknown_model_type_is_named_back_with_the_real_ones(project):
    """A typo must not fall through to a default nobody asked for."""
    with pytest.raises(ValueError) as excinfo:
        al.retrain_round(project["db"], "annotate", model_type="xgboost",
                         save_model=False)
    message = str(excinfo.value)
    assert "'xgboost'" in message
    assert "logistic_regression" in message and "random_forest" in message


def test_a_round_fits_classes_that_were_annotated_as_text(tmp_path):
    """A class named 'infected' is a class, not an unusable label.

    ``annotate`` is INTEGER when the Annotate app creates it, but a
    database migrated from another tool carries names, and the loop has to
    fit on those rather than turn them into one class (which would trip
    the "keep annotating until the other one appears" refusal on a
    perfectly well-labelled screen).
    """
    project = _make_project(tmp_path, annotation_type="TEXT",
                            class_names={0: "clean", 1: "infected"})
    result = al.retrain_round(project["db"], "annotate", save_model=False,
                              write_card=False, round_index=0)

    assert result.classes == ["clean", "infected"]
    assert result.report["classes"] == ["clean", "infected"]
    assert result.n_labels == 24
    assert set(result.per_class) == {"clean", "infected"}
    # The score columns stay positional — one per class, in class order.
    assert result.score_columns == ["al_prob_0", "al_prob_1"]
    curve = al.learning_curve(project["db"], "annotate")
    assert set(curve["per_class"].iloc[-1]) == {"clean", "infected"}


def test_a_three_class_round_writes_one_probability_per_class(tmp_path):
    """Three classes must produce three columns, not one collapsed score.

    ``build_queue`` re-ranks on ``al_prob_*``; if a three-class round wrote
    a single positive-class column, entropy would be computed over a
    two-class proxy and the queue would rank on a distribution that does
    not exist.
    """
    project = _make_project(tmp_path, per_well=6, n_classes=3,
                            class_of=lambda i, w, o: o % 3)
    result = al.retrain_round(project["db"], "annotate", save_model=False,
                              write_card=False, round_index=0)

    assert result.score_columns == ["al_prob_0", "al_prob_1", "al_prob_2"]
    assert result.scored == 24
    assert result.report["classes"] == ["0", "1", "2"]

    con = sqlite3.connect(project["db"])
    try:
        stored = pd.read_sql_query(
            "SELECT al_prob_0, al_prob_1, al_prob_2 FROM png_list", con)
    finally:
        con.close()
    assert len(stored) == 24
    assert stored.notna().all().all()
    assert stored.sum(axis=1).to_numpy() == pytest.approx(np.ones(24))

    # And the queue reads all three back, in class order, in preference to
    # anything an older Classify run left behind.
    _label(project["db"], {c["png_path"]: None for c in project["crops"]})
    queue = al.build_queue(project["db"], "annotate", diversity="none")
    assert queue.attrs["spacr_active_learning"]["pred_columns"] == \
        ["al_prob_0", "al_prob_1", "al_prob_2"]
    assert len(queue) == 24


def test_a_round_that_cannot_save_its_model_says_so_and_still_records(project):
    """A failed dump must not lose the round that produced the numbers.

    The held-out score is the point of the round; losing it because the
    output directory was in a bad state would silently break the learning
    curve, which is the one artefact that tells the annotator to stop.
    """
    model_dir = os.path.join(project["root"], "al_models")
    os.makedirs(os.path.join(model_dir,
                             "round_007_logistic_regression.joblib"))

    result = al.retrain_round(project["db"], "annotate", save_model=True,
                              model_dir=model_dir, round_index=7)

    assert result.model_path == ""
    assert result.card_path == ""
    assert any("Could not save the round model" in n for n in result.notes)
    curve = al.learning_curve(project["db"], "annotate")
    assert curve["round"].tolist() == [7]
    assert curve["model_path"].iloc[0] == ""
    assert float(curve["holdout_accuracy"].iloc[0]) == result.accuracy


def test_a_round_that_cannot_write_its_card_keeps_the_model(project):
    """A card is documentation; losing it must not lose the checkpoint."""
    model_dir = os.path.join(project["root"], "al_models")
    os.makedirs(model_dir)
    # The card is written beside the model as <stem>.card.json.
    os.makedirs(os.path.join(model_dir,
                             "round_002_logistic_regression.card.json"))

    result = al.retrain_round(project["db"], "annotate", save_model=True,
                              write_card=True, model_dir=model_dir,
                              round_index=2)

    assert result.model_path.endswith("round_002_logistic_regression.joblib")
    assert os.path.isfile(result.model_path)
    assert result.card_path == ""
    assert any("card could not be written" in n for n in result.notes)


def test_a_grouped_split_holds_out_whole_wells_when_no_fold_is_stratified(
        tmp_path):
    """Class perfectly confounded with well: the split must still not leak.

    Two wells are all class 0 and two are all class 1, so no stratified
    grouped fold can contain both classes. The fallback has to keep the
    grouping — a well on both sides of the split is exactly the leak that
    makes active-learning accuracies look wonderful — and it has to say
    that the held-out class balance is now whatever the groups gave.
    """
    project = _make_project(
        tmp_path, per_well=6,
        class_of=lambda i, well, obj: 0 if well < 2 else 1)

    result = al.retrain_round(project["db"], "annotate", group_by="well",
                              holdout=0.25, save_model=False,
                              write_card=False, round_index=0)

    assert "GroupShuffleSplit" in result.split_rule
    assert "no group appears on both sides" in result.split_rule
    assert any("held-out class balance" in n for n in result.notes)
    # One whole well was held out, and a well here is one class only.
    support = result.report["class_support"]
    assert sorted(support) == [0, 6], support
    assert result.report["n"] == 6


# ---------------------------------------------------------------------------
# RoundResult — what the screen prints
# ---------------------------------------------------------------------------

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


def test_coverage_without_plate_metadata_says_so_rather_than_inventing_one(
        tmp_path):
    """No plate map is a caveat on every number below it, so it is a note.

    A crop table with no plate/row/column cannot attribute a label to a
    well, and silently reporting "1 well" would be a claim the data does
    not support — the concentration numbers under it would read as
    "perfectly concentrated" when they mean "unknown".
    """
    db = _plain_png_list(
        str(tmp_path / "measurements.db"),
        [{"png_path": f"/c/{i}.png", "annotate": i % 2} for i in range(12)],
        ["png_path", "annotate"])

    coverage = al.annotation_coverage(db, "annotate")
    meta = coverage.attrs["spacr_annotation_coverage"]

    assert meta["n_annotated"] == 12
    assert meta["by_plate"] == {"(unknown)": 12}
    assert meta["by_well"] == {"(unknown)": 12}
    assert meta["plates_total"] == 1 and meta["wells_total"] == 1
    assert any("no plateID column" in n for n in meta["notes"])
    assert "(unknown)" in al.format_coverage_summary(coverage)


def test_coverage_refuses_a_crop_table_with_no_key_column(tmp_path):
    """A label that cannot be attributed to a crop cannot be attributed at all.

    An empty report would read as "nothing annotated yet", which is a
    different fact about the screen and the wrong one to act on.
    """
    db = _plain_png_list(str(tmp_path / "measurements.db"),
                         [{"prcfo": "p1_r1_c1_f1_o1", "annotate": 1}],
                         ["prcfo", "annotate"])
    with pytest.raises(ValueError, match="no 'png_path' column"):
        al.annotation_coverage(db, "annotate")


def test_coverage_names_a_class_that_never_left_one_well(tmp_path):
    """All of a class from one well cannot be told apart from the well.

    This is the different-in-kind case from "mostly one well": there is no
    second well to compare against at all, so a random split of these
    objects reports an accuracy that cannot transfer, and the note has to
    say that rather than only quoting a share.
    """
    rows = [{"png_path": f"/c/a{i}.png", "plateID": "p1", "rowID": "r1",
             "columnID": "c1", "annotate": 1} for i in range(12)]
    rows += [{"png_path": f"/c/b{i}.png", "plateID": "p1", "rowID": "r2",
              "columnID": f"c{i % 4 + 1}", "annotate": 0} for i in range(12)]
    db = _plain_png_list(str(tmp_path / "measurements.db"), rows,
                         ["png_path", "plateID", "rowID", "columnID",
                          "annotate"])

    meta = al.annotation_coverage(db, "annotate").attrs[
        "spacr_annotation_coverage"]

    assert meta["concentration"]["1"]["n_groups"] == 1
    assert meta["concentration"]["1"]["hhi"] == pytest.approx(1.0)
    assert meta["concentration"]["1"]["effective_groups"] == pytest.approx(1.0)
    note = [n for n in meta["notes"] if n.startswith("Class 1:")]
    assert len(note) == 1
    assert "single well p1/r1/c1" in note[0]
    assert "does not transfer" in note[0]
    # Class 0 came from four wells, so it gets no such note.
    assert not [n for n in meta["notes"] if n.startswith("Class 0:")]


def test_coverage_warns_when_one_class_is_a_fraction_of_the_other(tmp_path):
    """5:1 makes the aggregate accuracy a description of the majority.

    Uncertainty sampling on a 98 %-negative screen produces exactly this,
    and the annotator can only rebalance if the report says it happened.
    """
    rows = [{"png_path": f"/c/n{i}.png", "plateID": "p1", "rowID": "r1",
             "columnID": f"c{i % 6 + 1}", "annotate": 0} for i in range(30)]
    rows += [{"png_path": f"/c/p{i}.png", "plateID": "p1", "rowID": "r2",
              "columnID": f"c{i % 5 + 1}", "annotate": 1} for i in range(5)]
    db = _plain_png_list(str(tmp_path / "measurements.db"), rows,
                         ["png_path", "plateID", "rowID", "columnID",
                          "annotate"])

    meta = al.annotation_coverage(db, "annotate").attrs[
        "spacr_annotation_coverage"]

    assert meta["by_class"] == {"0": 30, "1": 5}
    balance = [n for n in meta["notes"] if n.startswith("Class balance")]
    assert len(balance) == 1
    assert "30:5" in balance[0]
    assert "smallest class has 5 labels" in balance[0]
    assert "per-class accuracy" in balance[0]


def test_coverage_reports_a_text_class_under_its_own_name(tmp_path):
    """A class written as text stays that text; 1.0 and 1 stay one class.

    ``annotate`` is INTEGER when the Annotate app makes it, but a database
    migrated from elsewhere can carry names, and a coverage report that
    stringified them differently per row would split one class in two.
    """
    rows = [{"png_path": f"/c/{i}.png", "plateID": "p1", "rowID": "r1",
             "columnID": "c1", "annotate": "infected" if i % 2 else "clean"}
            for i in range(8)]
    db = _plain_png_list(str(tmp_path / "measurements.db"), rows,
                         ["png_path", "plateID", "rowID", "columnID",
                          "annotate"], annotation_type="TEXT")

    coverage = al.annotation_coverage(db, "annotate")
    meta = coverage.attrs["spacr_annotation_coverage"]

    assert meta["by_class"] == {"clean": 4, "infected": 4}
    assert meta["n_classes"] == 2
    assert sorted(coverage["class"].tolist()) == ["clean", "infected"]
    assert "infected" in al.format_coverage_summary(coverage)


def test_coverage_summary_lists_every_plate_when_there_is_more_than_one(
        tmp_path):
    """Two plates is the first case where "per plate" is worth printing.

    A model trained on labels from one plate of a two-plate screen has
    learned that plate's staining; the per-plate line is what makes the
    imbalance visible before the model is fitted.
    """
    rows = [{"png_path": f"/c/{plate}_{i}.png", "plateID": plate,
             "rowID": "r1", "columnID": f"c{i % 3 + 1}", "annotate": i % 2}
            for plate in ("plate1", "plate2")
            for i in range(6 if plate == "plate1" else 2)]
    db = _plain_png_list(str(tmp_path / "measurements.db"), rows,
                         ["png_path", "plateID", "rowID", "columnID",
                          "annotate"])

    coverage = al.annotation_coverage(db, "annotate")
    text = al.format_coverage_summary(coverage)

    assert coverage.attrs["spacr_annotation_coverage"]["by_plate"] == \
        {"plate1": 6, "plate2": 2}
    per_plate = [line for line in text.splitlines()
                 if line.startswith("Per plate:")]
    assert per_plate == ["Per plate: plate1: 6 · plate2: 2"]
    assert "2/2 plates" in text


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
# holdout_report — the shapes a head can emit
# ---------------------------------------------------------------------------

def test_a_single_probability_column_is_read_as_the_positive_class():
    """``(N, 1)`` and ``(N,)`` are the same binary head, and must agree.

    ``merge_cv_predictions`` writes one REAL column, and a caller who
    slices it out of a frame gets ``(N, 1)`` rather than ``(N,)``. Reading
    that column as "class 0" would invert every prediction while still
    producing a confusion matrix of the right shape.
    """
    truth = [0, 1, 1, 0, 1]
    flat = np.array([0.1, 0.9, 0.2, 0.3, 0.8])

    column = al.holdout_report(truth, flat.reshape(-1, 1))
    assert column == al.holdout_report(truth, flat)
    # Hand-computed: predictions are 0,1,0,0,1 against truth 0,1,1,0,1.
    assert column["confusion_matrix"] == [[2, 0], [1, 2]]
    assert column["accuracy"] == pytest.approx(4 / 5)
    assert column["class_support"] == [2, 3]
    assert column["predicted_support"] == [3, 2]


def test_holdout_report_on_an_empty_held_out_set_is_nan_not_zero():
    """No held-out objects is an unknown accuracy, not a perfect one."""
    report = al.holdout_report([], np.zeros((0, 2)))
    assert report["n"] == 0
    assert np.isnan(report["accuracy"])
    assert report["confusion_matrix"] == [[0, 0], [0, 0]]


# ---------------------------------------------------------------------------
# crops_for_object_keys — the database half of the routing contract
# ---------------------------------------------------------------------------

def _routing_db(tmp_path, rows, columns):
    return _plain_png_list(str(tmp_path / "measurements.db"), rows, columns)


def test_a_timelapse_key_resolves_to_the_crop_of_that_frame(tmp_path):
    """The same object at two timepoints is two crops, and two keys.

    Dropping the timepoint collapses every frame of a tracked object onto
    one key, so a click on frame 12 of a scatter plot opens frame 1 — the
    bug this codebase has hit from the other direction more than once.
    Keys are composed by the real producer, ``selection.object_keys``.
    """
    rows = [{"png_path": f"/crops/cell_png/p1_r1_c1_f1_t{t}_o7.png",
             "plateID": "p1", "rowID": "r1", "columnID": "c1",
             "fieldID": "f1", "timeID": f"t{t}", "cell_id": "o7",
             "annotate": None} for t in (1, 2)]
    db = _routing_db(tmp_path, rows,
                     ["png_path", "plateID", "rowID", "columnID", "fieldID",
                      "timeID", "cell_id", "annotate"])
    frame = pd.DataFrame([
        {"plateID": "p1", "rowID": "r1", "columnID": "c1", "fieldID": "f1",
         "timeID": "t2", "object_label": 7},
        {"plateID": "p1", "rowID": "r1", "columnID": "c1", "fieldID": "f1",
         "timeID": "t1", "object_label": 7},
    ])
    keys = list(selection.object_keys(frame, timelapse=True))
    assert keys == ["p1_r1_c1_f1_t2_7", "p1_r1_c1_f1_t1_7"]

    resolved = al.crops_for_object_keys(db, keys, timelapse=True)
    assert [p for p, _ in resolved] == [
        "/crops/cell_png/p1_r1_c1_f1_t2_o7.png",
        "/crops/cell_png/p1_r1_c1_f1_t1_o7.png"]
    # Without timelapse=True the field-level key names one crop only.
    assert len(al.crops_for_object_keys(db, ["p1_r1_c1_f1_7"])) == 1


def test_crops_for_object_keys_honours_the_image_type_filter(tmp_path):
    """One object, two crop modes: the filter decides which one opens.

    The Annotate screen paginates one crop mode at a time. A routed
    selection that ignored the filter would hand it a nucleus crop while
    the screen was showing cells, and the annotator would label an object
    they were not looking at.
    """
    rows = [{"png_path": "/crops/cell_png/p1_r1_c1_f1_o7.png",
             "plateID": "p1", "rowID": "r1", "columnID": "c1",
             "fieldID": "f1", "cell_id": "o7", "annotate": None},
            {"png_path": "/crops/nucleus_png/p1_r1_c1_f1_o7.png",
             "plateID": "p1", "rowID": "r1", "columnID": "c1",
             "fieldID": "f1", "cell_id": "o7", "annotate": None}]
    db = _routing_db(tmp_path, rows,
                     ["png_path", "plateID", "rowID", "columnID", "fieldID",
                      "cell_id", "annotate"])

    assert al.crops_for_object_keys(db, ["p1_r1_c1_f1_7"],
                                    image_type="nucleus_png") == \
        [("/crops/nucleus_png/p1_r1_c1_f1_o7.png", None)]
    assert al.crops_for_object_keys(db, ["p1_r1_c1_f1_7"],
                                    image_type="cell_png") == \
        [("/crops/cell_png/p1_r1_c1_f1_o7.png", None)]
    assert al.crops_for_object_keys(db, ["p1_r1_c1_f1_7"],
                                    image_type="pathogen_png") == []


def test_a_bare_o_in_an_id_column_mints_no_field_level_key(tmp_path):
    """'o' with no number is not object 0, and must not key on the field.

    ``_object_label`` strips the ``'o'`` prefix that ``png_list`` writes;
    what is left has to be a number. A truncated id that reduced to an
    empty label would compose the key ``'p1_r1_c1_f1_'`` — which every
    other degenerate row in that field would also compose, so one routed
    key would open whichever of them the table happened to list first.
    """
    rows = [{"png_path": "/crops/cell_png/p1_r1_c1_f1_o5.png",
             "prcfo": "p1_r1_c1_f1_o5", "plateID": "p1", "rowID": "r1",
             "columnID": "c1", "fieldID": "f1", "cell_id": "o",
             "annotate": None}]
    db = _routing_db(tmp_path, rows,
                     ["png_path", "prcfo", "plateID", "rowID", "columnID",
                      "fieldID", "cell_id", "annotate"])

    assert al.crops_for_object_keys(db, ["p1_r1_c1_f1_"]) == []
    # The crop is still reachable: prcfo carries the label the id column lost.
    assert al.crops_for_object_keys(db, ["p1_r1_c1_f1_5"]) == \
        [("/crops/cell_png/p1_r1_c1_f1_o5.png", None)]


def test_crops_for_object_keys_refuses_a_table_with_no_key_column(tmp_path):
    """No crop key means no crop to open; that is an error, not zero hits.

    Returning an empty list would look exactly like "those objects have no
    crops", and the caller would go looking for the objects instead of the
    misnamed column.
    """
    db = _plain_png_list(str(tmp_path / "measurements.db"),
                         [{"prcfo": "p1_r1_c1_f1_o1", "annotate": 1}],
                         ["prcfo", "annotate"])
    with pytest.raises(ValueError, match="no 'png_path' column"):
        al.crops_for_object_keys(db, ["p1_r1_c1_f1_1"])


def test_a_crop_row_that_names_two_object_types_claims_neither(tmp_path):
    """Two id columns filled is a row that does not say what it is.

    Which id column holds the label is how a crop states its object type.
    A row with two filled has contradicted itself, and inventing a type
    from the first column would route a nucleus key to a pathogen crop —
    worse than the untyped fallback, which is honest about being
    under-specified.
    """
    rows = [{"png_path": "/crops/p1_r1_c1_f1_o3.png", "plateID": "p1",
             "rowID": "r1", "columnID": "c1", "fieldID": "f1",
             "nucleus_id": "o3", "pathogen_id": "o3", "annotate": None}]
    db = _routing_db(tmp_path, rows,
                     ["png_path", "plateID", "rowID", "columnID", "fieldID",
                      "nucleus_id", "pathogen_id", "annotate"])

    assert al.crops_for_object_keys(db, ["p1_r1_c1_f1_3"]) == \
        [("/crops/p1_r1_c1_f1_o3.png", None)]
    # The typed keys fall back to the untyped one rather than resolving a
    # type the row never stated.
    assert al.crops_for_object_keys(db, ["p1_r1_c1_f1_nucleus3"]) == \
        [("/crops/p1_r1_c1_f1_o3.png", None)]
    assert al.crops_for_object_keys(db, ["p1_r1_c1_f1_pathogen3"]) == \
        [("/crops/p1_r1_c1_f1_o3.png", None)]


# ---------------------------------------------------------------------------
# Defects found while writing the above, now fixed. Each asserts the CORRECT
# behaviour, and each names the wrong answer it used to give — that number is
# the regression to watch for, not the mechanism that produced it.
# ---------------------------------------------------------------------------

def test_an_unknown_group_by_is_refused_the_way_an_unknown_diversity_is(
        project):
    """A group_by spelling this module does not know is a hard error.

    ``group_by='Well'`` used to be accepted in silence: ``_SPLIT_GROUPS`` is
    matched exactly, so the column list came out empty, the groups became
    one-per-object, and the split_rule written to the learning curve AND the
    model card read "StratifiedGroupKFold(4) over 24 groups — no group
    appears on both sides". That is a per-object random split wearing the
    label of the grouped one this module exists to enforce — the exact
    optimistic number, with provenance asserting the opposite.
    ``build_queue`` raises for an unknown ``diversity=``; this raises too.
    """
    with pytest.raises(ValueError, match="group_by") as excinfo:
        al.retrain_round(project["db"], "annotate", group_by="Well",
                         save_model=False, write_card=False)
    message = str(excinfo.value)
    assert "'Well'" in message
    # The valid names, so the fix is in the message rather than the source.
    assert "well" in message and "plate" in message and "field" in message
    # Nothing was recorded: a refused round must not leave a curve point.
    assert al.learning_curve(project["db"], "annotate").empty


def test_group_by_none_records_a_random_split_as_a_random_one(project):
    """The honest spelling of "not grouped" is the same on every path.

    ``group_by='none'`` reached the same singleton-group vector as the
    unknown spelling did, so the documented way to ask for an ungrouped
    split also recorded itself as "no group appears on both sides". The
    rule now says NOT grouped, and a note says what that costs.
    """
    result = al.retrain_round(project["db"], "annotate", group_by="none",
                              save_model=False, write_card=False,
                              round_index=0)
    assert "NOT grouped" in result.split_rule
    assert "no group appears on both sides" not in result.split_rule
    # 'none' is still accepted, and now reports itself as the level it IS
    # -- 'cell' -- because a model card should record the deliberate choice
    # of the leakiest level, not the absence of a strategy.
    assert any("group_by='cell'" in note and "optimistic" in note
               for note in result.notes)
    curve = al.learning_curve(project["db"], "annotate")
    assert "NOT grouped" in curve["split_rule"].iloc[-1]


def test_a_group_by_the_crop_table_cannot_honour_is_not_called_grouped(
        tmp_path):
    """A recognised strategy with none of its columns is still not grouped.

    The refusal above only covers the misspelling. A crop table with no
    plate map cannot group by well however correctly the caller spelled it,
    and the rule has to say so rather than claim a grouping over 24 groups
    of one object each.
    """
    db = _plain_png_list(
        str(tmp_path / "measurements.db"),
        [{"png_path": f"/crops/cell_png/{i}.png", "prcfo": f"p_r1_c1_f1_o{i}",
          "annotate": i % 2} for i in range(24)],
        ["png_path", "prcfo", "annotate"])
    features = pd.DataFrame(
        {"f0": [float(i % 2) * 10 + i * 0.01 for i in range(24)],
         "f1": [float(i) for i in range(24)]},
        index=[f"/crops/cell_png/{i}.png" for i in range(24)])
    features.index.name = "png_path"

    result = al.retrain_round(db, "annotate", features=features,
                              group_by="well", save_model=False,
                              write_card=False, round_index=0)

    assert "NOT grouped" in result.split_rule
    assert "no group appears on both sides" not in result.split_rule
    assert any("none of the columns" in note and "plateID" in note
               for note in result.notes)


def test_holdout_report_scores_every_row_it_says_it_scored():
    """``n`` and the supports describe one population, always.

    A true class outside the probability matrix's columns used to be masked
    out of the confusion matrix while ``n`` still counted it, so this set —
    which the model gets 2 of 3 right — reported ``n=3``, ``accuracy=1.0``
    and a support summing to 2. The ``accuracy == trace/total`` invariant
    still checked out, because the missing row was missing from both sides.
    The row is now an error the model cannot avoid: class 2 has a row in the
    matrix and an empty column, because a two-column head can never predict
    it.
    """
    report = al.holdout_report([0, 1, 2], np.array([[0.9, 0.1],
                                                    [0.2, 0.8],
                                                    [0.6, 0.4]]))
    assert sum(report["class_support"]) == report["n"]
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["n"] == 3
    assert report["num_classes"] == 3 and report["head_classes"] == 2
    assert report["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [1, 0, 0]]
    # The class the head cannot reach is never predicted, and scores 0.
    assert report["predicted_support"] == [2, 1, 0]
    assert report["per_class_accuracy"] == pytest.approx([1.0, 1.0, 0.0])
    # And the card is told why, rather than left to infer it from a shape.
    assert any("can never be predicted" in note for note in report["notes"])


def test_holdout_report_excludes_a_negative_class_and_counts_it_out_loud():
    """-1 is not a class; it leaves the totals rather than joining them.

    The rule is the same one the row above enforces from the other side: a
    row that is not in the matrix must not be in ``n`` either. A negative id
    cannot have a row in a confusion matrix at all, so it leaves — and says
    so, instead of shrinking the denominator in silence.
    """
    report = al.holdout_report([0, 1, -1], np.array([[0.9, 0.1],
                                                     [0.2, 0.8],
                                                     [0.6, 0.4]]))
    assert report["n"] == 2
    assert report["n_unscored"] == 1
    assert sum(report["class_support"]) == report["n"]
    assert report["accuracy"] == pytest.approx(1.0)
    assert any("negative class id" in note for note in report["notes"])


def test_holdout_report_refuses_labels_and_scores_of_different_lengths():
    """Mismatched lengths compare one object's label to another's score."""
    with pytest.raises(ValueError, match="different objects"):
        al.holdout_report([0, 1, 0], np.array([[0.9, 0.1], [0.2, 0.8]]))


def test_an_escaped_object_key_still_finds_its_crop(tmp_path):
    """A key whose metadata needed escaping resolves to its crop.

    ``spacr.selection`` percent-escapes the key separator inside a component
    (fieldID ``'f_1'`` → key ``'plate1_r1_c1_f%5F1_cell7'``) so two distinct
    objects cannot share one key. ``crops_for_object_keys`` composed its
    lookup keys raw with ``'_'.join``, so the escaped key matched nothing:
    the crop was dropped from a routed selection without a word while its
    neighbours opened normally, and the user got fewer crops than they
    picked.
    """
    rows = [{"png_path": "/crops/cell_png/plate1_r1_c1_f_1_o7.png",
             "prcfo": "plate1_r1_c1_f_1_o7", "plateID": "plate1",
             "rowID": "r1", "columnID": "c1", "fieldID": "f_1",
             "cell_id": "o7", "annotate": None}]
    db = _routing_db(tmp_path, rows,
                     ["png_path", "prcfo", "plateID", "rowID", "columnID",
                      "fieldID", "cell_id", "annotate"])
    frame = pd.DataFrame([{"plateID": "plate1", "rowID": "r1",
                           "columnID": "c1", "fieldID": "f_1",
                           "object_label": 7}])
    keys = list(selection.object_keys(frame, object_type="cell"))
    assert keys == ["plate1_r1_c1_f%5F1_cell7"]

    assert al.crops_for_object_keys(db, keys) == \
        [("/crops/cell_png/plate1_r1_c1_f_1_o7.png", None)]
    # The untyped spelling of the same key, which is what an older view
    # sends, resolves to the same crop.
    untyped = list(selection.untyped_object_keys(frame))
    assert untyped == ["plate1_r1_c1_f%5F1_7"]
    assert al.crops_for_object_keys(db, untyped) == \
        [("/crops/cell_png/plate1_r1_c1_f_1_o7.png", None)]
    # And the raw spelling still resolves: keys composed before the escape
    # existed have to go on meaning what they meant.
    assert al.crops_for_object_keys(db, ["plate1_r1_c1_f_1_7"]) == \
        [("/crops/cell_png/plate1_r1_c1_f_1_o7.png", None)]


def test_the_escaped_spelling_wins_when_two_fields_spell_one_key(tmp_path):
    """Carrying both spellings must not reintroduce the collision.

    A field literally named ``'f%5F1'`` spells its RAW key exactly the way a
    field named ``'f_1'`` spells its ESCAPED one — which is why the escape
    percent-encodes ``'%'`` in the first place. The escaped spelling is what
    a producer emits today, so it is the one that wins; otherwise fixing the
    dropped crop would have swapped it for the wrong crop, which is worse.
    """
    rows = [{"png_path": "/crops/cell_png/literal_percent.png",
             "prcfo": "plate1_r1_c1_f%5F1_o7", "plateID": "plate1",
             "rowID": "r1", "columnID": "c1", "fieldID": "f%5F1",
             "cell_id": "o7", "annotate": None},
            {"png_path": "/crops/cell_png/literal_underscore.png",
             "prcfo": "plate1_r1_c1_f_1_o7", "plateID": "plate1",
             "rowID": "r1", "columnID": "c1", "fieldID": "f_1",
             "cell_id": "o7", "annotate": None}]
    db = _routing_db(tmp_path, rows,
                     ["png_path", "prcfo", "plateID", "rowID", "columnID",
                      "fieldID", "cell_id", "annotate"])

    frame = pd.DataFrame([{"plateID": "plate1", "rowID": "r1",
                           "columnID": "c1", "fieldID": f, "object_label": 7}
                          for f in ("f%5F1", "f_1")])
    keys = list(selection.object_keys(frame))
    # The two fields have distinct keys, which is the whole point of escaping
    # '%' as well as the separator.
    assert keys == ["plate1_r1_c1_f%255F1_7", "plate1_r1_c1_f%5F1_7"]

    assert al.crops_for_object_keys(db, keys) == [
        ("/crops/cell_png/literal_percent.png", None),
        ("/crops/cell_png/literal_underscore.png", None)]


def test_the_coverage_denominator_describes_the_filtered_population(tmp_path):
    """Numerator and denominator come from the same population.

    ``n_rows`` used to be counted before the ``image_type`` filter ran, so
    ``format_coverage_summary`` printed "12 of 20 crops annotated" for a
    filtered population of 12 in which nothing was left to annotate — and,
    unlike ``build_queue``, recorded no note that the filter had excluded
    anything.
    """
    rows = [{"png_path": f"/crops/cell_png/{i}.png", "plateID": "p1",
             "rowID": "r1", "columnID": "c1", "annotate": i % 2}
            for i in range(12)]
    rows += [{"png_path": f"/crops/nucleus_png/{i}.png", "plateID": "p1",
              "rowID": "r1", "columnID": "c1", "annotate": None}
             for i in range(8)]
    db = _plain_png_list(str(tmp_path / "measurements.db"), rows,
                         ["png_path", "plateID", "rowID", "columnID",
                          "annotate"])

    coverage = al.annotation_coverage(db, "annotate", image_type="cell_png")
    meta = coverage.attrs["spacr_annotation_coverage"]
    assert meta["n_annotated"] == 12
    assert meta["n_rows"] == 12
    # The total is kept, so nothing is lost — it is just not the denominator.
    assert meta["n_rows_unfiltered"] == 20
    assert meta["image_type"] == "cell_png"
    assert any("excluded 8 of 20 crops" in note for note in meta["notes"])
    assert "12 of 12 crops annotated" in al.format_coverage_summary(coverage)

    # Unfiltered, the denominator is the whole table and there is no note.
    whole = al.annotation_coverage(db, "annotate")
    whole_meta = whole.attrs["spacr_annotation_coverage"]
    assert whole_meta["n_rows"] == 20 and whole_meta["n_annotated"] == 12
    assert not [n for n in whole_meta["notes"] if "image_type" in n]
