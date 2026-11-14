"""Table-driven tests of an active-learning round in ``spacr.active_learning``.

A round joins the crop table (``png_list``) to the measured objects, fits a
model on the annotated crops, scores every crop and records what it did.  Each
unit is tested on exact values: the feature row each crop gets (checked
against a tag unique to each object), what a round fits and records for each
kind of screen, every refusal with its message, how object keys route to
crops, the coverage report and its summary, the held-out report and how an
estimator's probability columns are re-indexed.  One general test per unit,
walking a case table with pytest's ``subtests`` fixture.
No network, no GPU, no torch: sklearn on a few dozen synthetic rows.
"""
from __future__ import annotations

import os
import re
import sqlite3

import numpy as np
import pandas as pd
import pytest

import spacr.active_learning as al
from spacr import selection
from tests.test_an_active_learning_round_says_what_it_could_not_use import \
    _three_class_db
from tests.test_issue152_suggest_features import _project as _two_table_project

WELLS = (("r1", "c1"), ("r1", "c2"), ("r2", "c1"), ("r2", "c2"))
PNG_COLUMNS = ["png_path", "prcfo", "file_name", "plateID", "rowID",
               "columnID", "fieldID", "cell_id"]


def _make_project(root, per_well=6, n_classes=2, class_of=None,
                  plate="plate1", annotation_type="INTEGER", class_names=None):
    """``<root>/measurements/measurements.db`` with a cell table + png_list.

    One measured cell per crop; ``cell_area`` carries the class signal and
    ``cell_tag`` is unique per object, so a join can be checked row by row.
    """
    rng = np.random.default_rng(0)
    meas = os.path.join(str(root), "measurements")
    os.makedirs(meas, exist_ok=True)
    db = os.path.join(meas, "measurements.db")
    cells, crops, index = [], [], 0
    for well_index, (row_id, column_id) in enumerate(WELLS):
        for obj in range(1, per_well + 1):
            prcf = f"{plate}_{row_id}_{column_id}_f1"
            cls = class_of(well_index, obj) if class_of else index % n_classes
            tag = float(well_index * 100 + obj)
            cells.append({
                "plateID": plate, "rowID": row_id, "columnID": column_id,
                "fieldID": "f1", "object_label": obj, "prcf": prcf,
                "prc": f"{plate}_{row_id}_{column_id}",
                "cell_area": 500.0 + 400.0 * cls + float(rng.normal(0, 5)),
                "cell_tag": tag, "cell_noise": float(rng.normal(0, 1))})
            label = class_names[int(cls)] if class_names else int(cls)
            crops.append({
                "png_path": f"/crops/cell_png/{prcf}_o{obj}.png",
                "prcfo": f"{prcf}_o{obj}", "file_name": f"{prcf}_o{obj}.png",
                "plateID": plate, "rowID": row_id, "columnID": column_id,
                "fieldID": "f1", "cell_id": f"o{obj}", "annotate": label,
                "_tag": tag})
            index += 1
    con = sqlite3.connect(db)
    try:
        pd.DataFrame(cells).to_sql("cell", con, index=False)
        columns = PNG_COLUMNS + ["annotate"]
        decls = [f'"{c}" TEXT' for c in PNG_COLUMNS] + [f'"annotate" {annotation_type}']
        con.execute(f'CREATE TABLE png_list ({", ".join(decls)})')
        con.executemany(
            f'INSERT INTO png_list ({", ".join(columns)}) VALUES ({", ".join("?" * len(columns))})',
            [tuple(row.get(c) for c in columns) for row in crops])
        con.commit()
    finally:
        con.close()
    return {"db": db, "root": str(root), "crops": crops}


def _plain_png_list(db, rows, columns, annotation_type="INTEGER"):
    """A bare ``png_list`` with the given columns (``annotate`` typed)."""
    con = sqlite3.connect(db)
    try:
        decls = ", ".join(f'"{c}" {annotation_type if c == "annotate" else "TEXT"}'
                          for c in columns)
        con.execute(f"CREATE TABLE png_list ({decls})")
        con.executemany(f'INSERT INTO png_list VALUES ({", ".join("?" * len(columns))})',
                        [tuple(r.get(c) for c in columns) for r in rows])
        con.commit()
    finally:
        con.close()
    return db


def _sql(db, *statements):
    con = sqlite3.connect(db)
    try:
        for statement in statements:
            con.execute(statement)
        con.commit()
    finally:
        con.close()


def _query(db, sql):
    con = sqlite3.connect(db)
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


def _png_columns(db):
    con = sqlite3.connect(db)
    try:
        return [r[1] for r in con.execute("PRAGMA table_info(png_list)")]
    finally:
        con.close()


BALANCE_NOTE = ("balance=none: class imbalance is handled by the estimator's "
                "`class_weight='balanced'`, which reweights rather than drops.")
QUARTER_RULE = ("split_by=well: held out 1 of 4 well group(s) (25.0%) and 6 of 24 cells "
                "(25.0%); requested 25.0%. StratifiedGroupKFold(3) over 4 well groups; "
                "no group appears on both sides")
ROUNDS_NOTE = ("No annotation_rounds rows for 'annotate', so no label can be attributed to an "
               "annotation round. Rounds are recorded from the Annotate screen; labels written "
               "before that read as 'unrecorded'.")
DRY = dict(save_model=False, write_card=False, round_index=0)


def _rounded(value):
    """Floats rounded to 9 places, recursively, so whole reports compare exactly."""
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    return value


def _folder(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _expect_tags(features, crops, columns):
    """Every crop's row carries its own object's tag."""
    assert len(features) == len(crops)
    assert features.index.name == "png_path"
    assert list(features.columns) == columns
    for crop in crops:
        assert features.loc[crop["png_path"], "cell_tag"] == crop["_tag"]


def test_round_features_gives_each_crop_its_own_objects_row(subtests, tmp_path):
    """The prcfo join is exact: one numeric row per measured crop, unmeasured crops dropped."""
    with subtests.test(msg="one_cell_per_crop"):
        project = _make_project(tmp_path / "plain")
        features = al.round_features(project["db"])
        _expect_tags(features, project["crops"], ["cell_area", "cell_tag", "cell_noise"])
        assert all(np.issubdtype(dtype, np.number) for dtype in features.dtypes)

    with subtests.test(msg="crop_without_a_measured_object_is_dropped"):
        project = _make_project(tmp_path / "extra")
        _sql(project["db"],
             "INSERT INTO png_list (png_path, prcfo, plateID, rowID, columnID, fieldID, cell_id, annotate) "
             "VALUES ('/crops/cell_png/plate1_r9_c9_f1_o1.png', 'plate1_r9_c9_f1_o1', 'plate1', 'r9', 'c9', "
             "'f1', 'o1', 1)")
        features = al.round_features(project["db"])
        _expect_tags(features, project["crops"], ["cell_area", "cell_tag", "cell_noise"])

    two_table = ["cell_area", "cell_channel_1_mean_intensity", "cell_channel_2_mean_intensity",
                 "cell_tag", "nucleus_area", "nucleus_channel_1_mean_intensity", "nucleus_tag"]
    with subtests.test(msg="child_means_without_a_limit"):
        db, crops = _two_table_project(_folder(tmp_path / "children"))
        features = al.round_features(db, tables=["cell", "nucleus"], nuclei_limit=None)
        _expect_tags(features, crops, two_table)
        for crop in crops:
            # Nucleus n lies in cell n, and its tag is the cell's plus 0.5.
            assert features.loc[crop["png_path"], "nucleus_tag"] == crop["_tag"] + 0.5

    with subtests.test(msg="children_without_a_cell_link_are_missing_not_borrowed"):
        db, crops = _two_table_project(_folder(tmp_path / "unlinked"))
        nuclei = _query(db, "SELECT * FROM nucleus").drop(columns=["cell_id"])
        con = sqlite3.connect(db)
        try:
            nuclei.to_sql("nucleus", con, index=False, if_exists="replace")
        finally:
            con.close()
        features = al.round_features(db, tables=["cell", "nucleus"])
        _expect_tags(features, crops, two_table)
        assert features.filter(like="nucleus").isna().all().all()
        assert features.filter(like="cell").notna().all().all()


def test_round_features_refuses_tables_it_cannot_join(subtests, tmp_path):
    """No prcfo, no object table or no measured object is a named refusal."""
    no_prcfo = str(tmp_path / "no_prcfo.db")
    _sql(no_prcfo, 'CREATE TABLE png_list ("png_path" TEXT, "annotate" INT)',
         "INSERT INTO png_list VALUES ('/c/a.png', 1)", 'CREATE TABLE cell ("prcf" TEXT)')
    no_tables = str(tmp_path / "no_tables.db")
    _sql(no_tables, 'CREATE TABLE png_list ("png_path" TEXT, "prcfo" TEXT)',
         "INSERT INTO png_list VALUES ('/c/a.png', 'p_r1_c1_f1_o1')")
    unmeasured, _ = _two_table_project(_folder(tmp_path / "unmeasured"))
    _sql(unmeasured, "DROP TABLE cell", "CREATE TABLE cell (prcf TEXT, object_label REAL, cell_area REAL)",
         "INSERT INTO cell VALUES ('plate1_r1_c1_f1', NULL, 5.0)")
    cases = [
        ("crop_table_without_prcfo", no_prcfo, {},
         "'png_list' has no 'prcfo' column, so a crop cannot be matched to its measurements. "
         "Re-run Measure with save_png=True."),
        ("no_object_tables", no_tables, {},
         "no_tables.db has none of the object tables cell, nucleus, pathogen, cytoplasm, so there "
         "are no features to fit on. Run Measure first, or pass features= explicitly."),
        ("object_table_without_a_measured_object", unmeasured, {"tables": ["cell"]},
         "png_list lists 24 crops, but cell hold no measured object, so there are no features to "
         "fit on. Check that Measure finished for these plates."),
    ]
    for case_id, db, kwargs, message in cases:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
                al.round_features(db, **kwargs)


def _round(db, column="annotate", **kwargs):
    result = al.retrain_round(db, column, **{**DRY, **kwargs})
    return {"n_labels": result.n_labels, "scored": result.scored, "accuracy": result.accuracy,
            "classes": result.classes, "report_classes": result.report["classes"],
            "support": result.report["class_support"], "n": result.report["n"],
            "score_columns": result.score_columns, "model_type": result.model_type,
            "split_rule": result.split_rule, "notes": result.notes,
            "confusion": result.report["confusion_matrix"], "per_class": result.per_class}


def _two_class(**overrides):
    base = {"n_labels": 24, "scored": 24, "accuracy": 1.0, "classes": ["0", "1"],
            "report_classes": ["0", "1"], "support": [3, 3], "n": 6,
            "score_columns": ["al_prob_0", "al_prob_1"], "model_type": "logistic_regression",
            "split_rule": QUARTER_RULE, "notes": [BALANCE_NOTE, QUARTER_RULE],
            "confusion": [[3, 0], [0, 3]], "per_class": {"0": 1.0, "1": 1.0}}
    base.update(overrides)
    if "split_rule" in overrides and "notes" not in overrides:
        base["notes"] = [BALANCE_NOTE, overrides["split_rule"]]
    return base


def test_retrain_round_fits_and_records_each_kind_of_screen(subtests, tmp_path):
    """A round's labels, split rule, classes and score columns are exact for each screen."""
    half_rule = ("split_by=well: held out 2 of 4 well group(s) (50.0%) and 12 of 24 cells (50.0%); "
                 "requested 25.0%. StratifiedGroupKFold(2) over 4 well groups; no group appears on both sides")
    cell_rule = ("split_by=cell: held out 6 of 24 object group(s) (25.0%) and 6 of 24 cells (25.0%); "
                 "requested 25.0%. stratified random split of objects; sibling cells may cross")
    cases = [
        ("default_reads_its_own_features", {}, {}, _two_class()),
        ("random_forest", {}, {"model_type": "random_forest", "seed": 0}, _two_class(model_type="random_forest")),
        ("gradient_boosting", {}, {"model_type": "gradient_boosting", "seed": 0},
         _two_class(model_type="gradient_boosting")),
        ("classes_annotated_as_text", {"annotation_type": "TEXT", "class_names": {0: "clean", 1: "infected"}}, {},
         _two_class(classes=["clean", "infected"], report_classes=["clean", "infected"],
                    per_class={"clean": 1.0, "infected": 1.0})),
        ("three_classes_three_columns", {"n_classes": 3, "class_of": lambda w, o: o % 3}, {},
         _two_class(classes=["0", "1", "2"], report_classes=["0", "1", "2"], support=[2, 2, 2],
                    score_columns=["al_prob_0", "al_prob_1", "al_prob_2"],
                    confusion=[[2, 0, 0], [0, 2, 0], [0, 0, 2]], per_class={"0": 1.0, "1": 1.0, "2": 1.0})),
        # Class and well are confounded, so only a 50% grouped hold-out keeps both classes.
        ("grouped_split_expands_to_both_classes", {"class_of": lambda w, o: 0 if w < 2 else 1},
         {"group_by": "well", "holdout": 0.25},
         _two_class(split_rule=half_rule, support=[6, 6], n=12, confusion=[[6, 0], [0, 6]])),
        ("group_by_none_is_recorded_as_random", {}, {"group_by": "none"}, _two_class(split_rule=cell_rule)),
    ]
    for case_id, project_kwargs, round_kwargs, expected in cases:
        with subtests.test(msg=case_id):
            project = _make_project(tmp_path / case_id, **project_kwargs)
            assert _round(project["db"], **round_kwargs) == expected
            curve = al.learning_curve(project["db"], "annotate")
            assert curve["model_type"].tolist() == [expected["model_type"]]
            assert curve["split_rule"].tolist() == [expected["split_rule"]]
            assert curve["per_class"].tolist() == [expected["per_class"]]
            stored = _query(project["db"], "SELECT " + ", ".join(expected["score_columns"]) + " FROM png_list")
            assert stored.notna().all().all()
            np.testing.assert_allclose(stored.sum(axis=1).to_numpy(), np.ones(24))

    with subtests.test(msg="explicit_features_fit_the_same_round"):
        project = _make_project(tmp_path / "explicit")
        assert _round(project["db"], features=al.round_features(project["db"])) == _two_class()

    with subtests.test(msg="image_type_filter_reaches_the_fit"):
        project = _make_project(tmp_path / "two_modes")
        _sql(project["db"],
             "INSERT INTO png_list (png_path, prcfo, plateID, rowID, columnID, fieldID, cell_id, annotate) "
             "SELECT replace(png_path, 'cell_png', 'nucleus_png'), prcfo, plateID, rowID, columnID, "
             "fieldID, cell_id, annotate FROM png_list")
        assert _round(project["db"], image_type="cell_png") == _two_class()
        assert _round(project["db"])["n_labels"] == 48

    with subtests.test(msg="three_class_queue_reads_every_column"):
        project = _make_project(tmp_path / "queue", n_classes=3, class_of=lambda w, o: o % 3)
        _round(project["db"])
        _sql(project["db"], "UPDATE png_list SET annotate = NULL")
        queue = al.build_queue(project["db"], "annotate", diversity="none")
        assert queue.attrs["spacr_active_learning"]["pred_columns"] == ["al_prob_0", "al_prob_1", "al_prob_2"]
        assert len(queue) == 24

    with subtests.test(msg="write_scores_false_leaves_png_list_alone"):
        project = _make_project(tmp_path / "dry")
        before = _png_columns(project["db"])
        dry = al.retrain_round(project["db"], "annotate", write_scores=False, **DRY)
        assert (dry.score_columns, dry.scored, dry.n_labels) == ([], 0, 24)
        assert "Re-scored 0 crops into nothing" in dry.summary()
        assert _png_columns(project["db"]) == before
        # The same fit with writing on adds one column per class, and as a
        # second round with no new labels its verdict waits for more labels.
        wet = al.retrain_round(project["db"], "annotate", write_scores=True,
                               save_model=False, write_card=False, round_index=1)
        assert (wet.score_columns, wet.scored) == (["al_prob_0", "al_prob_1"], 24)
        assert _png_columns(project["db"]) == before + ["al_prob_0", "al_prob_1"]
        assert (wet.verdict.stop, wet.verdict.reason) == (
            False, "Only 0 labels since round 0; the rule waits for 50 before calling a plateau. "
                   "Held-out accuracy has moved +0.000 so far.")

    with subtests.test(msg="rejection_in_two_class_column_is_the_other_class"):
        db, rows, features = _three_class_db(tmp_path / "rejections")
        _sql(db, "UPDATE png_list SET annotate = 1 + (annotate % 2) WHERE annotate IS NOT NULL")
        result = al.retrain_round(db, "annotate", features=features, seed=0, save_model=False,
                                  write_card=False, rejections={rows[k][0]: 1 for k in (61, 63)})
        assert result.n_labels == 62
        assert result.notes[0] == ("2 rejected suggestions were fitted as the other class: in a "
                                   "two-class column a rejection of class 1 is an example of class 2.")
        assert not any("could not be fitted" in note for note in result.notes)


def test_retrain_round_refuses_requests_it_cannot_honour(subtests, tmp_path):
    """Unknown names and unjoinable inputs refuse by name and record no round."""
    project = _make_project(tmp_path / "screen")
    elsewhere = al.round_features(project["db"])
    elsewhere.index = [f"/somewhere/else/{i}.png" for i in range(len(elsewhere))]
    ungroupable = _plain_png_list(
        str(tmp_path / "ungroupable.db"),
        [{"png_path": f"/crops/cell_png/{i}.png", "prcfo": f"p_r1_c1_f1_o{i}", "annotate": i % 2}
         for i in range(24)], ["png_path", "prcfo", "annotate"])
    ungroupable_features = pd.DataFrame(
        {"f0": [float(i % 2) * 10 + i * 0.01 for i in range(24)], "f1": [float(i) for i in range(24)]},
        index=pd.Index([f"/crops/cell_png/{i}.png" for i in range(24)], name="png_path"))
    cases = [
        ("unknown_column", project["db"], "not_a_column", {},
         "'png_list' has no 'not_a_column' column — there are no labels to retrain on yet."),
        ("feature_matrix_shares_no_crop", project["db"], "annotate", {"features": elsewhere},
         "No crop in png_list has a row in the feature matrix, so nothing can be fitted: 24 crops are "
         "keyed '/crops/cell_png/plate1_r1_c1_f1_o1.png', '/crops/cell_png/plate1_r1_c1_f1_o2.png', "
         "'/crops/cell_png/plate1_r1_c1_f1_o3.png', and the 24 feature rows are keyed "
         "'/somewhere/else/0.png', '/somewhere/else/1.png', '/somewhere/else/2.png'. Check that "
         "Measure and the crop export ran over the same objects."),
        ("unknown_model_type", project["db"], "annotate", {"model_type": "xgbost"},
         "Unknown model_type 'xgbost'; use 'logistic_regression', 'random_forest', "
         "'gradient_boosting' or 'xgboost'."),
        ("unknown_group_by", project["db"], "annotate", {"group_by": "Well"},
         "Unknown group_by/cv_group_by split level 'Well'; use one of cell, field, well, plate. "
         "Names are exact and lower-case. 'none' remains an alias for 'cell'."),
        ("group_by_without_its_columns", ungroupable, "annotate",
         {"features": ungroupable_features, "group_by": "well"},
         "Cannot split png_list by well: missing identity columns ['plateID', 'rowID', 'columnID'], "
         "and no valid prcfo or crop path is available. A grouped design is therefore unverifiable "
         "and will not be replaced by a random split."),
    ]
    for case_id, db, column, kwargs, message in cases:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
                al.retrain_round(db, column, **{**DRY, **kwargs})
            assert al.learning_curve(db, column).empty


def test_retrain_round_keeps_the_round_when_a_write_fails(subtests, tmp_path):
    """A directory in the way of the model or the card is a note; the round is still recorded."""
    cases = [
        ("model_cannot_be_saved", 7, "round_007_logistic_regression.joblib", {},
         "Could not save the round model ([Errno 21] Is a directory: '{blocker}').", False),
        ("card_cannot_be_written", 2, "round_002_logistic_regression.card.json", {"write_card": True},
         "Round model card could not be written ([Errno 21] Is a directory: '{blocker}').", True),
    ]
    for case_id, round_index, blocker_name, kwargs, note, model_saved in cases:
        with subtests.test(msg=case_id):
            project = _make_project(tmp_path / case_id)
            model_dir = os.path.join(project["root"], "al_models")
            blocker = os.path.join(model_dir, blocker_name)
            os.makedirs(blocker)
            result = al.retrain_round(project["db"], "annotate", save_model=True,
                                      model_dir=model_dir, round_index=round_index, **kwargs)
            model_path = os.path.join(model_dir, f"round_{round_index:03d}_logistic_regression.joblib")
            assert result.model_path == (model_path if model_saved else "")
            assert os.path.isfile(model_path) is model_saved
            assert result.card_path == ""
            assert result.notes == [BALANCE_NOTE, QUARTER_RULE, note.format(blocker=blocker)]
            curve = al.learning_curve(project["db"], "annotate")
            assert curve[["round", "model_path", "holdout_accuracy"]].to_dict("list") == {
                "round": [round_index], "model_path": [result.model_path], "holdout_accuracy": [1.0]}


def _crop(path, **meta):
    return {"png_path": path, "plateID": "p1", "rowID": "r1", "columnID": "c1",
            "fieldID": "f1", "annotate": None, **meta}


def test_crops_for_object_keys_routes_each_key_to_its_crop(subtests, tmp_path):
    """Keys from spacr.selection resolve to exactly the crop they name, or to none."""
    frames = pd.DataFrame([{"plateID": "p1", "rowID": "r1", "columnID": "c1", "fieldID": "f1",
                            "timeID": f"t{t}", "object_label": 7} for t in (2, 1)])
    timelapse_keys = list(selection.object_keys(frames, timelapse=True))
    assert timelapse_keys == ["p1_r1_c1_f1_t2_7", "p1_r1_c1_f1_t1_7"]
    escaped_frame = pd.DataFrame([{"plateID": "plate1", "rowID": "r1", "columnID": "c1",
                                   "fieldID": "f_1", "object_label": 7}])
    typed_escaped = list(selection.object_keys(escaped_frame, object_type="cell"))
    untyped_escaped = list(selection.untyped_object_keys(escaped_frame))
    assert (typed_escaped, untyped_escaped) == (["plate1_r1_c1_f%5F1_cell7"], ["plate1_r1_c1_f%5F1_7"])

    timelapse = _plain_png_list(
        str(tmp_path / "timelapse.db"),
        [_crop(f"/crops/cell_png/p1_r1_c1_f1_t{t}_o7.png", timeID=f"t{t}", cell_id="o7") for t in (1, 2)],
        ["png_path", "plateID", "rowID", "columnID", "fieldID", "timeID", "cell_id", "annotate"])
    modes = _plain_png_list(
        str(tmp_path / "modes.db"),
        [_crop(f"/crops/{mode}/p1_r1_c1_f1_o7.png", cell_id="o7") for mode in ("cell_png", "nucleus_png")],
        ["png_path", "plateID", "rowID", "columnID", "fieldID", "cell_id", "annotate"])
    bare_o = _plain_png_list(
        str(tmp_path / "bare_o.db"),
        [_crop("/crops/cell_png/p1_r1_c1_f1_o5.png", prcfo="p1_r1_c1_f1_o5", cell_id="o")],
        ["png_path", "prcfo", "plateID", "rowID", "columnID", "fieldID", "cell_id", "annotate"])
    two_types = _plain_png_list(
        str(tmp_path / "two_types.db"),
        [_crop("/crops/p1_r1_c1_f1_o3.png", nucleus_id="o3", pathogen_id="o3")],
        ["png_path", "plateID", "rowID", "columnID", "fieldID", "nucleus_id", "pathogen_id", "annotate"])
    escaped = _plain_png_list(
        str(tmp_path / "escaped.db"),
        [{"png_path": "/crops/cell_png/plate1_r1_c1_f_1_o7.png", "prcfo": "plate1_r1_c1_f_1_o7",
          "plateID": "plate1", "rowID": "r1", "columnID": "c1", "fieldID": "f_1", "cell_id": "o7",
          "annotate": None}],
        ["png_path", "prcfo", "plateID", "rowID", "columnID", "fieldID", "cell_id", "annotate"])
    escaped_crop = [("/crops/cell_png/plate1_r1_c1_f_1_o7.png", None)]
    cases = [
        ("timelapse_key_names_its_frame", timelapse, timelapse_keys, {"timelapse": True},
         [("/crops/cell_png/p1_r1_c1_f1_t2_o7.png", None), ("/crops/cell_png/p1_r1_c1_f1_t1_o7.png", None)]),
        ("field_level_key_without_timelapse", timelapse, ["p1_r1_c1_f1_7"], {},
         [("/crops/cell_png/p1_r1_c1_f1_t1_o7.png", None)]),
        ("image_type_nucleus", modes, ["p1_r1_c1_f1_7"], {"image_type": "nucleus_png"},
         [("/crops/nucleus_png/p1_r1_c1_f1_o7.png", None)]),
        ("image_type_cell", modes, ["p1_r1_c1_f1_7"], {"image_type": "cell_png"},
         [("/crops/cell_png/p1_r1_c1_f1_o7.png", None)]),
        ("image_type_with_no_crop", modes, ["p1_r1_c1_f1_7"], {"image_type": "pathogen_png"}, []),
        ("bare_o_mints_no_field_key", bare_o, ["p1_r1_c1_f1_"], {}, []),
        ("bare_o_crop_reached_through_prcfo", bare_o, ["p1_r1_c1_f1_5"], {},
         [("/crops/cell_png/p1_r1_c1_f1_o5.png", None)]),
        ("two_object_types_untyped_key", two_types, ["p1_r1_c1_f1_3"], {},
         [("/crops/p1_r1_c1_f1_o3.png", None)]),
        ("two_object_types_nucleus_key", two_types, ["p1_r1_c1_f1_nucleus3"], {}, []),
        ("two_object_types_pathogen_key", two_types, ["p1_r1_c1_f1_pathogen3"], {}, []),
        ("escaped_typed_key", escaped, typed_escaped, {}, escaped_crop),
        ("escaped_untyped_key", escaped, untyped_escaped, {}, escaped_crop),
        ("raw_key_still_resolves", escaped, ["plate1_r1_c1_f_1_7"], {}, escaped_crop),
    ]
    for case_id, db, keys, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            assert al.crops_for_object_keys(db, keys, **kwargs) == expected

    with subtests.test(msg="table_without_png_path"):
        db = _plain_png_list(str(tmp_path / "nokey.db"), [{"prcfo": "p1_r1_c1_f1_o1", "annotate": 1}],
                             ["prcfo", "annotate"])
        with pytest.raises(ValueError, match="^'png_list' has no 'png_path' column, so an object key "
                                             "cannot be resolved to a crop.$"):
            al.crops_for_object_keys(db, ["p1_r1_c1_f1_1"])


COVERAGE_COLUMNS = ["png_path", "plateID", "rowID", "columnID", "annotate"]


def _well_row(path, plate, row_id, column_id, label):
    return {"png_path": path, "plateID": plate, "rowID": row_id, "columnID": column_id, "annotate": label}


COVERAGE_SCREENS = {
    "no_plate_metadata": ([{"png_path": f"/c/{i}.png", "annotate": i % 2} for i in range(12)],
                          ["png_path", "annotate"], "INTEGER"),
    "class_from_one_well": ([_well_row(f"/c/a{i}.png", "p1", "r1", "c1", 1) for i in range(12)]
                            + [_well_row(f"/c/b{i}.png", "p1", "r2", f"c{i % 4 + 1}", 0) for i in range(12)],
                            COVERAGE_COLUMNS, "INTEGER"),
    "imbalanced": ([_well_row(f"/c/n{i}.png", "p1", "r1", f"c{i % 6 + 1}", 0) for i in range(30)]
                   + [_well_row(f"/c/p{i}.png", "p1", "r2", f"c{i % 5 + 1}", 1) for i in range(5)],
                   COVERAGE_COLUMNS, "INTEGER"),
    "text_classes": ([_well_row(f"/c/{i}.png", "p1", "r1", "c1", "infected" if i % 2 else "clean")
                      for i in range(8)], COVERAGE_COLUMNS, "TEXT"),
    "two_plates": ([_well_row(f"/c/{p}_{i}.png", p, "r1", f"c{i % 3 + 1}", i % 2)
                    for p in ("plate1", "plate2") for i in range(6 if p == "plate1" else 2)],
                   COVERAGE_COLUMNS, "INTEGER"),
    "two_crop_modes": ([_well_row(f"/crops/cell_png/{i}.png", "p1", "r1", "c1", i % 2) for i in range(12)]
                       + [_well_row(f"/crops/nucleus_png/{i}.png", "p1", "r1", "c1", None) for i in range(8)],
                       COVERAGE_COLUMNS, "INTEGER"),
}


def _coverage_db(tmp_path, screen):
    rows, columns, kind = COVERAGE_SCREENS[screen]
    folder = tmp_path / screen
    folder.mkdir(exist_ok=True)
    db = folder / "m.db"
    if not db.exists():
        _plain_png_list(str(db), rows, columns, annotation_type=kind)
    return str(db)


def test_annotation_coverage_counts_the_filtered_population_and_names_its_caveats(subtests, tmp_path):
    """Counts, concentration and notes describe exactly the crops the filter kept."""
    no_plate_note = ("png_list has no plateID column, so nothing here can be attributed to a plate; "
                     "every label is reported under '(unknown)'.")
    one_well_note = ("Class 1: all 12 labels come from the single well p1/r1/c1. There is no way to tell "
                     "the class apart from the well, and a random train/test split of these objects will "
                     "report an accuracy that does not transfer.")
    balance_note = ("Class balance is 30:5 (0, 1). The smallest class has 5 labels; per-class accuracy, "
                    "not the aggregate, is the number to read on a model trained here.")
    cases = [
        ("no_plate_metadata", None,
         dict(n_annotated=12, n_rows=12, by_plate={"(unknown)": 12}, by_well={"(unknown)": 12},
              plates_total=1, wells_total=1, by_class={"0": 6, "1": 6}, notes=[no_plate_note, ROUNDS_NOTE])),
        ("class_from_one_well", None,
         dict(by_class={"0": 12, "1": 12}, wells_total=5, notes=[ROUNDS_NOTE, one_well_note],
              concentration={
                  "0": {"n": 12, "n_groups": 4, "top": "p1/r2/c1", "top_n": 3, "top_share": 0.25,
                        "hhi": 0.25, "effective_groups": 4.0},
                  "1": {"n": 12, "n_groups": 1, "top": "p1/r1/c1", "top_n": 12, "top_share": 1.0,
                        "hhi": 1.0, "effective_groups": 1.0},
                  "__all__": {"n": 24, "n_groups": 5, "top": "p1/r1/c1", "top_n": 12, "top_share": 0.5,
                              "hhi": 0.3125, "effective_groups": 3.2}})),
        ("imbalanced", None, dict(by_class={"0": 30, "1": 5}, notes=[ROUNDS_NOTE, balance_note])),
        ("text_classes", None, dict(by_class={"clean": 4, "infected": 4}, n_classes=2)),
        ("two_plates", None, dict(by_plate={"plate1": 6, "plate2": 2}, plates_total=2)),
        ("two_crop_modes", "cell_png",
         dict(n_annotated=12, n_rows=12, n_rows_unfiltered=20, image_type="cell_png",
              notes=["image_type='cell_png' excluded 8 of 20 crops before counting; every number below "
                     "describes the 12 that matched.", ROUNDS_NOTE])),
        ("two_crop_modes", None,
         dict(n_annotated=12, n_rows=20, n_rows_unfiltered=20, image_type=None, notes=[ROUNDS_NOTE])),
    ]
    for screen, image_type, expected in cases:
        with subtests.test(msg=f"{screen}_{image_type}"):
            coverage = al.annotation_coverage(_coverage_db(tmp_path, screen), "annotate", image_type=image_type)
            meta = coverage.attrs["spacr_annotation_coverage"]
            assert _rounded({k: meta[k] for k in expected}) == _rounded(expected)
            # One row per class and well; the rows of a class add up to its label count.
            assert coverage.groupby("class")["n"].sum().to_dict() == meta["by_class"]

    with subtests.test(msg="filter_that_keeps_every_crop_has_no_note"):
        project = _make_project(tmp_path / "screen")
        for image_type, rows, by_class, notes in (
                ("cell_png", 24, {"0": 12, "1": 12}, [ROUNDS_NOTE]),
                ("_r1_", 12, {"0": 6, "1": 6},
                 ["image_type='_r1_' excluded 12 of 24 crops before counting; every number below "
                  "describes the 12 that matched.", ROUNDS_NOTE])):
            meta = al.annotation_coverage(project["db"], "annotate", image_type=image_type).attrs[
                "spacr_annotation_coverage"]
            assert (meta["n_rows"], meta["n_rows_unfiltered"], meta["n_annotated"],
                    meta["by_class"], meta["notes"]) == (rows, 24, rows, by_class, notes)

    with subtests.test(msg="table_without_png_path"):
        db = _plain_png_list(str(tmp_path / "nokey.db"), [{"prcfo": "p1_r1_c1_f1_o1", "annotate": 1}],
                             ["prcfo", "annotate"])
        with pytest.raises(ValueError, match="^'png_list' has no 'png_path' column, so an annotation "
                                             "cannot be attributed to a crop.$"):
            al.annotation_coverage(db, "annotate")


def test_format_coverage_summary_prints_counts_classes_and_plates(subtests, tmp_path):
    """The summary's header, per-class rows, per-plate line and notes come from the coverage numbers."""
    cases = [
        ("no_plate_metadata", None,
         "12 of 12 crops annotated · 2 classes · 1/1 plates · 1/1 wells",
         ["  0                6   50.0%       1       1             (unknown)     100.0%",
          "  1                6   50.0%       1       1             (unknown)     100.0%"], None),
        ("text_classes", None,
         "8 of 8 crops annotated · 2 classes · 1/1 plates · 1/1 wells",
         ["  clean            4   50.0%       1       1              p1/r1/c1     100.0%",
          "  infected         4   50.0%       1       1              p1/r1/c1     100.0%"], None),
        ("two_plates", None,
         "8 of 8 crops annotated · 2 classes · 2/2 plates · 5/5 wells",
         ["  0                4   50.0%       4       2          plate1/r1/c1      25.0%",
          "  1                4   50.0%       4       2          plate1/r1/c2      25.0%"],
         "Per plate: plate1: 6 · plate2: 2"),
        ("two_crop_modes", "cell_png",
         "12 of 12 crops annotated · 2 classes · 1/1 plates · 1/1 wells",
         ["  0                6   50.0%       1       1              p1/r1/c1     100.0%",
          "  1                6   50.0%       1       1              p1/r1/c1     100.0%"], None),
    ]
    for screen, image_type, header, class_rows, per_plate in cases:
        with subtests.test(msg=screen):
            coverage = al.annotation_coverage(_coverage_db(tmp_path, screen), "annotate", image_type=image_type)
            lines = al.format_coverage_summary(coverage).splitlines()
            assert lines[:2] == ["Annotation coverage — m.db [annotate]", header]
            assert lines[3:5] == ["Per class:", "  class       labels   share   wells  plates          busiest well  its share"]
            assert lines[5:5 + len(class_rows)] == class_rows
            assert [line for line in lines if line.startswith("Per plate:")] == ([per_plate] if per_plate else [])
            notes = coverage.attrs["spacr_annotation_coverage"]["notes"]
            assert lines[-len(notes):] == [f"! {note}" for note in notes]


def test_holdout_report_scores_exactly_the_rows_it_counts(subtests):
    """n, supports and the confusion matrix describe one population for every head shape."""
    flat = np.array([0.1, 0.9, 0.2, 0.3, 0.8])
    two_head = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    binary = dict(n=5, n_unscored=0, num_classes=2, head_classes=2, classes=["class_0", "class_1"],
                  accuracy=0.8, f1_macro=0.8, per_class_accuracy=[1.0, 2 / 3], class_support=[2, 3],
                  predicted_support=[3, 2], confusion_matrix=[[2, 0], [1, 2]], notes=[])
    cases = [
        ("single_column_is_the_positive_class", [0, 1, 1, 0, 1], flat.reshape(-1, 1), binary),
        ("flat_vector_is_the_same_head", [0, 1, 1, 0, 1], flat, binary),
        ("class_the_head_cannot_reach_is_an_error", [0, 1, 2], two_head,
         dict(n=3, n_unscored=0, num_classes=3, head_classes=2, classes=["class_0", "class_1", "class_2"],
              accuracy=2 / 3, f1_macro=5 / 9, per_class_accuracy=[1.0, 1.0, 0.0], class_support=[1, 1, 1],
              predicted_support=[2, 1, 0], confusion_matrix=[[1, 0, 0], [0, 1, 0], [1, 0, 0]],
              notes=["The held-out labels name 3 classes but the probability matrix has 2 columns, so "
                     "classes 2..2 can never be predicted and every object in them counts as an error. "
                     "Score this set with the head it was labelled for."])),
        ("negative_class_id_leaves_the_totals", [0, 1, -1], two_head,
         dict(n=2, n_unscored=1, num_classes=2, head_classes=2, classes=["class_0", "class_1"],
              accuracy=1.0, f1_macro=1.0, per_class_accuracy=[1.0, 1.0], class_support=[1, 1],
              predicted_support=[1, 1], confusion_matrix=[[1, 0], [0, 1]],
              notes=["1 of 3 held-out rows carry a negative class id, which is not a class: they are "
                     "excluded from every number here, including n."])),
    ]
    for case_id, truth, scores, expected in cases:
        with subtests.test(msg=case_id):
            assert _rounded(al.holdout_report(truth, scores)) == _rounded(expected)

    with subtests.test(msg="empty_held_out_set_is_nan"):
        report = al.holdout_report([], np.zeros((0, 2)))
        assert np.isnan(report["accuracy"]) and np.isnan(report["f1_macro"])
        assert {k: report[k] for k in ("n", "class_support", "confusion_matrix")} == {
            "n": 0, "class_support": [0, 0], "confusion_matrix": [[0, 0], [0, 0]]}

    with subtests.test(msg="labels_and_scores_of_different_lengths"):
        with pytest.raises(ValueError, match=re.escape(
                "3 held-out labels but 2 rows of scores; they name different objects, so no confusion "
                "matrix built from them would mean anything.")):
            al.holdout_report([0, 1, 0], np.array([[0.9, 0.1], [0.2, 0.8]]))


class _Model:
    """An estimator double whose ``predict_proba`` repeats one row."""

    def __init__(self, row, classes=None):
        self._row = row
        if classes is not None:
            self.classes_ = np.array(classes)

    def predict_proba(self, x):
        if len(x) == 0:
            raise AssertionError("an empty batch must not reach the model")
        return np.tile(self._row, (len(x), 1))


def test_predict_proba_reindexes_columns_by_the_models_classes(subtests):
    """Each class id lands in its own column; unknown or negative ids drop, unseen ones are zero."""
    cases = [
        ("class_beyond_the_round_is_dropped", _Model([0.5, 0.2, 0.3], [0, 1, 2]), 2, [0.5, 0.2]),
        ("class_the_fold_never_saw_is_zero", _Model([0.4, 0.6], [0, 2]), 3, [0.4, 0.0, 0.6]),
        ("negative_class_does_not_wrap", _Model([0.7, 0.1, 0.2], [-1, 0, 1]), 2, [0.1, 0.2]),
        ("reordered_classes_are_respected", _Model([0.3, 0.7], [1, 0]), 2, [0.7, 0.3]),
        ("no_classes_attribute_is_positional", _Model([0.3, 0.7]), 2, [0.3, 0.7]),
        ("no_classes_attribute_padded", _Model([0.3, 0.7]), 3, [0.3, 0.7, 0.0]),
    ]
    for case_id, model, n_classes, row in cases:
        with subtests.test(msg=case_id):
            out = al._predict_proba(model, np.zeros((4, 2)), n_classes)
            np.testing.assert_allclose(out, np.tile(row, (4, 1)))

    with subtests.test(msg="empty_batch_has_the_class_width"):
        out = al._predict_proba(_Model([0.5, 0.2, 0.3]), np.empty((0, 4)), n_classes=3)
        assert out.shape == (0, 3)
