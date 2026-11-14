"""Table-driven tests of class rules, the confusion matrix and the agreement reader.

Three small modules that sit around a trained classifier: ``spacr.classify_classes``
turns a settings file into class rules, ``spacr.confusion`` builds the
true-by-predicted matrix and the confidence threshold that splits one of its
cells, and ``spacr.agreement`` reads annotator columns from ``png_list``
through a read-only connection.  One general test per unit, each walking a case
table with pytest's ``subtests`` fixture and asserting exact values.
"""
from __future__ import annotations

import os
import re
import sqlite3

import pandas as pd
import pytest

from spacr import classify_classes as cc
from spacr import confusion as C
from spacr.agreement import PNG_KEY, PNG_TABLE, _read_only_uri, annotation_columns

_META = ("png_path", "file_name", "plateID", "rowID", "columnID", "fieldID",
         "prcfo", "cell_id")


def _predictions():
    """A small evaluation bundle in the shape spaCR writes."""
    rows = [
        ("p1_A01_f1_cell1", "infected", "infected", 0.95, "f1", "A01", "p1"),
        ("p1_A01_f1_cell2", "infected", "uninfected", 0.91, "f1", "A01", "p1"),
        ("p1_A01_f1_cell3", "infected", "uninfected", 0.55, "f1", "A01", "p1"),
        ("p1_A02_f2_cell4", "uninfected", "infected", 0.80, "f2", "A02", "p1"),
    ]
    return pd.DataFrame(rows, columns=[
        "object_key", C.TRUE_COLUMN, C.PREDICTED_COLUMN, C.CONFIDENCE_COLUMN,
        "field", "well", "plate"])


def _make_db(path, columns, rows):
    """``png_list`` with the spaCR metadata plus one column per name given."""
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    try:
        meta_sql = ", ".join(f'"{c}" TEXT' for c in _META)
        con.execute(f"CREATE TABLE {PNG_TABLE} ({meta_sql})")
        for name, kind in columns:
            con.execute(f'ALTER TABLE {PNG_TABLE} ADD COLUMN "{name}" {kind}')
        payload = []
        for i, values in enumerate(rows):
            crop = f"/data/plate1/cell_png/plate1_A01_1_{i}.png"
            payload.append((crop, os.path.basename(crop), "plate1", "r1", "c1",
                            "f1", f"plate1_A01_1_o{i}", f"o{i}", *values))
        placeholders = ", ".join("?" * (len(_META) + len(columns)))
        con.executemany(f"INSERT INTO {PNG_TABLE} VALUES ({placeholders})", payload)
        con.commit()
    finally:
        con.close()
    return path


def test_normalize_settings_derives_one_rule_per_named_class(subtests):
    """Rules come only from what the settings name; blanks and bare strings make no extra class."""
    cases = [
        ("annotation_without_column", {"dataset_mode": "annotation"}, None, ()),
        ("single_annotation_value_as_string",
         {"dataset_mode": "annotation", "annotation_column": "test", "annotation_values": "positive"},
         {"class_positive": {"column": "test", "value": "positive"}},
         (("class_positive", "test", "positive"),)),
        ("two_annotation_values",
         {"dataset_mode": "annotation", "annotation_column": "test", "annotation_values": ["a", "b"]},
         {"class_a": {"column": "test", "value": "a"}, "class_b": {"column": "test", "value": "b"}},
         (("class_a", "test", "a"), ("class_b", "test", "b"))),
        ("metadata_with_blank_negative_control",
         {"dataset_mode": "metadata", "location_column": "column_name",
          "negative_control_id": "", "positive_control_id": "c3"},
         {"positive control": {"column": "column_name", "value": "c3"}},
         (("positive control", "column_name", "c3"),)),
        ("metadata_with_both_controls",
         {"dataset_mode": "metadata", "location_column": "column_name",
          "negative_control_id": "c1", "positive_control_id": "c3"},
         {"negative control": {"column": "column_name", "value": "c1"},
          "positive control": {"column": "column_name", "value": "c3"}},
         (("negative control", "column_name", "c1"), ("positive control", "column_name", "c3"))),
    ]
    for case_id, settings, classes, rules in cases:
        with subtests.test(msg=case_id):
            out = cc.normalize_settings(dict(settings))
            assert out.get(cc.CLASSES) == classes
            assert tuple((r.name, r.column, r.value) for r in cc.class_rules(out)) == rules
            if classes is not None:
                assert out["class_names"] == out["class_folder_names"] == list(classes)


def test_confusion_counts_is_the_declared_square_matrix(subtests):
    """Every declared class gets a row and a column, even one nobody predicted or saw."""
    cases = [
        ("observed_classes", _predictions(), {}, ["infected", "uninfected"], [[1, 2], [1, 0]]),
        ("empty_table_declared_classes", _predictions().iloc[:0], {"classes": ["a", "b"]},
         ["a", "b"], [[0, 0], [0, 0]]),
        ("declared_class_never_seen", _predictions(), {"classes": ["infected", "uninfected", "other"]},
         ["infected", "uninfected", "other"], [[1, 2, 0], [1, 0, 0], [0, 0, 0]]),
    ]
    for case_id, frame, kwargs, labels, counts in cases:
        with subtests.test(msg=case_id):
            matrix = C.confusion_counts(frame, **kwargs)
            assert list(matrix.index) == list(matrix.columns) == labels
            assert matrix.index.name == C.TRUE_COLUMN
            assert matrix.to_numpy().tolist() == counts

    with subtests.test(msg="not_an_evaluation_bundle"):
        message = ("this prediction table is missing ['true_class', 'predicted_class']; it has "
                   "['object_key', 'score']... Bundles written by "
                   "spacr.classifier_evaluation.evaluate_predictions carry them.")
        with pytest.raises(C.ConfusionError, match=f"^{re.escape(message)}$"):
            C.confusion_counts(pd.DataFrame({"object_key": ["a"], "score": [1.0]}))


def test_confidence_threshold_is_halfway_from_chance_to_certainty(subtests):
    """The threshold is (1 + 1/n) / 2 for n classes; a cell derives it unless given one."""
    for n_classes, expected in ((2, 0.75), (3, 2 / 3), (4, 0.625), (10, 0.55)):
        with subtests.test(msg=f"{n_classes}_classes"):
            assert C.confidence_threshold(n_classes) == pytest.approx(expected)

    for n_classes in (0, 1):
        with subtests.test(msg=f"{n_classes}_classes_refused"):
            with pytest.raises(C.ConfusionError,
                               match=f"^a confusion matrix needs at least two classes, got {n_classes}$"):
                C.confidence_threshold(n_classes)

    cells = [
        ("derived_from_two_classes", {}, 0.75, 1, 1),
        ("told_four_classes", {"n_classes": 4}, 0.625, 1, 1),
        ("explicit_threshold", {"threshold": 0.5}, 0.5, 2, 0),
    ]
    for case_id, kwargs, threshold, n_high, n_low in cells:
        with subtests.test(msg=case_id):
            cell = C.ConfusionCell.build(_predictions(), "infected", "uninfected", **kwargs)
            assert cell.threshold == pytest.approx(threshold)
            assert (len(cell), cell.is_error, len(cell.high), len(cell.low)) == (2, True, n_high, n_low)


def test_read_only_uri_escapes_the_path_and_says_read_only(subtests):
    """Separators become '/', reserved characters are percent-encoded, mode=ro is appended."""
    cases = [
        ("posix_path", "/data/run1/measurements/measurements.db",
         "file:/data/run1/measurements/measurements.db?mode=ro"),
        ("windows_separators", r"C:\data\run1\measurements.db", "file:C:/data/run1/measurements.db?mode=ro"),
        ("space_and_fragment_marker", "/data/my runs/plate #2/measurements.db",
         "file:/data/my%20runs/plate%20%232/measurements.db?mode=ro"),
        ("query_and_percent", "/data/a?b%c.db", "file:/data/a%3Fb%25c.db?mode=ro"),
    ]
    for case_id, path, uri in cases:
        with subtests.test(msg=case_id):
            assert _read_only_uri(path) == uri


def test_annotation_columns_offers_only_label_like_columns(subtests, tmp_path):
    """A column past the class cap, or the row key, is never offered as an annotator."""
    db = _make_db(tmp_path / "run" / "measurements.db",
                  [("alice", "INTEGER"), ("cell_area", "INTEGER")],
                  [(1, i) for i in range(40)])
    cases = [
        ("measurement_over_the_cap", {"max_classes": 10, "min_labelled": 1}, ["alice"]),
        ("cap_raised_lets_it_through", {"max_classes": 100, "min_labelled": 1}, ["alice", "cell_area"]),
    ]
    for case_id, kwargs, offered in cases:
        with subtests.test(msg=case_id):
            got = annotation_columns(db, **kwargs)
            assert got == offered
            assert PNG_KEY not in got
