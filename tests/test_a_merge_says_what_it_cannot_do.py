"""What a table merge and an xD projection refuse, and what they say.

Each of these is a message a user reads instead of a traceback: the
aggregation that does not exist, the table the database does not hold, the
columns that are too empty to project. The wording matters because it is
what tells them which setting to change.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.merge_tables import (AGGREGATIONS, MergeError, MergePolicy,
                                ReductionError, _apply_na_policy,
                                aggregation_for, merge_tables, object_keys,
                                reduce_dimensions)


def test_an_aggregation_that_does_not_exist_lists_the_ones_that_do():
    with pytest.raises(MergeError) as caught:
        aggregation_for("cell_area", overrides={"cell_area": "geomean"})

    assert "'geomean' is not an aggregation" in str(caught.value)
    assert str(list(AGGREGATIONS)) in str(caught.value)


def test_an_override_that_exists_beats_the_rule():
    assert aggregation_for("cell_area", overrides={"cell_area": "max"}) \
        == "max"


def test_the_crop_tables_object_spelling_is_read_as_the_same_integer():
    assert list(object_keys(pd.Series(["o5", "o7"]))) == [5, 7]


def test_plain_numeric_text_converts_without_the_png_translator():
    assert list(object_keys(pd.Series(["5", "7"]))) == [5, 7]


def test_a_key_that_is_neither_becomes_missing_rather_than_fatal():
    keys = object_keys(pd.Series(["5", "not a key"]))

    assert keys[0] == 5
    assert pd.isna(keys[1]), (
        "an unreadable key must not match, and must not stop the merge")


def _database(tmp_path, tables):
    path = str(tmp_path / "measurements.db")
    with sqlite3.connect(path) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return path


def test_a_missing_primary_table_names_what_the_database_does_hold(tmp_path):
    nucleus = pd.DataFrame({
        "plateID": ["p1"], "rowID": ["r1"], "columnID": ["c1"],
        "fieldID": ["f1"], "object_label": [1], "nucleus_area": [10.0],
    })
    path = _database(tmp_path, {"nucleus": nucleus})

    with pytest.raises(MergeError) as caught:
        merge_tables(path, ["nucleus"], policy=MergePolicy(primary="cell"))

    assert "no 'cell' table to merge onto" in str(caught.value)
    assert "nucleus" in str(caught.value), (
        "the message names the tables that ARE there")


def test_the_zero_policy_fills_every_numeric_column_not_only_the_counts():
    frame = pd.DataFrame({
        "cell_area": [1.0, np.nan],
        "pathogen_count": [2.0, np.nan],
        "wellID": ["A1", None],
    })

    filled = _apply_na_policy(frame.copy(), MergePolicy(na="zero"))

    assert filled["cell_area"].tolist() == [1.0, 0.0]
    assert filled["pathogen_count"].tolist() == [2.0, 0.0]
    assert filled["wellID"].isna().iloc[1], "a text column has no zero"


def _measurements(rows=8, columns=3):
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {f"m{i}": rng.normal(size=rows) for i in range(columns)})


def test_a_reduction_method_that_is_not_offered_lists_the_ones_that_are():
    with pytest.raises(ReductionError, match="'ica' is not one of"):
        reduce_dimensions(_measurements(), ["m0", "m1"], method="ica")


def test_one_measurement_is_not_a_projection():
    with pytest.raises(ReductionError,
                       match="needs at least two measurements"):
        reduce_dimensions(_measurements(), ["m0", "not_a_column"])


def test_columns_that_are_mostly_empty_are_left_out_and_counted():
    frame = _measurements(rows=10, columns=3)
    frame.loc[1:, "m1"] = np.nan
    frame.loc[1:, "m2"] = np.nan

    with pytest.raises(ReductionError) as caught:
        reduce_dimensions(frame, ["m0", "m1", "m2"], min_coverage=0.5)

    assert "only 1 of 3 measurement(s) are at least 50% complete" \
        in str(caught.value)


def test_two_objects_are_not_enough_to_project():
    frame = _measurements(rows=6, columns=2)
    frame.loc[2:, :] = np.nan

    with pytest.raises(ReductionError) as caught:
        reduce_dimensions(frame, ["m0", "m1"], min_coverage=0.0)

    assert "only 2 object(s) have any of these measurements" \
        in str(caught.value)
