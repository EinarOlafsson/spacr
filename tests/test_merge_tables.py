"""Merging object tables, aggregated by what each column measures.

The centre of this file is one test: a cell with four pathogens where the sum,
the min, the max and the mean of a measurement are all DIFFERENT numbers, so
an aggregation that picks the wrong one cannot pass by coincidence.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.merge_tables import (
    AGGREGATIONS, MAX, MEAN, MEDIAN, MIN, SUM, MergeError, MergePolicy,
    ReductionError, aggregation_for, aggregation_plan, merge_tables,
    mergeable_tables, reduce_dimensions,
)


def _cells(n=2):
    return pd.DataFrame({
        "plateID": ["p1"] * n, "rowID": ["A"] * n,
        "columnID": ["1"] * n, "fieldID": ["f1"] * n,
        "object_label": range(1, n + 1),
        "area": [100.0, 200.0][:n],
        "mean_intensity": [10.0, 20.0][:n],
    })


def _pathogens():
    """Four pathogens in cell 1, one in cell 2.

    The four areas sum to 100, their min is 10, max is 40 and mean is 25 --
    four different numbers, so no rule can pass by accident.
    """
    return pd.DataFrame({
        "plateID": ["p1"] * 5, "rowID": ["A"] * 5,
        "columnID": ["1"] * 5, "fieldID": ["f1"] * 5,
        "cell_id": [1, 1, 1, 1, 2],
        "object_label": [1, 2, 3, 4, 1],
        "area": [10.0, 20.0, 30.0, 40.0, 7.0],
        "min_intensity": [10.0, 20.0, 30.0, 40.0, 1.0],
        "max_intensity": [10.0, 20.0, 30.0, 40.0, 2.0],
        "mean_intensity": [10.0, 20.0, 30.0, 40.0, 3.0],
        "integrated_intensity": [10.0, 20.0, 30.0, 40.0, 4.0],
    })


def _db(tmp_path, tables, name="measurements.db"):
    path = str(tmp_path / name)
    with sqlite3.connect(path) as db:
        for table, frame in tables.items():
            frame.to_sql(table, db, index=False)
    return path


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("column, expected", [
    ("area", SUM),
    ("perimeter", SUM),
    ("integrated_intensity", SUM),
    ("count_pathogen", SUM),
    ("min_intensity", MIN),
    ("channel_1_min_intensity", MIN),
    ("max_intensity", MAX),
    ("mean_intensity", MEAN),
    ("median_intensity", MEDIAN),
    ("eccentricity", MEAN),
    ("centroid_x", MEAN),
])
def test_each_measurement_gets_the_aggregation_it_deserves(column, expected):
    assert aggregation_for(column) == expected


def test_text_takes_the_first_value_whatever_it_is_called():
    assert aggregation_for("area", numeric=False) == "first"


def test_an_override_beats_the_rule():
    """A default that is right most of the time is a wrong answer nobody can
    find the rest of the time."""
    assert aggregation_for("area", overrides={"area": MEAN}) == MEAN


def test_an_unknown_aggregation_is_refused():
    with pytest.raises(MergeError, match="not an aggregation"):
        aggregation_for("area", overrides={"area": "median-ish"})


def test_the_plan_is_visible_so_it_can_be_overridden():
    frame = pd.DataFrame({"area": [1.0], "min_intensity": [1.0], "name": ["a"]})
    plan = aggregation_plan(frame)
    assert plan == {"area": SUM, "min_intensity": MIN, "name": "first"}
    assert set(plan.values()) <= set(AGGREGATIONS)


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------

def test_four_pathogens_in_one_cell_roll_up_by_measurement(tmp_path):
    """The test the whole module exists for. `io._read_and_join_tables` used
    mean for every one of these, which turns a total into an average and a
    minimum into a mean of minima."""
    path = _db(tmp_path, {"cell": _cells(), "pathogen": _pathogens()})
    merged = merge_tables(path, ["cell", "pathogen"]).set_index("object_label")

    assert merged.loc[1, "pathogen_area"] == 100.0, "areas did not sum"
    assert merged.loc[1, "pathogen_min_intensity"] == 10.0, "min is not the min"
    assert merged.loc[1, "pathogen_max_intensity"] == 40.0, "max is not the max"
    assert merged.loc[1, "pathogen_mean_intensity"] == 25.0
    assert merged.loc[1, "pathogen_integrated_intensity"] == 100.0
    assert merged.loc[1, "pathogen_count"] == 4


def test_measurements_from_different_objects_are_told_apart(tmp_path):
    """"i want to be able to have a cell measurement on one axis nuclear on
    another and pathogen on a thired"."""
    nucleus = _pathogens().assign(cell_id=[1, 1, 1, 1, 2])
    path = _db(tmp_path, {"cell": _cells(), "nucleus": nucleus,
                          "pathogen": _pathogens()})
    merged = merge_tables(path, ["cell", "nucleus", "pathogen"])

    assert "cell_area" in merged.columns
    assert "nucleus_area" in merged.columns
    assert "pathogen_area" in merged.columns
    assert merged["cell_area"].notna().all()


def test_one_row_per_primary_object(tmp_path):
    path = _db(tmp_path, {"cell": _cells(), "pathogen": _pathogens()})
    assert len(merge_tables(path, ["cell", "pathogen"])) == 2


def test_the_primary_object_is_a_choice(tmp_path):
    """Rolling cells onto pathogens is a legitimate thing to want, and gives a
    different table."""
    path = _db(tmp_path, {"cell": _cells(), "pathogen": _pathogens()})
    onto_pathogen = merge_tables(path, ["cell", "pathogen"],
                                 policy=MergePolicy(primary="pathogen"))
    assert len(onto_pathogen) == 5, "it did not roll up onto the pathogen"


def test_a_missing_primary_table_says_what_the_database_has(tmp_path):
    path = _db(tmp_path, {"cell": _cells()})
    with pytest.raises(MergeError, match="no 'nucleus' table"):
        merge_tables(path, ["nucleus"], policy=MergePolicy(primary="nucleus"))


def test_a_child_with_no_parent_link_is_named_and_skipped(tmp_path):
    """Measured without a parent mask, the roll-up is not empty -- it is
    undefined. One unlinkable table must not cost the user the others."""
    orphan = _pathogens().drop(columns=["cell_id"])
    path = _db(tmp_path, {"cell": _cells(), "pathogen": orphan})
    merged = merge_tables(path, ["cell", "pathogen"])
    assert "cell_area" in merged.columns
    assert not any(c.startswith("pathogen_") for c in merged.columns)


def test_a_cell_with_no_children_keeps_its_row(tmp_path):
    lonely = _pathogens().iloc[:4]          # all four belong to cell 1
    path = _db(tmp_path, {"cell": _cells(), "pathogen": lonely})
    merged = merge_tables(path, ["cell", "pathogen"]).set_index("object_label")
    assert len(merged) == 2
    assert merged.loc[2, "pathogen_count"] == 0, "a childless cell has 0 children"
    assert pd.isna(merged.loc[2, "pathogen_mean_intensity"]), (
        "a cell with no pathogens has no pathogen intensity -- zero would be "
        "a measurement that was never made")


def test_the_na_policy_can_drop_or_zero(tmp_path):
    lonely = _pathogens().iloc[:4]
    path = _db(tmp_path, {"cell": _cells(), "pathogen": lonely})

    dropped = merge_tables(path, ["cell", "pathogen"],
                           policy=MergePolicy(na="drop"))
    assert len(dropped) == 1

    zeroed = merge_tables(path, ["cell", "pathogen"],
                          policy=MergePolicy(na="zero")).set_index("object_label")
    assert zeroed.loc[2, "pathogen_mean_intensity"] == 0.0


def test_an_override_reaches_the_merge(tmp_path):
    path = _db(tmp_path, {"cell": _cells(), "pathogen": _pathogens()})
    merged = merge_tables(
        path, ["cell", "pathogen"],
        policy=MergePolicy(overrides={"area": MEAN})).set_index("object_label")
    assert merged.loc[1, "pathogen_area"] == 25.0


def test_only_the_object_tables_present_are_offered(tmp_path):
    path = _db(tmp_path, {"cell": _cells(), "pathogen": _pathogens()})
    assert mergeable_tables(path) == ("cell", "pathogen")


# ---------------------------------------------------------------------------
# xD: reduction
# ---------------------------------------------------------------------------

def _wide(n=200):
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, n)
    return pd.DataFrame({
        "a": base + rng.normal(0, 0.05, n),
        "b": base * 2 + rng.normal(0, 0.05, n),
        "c": rng.normal(0, 1, n),
        "d": rng.normal(0, 1, n),
    })


def test_pca_returns_ordinary_columns_to_gate_on():
    """A gate on PC1 vs PC2 has to be the same kind of object as a gate on
    area vs intensity, or none of the gate tools work in xD."""
    out = reduce_dimensions(_wide(), ["a", "b", "c", "d"], components=2)
    assert list(out.columns) == ["PC1", "PC2"]
    assert len(out) == 200
    assert out["PC1"].notna().all()


def test_pca_finds_the_structure_that_is_there():
    """a and b are the same signal; the first component should carry most of
    the variance."""
    out = reduce_dimensions(_wide(), ["a", "b", "c", "d"], components=3)
    explained = out.attrs["explained_variance"]
    assert explained[0] > explained[1] > 0


def test_scaling_stops_the_biggest_numbers_winning():
    """Without it, a measurement whose numbers are larger dominates every
    component regardless of what it means."""
    frame = _wide()
    frame["huge"] = frame["c"] * 1e6
    scaled = reduce_dimensions(frame, ["a", "b", "huge"], scale=True)
    unscaled = reduce_dimensions(frame, ["a", "b", "huge"], scale=False)
    assert not np.allclose(scaled["PC1"].to_numpy(),
                           unscaled["PC1"].to_numpy())


def test_rows_that_cannot_be_projected_keep_their_place():
    """Otherwise adding components silently drops rows out from under every
    other column in the table."""
    frame = _wide(50)
    frame.loc[0, "a"] = np.nan
    out = reduce_dimensions(frame, ["a", "b", "c"])
    assert len(out) == 50
    assert pd.isna(out.loc[0, "PC1"])
    assert out["PC1"].notna().sum() == 49


def test_one_column_is_not_a_projection():
    with pytest.raises(ReductionError, match="at least two"):
        reduce_dimensions(_wide(), ["a"])


def test_too_few_complete_rows_says_so():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
    with pytest.raises(ReductionError, match="nothing to project"):
        reduce_dimensions(frame, ["a", "b"])


def test_an_unknown_method_is_refused():
    with pytest.raises(ReductionError, match="not one of"):
        reduce_dimensions(_wide(), ["a", "b"], method="magic")
