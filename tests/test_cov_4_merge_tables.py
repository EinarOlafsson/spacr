"""Merging object tables says what it dropped and refuses what it cannot link.

One row of the merged table is one object, and every branch here decides
which objects survive. An inner join is a filter; a child table with no
parent link cannot be rolled up at all; a projection built from columns that
share no rows describes nothing. Each of them has to be visible -- a merge
that quietly halves the population is how a result gets reported for a
subgroup nobody chose.
"""
from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import merge_tables as mt
from spacr.merge_tables import (MergeError, MergePolicy, ReductionError,
                                merge_tables, missingness_leak,
                                reduce_dimensions, roll_up)


# -- policy ------------------------------------------------------------------

def test_an_unknown_childless_policy_is_refused_at_construction():
    """"keep", "zero" and "drop" are not interchangeable; a typo must stop."""
    with pytest.raises(MergeError) as excinfo:
        MergePolicy(na="ignore")
    assert "keep" in str(excinfo.value) and "drop" in str(excinfo.value)


def test_a_policy_whose_childless_rule_is_unrecognised_changes_nothing():
    """The final fall-through leaves the frame exactly as it arrived."""
    class _Unvalidated:
        na = "something else"

    frame = pd.DataFrame({"cell_area": [1.0, None], "nucleus_count": [1, None]})
    out = mt._apply_na_policy(frame.copy(), _Unvalidated())
    pd.testing.assert_frame_equal(out, frame)


# -- key alignment -----------------------------------------------------------

def test_a_key_missing_from_one_side_is_left_alone():
    """Coercing a column that only one side has would raise on the other."""
    left = pd.DataFrame({"plateID": [1, 2], "object_label": [1, 2]})
    right = pd.DataFrame({"plateID": ["1", "2"]})
    mt._align_keys(left, right, ["plateID", "object_label"])
    assert left["plateID"].dtype == right["plateID"].dtype
    assert list(left["object_label"]) == [1, 2]


# -- roll-up -----------------------------------------------------------------

def test_a_child_with_no_parent_link_says_what_to_re_run():
    """"Cannot merge" is unactionable; naming the missing column is not."""
    child = pd.DataFrame({"plateID": ["p1"], "nucleus_area": [3.0]})
    with pytest.raises(MergeError) as excinfo:
        roll_up(child, ["plateID", "cell_id"], name="nucleus",
                policy=MergePolicy())
    assert "cell_id" in str(excinfo.value)
    assert "re-run Measure" in str(excinfo.value)


# -- merging a database ------------------------------------------------------

def _db(tmp_path, tables):
    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return str(path)


def _cells(n=3):
    return pd.DataFrame({
        "plateID": ["p1"] * n, "rowID": ["r1"] * n,
        "columnID": ["c1"] * n, "fieldID": ["f1"] * n,
        "object_label": list(range(1, n + 1)),
        "cell_area": [10.0 * i for i in range(1, n + 1)],
    })


def test_a_primary_table_without_an_object_key_cannot_be_merged_onto(tmp_path):
    """Without object_label there is nothing for a child to attach to."""
    path = _db(tmp_path, {"cell": _cells().drop(columns=["object_label"])})
    with pytest.raises(MergeError) as excinfo:
        merge_tables(path, ["cell"])
    assert "object_label" in str(excinfo.value)


def test_crop_paths_are_merged_even_when_named_as_a_bare_string(tmp_path):
    """``tables`` is a sequence of names, and a bare name is one."""
    png = pd.DataFrame({
        "plateID": ["p1"], "rowID": ["r1"], "columnID": ["c1"],
        "fieldID": ["f1"], "cell_id": [1], "png_path": ["/crops/1.png"]})
    path = _db(tmp_path, {"cell": _cells(), "png_list": png})
    merged = merge_tables(path, "png_list")
    assert "png_list_path" in merged.columns, list(merged.columns)
    assert merged.loc[merged["object_label"] == 1,
                      "png_list_path"].iloc[0] == "/crops/1.png"


def test_crops_with_no_object_id_column_are_reported_and_skipped(tmp_path,
                                                                caplog):
    """A png_list that cannot name its object contributes no paths."""
    png = pd.DataFrame({
        "plateID": ["p1"], "rowID": ["r1"], "columnID": ["c1"],
        "fieldID": ["f1"], "png_path": ["/crops/1.png"]})
    path = _db(tmp_path, {"cell": _cells(), "png_list": png})
    with caplog.at_level(logging.INFO, logger="spacr.merge_tables"):
        merged = merge_tables(path, ["png_list"])
    assert not [c for c in merged.columns if "png" in c], list(merged.columns)
    assert "no object id column" in caplog.text


def test_an_inner_join_says_how_many_objects_it_removed(tmp_path, caplog):
    """A filter that removes a third of the population must not be silent."""
    nucleus = pd.DataFrame({
        "plateID": ["p1"], "rowID": ["r1"], "columnID": ["c1"],
        "fieldID": ["f1"], "cell_id": [1], "object_label": [1],
        "nucleus_area": [4.0]})
    path = _db(tmp_path, {"cell": _cells(3), "nucleus": nucleus})
    with caplog.at_level(logging.INFO, logger="spacr.merge_tables"):
        merged = merge_tables(path, ["nucleus"])
    assert len(merged) == 1
    assert "had no nucleus row and were removed" in caplog.text


def test_a_rollup_that_cannot_produce_an_object_key_is_skipped(
        tmp_path, caplog, monkeypatch):
    """One unlinkable table must not cost the user the others."""
    nucleus = pd.DataFrame({
        "plateID": ["p1"], "rowID": ["r1"], "columnID": ["c1"],
        "fieldID": ["f1"], "cell_id": [1], "nucleus_area": [4.0]})
    path = _db(tmp_path, {"cell": _cells(2), "nucleus": nucleus})

    real_roll_up = mt.roll_up

    def _keyless(child, keys, *, name, policy):
        rolled = real_roll_up(child, keys, name=name, policy=policy)
        return rolled.drop(columns=["cell_id"])

    monkeypatch.setattr(mt, "roll_up", _keyless)
    with caplog.at_level(logging.INFO, logger="spacr.merge_tables"):
        merged = merge_tables(path, ["nucleus"])
    assert "cannot be joined to cell on an object key" in caplog.text
    assert len(merged) == 2
    assert not [c for c in merged.columns if c.startswith("nucleus_")]


# -- projections -------------------------------------------------------------

def _wide(rows=12):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "cell_area": rng.normal(size=rows),
        "cell_perimeter": rng.normal(size=rows),
        "cell_intensity": rng.normal(size=rows),
    })


def test_columns_with_no_values_in_common_cannot_be_projected():
    """Two all-empty columns pass a zero coverage bar and still say nothing."""
    frame = pd.DataFrame({"a": [np.nan] * 5, "b": [np.nan] * 5,
                          "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.raises(ReductionError) as excinfo:
        reduce_dimensions(frame, ["a", "b"], min_coverage=0.0)
    assert "no two measurements" in str(excinfo.value)


def test_a_umap_projection_uses_the_installed_reducer(monkeypatch):
    """UMAP's own settings must reach it, not be silently defaulted."""
    from spacr import utils as _utils
    seen = {}

    class _FakeUMAP:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def fit_transform(self, values):
            return np.asarray(values)[:, :2]

    monkeypatch.setattr(_utils, "umap",
                        type("m", (), {"UMAP": _FakeUMAP}))
    out = reduce_dimensions(_wide(), list(_wide().columns), method="umap",
                            n_neighbors=4, min_dist=0.3)
    assert list(out.columns) == ["UMAP1", "UMAP2"]
    assert seen["n_neighbors"] == 4
    assert seen["min_dist"] == pytest.approx(0.3)


def test_a_missing_umap_says_pca_is_always_available(monkeypatch):
    """The refusal has to name the way out, not just the missing package."""
    from spacr import utils as _utils
    monkeypatch.setattr(_utils, "umap", type("m", (), {}))
    with pytest.raises(ReductionError) as excinfo:
        reduce_dimensions(_wide(), list(_wide().columns), method="umap")
    assert "PCA is always available" in str(excinfo.value)


def test_a_tsne_projection_clamps_its_perplexity_to_the_selection():
    """sklearn raises when perplexity reaches the sample size."""
    out = reduce_dimensions(_wide(rows=12), list(_wide().columns),
                            method="tsne", perplexity=500.0)
    assert list(out.columns) == ["tSNE1", "tSNE2"]
    assert len(out) == 12


# -- leak diagnostics --------------------------------------------------------

def test_a_projection_with_no_spread_reports_no_leak():
    """Every object on one point has no radius to measure a gap against."""
    components = pd.DataFrame({"PC1": [0.0] * 4, "PC2": [0.0] * 4})
    frame = pd.DataFrame({"a": [1.0, np.nan, 1.0, np.nan]})
    out = missingness_leak(components, frame, ["a"], min_objects=1)
    assert out.empty
    assert list(out.columns) == ["column", "missing_fraction", "centroid_gap",
                                 "dispersion_ratio", "severity"]


def test_the_spread_of_no_points_is_zero():
    """A group with no members has no dispersion, and nan would poison the
    ratio it is divided into."""
    assert mt._spread(np.zeros((0, 2))) == 0.0


def test_group_shares_need_two_columns_that_clear_the_coverage_bar():
    """One usable measurement is not a projection to apportion."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan] * 3})
    out = mt.group_variance_share(frame, {"shape": ["a", "b"]})
    assert out.empty
    assert list(out.columns) == ["share", "columns"]


def test_group_shares_need_three_objects_to_describe():
    """Two rows cannot show which group carries the variance."""
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = mt.group_variance_share(frame, {"shape": ["a", "b"]})
    assert out.empty
