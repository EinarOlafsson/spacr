"""Filters built from a database that will not cooperate.

The bootstrap has two routes: copy the relationships table, or -- when that
cannot be built at all -- reassemble the identity from the object tables
directly. The second route is what a read-only database on a cluster share
takes, and it is exercised here with a real read-only file rather than a
stand-in. The rest are the refusals: a png_list with no path, a filters table
written by an older spaCR, a gate that does not apply to the table it is
being annotated onto.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from spacr import filters as flt


def _objects(n=4, start=1, field="f1"):
    return pd.DataFrame({
        "plateID": ["p1"] * n,
        "rowID": ["A"] * n,
        "columnID": ["1"] * n,
        "fieldID": [field] * n,
        "object_label": list(range(start, start + n)),
        "area": np.linspace(10.0, 60.0, n),
    })


def _db(tmp_path, tables, name="measurements.db"):
    path = str(tmp_path / name)
    with sqlite3.connect(path) as db:
        for table, frame in tables.items():
            frame.to_sql(table, db, index=False)
    return path


# ---------------------------------------------------------------------------
# png_list, and the ways it declines to supply crop paths


def test_a_png_list_with_no_path_column_carries_no_crops(tmp_path):
    """A crop table with no filename cannot say where a crop is."""
    png = pd.DataFrame({"plateID": ["p1"], "rowID": ["A"], "columnID": ["1"],
                        "fieldID": ["f1"], "cell_id": ["o1"]})
    path = _db(tmp_path, {"cell": _objects(), "png_list": png})

    assert flt._png_paths(path) is None


def test_a_png_list_with_no_object_id_carries_no_crops(tmp_path):
    """A path with nothing to attach it to is worse than no path."""
    png = pd.DataFrame({"plateID": ["p1"], "rowID": ["A"], "columnID": ["1"],
                        "fieldID": ["f1"], "png_path": ["/crops/a.png"]})
    path = _db(tmp_path, {"cell": _objects(), "png_list": png})

    assert flt._png_paths(path) is None
    assert flt.png_crop_type(path) is None


def test_a_timelapse_png_list_carries_its_timepoint(tmp_path):
    """Crop paths join on time too, or frame 2 gets frame 1's picture."""
    png = pd.DataFrame({
        "plateID": ["p1", "p1"], "rowID": ["A", "A"], "columnID": ["1", "1"],
        "fieldID": ["f1", "f1"], "timeID": [1, 2], "cell_id": ["o1", "o1"],
        "png_path": ["/crops/t1_o1.png", "/crops/t2_o1.png"]})
    path = _db(tmp_path, {"cell": _objects(), "png_list": png})

    paths = flt._png_paths(path)

    assert "timeID" in paths.columns
    assert list(paths["timeID"]) == [1, 2]
    assert list(paths["object_label"]) == [1, 1]


def test_crops_of_an_unknown_object_type_are_attached_to_nothing():
    """Matching labels blind is what handed nucleus 2 the crop of cell 2."""
    frame = pd.DataFrame({"object_label": [1, 2], "object_type": ["cell"] * 2})
    paths = pd.DataFrame({"object_label": [1, 2],
                          "png_path": ["/a.png", "/b.png"]})

    attached = flt._attach_png_paths(frame, paths, ["object_label"], None)

    assert "png_path" not in attached.columns
    assert attached is frame


def test_crops_are_attached_to_nothing_when_the_frame_has_no_type_axis():
    """Same refusal from the other side: nothing says what a row is."""
    frame = pd.DataFrame({"object_label": [1, 2]})
    paths = pd.DataFrame({"object_label": [1, 2],
                          "png_path": ["/a.png", "/b.png"]})

    attached = flt._attach_png_paths(frame, paths, ["object_label"], "cell")

    assert "png_path" not in attached.columns


# ---------------------------------------------------------------------------
# Relationships


def test_a_stored_relationships_table_is_read_not_rebuilt(tmp_path):
    """Rebuilding on every gate would re-read every object table."""
    path = _db(tmp_path, {"cell": _objects()})
    first = flt.ensure_relationships_table(path)
    with sqlite3.connect(path) as db:
        db.execute('UPDATE "relationships" SET object_type = "marked"')

    again = flt.ensure_relationships_table(path)

    assert len(again) == len(first)
    assert set(again["object_type"]) == {"marked"}


# ---------------------------------------------------------------------------
# Writing gates and annotations onto a filters table that predates them


def test_re_annotating_replaces_the_column(tmp_path):
    """Two columns for one annotation would leave no way to tell them apart."""
    path = _db(tmp_path, {"cell": _objects()})
    frame = flt.read_identity(path, "cell")

    flt.export_annotation(path, frame, pd.Series(["a"] * len(frame)), "state")
    name, marked = flt.export_annotation(
        path, frame, pd.Series(["b"] * len(frame)), "state")

    stored = flt.ensure_filters_table(path)
    assert name == "state"
    assert marked == len(frame)
    assert [c for c in stored.columns if c.startswith("state")] == ["state"]
    assert set(stored["state"]) == {"b"}


# ---------------------------------------------------------------------------
# Sampling and counting


def test_the_row_count_is_what_a_sample_is_a_fraction_of(tmp_path):
    """The denominator the GUI shows beside the sampling setting."""
    path = _db(tmp_path, {"cell": _objects(n=7)})

    assert flt.row_count(path, "cell") == 7
