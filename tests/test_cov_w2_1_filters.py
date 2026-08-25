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

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import filters as flt
from spacr.filters import FilterError


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


@pytest.fixture
def read_only_db(tmp_path):
    """A database nothing may write to, as on a mounted results share."""
    made = {}

    def _make(tables):
        path = _db(tmp_path, tables)
        os.chmod(path, 0o444)
        os.chmod(tmp_path, 0o555)
        made["dir"] = tmp_path
        made["path"] = path
        return path

    yield _make
    if made:
        os.chmod(made["dir"], 0o755)
        os.chmod(made["path"], 0o644)


# ---------------------------------------------------------------------------
# Reading identity


def test_a_table_whose_rows_cannot_be_told_apart_is_refused(tmp_path):
    """Without an object label there is nothing to attach a filter to."""
    path = _db(tmp_path, {"settings": pd.DataFrame({"key": ["a"],
                                                    "value": ["b"]})})

    with pytest.raises(FilterError, match="no object_label column"):
        flt.read_identity(path, "settings")


def test_a_timelapse_keeps_its_timepoint_in_the_join_key():
    """The same label recurs every frame; without time the join is many-to-many."""
    frame = pd.DataFrame(columns=["plateID", "rowID", "columnID", "fieldID",
                                  "timeID", "object_label", "area"])

    assert flt.key_columns(frame) == [
        "plateID", "rowID", "columnID", "fieldID", "timeID", "object_label"]


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
# The fallback: a database that cannot be written to


def test_a_read_only_database_is_still_gateable(read_only_db):
    """The relationships table cannot be created, so the identity is
    reassembled from the object tables instead."""
    path = read_only_db({"cell": _objects(), "nucleus": _objects(n=2)})

    frame = flt.build_filters_frame(path)

    assert "object_type" not in frame.columns, "took the relationships route"
    assert set(frame["in_cell"]) == {1}
    assert list(frame["in_nucleus"]) == [1, 1, 0, 0]
    assert flt.key_columns(frame) == ["plateID", "rowID", "columnID",
                                      "fieldID", "object_label"]


def test_the_fallback_carries_crop_paths_only_to_the_cropped_type(
        read_only_db):
    """`in_cell = 0` with a cell crop attached is the mismatch to prevent."""
    nucleus = _objects(n=2, start=90)
    png = pd.DataFrame({
        "plateID": ["p1"] * 2, "rowID": ["A"] * 2, "columnID": ["1"] * 2,
        "fieldID": ["f1"] * 2, "cell_id": ["o1", "o90"],
        "png_path": ["/crops/cell_1.png", "/crops/cell_90.png"]})
    path = read_only_db({"cell": _objects(), "nucleus": nucleus,
                         "png_list": png})

    frame = flt.build_filters_frame(path).set_index("object_label")

    assert frame.loc[1, "png_path"] == "/crops/cell_1.png"
    assert frame.loc[90, "in_cell"] == 0
    assert frame.loc[90, "png_path"] is None


def test_the_fallback_skips_a_table_whose_identity_will_not_read(
        read_only_db, monkeypatch):
    """One unreadable table must not cost the whole bootstrap."""
    path = read_only_db({"cell": _objects(), "nucleus": _objects(n=2)})
    real = flt.read_identity

    def _refuse_nucleus(db_path, table):
        if table == "nucleus":
            raise FilterError("nucleus went away")
        return real(db_path, table)

    monkeypatch.setattr(flt, "read_identity", _refuse_nucleus)

    frame = flt.build_filters_frame(path)

    assert "in_nucleus" not in frame.columns
    assert list(frame["in_cell"]) == [1, 1, 1, 1]


def test_the_fallback_says_so_when_no_table_can_be_read(read_only_db,
                                                        monkeypatch):
    """A refusal that names the tables it looked at is fixable."""
    path = read_only_db({"cell": _objects(), "nucleus": _objects(n=2)})

    def _refuse(db_path, table):
        raise FilterError("unreadable")

    monkeypatch.setattr(flt, "read_identity", _refuse)

    with pytest.raises(FilterError, match="none of cell, nucleus"):
        flt.build_filters_frame(path)


def test_the_fallback_skips_a_table_sharing_no_identity(read_only_db,
                                                        monkeypatch):
    """A table keyed on something else entirely is not merged blind."""
    path = read_only_db({"cell": _objects(), "nucleus": _objects(n=2)})
    real = flt.read_identity

    def _rename_nucleus(db_path, table):
        frame = real(db_path, table)
        if table == "nucleus":
            return frame.rename(columns={
                name: f"nucleus_{name}" for name in frame.columns})
        return frame

    monkeypatch.setattr(flt, "read_identity", _rename_nucleus)

    frame = flt.build_filters_frame(path)

    assert "in_nucleus" not in frame.columns


def test_the_fallback_carries_no_crops_when_png_list_shares_no_key(
        read_only_db, monkeypatch):
    """No shared key means no join; anything else would attach by position."""
    path = read_only_db({"cell": _objects()})
    monkeypatch.setattr(flt, "_png_paths", lambda db_path: pd.DataFrame({
        "crop_number": [1, 2], "png_path": ["/a.png", "/b.png"]}))

    frame = flt.build_filters_frame(path)

    assert "png_path" not in frame.columns


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


def _legacy_filters(path):
    """A `filters` table from an older spaCR, keyed on nothing usable."""
    with sqlite3.connect(path) as db:
        pd.DataFrame({"plateID": ["p1"], "note": ["kept"]}).to_sql(
            "filters", db, index=False)


def test_a_gate_cannot_be_written_onto_a_filters_table_with_no_object_key(
        tmp_path):
    """Merging on the well alone would mark every object in it."""
    path = _db(tmp_path, {"cell": _objects()})
    _legacy_filters(path)
    frame = flt.read_identity(path, "cell")

    with pytest.raises(FilterError, match="share no object key"):
        flt.export_gate(path, frame, np.ones(len(frame), bool), "big")


def test_an_annotation_cannot_be_written_onto_one_either(tmp_path):
    """The same refusal, so a filter and an annotation behave alike."""
    path = _db(tmp_path, {"cell": _objects()})
    _legacy_filters(path)
    frame = flt.read_identity(path, "cell")
    labels = pd.Series(["live"] * len(frame))

    with pytest.raises(FilterError, match="share no object key"):
        flt.export_annotation(path, frame, labels, "state")


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
# Annotating from gates


def test_a_gate_that_does_not_apply_names_itself(tmp_path):
    """The user picked a gate drawn on a column this table does not have."""
    from spacr.qt.widgets.gate_spec import GateSet, RectGate

    gates = GateSet().add(RectGate(name="big", x_column="area",
                                   y_column="perimeter",
                                   x_low=0.0, x_high=100.0,
                                   y_low=0.0, y_high=100.0))
    frame = pd.DataFrame({"area": [1.0, 2.0]})

    with pytest.raises(FilterError, match="gate 'big' cannot be applied"):
        flt.annotate_from_gates(frame, gates, ["big"])


# ---------------------------------------------------------------------------
# Sampling and counting


def test_a_sample_fraction_outside_zero_to_one_is_refused(tmp_path):
    """`read_sampled` refuses before it reads, not after."""
    path = _db(tmp_path, {"cell": _objects()})

    with pytest.raises(FilterError, match="not a fraction between 0 and 1"):
        flt.read_sampled(path, "cell", fraction=0.0)


def test_a_row_cap_still_applies_when_sampling_falls_back_to_pandas(tmp_path):
    """A table shadowing every row id alias is read whole, then capped."""
    frame = pd.DataFrame({
        "rowid": range(200), "oid": range(200), "_rowid_": range(200),
        "object_label": range(200), "area": np.arange(200.0)})
    path = _db(tmp_path, {"cell": frame})

    sampled = flt.read_sampled(path, "cell", fraction=0.25, limit=10)

    assert len(sampled) == 10
    assert list(sampled["object_label"]) == list(range(0, 40, 4))


def test_the_row_count_is_what_a_sample_is_a_fraction_of(tmp_path):
    """The denominator the GUI shows beside the sampling setting."""
    path = _db(tmp_path, {"cell": _objects(n=7)})

    assert flt.row_count(path, "cell") == 7
