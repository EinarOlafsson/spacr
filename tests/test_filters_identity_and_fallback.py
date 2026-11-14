"""Table-driven tests of how ``spacr.filters`` identifies objects in a database.

Identity is what every gate and annotation is merged on: which table anchors
the metadata, which columns form the join key, what is read from an object
table, how the ``filters`` frame is bootstrapped -- through the relationships
table, or, when that cannot be built (a read-only results share, an
unexpected schema), straight from the object tables -- and how a sampled read
takes its rows.  One general test per unit, each walking a case table with
pytest's ``subtests`` fixture and asserting exact frames, keys and refusal
messages.  Every database is a tiny SQLite file written per case.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import filters as flt
from spacr.filters import FilterError

WELL = ["plateID", "rowID", "columnID", "fieldID"]
KEY = WELL + ["object_label"]


def _objects(n=4, start=1):
    """``n`` objects of one field, labelled ``start..start+n-1``."""
    return pd.DataFrame({
        "plateID": ["p1"] * n, "rowID": ["A"] * n, "columnID": ["1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": list(range(start, start + n)),
        "area": [10.0 * i for i in range(start, start + n)],
    })


def _db(folder, tables, name="measurements.db"):
    """Write ``tables`` ({name: frame}) into a fresh SQLite file."""
    os.makedirs(str(folder), exist_ok=True)
    path = str(folder / name)
    with sqlite3.connect(path) as db:
        for table, frame in tables.items():
            frame.to_sql(table, db, index=False)
    return path


def _png(ids, column="cell_id"):
    """A ``png_list`` with one crop per object id ``o<i>``."""
    return pd.DataFrame({
        "plateID": ["p1"] * len(ids), "rowID": ["A"] * len(ids),
        "columnID": ["1"] * len(ids), "fieldID": ["f1"] * len(ids),
        column: [f"o{i}" for i in ids],
        "png_path": [f"/crops/{i}.png" for i in ids]})


# ---------------------------------------------------------------------------
# choose_anchor
# ---------------------------------------------------------------------------

def test_choose_anchor_follows_the_preference_order_and_refuses_nothing(subtests):
    """The anchor is the first OBJECT_TABLES entry present; none present is refused."""
    cases = [
        ("cell wins wherever it is listed", ("nucleus", "cell", "pathogen"), "cell"),
        ("only nuclei anchor on nuclei", ("nucleus",), "nucleus"),
        ("pathogen before cytoplasm", ("cytoplasm", "pathogen"), "pathogen"),
        ("cytoplasm alone", ("cytoplasm",), "cytoplasm"),
        ("unknown names are ignored", ("notes", "nucleus"), "nucleus"),
    ]
    for case_id, tables, expected in cases:
        with subtests.test(msg=case_id):
            assert flt.choose_anchor(tables) == expected

    for case_id, tables in [("no tables", ()), ("no object tables", ("notes", "png_list"))]:
        with subtests.test(msg=case_id):
            with pytest.raises(FilterError) as caught:
                flt.choose_anchor(tables)
            assert str(caught.value) == (
                "this database has no object table to build filters from; "
                "looked for " + ", ".join(flt.OBJECT_TABLES))
            assert flt.OBJECT_TABLES[:4] == ("cell", "nucleus", "pathogen", "cytoplasm")


# ---------------------------------------------------------------------------
# key_columns
# ---------------------------------------------------------------------------

def test_key_columns_are_the_canonical_identity_in_join_order(subtests):
    """Exactly the canonical identity names present, well keys then time then label."""
    cases = [
        ("full identity, measurements dropped", KEY + ["area"], KEY),
        ("reordered input keeps join order",
         ["object_label", "fieldID", "area", "plateID", "columnID", "rowID"], KEY),
        ("a timelapse keeps its timepoint before the label",
         ["plateID", "rowID", "columnID", "fieldID", "timeID", "object_label", "area"],
         WELL + ["timeID", "object_label"]),
        ("a timepoint without a label", ["plateID", "timeID"], ["plateID", "timeID"]),
        ("aliases are not resolved here", ["plate", "row", "column", "field",
                                           "object_label"], ["object_label"]),
        ("only some well keys", ["plateID", "fieldID", "object_label"],
         ["plateID", "fieldID", "object_label"]),
        ("measurements only", ["area", "intensity"], []),
        ("no columns", [], []),
    ]
    for case_id, columns, expected in cases:
        with subtests.test(msg=case_id):
            assert flt.key_columns(pd.DataFrame(columns=columns)) == expected


# ---------------------------------------------------------------------------
# read_identity
# ---------------------------------------------------------------------------

def test_read_identity_returns_only_the_canonical_identity_or_refuses(tmp_path, subtests):
    """Identity columns only, renamed to canonical names; no object label is refused."""
    aliased = pd.DataFrame({"plate": ["p1", "p1"], "row": ["A", "B"],
                            "col": ["1", "2"], "field": ["f1", "f2"],
                            "time_id": [1, 2], "OBJECT_LABEL": [5, 6],
                            "intensity": [0.1, 0.2]})
    cases = [
        ("canonical spellings, measurements left out", {"cell": _objects(3)}, "cell",
         pd.DataFrame({"plateID": ["p1"] * 3, "rowID": ["A"] * 3,
                       "columnID": ["1"] * 3, "fieldID": ["f1"] * 3,
                       "object_label": [1, 2, 3]})),
        ("aliases read under their canonical names", {"cell": aliased}, "cell",
         pd.DataFrame({"plateID": ["p1", "p1"], "rowID": ["A", "B"],
                       "columnID": ["1", "2"], "fieldID": ["f1", "f2"],
                       "timeID": [1, 2], "object_label": [5, 6]})),
        ("a table with only a label", {"nucleus": pd.DataFrame(
            {"object_label": [7], "area": [1.0]})}, "nucleus",
         pd.DataFrame({"object_label": [7]})),
    ]
    for case_id, tables, table, expected in cases:
        with subtests.test(msg=case_id):
            path = _db(tmp_path / case_id.replace(" ", "_").replace(",", ""), tables)
            pd.testing.assert_frame_equal(flt.read_identity(path, table), expected)

    refusals = [
        ("rows that cannot be told apart", "settings",
         {"settings": pd.DataFrame({"key": ["a"], "value": ["b"]})}),
        ("a well-keyed table without labels", "cell",
         {"cell": _objects(2).drop(columns="object_label")}),
        ("a table that is not there", "pathogen", {"cell": _objects(2)}),
    ]
    for case_id, table, tables in refusals:
        with subtests.test(msg=case_id):
            path = _db(tmp_path / case_id.replace(" ", "_"), tables)
            with pytest.raises(FilterError) as caught:
                flt.read_identity(path, table)
            assert str(caught.value) == (
                f"table {table!r} has no object_label column, so its rows "
                f"cannot be identified")


def test_a_gate_or_annotation_onto_a_filters_table_with_no_object_key_is_refused(
        tmp_path, subtests):
    """A legacy ``filters`` table keyed on the well alone takes no gate or annotation."""
    cases = [
        ("gate", lambda path, frame: flt.export_gate(
            path, frame, np.ones(len(frame), bool), "big"),
         "the filters table and this measurement table share no object key, "
         "so the gate cannot be merged onto it"),
        ("annotation", lambda path, frame: flt.export_annotation(
            path, frame, pd.Series(["live"] * len(frame)), "state"),
         "the filters table and this measurement table share no object key"),
    ]
    for case_id, write, message in cases:
        with subtests.test(msg=case_id):
            legacy = pd.DataFrame({"plateID": ["p1"], "note": ["kept"]})
            path = _db(tmp_path / case_id, {"cell": _objects(), "filters": legacy})
            frame = flt.read_identity(path, "cell")
            with pytest.raises(FilterError) as caught:
                write(path, frame)
            assert str(caught.value) == message
            with sqlite3.connect(path) as db:
                stored = pd.read_sql_query('SELECT * FROM "filters"', db)
            pd.testing.assert_frame_equal(stored, legacy)


# ---------------------------------------------------------------------------
# build_filters_frame
# ---------------------------------------------------------------------------

def test_build_filters_frame_falls_back_on_surprises_but_not_on_refusals(
        tmp_path, monkeypatch, subtests):
    """Relationships route when it works; object tables on any other error; FilterError propagates."""
    def explode(_path):
        raise RuntimeError("relationships table is not readable")

    def refuse(_path):
        raise FilterError("this database has no object table")

    with subtests.test(msg="relationships route: one row per object"):
        path = _db(tmp_path / "rel", {"cell": _objects(4)})
        frame = flt.build_filters_frame(path)
        assert len(frame) == 4
        assert list(frame["object_label"]) == [1, 2, 3, 4]
        assert set(frame["object_type"]) == {"cell"}
        assert list(frame["in_cell"]) == [1, 1, 1, 1]

    with subtests.test(msg="an unexpected error falls back to the object tables"):
        path = _db(tmp_path / "boom", {"cell": _objects(3)})
        with monkeypatch.context() as patch:
            patch.setattr(flt, "build_filters_from_relationships", explode)
            frame = flt.build_filters_frame(path)
        expected = _objects(3)[KEY].assign(in_cell=[1, 1, 1])
        pd.testing.assert_frame_equal(frame, expected)

    with subtests.test(msg="a FilterError is not swallowed by the fallback"):
        path = _db(tmp_path / "refuse", {"cell": _objects()})
        with monkeypatch.context() as patch:
            patch.setattr(flt, "build_filters_from_relationships", refuse)
            with pytest.raises(FilterError) as caught:
                flt.build_filters_frame(path)
        assert str(caught.value) == "this database has no object table"


@pytest.fixture
def read_only_db(tmp_path):
    """Make databases nothing may write to, as on a mounted results share."""
    made = []

    def _make(case_id, tables):
        folder = tmp_path / case_id.replace(" ", "_")
        path = _db(folder, tables)
        os.chmod(path, 0o444)
        os.chmod(folder, 0o555)
        made.append((folder, path))
        return path

    yield _make
    for folder, path in made:
        os.chmod(folder, 0o755)
        os.chmod(path, 0o644)


def test_the_object_table_fallback_merges_flags_and_crop_paths(
        read_only_db, monkeypatch, subtests):
    """A read-only database is bootstrapped from its object tables: exact flags and crops."""
    real = flt.read_identity

    def refuse_nucleus(db_path, table):
        if table == "nucleus":
            raise FilterError("nucleus went away")
        return real(db_path, table)

    def refuse_all(db_path, table):
        raise FilterError("unreadable")

    def rename_nucleus(db_path, table):
        frame = real(db_path, table)
        if table == "nucleus":
            return frame.rename(columns={c: f"nucleus_{c}" for c in frame.columns})
        return frame

    cell_and_two_nuclei = lambda: {"cell": _objects(), "nucleus": _objects(n=2)}
    cases = [
        # case id, tables, patches, expected frame columns -> values
        ("cell and nucleus merge with presence flags", cell_and_two_nuclei, {},
         {"object_label": [1, 2, 3, 4], "in_cell": [1, 1, 1, 1],
          "in_nucleus": [1, 1, 0, 0]}),
        ("crop paths reach only the cropped type",
         lambda: {"cell": _objects(), "nucleus": _objects(n=2, start=90),
                  "png_list": _png([1, 90])}, {},
         {"object_label": [1, 2, 3, 4, 90, 91], "in_cell": [1, 1, 1, 1, 0, 0],
          "in_nucleus": [0, 0, 0, 0, 1, 1],
          "png_path": ["/crops/1.png", None, None, None, None, None]}),
        ("a crop type with no flag keeps its label-matched paths",
         lambda: {"cell": _objects(2), "png_list": _png([1, 2], "pathogen_id")}, {},
         {"object_label": [1, 2], "in_cell": [1, 1],
          "png_path": ["/crops/1.png", "/crops/2.png"]}),
        ("an unreadable table is skipped", cell_and_two_nuclei,
         {"read_identity": refuse_nucleus},
         {"object_label": [1, 2, 3, 4], "in_cell": [1, 1, 1, 1]}),
        ("a table sharing no identity is not merged blind", cell_and_two_nuclei,
         {"read_identity": rename_nucleus},
         {"object_label": [1, 2, 3, 4], "in_cell": [1, 1, 1, 1]}),
        ("a png_list sharing no key carries no crops",
         lambda: {"cell": _objects()},
         {"_png_paths": lambda db_path: pd.DataFrame(
             {"crop_number": [1, 2], "png_path": ["/a.png", "/b.png"]})},
         {"object_label": [1, 2, 3, 4], "in_cell": [1, 1, 1, 1]}),
    ]
    for case_id, tables, patches, expected in cases:
        with subtests.test(msg=case_id):
            path = read_only_db(case_id, tables())
            with monkeypatch.context() as patch:
                for name, value in patches.items():
                    patch.setattr(flt, name, value)
                frame = flt.build_filters_frame(path)
            assert "object_type" not in frame.columns, "took the relationships route"
            assert list(frame.columns) == WELL + list(expected)
            assert flt.key_columns(frame) == KEY
            assert frame[WELL].drop_duplicates().values.tolist() == [["p1", "A", "1", "f1"]]
            for column, values in expected.items():
                got = [None if pd.isna(v) else v for v in frame[column].tolist()]
                assert got == values, column

    with subtests.test(msg="no table readable names the tables it looked at"):
        path = read_only_db("none readable", cell_and_two_nuclei())
        with monkeypatch.context() as patch:
            patch.setattr(flt, "read_identity", refuse_all)
            with pytest.raises(FilterError) as caught:
                flt.build_filters_frame(path)
        assert str(caught.value) == (
            "none of cell, nucleus carries the identity columns a filter needs")


# ---------------------------------------------------------------------------
# read_sampled
# ---------------------------------------------------------------------------

def test_read_sampled_takes_a_systematic_share_under_an_optional_cap(tmp_path, subtests):
    """Exact rows for each fraction/limit pair, in SQL or the pandas fallback."""
    hundred = _db(tmp_path / "hundred", {"cell": _objects(n=100)})
    ten = _db(tmp_path / "ten", {"cell": _objects(n=10)})
    shadowed = _db(tmp_path / "shadowed", {"cell": pd.DataFrame({
        "rowid": range(200), "oid": range(200), "_rowid_": range(200),
        "object_label": range(200), "area": np.arange(200.0)})})
    cases = [
        # case id, db, kwargs, expected object labels
        ("nothing asked reads the whole table", ten, {}, list(range(1, 11))),
        ("a limit caps the rows", ten, {"limit": 4}, [1, 2, 3, 4]),
        ("a limit above the size reads everything", ten, {"limit": 50},
         list(range(1, 11))),
        ("a fraction of 1 is everything", ten, {"fraction": 1.0}, list(range(1, 11))),
        ("a fifth is every fifth row id", hundred, {"fraction": 0.2},
         list(range(5, 101, 5))),
        ("a half is every second row id", ten, {"fraction": 0.5}, [2, 4, 6, 8, 10]),
        ("a fraction near 1 still steps by 2, then the limit applies", hundred,
         {"fraction": 0.9, "limit": 5}, [2, 4, 6, 8, 10]),
        ("shadowed row ids sample in pandas, then cap", shadowed,
         {"fraction": 0.25, "limit": 10}, list(range(0, 40, 4))),
        ("shadowed row ids sample in pandas without a cap", shadowed,
         {"fraction": 0.1}, list(range(0, 200, 10))),
        ("shadowed row ids, no fraction, still capped in SQL", shadowed,
         {"limit": 3}, [0, 1, 2]),
    ]
    for case_id, path, kwargs, labels in cases:
        with subtests.test(msg=case_id):
            frame = flt.read_sampled(path, "cell", **kwargs)
            assert frame["object_label"].tolist() == labels
            assert list(frame.index) == list(range(len(labels)))

    for bad in (0.0, -0.5, 1.5):
        with subtests.test(msg=f"fraction {bad} is refused before reading"):
            with pytest.raises(FilterError) as caught:
                flt.read_sampled(str(tmp_path / "missing.db"), "cell", fraction=bad)
            assert str(caught.value) == (
                f"sample fraction {bad!r} is not a fraction between 0 and 1")
