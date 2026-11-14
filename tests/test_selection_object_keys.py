"""Table-driven tests of the object keys linked views use to name the same rows.

A key built by ``spacr.selection.object_keys`` is what a lasso in one view uses
to highlight the same objects in another, and what
``spacr.active_learning.crops_for_object_keys`` resolves to crop files.  The
cases here are the frames the ordinary path never produces: a type column
that is entirely blank (untyped, not typed with an empty prefix), an empty
frame (an empty answer), a frame missing a key column (a refusal naming the
column), and field names that collide unless the separator and ``%`` are
both escaped.  Pure pandas plus one real SQLite ``png_list``.
"""
from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd
import pytest

import spacr.active_learning as al
from spacr.selection import (FilterError, Selection, object_keys,
                             untyped_object_keys)

UNTYPED = ["p1_r1_c1_f1_1", "p1_r1_c1_f1_2"]


def _frame(**extra) -> pd.DataFrame:
    """Two objects in one field, plus any extra columns."""
    data = {"plateID": ["p1", "p1"], "rowID": ["r1", "r1"],
            "columnID": ["c1", "c1"], "fieldID": ["f1", "f1"],
            "object_label": [1, 2]}
    data.update(extra)
    return pd.DataFrame(data)


def _missing(column, rows=2):
    """The start of the refusal for a frame lacking ``column``."""
    return re.escape(f"cannot build object keys: {rows}-row frame is missing "
                     f"['{column}']")


def test_object_keys_compose_one_stable_key_per_row(subtests):
    """Blank types stay untyped, stated types prefix the label, collisions are escaped."""
    escaped = pd.DataFrame([{"plateID": "plate1", "rowID": "r1",
                             "columnID": "c1", "fieldID": f, "object_label": 7}
                            for f in ("f%5F1", "f_1")])
    cases = [
        ("no type column", _frame(), {}, UNTYPED, UNTYPED),
        ("blank type column of None and ''",
         _frame(object_type=[None, ""]), {}, UNTYPED, UNTYPED),
        ("blank type column of None", _frame(object_type=[None, None]), {},
         UNTYPED, UNTYPED),
        ("blank type column of ''", _frame(object_type=["", ""]), {},
         UNTYPED, UNTYPED),
        ("blank type column of nan/none text",
         _frame(object_type=["nan", "none"]), {}, UNTYPED, UNTYPED),
        ("blank type column of NaN/null",
         _frame(object_type=[float("nan"), "null"]), {}, UNTYPED, UNTYPED),
        ("a stated type column prefixes the label",
         _frame(object_type=["nucleus", ""]), {},
         ["p1_r1_c1_f1_nucleus1", "p1_r1_c1_f1_2"], UNTYPED),
        ("object_type= types every row",
         _frame(), {"object_type": "cell"},
         ["p1_r1_c1_f1_cell1", "p1_r1_c1_f1_cell2"], UNTYPED),
        ("a timelapse key carries the timepoint",
         _frame(timeID=["t1", "t2"]), {"timelapse": True},
         ["p1_r1_c1_f1_t1_1", "p1_r1_c1_f1_t2_2"],
         ["p1_r1_c1_f1_t1_1", "p1_r1_c1_f1_t2_2"]),
        ("'%' and the separator are both escaped so two fields stay apart",
         escaped, {}, ["plate1_r1_c1_f%255F1_7", "plate1_r1_c1_f%5F1_7"],
         ["plate1_r1_c1_f%255F1_7", "plate1_r1_c1_f%5F1_7"]),
    ]
    for case_id, frame, kwargs, typed, untyped in cases:
        with subtests.test(msg=case_id):
            assert list(object_keys(frame, **kwargs)) == typed
            untyped_kwargs = {k: v for k, v in kwargs.items()
                              if k == "timelapse"}
            assert list(untyped_object_keys(frame, **untyped_kwargs)) == untyped

    with subtests.test(msg="an empty frame has no keys"):
        for build in (object_keys, untyped_object_keys):
            keys = build(_frame().iloc[:0])
            assert (len(keys), keys.dtype) == (0, object)

    refusals = [
        ("a missing key column is named", _frame().drop(columns=["fieldID"]),
         {}, _missing("fieldID")),
        ("a timelapse frame without timeID names it", _frame(),
         {"timelapse": True}, _missing("timeID")),
    ]
    for case_id, frame, kwargs, message in refusals:
        for build in (object_keys, untyped_object_keys):
            with subtests.test(msg=f"{case_id} ({build.__name__})"):
                with pytest.raises(FilterError, match=message):
                    build(frame, **kwargs)


def test_selection_from_frame_matches_exactly_the_rows_it_was_made_from(
        subtests):
    """A selection marks its own rows; a frame missing a key column is refused by name."""
    frame = _frame()
    cases = [
        ("the first row only", Selection.from_frame(frame.iloc[:1]), frame,
         [True, False]),
        ("both rows", Selection.from_frame(frame), frame, [True, True]),
        ("an empty lasso is active and matches nothing",
         Selection.from_frame(frame.iloc[:0]), frame, [False, False]),
        ("the resting selection keeps every row", Selection.none(), frame,
         [True, True]),
        ("an untyped key matches a typed row",
         Selection.from_frame(frame.iloc[1:]),
         _frame(object_type=["cell", "cell"]), [False, True]),
    ]
    for case_id, chosen, target, expected in cases:
        with subtests.test(msg=case_id):
            mask = chosen.mask_for(target)
            assert mask.dtype == bool
            np.testing.assert_array_equal(mask, expected)

    with subtests.test(msg="a frame lacking a key column says which one"):
        with pytest.raises(FilterError, match=_missing("object_label")):
            Selection.from_frame(frame).mask_for(
                frame.drop(columns=["object_label"]))


def test_escaped_keys_open_the_crop_of_the_field_they_name(tmp_path):
    """The escaped spelling wins when a raw and an escaped field spell one key."""
    columns = ["png_path", "prcfo", "plateID", "rowID", "columnID", "fieldID",
               "cell_id", "annotate"]
    rows = [("/crops/cell_png/literal_percent.png", "plate1_r1_c1_f%5F1_o7",
             "plate1", "r1", "c1", "f%5F1", "o7", None),
            ("/crops/cell_png/literal_underscore.png", "plate1_r1_c1_f_1_o7",
             "plate1", "r1", "c1", "f_1", "o7", None)]
    db = str(tmp_path / "measurements.db")
    con = sqlite3.connect(db)
    try:
        con.execute("CREATE TABLE png_list (%s)" % ", ".join(
            f'"{c}" {"INTEGER" if c == "annotate" else "TEXT"}'
            for c in columns))
        con.executemany(f"INSERT INTO png_list VALUES ({', '.join('?' * 8)})",
                        rows)
        con.commit()
    finally:
        con.close()

    frame = pd.DataFrame([{"plateID": "plate1", "rowID": "r1",
                           "columnID": "c1", "fieldID": f, "object_label": 7}
                          for f in ("f%5F1", "f_1")])
    assert al.crops_for_object_keys(db, list(object_keys(frame))) == [
        ("/crops/cell_png/literal_percent.png", None),
        ("/crops/cell_png/literal_underscore.png", None)]
