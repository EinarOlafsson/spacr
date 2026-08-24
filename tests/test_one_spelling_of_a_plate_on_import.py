"""A doubled ``pp`` plate prefix is collapsed when a database is read.

Asked for: "whenever pplate is found in a database or csv change to plate,
add this to the import logic."

A screen written by an older run stamps its plate ``pplate1`` and
everything computed since stamps it ``plate1``. The two then do not join,
and the failure is silent: an ML run over 60,816 real cells fitted, scored,
explained and plotted, then merged 0 rows and reported that its own
database "probably comes from a different experiment than png_list".

``schema.canonical_plate_id`` is the one rule and ``normalise_plate_columns``
applies it to every plate-bearing column -- plateID, prc, prcf, prcfo. It
had two callers and neither was the database reader every module goes
through.

APPLIED ON READ. Nothing on disk is rewritten, so an old database keeps
working unchanged and a re-read of it produces the same keys as a fresh
run -- which is the only thing that makes the two join.
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

from spacr import schema


@pytest.mark.parametrize("stored,expected", [
    ("pplate1", "plate1"),
    ("plate1", "plate1"),
    ("pplate1_r8_c19_f11", "plate1_r8_c19_f11"),
    ("pplate1_r8_c19_f11_o84", "plate1_r8_c19_f11_o84"),
    ("p1", "p1"),                 # a single p is a real plate name
    ("plate_1", "plate_1"),
])
def test_the_rule_collapses_only_a_doubled_prefix(stored, expected):
    assert schema.canonical_plate_id(stored) == expected


def test_every_plate_bearing_column_is_covered():
    assert set(schema.PLATE_BEARING_COLUMNS) >= {"plateID", "prc", "prcf",
                                                 "prcfo"}


def test_a_frame_is_normalised_across_all_of_them():
    frame = pd.DataFrame({
        "plateID": ["pplate1"], "prcf": ["pplate1_r1_c1_f1"],
        "prcfo": ["pplate1_r1_c1_f1_o2"], "area": [10.0],
    })

    schema.normalise_plate_columns(frame)

    assert frame["plateID"].iloc[0] == "plate1"
    assert frame["prcf"].iloc[0] == "plate1_r1_c1_f1"
    assert frame["prcfo"].iloc[0] == "plate1_r1_c1_f1_o2"
    assert frame["area"].iloc[0] == 10.0, "a measurement was touched"


def test_the_database_reader_normalises(tmp_path):
    """The path every module goes through, which is where it was missing."""
    from spacr.io import _read_and_join_tables

    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as db:
        pd.DataFrame({
            "plateID": ["pplate1"] * 2, "rowID": ["r1"] * 2,
            "columnID": ["c1"] * 2, "fieldID": ["f1"] * 2,
            "prcf": ["pplate1_r1_c1_f1"] * 2,
            "object_label": [1, 2], "cell_area": [10.0, 20.0],
        }).to_sql("cell", db, index=False)

    frame = _read_and_join_tables(str(path), keep_uninfected=True,
                                  require_crops=False)

    assert frame is not None and len(frame) == 2
    assert set(frame["plateID"]) == {"plate1"}
    assert set(frame["prcf"]) == {"plate1_r1_c1_f1"}


def test_nothing_on_disk_is_rewritten(tmp_path):
    """An old database keeps working; the repair is on read."""
    from spacr.io import _read_and_join_tables

    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as db:
        pd.DataFrame({
            "plateID": ["pplate1"], "rowID": ["r1"], "columnID": ["c1"],
            "fieldID": ["f1"], "object_label": [1], "cell_area": [10.0],
        }).to_sql("cell", db, index=False)

    _read_and_join_tables(str(path), keep_uninfected=True,
                          require_crops=False)

    with sqlite3.connect(path) as db:
        on_disk = pd.read_sql("SELECT plateID FROM cell", db)
    assert on_disk["plateID"].iloc[0] == "pplate1"


def test_a_csv_is_repaired_on_read_too(tmp_path):
    """`tabular.read_table` carries the same rule, on by default."""
    from spacr import tabular

    path = tmp_path / "scores.csv"
    pd.DataFrame({"plateID": ["pplate1"], "pred": [0.5]}).to_csv(path,
                                                                 index=False)

    frame = tabular.read_table(str(path))

    assert frame["plateID"].iloc[0] == "plate1"
