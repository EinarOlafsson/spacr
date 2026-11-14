"""What the plate merge tells the user about a merge it has already done.

Two units of :mod:`spacr.plate_measurements`:

* :func:`~spacr.plate_measurements.describe_identifier_refusal` -- the one
  sentence that says why a varying text identifier was left out: with or
  without an example, truncated to three values with the hidden count, and
  always a finished sentence;
* :func:`~spacr.plate_measurements.merge_plate_databases` -- the anchor's row
  counts, screens and shared plates are found by NAME in ``tables`` (and fall
  through to empty when the anchor record is absent); same-named columns lost
  by two child tables keep their table prefixes; a refused identifier reaches
  the log with or without a ``report`` callback, in the same words.

One ``measurements.db`` per plate under its own folder, as spaCR writes them.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import replace

import pandas as pd

from spacr.multi_database import SCREEN_COLUMN
from spacr.plate_measurements import (describe_identifier_refusal,
                                      merge_plate_databases)

_IDENTITY = {"rowID": "A", "columnID": "1", "fieldID": "f1"}
_REFUSAL_TAIL = ("is a text identifier that differs WITHIN {groups} group(s), "
                 "so it was left out rather than set to one of its values — "
                 "picking one invents provenance")


def _refusal(table, column, groups, example=""):
    line = f"  {table}: {column} " + _REFUSAL_TAIL.format(groups=groups)
    return line + (f" (e.g. {example})" if example else "") + "."


def test_describe_identifier_refusal_is_one_finished_sentence(subtests):
    """Example clause, truncation count and full stop for every detail shape."""
    eight = [f"object_{i}" for i in range(8)]
    cases = [
        ("no example key", "pathogen", "path_name", {"groups": 3},
         _refusal("pathogen", "path_name", 3)),
        ("an empty example list", "pathogen", "path_name",
         {"groups": 3, "examples": []}, _refusal("pathogen", "path_name", 3)),
        ("no group count", "cell", "prcfo", {},
         _refusal("cell", "prcfo", 0)),
        ("a two-value example", "pathogen", "path_name",
         {"groups": 3, "examples": [(("plate1", "A01"), ["a.tif", "b.tif"])]},
         _refusal("pathogen", "path_name", 3, "plate1/A01: a.tif, b.tif")),
        ("exactly three values state no count", "cell", "prcfo",
         {"groups": 1, "examples": [(("plate1", "A01"), ["a", "b", "c"])]},
         _refusal("cell", "prcfo", 1, "plate1/A01: a, b, c")),
        ("a truncated example says how many it hid", "cell", "prcfo",
         {"groups": 4, "examples": [(("plate1", "A01"), eight),
                                    (("plate2", "B02"), ["z"])]},
         _refusal("cell", "prcfo", 4,
                  "plate1/A01: object_0, object_1, object_2, … (8 values)")),
        ("thousands are grouped", "cell", "prcfo", {"groups": 12345},
         _refusal("cell", "prcfo", "12,345")),
    ]
    for case_id, table, column, detail, expected in cases:
        with subtests.test(msg=case_id):
            assert describe_identifier_refusal(table, column, detail) == expected


def _ids(n):
    return {key: [value] * n for key, value in _IDENTITY.items()}


def _cells(plate, n=2, extra=None):
    """``n`` cells of one field, numbered from 1."""
    frame = pd.DataFrame({"plateID": [plate] * n, **_ids(n),
                          "object_label": list(range(1, n + 1)),
                          "area": [10.0 * (i + 1) for i in range(n)]})
    if extra is not None:
        frame[extra] = [1.0] * n
    return frame


def _children(plate, *, object_label=True, path_names=None):
    """Two children of cell 1 -- the many-per-cell side of the roll-up."""
    frame = pd.DataFrame({"plateID": [plate] * 2, **_ids(2),
                          "cell_id": [1, 1], "area": [1.0, 2.0]})
    if object_label:
        frame["object_label"] = [1, 2]
    if path_names is not None:
        frame["path_name"] = list(path_names)
    return frame


def _database(directory, tables):
    """Write ``{table: frame}`` to ``<directory>/measurements.db``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = str(directory / "measurements.db")
    with sqlite3.connect(path) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return path


def _logged(caplog, needle):
    return [record.getMessage() for record in caplog.records
            if record.name == "spacr.plate_measurements"
            and needle in record.getMessage()]


def test_merge_plate_databases_discloses_what_it_did(subtests, tmp_path,
                                                     caplog):
    """Anchor numbers, dropped columns and refusals are reported exactly."""
    with subtests.test(msg="the anchor numbers are found by name"):
        attached = {
            name: _database(tmp_path / "anchor" / name,
                            {"cell": _cells("plate1"),
                             "pathogen": _children("plate1")})
            for name in ("kd", "oe")}
        merge = merge_plate_databases(attached, ["pathogen"],
                                      screens={"kd": "kd", "oe": "oe"})
        assert [entry.table for entry in merge.tables] == ["cell", "pathogen"]
        assert len(merge.sources) == 2
        assert merge.rows_read_per_source == dict.fromkeys(merge.sources, 2)
        assert merge.shared_plates_across_screens == {"plate1": ("kd", "oe")}
        assert sorted(merge.frame[SCREEN_COLUMN].unique()) == ["kd", "oe"]
        # The same merge with the anchor record last.
        shuffled = replace(merge, tables=tuple(reversed(merge.tables)))
        assert shuffled.tables[0].table == "pathogen"
        assert shuffled.sources == merge.sources
        assert shuffled.rows_read_per_source == merge.rows_read_per_source
        assert shuffled.shared_plates_across_screens == {
            "plate1": ("kd", "oe")}
        assert shuffled.describe() == merge.describe()
        assert "more than one SCREEN" in shuffled.describe()

    with subtests.test(msg="no anchor record reports nothing, not a guess"):
        path = _database(tmp_path / "solo" / "plate1",
                         {"cell": _cells("plate1"),
                          "pathogen": _children("plate1")})
        merge = merge_plate_databases({"plate1": path}, ["pathogen"])
        children = tuple(e for e in merge.tables if e.table != "cell")
        without_anchor = replace(merge, tables=children)
        assert len(merge.sources) == 1
        assert merge.rows_read_per_source == dict.fromkeys(merge.sources, 2)
        assert merge.shared_plates_across_screens == {}
        assert without_anchor.sources == ()
        assert without_anchor.rows_read_per_source == {}
        assert without_anchor.shared_plates_across_screens == {}

    with subtests.test(msg="each child table names its own lost label"):
        first = _database(tmp_path / "lost" / "plate1", {
            "cell": _cells("plate1", extra="stain"),
            "pathogen": _children("plate1"),
            "nucleus": _children("plate1")})
        second = _database(tmp_path / "lost" / "plate2", {
            "cell": _cells("plate2"),
            "pathogen": _children("plate2", object_label=False),
            "nucleus": _children("plate2", object_label=False)})
        merge = merge_plate_databases({"plate1": first, "plate2": second},
                                      ["pathogen", "nucleus"])
        lost = {entry.table: entry.dropped for entry in merge.tables}
        assert lost == {"cell": ("stain",), "pathogen": ("object_label",),
                        "nucleus": ("object_label",)}
        assert merge.dropped_columns == (
            "cell_stain", "nucleus_object_label", "pathogen_object_label")
        assert ("3 measurement(s) present in only some databases were dropped"
                in merge.describe())
        assert "object_label" in merge.frame
        assert "cell_object_label" not in merge.frame
        assert merge.frame.attrs["dropped_columns"] == merge.dropped_columns

    with subtests.test(msg="a refusal is logged when no report was given"):
        caplog.clear()
        cell = _cells("plate1")
        pathogen = _children("plate1", path_names=["a.tif", "b.tif"])
        pathogen["file_name"] = ["one.tif", "one.tif"]
        path = _database(tmp_path / "headless" / "plate1",
                         {"cell": cell, "pathogen": pathogen})
        with caplog.at_level(logging.INFO, logger="spacr.plate_measurements"):
            merge = merge_plate_databases({"plate1": path}, ["pathogen"])
        logged = _logged(caplog, "invents provenance")
        assert len(logged) == 1
        sentence = _refusal("pathogen", "path_name", 1).strip()[:-1]
        # The example key is the group's identity, ending in the cell label.
        assert logged[0].startswith(sentence + " (e.g. plate1/A/1/f1/")
        assert logged[0].endswith("/1: a.tif, b.tif).")
        assert "pathogen_path_name" not in merge.frame.columns
        assert merge.frame.attrs["refused_identifiers"] == {
            "pathogen": ("path_name",)}
        cell_one = merge.frame[merge.frame["object_label"] == 1].iloc[0]
        assert cell_one["pathogen_file_name"] == "one.tif"
        assert cell_one["pathogen_area"] == 3.0

    with subtests.test(msg="the same refusal reaches a report callback"):
        caplog.clear()
        path = _database(tmp_path / "reported" / "plate1",
                         {"cell": _cells("plate1", n=1),
                          "pathogen": _children(
                              "plate1", path_names=["a.tif", "b.tif"])})
        said = []
        with caplog.at_level(logging.INFO, logger="spacr.plate_measurements"):
            merge = merge_plate_databases({"plate1": path}, ["pathogen"],
                                          report=said.append)
        refusals = [line for line in said if "invents provenance" in line]
        assert len(refusals) == 1
        assert refusals == _logged(caplog, "invents provenance")
        assert "pathogen_path_name" not in merge.frame.columns
        assert merge.frame["pathogen_area"].tolist() == [3.0]
