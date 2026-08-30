"""The merge disclosure, in the four places it can still say the wrong thing.

Every assertion here is about what the Measurements tab TELLS the user about a
merge it has already done, which is the half of instruction 130 that decides
whether a wrong number is noticed:

* the anchor's row counts, screens and shared plate ids are looked up by NAME
  in :attr:`spacr.plate_measurements.PlateMerge.tables`, so a panel that
  filters or re-orders those records still gets the anchor's numbers rather
  than the first table's or none at all;
* a column that two child tables both lost is one loss to report, not two;
* :func:`~spacr.plate_measurements.describe_identifier_refusal` still produces
  a sentence when the detail it is handed carries no example;
* a refused text identifier reaches the LOG even when the caller passed no
  ``report`` callback -- the headless path, where the only record of a dropped
  column is the log line.

The fixtures mirror :mod:`tests.test_plate_measurements`: one
``measurements.db`` per plate under its own folder, as spaCR writes them.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import replace

import pandas as pd

from spacr.multi_database import SCREEN_COLUMN
from spacr.plate_measurements import describe_identifier_refusal, merge_plate_databases

_IDENTITY = {"rowID": "A", "columnID": "1", "fieldID": "f1"}


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


# --------------------------------------------------------------------------- #
#  The anchor record is found by name, wherever it sits in `tables`
# --------------------------------------------------------------------------- #

def test_the_anchor_numbers_are_found_by_name_not_by_position(tmp_path):
    """`sources`, `rows_read_per_source` and `shared_plates_across_screens`
    all report the ANCHOR's plan, and each walks `tables` looking for it. The
    merge happens to build that record first today; a panel that hands back a
    filtered or re-sorted tuple (dropping a table the user unticked, listing
    the anchor last) must still get 3,000 cell rows and the two screens that
    share plate1 -- not the pathogen table's numbers and not an empty dict,
    which would read as "no screen shares a plate" and be untrue.
    """
    attached = {"kd": _database(tmp_path / "kd", {"cell": _cells("plate1"),
                                                  "pathogen": _children("plate1")}),
                "oe": _database(tmp_path / "oe", {"cell": _cells("plate1"),
                                                  "pathogen": _children("plate1")})}

    merge = merge_plate_databases(attached, ["pathogen"],
                                  screens={"kd": "kd", "oe": "oe"})

    # As built today the anchor is the first record, so the loops return on
    # their first iteration and never step past a non-anchor table.
    assert [entry.table for entry in merge.tables] == ["cell", "pathogen"]
    assert len(merge.sources) == 2
    assert merge.rows_read_per_source == dict.fromkeys(merge.sources, 2)
    assert merge.shared_plates_across_screens == {"plate1": ("kd", "oe")}
    assert sorted(merge.frame[SCREEN_COLUMN].unique()) == ["kd", "oe"]

    # The same merge, with the anchor last -- the shape a caller produces by
    # sorting or filtering the records before reading them.
    shuffled = replace(merge, tables=tuple(reversed(merge.tables)))

    assert shuffled.tables[0].table == "pathogen"
    assert shuffled.sources == merge.sources
    assert shuffled.rows_read_per_source == merge.rows_read_per_source
    assert (shuffled.shared_plates_across_screens
            == {"plate1": ("kd", "oe")})
    # And the sentence a user reads is the same sentence.
    assert shuffled.describe().splitlines()[0] == merge.describe().splitlines()[0]
    assert "more than one SCREEN" in shuffled.describe()


def test_a_pathogen_only_record_set_reports_no_anchor_rather_than_a_guess(
        tmp_path):
    """The counterpart: when the anchor's record really is absent -- a caller
    that filtered it out -- the properties fall through to empty rather than
    reporting the pathogen table's row counts as the cell's. A panel showing
    pathogen counts under a "cell objects" heading is a wrong number that
    looks right, which is the failure this module exists to prevent."""
    attached = {"plate1": _database(tmp_path / "plate1",
                                    {"cell": _cells("plate1"),
                                     "pathogen": _children("plate1")})}

    merge = merge_plate_databases(attached, ["pathogen"])
    children = tuple(entry for entry in merge.tables if entry.table != "cell")
    without_anchor = replace(merge, tables=children)

    # Driven, not merely asserted absent: the intact merge answers with the
    # anchor's two cells, and only the filtered one answers with nothing.
    assert merge.rows_read_per_source == dict.fromkeys(merge.sources, 2)
    assert len(merge.sources) == 1
    assert without_anchor.sources == ()
    assert without_anchor.rows_read_per_source == {}
    assert without_anchor.shared_plates_across_screens == {}


# --------------------------------------------------------------------------- #
#  One column lost by two tables is one loss to report
# --------------------------------------------------------------------------- #

def test_a_column_two_tables_both_lost_is_named_once(tmp_path):
    """`columns='common'` drops what is not in every database, and the panel
    lists what it dropped. When two child tables lose the SAME column name the
    list must carry it once: a user reading "3 measurement(s) were dropped"
    over a list of two distinct names cannot tell whether the count or the
    list is wrong, and either way stops trusting the disclosure."""
    first = _database(tmp_path / "plate1", {
        "cell": _cells("plate1", extra="stain"),
        "pathogen": _children("plate1"),
        "nucleus": _children("plate1")})
    second = _database(tmp_path / "plate2", {
        "cell": _cells("plate2"),
        "pathogen": _children("plate2", object_label=False),
        "nucleus": _children("plate2", object_label=False)})

    merge = merge_plate_databases({"plate1": first, "plate2": second},
                                  ["pathogen", "nucleus"])

    lost = {entry.table: entry.dropped for entry in merge.tables}
    # The loss really did happen twice, once per child table...
    assert lost["pathogen"] == ("object_label",)
    assert lost["nucleus"] == ("object_label",)
    # ...and once for the anchor, under a name of its own.
    assert lost["cell"] == ("stain",)
    # NOTE: the child tables' loss is reported unprefixed while the frame
    # would carry it as `pathogen_object_label`; see the defect reported with
    # this file. What is pinned here is that it is listed ONCE.
    assert merge.dropped_columns == ("cell_stain", "object_label")
    assert len(merge.dropped_columns) == len(set(merge.dropped_columns))
    assert "2 measurement(s) present in only some databases were dropped" \
        in merge.describe()
    assert merge.frame.attrs["dropped_columns"] == merge.dropped_columns


# --------------------------------------------------------------------------- #
#  A refusal with no example is still a sentence
# --------------------------------------------------------------------------- #

def test_a_refusal_with_no_example_is_still_a_finished_sentence():
    """The example is optional -- a detail that has been round-tripped through
    a saved frame's attrs, or built by a caller that only counted the groups,
    carries none. The sentence still has to name the column, the group count
    and end in a full stop: a refusal that trails off in "left out rather than
    set to one of its values" reads as a truncated message rather than as the
    complete reason a column is missing from the user's frame."""
    without = describe_identifier_refusal("pathogen", "path_name",
                                          {"groups": 3})
    with_example = describe_identifier_refusal(
        "pathogen", "path_name",
        {"groups": 3, "examples": [(("plate1", "A01"), ["a.tif", "b.tif"])]})

    assert without.endswith("invents provenance.")
    assert "path_name is a text identifier that differs WITHIN 3 group(s)" \
        in without
    # The absence is real because the same call WITH an example produces the
    # clause that is missing here.
    assert "(e.g." not in without
    assert "(e.g. plate1/A01: a.tif, b.tif)" in with_example
    assert with_example.startswith(without[:-1])
    # An empty example list is the same case as no key at all.
    assert describe_identifier_refusal("pathogen", "path_name",
                                       {"groups": 3, "examples": []}) == without


# --------------------------------------------------------------------------- #
#  The headless path: refused, logged, no report callback
# --------------------------------------------------------------------------- #

def test_a_refused_identifier_is_logged_when_no_report_was_given(
        tmp_path, caplog):
    """`report` is the panel's line sink and it is optional -- a script, a
    test or the gate editor calls this merge without one. The column is still
    dropped in that case, so the log line is then the ONLY record that a
    measurement the user asked for is not in the frame. Lose it and a headless
    run silently returns fewer columns than the same merge in the GUI."""
    identity = dict(_IDENTITY)
    cell = pd.DataFrame({"plateID": ["plate1"] * 2,
                         **{k: [v] * 2 for k, v in identity.items()},
                         "object_label": [1, 2], "area": [10.0, 20.0]})
    pathogen = _children("plate1", path_names=["a.tif", "b.tif"])
    pathogen["file_name"] = ["one.tif", "one.tif"]
    path = _database(tmp_path / "plate1", {"cell": cell, "pathogen": pathogen})

    with caplog.at_level(logging.INFO, logger="spacr.plate_measurements"):
        merge = merge_plate_databases({"plate1": path}, ["pathogen"])

    logged = [record.getMessage() for record in caplog.records
              if record.name == "spacr.plate_measurements"]
    assert any("path_name is a text identifier that differs WITHIN 1 group(s)"
               in line for line in logged), logged
    assert any("invents provenance" in line for line in logged), logged
    # The line describes something that really happened to the frame.
    assert "pathogen_path_name" not in merge.frame.columns
    assert merge.frame.attrs["refused_identifiers"] == {
        "pathogen": ("path_name",)}
    # The identifier that is constant within the group is carried, and the
    # measurement beside it is untouched: only the ambiguous one went.
    cell_one = merge.frame[merge.frame["object_label"] == 1].iloc[0]
    assert cell_one["pathogen_file_name"] == "one.tif"
    assert cell_one["pathogen_area"] == 3.0


def test_the_same_refusal_reaches_a_report_callback_when_there_is_one(
        tmp_path, caplog):
    """The report sink and the log must say the same thing -- a user who
    reads the panel and a maintainer who reads the log are looking at one
    merge, and two different sentences about it is how a support question
    becomes unanswerable."""
    cell = _cells("plate1", n=1)
    pathogen = _children("plate1", path_names=["a.tif", "b.tif"])
    path = _database(tmp_path / "plate1", {"cell": cell, "pathogen": pathogen})

    said = []
    with caplog.at_level(logging.INFO, logger="spacr.plate_measurements"):
        merge = merge_plate_databases({"plate1": path}, ["pathogen"],
                                      report=said.append)

    refusals = [line for line in said if "invents provenance" in line]
    logged = [record.getMessage() for record in caplog.records
              if "invents provenance" in record.getMessage()]
    assert len(refusals) == 1, said
    assert refusals == logged
    assert "pathogen_path_name" not in merge.frame.columns
    assert merge.frame["pathogen_area"].tolist() == [3.0]
