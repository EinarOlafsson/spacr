"""The defensive branches in :mod:`spacr.anndata_export`.

Instruction 60. Eighteen statements, almost all of them ``except`` and
early-return paths that decide what an export does when a database is not
the shape it hoped for.

Every one is reachable, so none is a pragma candidate, and every one is
asserted on WHAT IT RETURNS rather than merely reached. That matters more
here than usual: these branches choose between "carry on with less" and
"stop", and a test that only executes them would not notice if one started
silently dropping the annotation columns the export exists to carry.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest


def _db(path, tables):
    with sqlite3.connect(str(path)) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return str(path)


def _cells(n=3):
    return pd.DataFrame({
        "plateID": ["plate1"] * n,
        "rowID": [f"r{i + 1}" for i in range(n)],
        "columnID": ["c1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": range(1, n + 1),
        "area": [10.0] * n,
    })


# --------------------------------------------------------------------------- #
#  _attach_png_labels -- five ways png_list can be unusable
# --------------------------------------------------------------------------- #

def test_an_anchor_with_no_png_id_column_is_left_alone(tmp_path):
    """An object type png_list does not key is not an error -- there is
    simply nothing to bring back."""
    from spacr.anndata_export import _attach_png_labels

    path = _db(tmp_path / "m.db", {"cell": _cells()})
    frame = _cells()

    out, added = _attach_png_labels(frame, path, "not_an_object_type",
                                    timelapse=False)

    assert added == []
    assert out.equals(frame)


def test_a_database_with_no_png_list_table_is_left_alone(tmp_path):
    """A measurements database written with save_png=False is legitimate;
    the export carries on without the annotation columns."""
    from spacr.anndata_export import _attach_png_labels

    path = _db(tmp_path / "m.db", {"cell": _cells()})
    frame = _cells()

    out, added = _attach_png_labels(frame, path, "cell", timelapse=False)

    assert added == []
    assert len(out) == len(frame)


def test_a_png_list_missing_the_key_columns_is_left_alone(tmp_path):
    """Without the field keys there is nothing to join ON, so joining would
    be inventing a correspondence."""
    from spacr.anndata_export import _attach_png_labels

    png = pd.DataFrame({"cell_id": ["plate1_r1_c1_f1_1"], "annotation": [1]})
    path = _db(tmp_path / "m.db", {"cell": _cells(), "png_list": png})

    out, added = _attach_png_labels(_cells(), path, "cell", timelapse=False)

    assert added == []


def test_a_png_list_whose_ids_carry_no_object_label_is_left_alone(tmp_path):
    """Rows whose id yields no label are dropped; if that empties the table
    there is nothing to attach."""
    from spacr.anndata_export import _attach_png_labels

    png = pd.DataFrame({
        "cell_id": ["not-a-key", "also-not-a-key"],
        "plateID": ["plate1"] * 2,
        "rowID": ["r1", "r2"],
        "columnID": ["c1"] * 2,
        "fieldID": ["f1"] * 2,
        "annotation": [1, 0],
    })
    path = _db(tmp_path / "m.db", {"cell": _cells(), "png_list": png})

    out, added = _attach_png_labels(_cells(), path, "cell", timelapse=False)

    assert added == []
    assert len(out) == 3


# --------------------------------------------------------------------------- #
#  _source_table -- attributing a column to the table it came from
# --------------------------------------------------------------------------- #

class _Entry:
    """What parse_column hands back: the two object types it recognised."""

    def __init__(self, object_type=None, object_type_2=None):
        self.object_type = object_type
        self.object_type_2 = object_type_2


def test_a_join_suffix_attributes_the_column():
    """``area_nucleus`` came from the nucleus table. The suffix is the only
    evidence left once parse_column recognised no object type."""
    from spacr.anndata_export import _source_table

    assert _source_table("area_nucleus", _Entry(),
                         ("cell", "nucleus")) == "nucleus"


def test_a_recognised_object_type_wins_over_the_suffix():
    """The other side of the same branch, so the suffix path is not passing
    because nothing else was tried."""
    from spacr.anndata_export import _source_table

    assert _source_table("area_nucleus", _Entry(object_type="cell"),
                         ("cell", "nucleus")) == "cell"


def test_a_count_column_names_what_it_counts():
    from spacr.anndata_export import _source_table

    assert _source_table("count_nucleus", _Entry(),
                         ("cell", "nucleus")) == "nucleus"


def test_an_unattributable_column_falls_back_to_the_anchor():
    """The anchor is a defensible guess; an empty string when there ARE
    tables would lose the attribution var is for."""
    from spacr.anndata_export import _source_table

    assert _source_table("area", _Entry(), ("cell", "nucleus")) == "cell"
    assert _source_table("area", _Entry(), ()) == ""


# --------------------------------------------------------------------------- #
#  _run_id_from_db -- provenance that must never stop an export
# --------------------------------------------------------------------------- #

def test_a_database_with_no_settings_history_yields_no_run_id(tmp_path):
    """Provenance is best-effort. A database with no history is exportable;
    it simply cannot say which run produced it."""
    from spacr.anndata_export import _run_id_from_db

    path = _db(tmp_path / "m.db", {"cell": _cells()})
    assert _run_id_from_db(path) == ""


def test_an_unreadable_database_yields_no_run_id_rather_than_raising(tmp_path):
    """The whole point of the except: a corrupt or absent file must not take
    the export down with it."""
    from spacr.anndata_export import _run_id_from_db

    missing = str(tmp_path / "nope.db")
    assert _run_id_from_db(missing) == ""

    broken = tmp_path / "broken.db"
    broken.write_bytes(b"this is not a database")
    assert _run_id_from_db(str(broken)) == ""


# --------------------------------------------------------------------------- #
#  _read_frame -- the join that returned nothing
# --------------------------------------------------------------------------- #

def test_a_join_that_returns_nothing_names_the_database_and_the_fix(tmp_path,
                                                                   monkeypatch):
    """The message has to carry the db path and the doctor command, because
    the caller is a user whose export just stopped and needs the next step.

    The join is forced to return None rather than contrived from data: what
    is under test is the REPORT, and making a real join fail this way would
    be testing spacr.io instead.
    """
    from spacr import io as spacr_io
    from spacr.anndata_export import _read_frame

    path = _db(tmp_path / "m.db", {"cell": _cells()})
    # Patched on spacr.io, not on anndata_export: _read_frame imports it
    # INSIDE the function, so the module attribute is the only one that binds.
    monkeypatch.setattr(spacr_io, "_read_and_join_tables",
                        lambda *a, **k: None)

    with pytest.raises(ValueError) as caught:
        _read_frame(path, ("cell",), None)

    message = str(caught.value)
    assert "m.db" in message
    assert "doctor" in message
