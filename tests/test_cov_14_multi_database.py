"""Naming a source and typing a column both stop rather than guess.

The source label for a merged embedding is the nearest ancestor folder whose
name is not a generic container. Two things can end that climb: running out of
path, and a parent that is its own parent (a drive or share root). Both must
end it, because the loop has no other exit and a source label is computed once
per database on the way into a merge.

Column typing follows SQLite's own affinity rules, whose last rule is that
anything declared but unrecognised is NUMERIC. Getting that wrong silently
changes how the column is aggregated in the merge.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from spacr import multi_database


def test_a_path_with_no_named_ancestor_has_no_source_label():
    """A file at the filesystem root has no folder to be named after."""
    assert multi_database._meaningful_parent("/measurements.db") == ""


def test_a_path_of_only_generic_folders_has_no_source_label():
    """Climbing off the top of a relative path ends with no label."""
    assert multi_database._meaningful_parent("data/measurements.db") == ""


def test_the_climb_stops_when_the_parent_stops_shrinking(monkeypatch):
    """A parent that is its own parent ends the climb instead of looping.

    A root whose directory name is itself -- what a drive or network share
    root is -- would otherwise make the ``while`` loop spin for ever with the
    GUI thread waiting on it.
    """
    real_dirname = os.path.dirname

    def _stuck_at_the_root(path):
        return "data" if path == "data" else real_dirname(path)

    monkeypatch.setattr(os.path, "dirname", _stuck_at_the_root)

    assert multi_database._meaningful_parent("data/measurements.db") == ""


def test_a_named_plate_folder_is_the_source_label():
    """The plate folder above ``measurements/`` is what names the source."""
    assert multi_database._meaningful_parent(
        "/screen/plate1/measurements/measurements.db") == "plate1"


def _database(tmp_path, declarations):
    path = tmp_path / "measurements.db"
    columns = ", ".join(f'"{name}" {kind}' for name, kind in declarations)
    with sqlite3.connect(path) as db:
        db.execute(f'CREATE TABLE object ({columns})')
    return str(path)


def test_an_unrecognised_declared_type_is_numeric(tmp_path):
    """SQLite's fifth affinity rule: anything else declared is NUMERIC.

    The merge picks its aggregation from the column's dtype, so a DECIMAL
    column typed as unknown would be reported to the user as combining one way
    and then combine another.
    """
    path = _database(tmp_path, [("well_score", "DECIMAL(10,5)"),
                                ("taken_at", "DATETIME")])

    kinds = multi_database.column_kinds(path, "object")

    assert kinds["well_score"] == "numeric"
    assert kinds["taken_at"] == "numeric"


def test_a_column_declared_with_no_type_stays_unknown(tmp_path):
    """Nothing declared is an absent answer, not a guessed one."""
    path = _database(tmp_path, [("anything", "")])

    assert multi_database.column_kinds(path, "object")["anything"] == "unknown"


def test_the_named_affinities_still_win(tmp_path):
    """The substring rules are checked before the catch-all."""
    path = _database(tmp_path, [("plateID", "VARCHAR(32)"),
                                ("area", "FLOAT"),
                                ("mask", "BLOB")])

    kinds = multi_database.column_kinds(path, "object")

    assert kinds["plateID"] == "text"
    assert kinds["area"] == "numeric"
    assert kinds["mask"] == "unknown"
