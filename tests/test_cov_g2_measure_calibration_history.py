"""Retained measurements and the intensity calibration that produced them.

Each case is one way an existing ``measurements.db`` cannot prove which
calibration its rows were measured with; every one is refused without
touching the database.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from spacr import measure as M

KEY = M._CALIBRATION_IDENTITY_KEY
FIELD = ("plate1", "r1", "c1", "f1")


def _db(tmp_path, *, recorded=None, rescale=None, rescale_columns=None,
        cell_columns=("plateID", "rowID", "columnID", "fieldID", "area")):
    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as con:
        con.execute(f'CREATE TABLE cell ({", ".join(cell_columns)})')
        con.execute(f'INSERT INTO cell VALUES ({", ".join("?" * len(cell_columns))})',
                    [*FIELD, *([1.0] * (len(cell_columns) - 4))][:len(cell_columns)])
        con.execute("CREATE TABLE settings (setting_key TEXT, setting_value TEXT)")
        for key, value in (recorded or {}).items():
            con.execute("INSERT INTO settings VALUES (?, ?)", (key, value))
        if rescale_columns is not None:
            con.execute(f'CREATE TABLE intensity_rescale ({", ".join(rescale_columns)})')
            for row in rescale or []:
                con.execute(f'INSERT INTO intensity_rescale VALUES '
                            f'({", ".join("?" * len(row))})', row)
    return str(path)


def _check(path, identity):
    M._validate_measurement_calibration_history({KEY: identity}, path)


def test_legacy_calibrated_rows_cannot_prove_their_references(tmp_path):
    path = _db(tmp_path, recorded={"intensity_calibration": "True"})
    with pytest.raises(ValueError, match="unverified intensity"):
        _check(path, None)


def test_calibration_without_a_provenance_column_is_refused(tmp_path):
    path = _db(tmp_path, rescale_columns=("plateID", "gain"))
    with pytest.raises(ValueError):
        _check(path, "abc")
    _check(path, None)


def test_a_field_without_calibration_is_refused_when_one_is_asked_for(tmp_path):
    columns = ("plateID", "rowID", "columnID", "fieldID", "intensity_calibration")
    path = _db(tmp_path, rescale_columns=columns, rescale=[(*FIELD, None)])
    with pytest.raises(ValueError):
        _check(path, "abc")
    _check(path, None)


def test_unreadable_provenance_is_refused(tmp_path):
    columns = ("plateID", "rowID", "columnID", "fieldID", "intensity_calibration")
    path = _db(tmp_path, rescale_columns=columns, rescale=[(*FIELD, "{not json")])
    with pytest.raises(ValueError):
        _check(path, "abc")


def _matching(identity="abc"):
    return json.dumps({"identity": identity})


def test_provenance_without_field_keys_is_refused(tmp_path):
    path = _db(tmp_path, rescale_columns=("intensity_calibration",),
               rescale=[(_matching(),)], recorded={KEY: "abc"})
    with pytest.raises(ValueError):
        _check(path, "abc")


def test_measured_rows_without_field_keys_are_refused(tmp_path):
    columns = ("plateID", "rowID", "columnID", "fieldID", "intensity_calibration")
    path = _db(tmp_path, rescale_columns=columns, rescale=[(*FIELD, _matching())],
               recorded={KEY: "abc"}, cell_columns=("plateID", "area"))
    with pytest.raises(ValueError):
        _check(path, "abc")


def test_timed_rows_need_timed_provenance(tmp_path):
    columns = ("plateID", "rowID", "columnID", "fieldID", "intensity_calibration")
    path = _db(tmp_path, rescale_columns=columns, rescale=[(*FIELD, _matching())],
               recorded={KEY: "abc"},
               cell_columns=("plateID", "rowID", "columnID", "fieldID", "timeID"))
    with pytest.raises(ValueError):
        _check(path, "abc")


def test_timed_rows_with_matching_timed_provenance_are_accepted(tmp_path):
    columns = ("plateID", "rowID", "columnID", "fieldID", "timeID",
               "intensity_calibration")
    path = _db(tmp_path, rescale_columns=columns,
               rescale=[(*FIELD, 1.0, _matching())], recorded={KEY: "abc"},
               cell_columns=("plateID", "rowID", "columnID", "fieldID", "timeID"))
    before = open(path, "rb").read()
    _check(path, "abc")
    assert open(path, "rb").read() == before
