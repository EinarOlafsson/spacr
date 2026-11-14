"""Table-driven tests of identity keys and object-table validation in ``spacr.schema``.

Four units: :func:`row_id` (row letters, indexes and prefixed ids to
``r<N>``), :func:`comparable_key_value` (one metadata cell reduced to its
comparison string), :func:`escape_field_stem_plate` (escaping only the plate
of a merged-stack stem) and :func:`validate_object_table_frame` (the writer
boundary that refuses frames that would corrupt a database).
"""
from __future__ import annotations

import itertools
import string

import numpy as np
import pandas as pd
import pytest

from spacr import metadata_resolution, schema


def _cell_frame(**extra) -> pd.DataFrame:
    """A minimal valid two-row ``cell`` object table, with overrides."""
    data = {
        "object_label": [1, 2],
        "plateID": ["plate1", "plate1"],
        "rowID": ["r1", "r1"],
        "columnID": ["c1", "c1"],
        "fieldID": ["f1", "f1"],
        "prcf": ["plate1_r1_c1_f1", "plate1_r1_c1_f1"],
        "file_name": ["plate1_A01_1", "plate1_A01_1"],
        "path_name": ["/data/plate1_A01_1.npy"] * 2,
        "cell_area": [10.0, 20.0],
    }
    data.update(extra)
    return pd.DataFrame(data)


def test_row_id_canonicalises_letters_indexes_and_prefixed_ids(subtests):
    """Every one- and two-letter row round-trips; other tokens take the general path."""
    letters = list(string.ascii_uppercase) + [
        a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    with subtests.test(msg="all 702 letter rows decode and round-trip"):
        assert len(letters) == 702
        for number, label in enumerate(letters, start=1):
            assert schema.row_index_from_letters(label) == number
            assert schema.row_id(label) == f"r{number}"
            assert schema.row_id(label.lower()) == f"r{number}"
            assert schema.letters_from_row_index(number) == label

    cases = [
        ("integer index", 1, "r1"),
        ("numeric string", "1", "r1"),
        ("zero-padded string", "003", "r3"),
        ("prefixed id round-trips", "r3", "r3"),
        ("prefixed id with two digits", "r12", "r12"),
        ("letter plus digit is a vendor prefix", "A1", "r1"),
        ("letters are stripped", " b ", "r2"),
        ("last two-letter row", "ZZ", "r702"),
        ("AB is row 28", "AB", "r28"),
        ("three letters are preserved, not decoded", "AAA", "rAAA"),
    ]
    for case_id, token, expected in cases:
        with subtests.test(msg=case_id):
            assert schema.row_id(token) == expected

    with subtests.test(msg="a token holding a digit is not row letters"):
        assert schema.row_index_from_letters("A1") is None
    with subtests.test(msg="an empty row is refused"):
        with pytest.raises(schema.KeyParseError,
                           match="cannot build a rowID from '': it is empty"):
            schema.row_id("")
    with subtests.test(msg="strict refuses a token with no integer"):
        with pytest.raises(schema.KeyParseError,
                           match="cannot build a rowID from 'AAA': it holds "
                                 "no integer"):
            schema.row_id("AAA", strict=True)


def test_comparable_key_value_reduces_spellings_of_one_value(subtests):
    """Missing values compare as blank, integral numbers lose their fraction, text is kept."""
    cases = [
        ("None", None, ""),
        ("python NaN", float("nan"), ""),
        ("numpy float32 NaN", np.float32("nan"), ""),
        ("the text NaN", "NaN", ""),
        ("empty text", "", ""),
        ("whitespace only", "   ", ""),
        ("integer", 1, "1"),
        ("integral float", 2.0, "2"),
        ("zero-padded padded text", " 01 ", "1"),
        ("fraction is kept", 1.5, "1.5"),
        ("fractional text", "2.50", "2.5"),
        ("negative", -3.0, "-3"),
        ("non-numeric text is kept stripped", " abc ", "abc"),
    ]
    for case_id, value, expected in cases:
        with subtests.test(msg=case_id):
            assert schema.comparable_key_value(value) == expected


def test_escape_field_stem_plate_escapes_only_the_plate(subtests):
    """The plate's separators are escaped; tails stay; short stems are refused."""
    cases = [
        ("plain stem is unchanged", "plate1_A01_1", False, "plate1_A01_1"),
        ("extension and folder are dropped", "/data/plate1_A01_1.npy", False,
         "plate1_A01_1"),
        ("plate separator is escaped", "my_plate_A01_1_1", True,
         "my%5Fplate_A01_1_1"),
        ("a vendor-prefixed fourth token is also read as time",
         "my_plate_A01_s1", False, "my_plate_A01_s1"),
        ("a numeric fourth token is read as time, not plate",
         "my_plate_A01_1", False, "my_plate_A01_1"),
        ("numeric trailing token is read as time", "p_1_A01_1_2", False,
         "p%5F1_A01_1_2"),
        ("timelapse stem", "exp_1_A01_1_1", True, "exp%5F1_A01_1_1"),
        ("multi-separator plate", "a_b_c_A01_1_1", True,
         "a%5Fb%5Fc_A01_1_1"),
    ]
    for case_id, stem, timelapse, expected in cases:
        with subtests.test(msg=case_id):
            assert schema.escape_field_stem_plate(
                stem, timelapse=timelapse) == expected

    refused = [
        ("no plate component", "r1_c1.npy", False,
         "cannot encode plate component in 'r1_c1': expected "
         "plate_well_field."),
        ("timelapse stem without its timepoint", "plate1_r1_c1", True,
         "cannot encode plate component in 'plate1_r1_c1': expected "
         "plate_well_field_time."),
    ]
    for case_id, stem, timelapse, message in refused:
        with subtests.test(msg=case_id):
            with pytest.raises(schema.KeyParseError) as excinfo:
                schema.escape_field_stem_plate(stem, timelapse=timelapse)
            assert str(excinfo.value) == message

    with subtests.test(msg="no stem of more than four parts escapes to itself"):
        # The migration skips stems of four parts or fewer; beyond that the
        # plate holds a raw separator, which escaping always changes.
        plates = ["exp_1", "a_b", "%5F", "a%5Fb", "_", "a__b", "p_1_2", "%",
                  "a%b", "exp%5F1", "x", "a%255Fb", "%5F%5F"]
        tried = 0
        for plate, well, field, time in itertools.product(
                plates, ["A01", "A_1", "%5F"], ["1", "2%5F"], ["1", "0"]):
            stem = f"{plate}_{well}_{field}_{time}"
            if len(stem.split("_")) <= 4:
                continue
            tried += 1
            assert schema.escape_field_stem_plate(stem, timelapse=True) != stem
        assert tried == 92


def test_validate_object_table_frame_accepts_a_valid_frame_and_refuses_corruption(
        subtests, monkeypatch):
    """A valid frame comes back equal; each corrupting shape raises with its exact cause."""
    with subtests.test(msg="a valid frame is returned unchanged"):
        pd.testing.assert_frame_equal(
            schema.validate_object_table_frame(_cell_frame(), "cell"),
            _cell_frame())

    frame = _cell_frame()
    cases = [
        ("a list is not a frame", [{"object_label": 1}], {},
         "cell must be validated from a pandas DataFrame, got list."),
        ("a dict is not a frame", {"object_label": [1]}, {},
         "cell must be validated from a pandas DataFrame, got dict."),
        ("a duplicated column label",
         pd.concat([frame, frame[["cell_area"]]], axis=1), {},
         "cell has duplicated column names: ['cell_area']."),
        ("a timepoint on a non-timelapse table",
         _cell_frame(timeID=["t1", "t2"]), {"timelapse": False},
         "cell is non-timelapse but unexpectedly carries 'timeID'."),
        ("a timelapse table without timepoints", _cell_frame(),
         {"timelapse": True},
         "cell is a timelapse table but has no 'timeID' column."),
        ("an empty well id names the offending rows",
         _cell_frame(rowID=["r1", "   "]), {},
         "cell.rowID must contain non-empty strings; invalid row indexes: "
         "[1]."),
    ]
    for case_id, value, kwargs, message in cases:
        with subtests.test(msg=case_id):
            with pytest.raises(schema.ObjectTableSchemaError) as excinfo:
                schema.validate_object_table_frame(value, "cell", **kwargs)
            assert str(excinfo.value) == message

    with subtests.test(msg="a missing required column is named"):
        with pytest.raises(schema.ObjectTableSchemaError) as excinfo:
            schema.validate_object_table_frame(
                _cell_frame().drop(columns=["path_name"]), "cell")
        assert str(excinfo.value).startswith(
            "cell is missing required canonical column(s) ['path_name']; ")
        assert isinstance(excinfo.value.__cause__,
                          metadata_resolution.MetadataResolutionRequired)

    with subtests.test(msg="an incomplete frame from the resolver is re-checked"):
        incomplete = _cell_frame().drop(columns=["prcf"])

        class _Result:
            frame = incomplete

        monkeypatch.setattr(metadata_resolution, "resolve_metadata_columns",
                            lambda *a, **k: _Result())
        with pytest.raises(schema.ObjectTableSchemaError) as excinfo:
            schema.validate_object_table_frame(incomplete, "cell")
        assert str(excinfo.value) == (
            "cell is missing required canonical column(s) ['prcf']; got "
            f"{list(incomplete.columns)}.")
