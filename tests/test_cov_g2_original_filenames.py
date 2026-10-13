"""Original-filename restoration: the identities it refuses and the maps it reads."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr import original_filenames as of
from tests.test_original_filenames import csv_map, modern


def test_a_database_that_is_not_sqlite_is_not_offered_as_a_map(tmp_path):
    fake = tmp_path / "measurements.db"
    fake.write_bytes(b"not a database at all" * 10)
    assert of.discover_maps(fake) == []


def test_a_row_with_two_times_or_a_conflicting_time_is_refused():
    key = ("plate1", "r1", "c1", "f1", None)
    with pytest.raises(ValueError, match="Conflicting timeID and time_id"):
        of._with_row_time(key, {"timeID": "1", "time_id": "2"}, "")
    with pytest.raises(ValueError, match="Conflicting time identity"):
        of._with_row_time(("plate1", "r1", "c1", "f1", "t1"),
                          {"timeID": "2"}, "")


def test_unparseable_identities_are_skipped_not_fatal():
    assert of._metadata_keys({"prcf": "not-a-prcf", "prcfo": "nor-this",
                              "plateID": "plate1", "rowID": "x",
                              "columnID": "y", "fieldID": "z"}) == []


def test_a_prcfo_identity_is_read():
    keys = of._metadata_keys({"prcfo": "plate1_r1_c1_f1_o3"})
    assert keys and keys[0][0] == "plate1"


def test_an_unreadable_or_unknown_map_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Not a supported"):
        of._load_map(b"a,b\n1,2\n", tmp_path / "x.csv")
    with pytest.raises(ValueError, match="Cannot read filename mapping"):
        of._load_map(b"\xff\xfe\x00" * 3, tmp_path / "x.csv")


def test_a_map_with_only_failed_rows_restores_nothing(tmp_path):
    rows = [dict(modern(), status="failed"), dict(modern(field=2), source="")]
    path = csv_map(tmp_path, rows)
    with pytest.raises(ValueError, match="no successful image conversions"):
        of.enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1"]}), path)


def test_a_row_with_a_bad_declared_field_is_refused(tmp_path):
    path = csv_map(tmp_path, [dict(modern(), well="??")])
    with pytest.raises(ValueError, match="Invalid field identity"):
        of.enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1"]}), path)


def test_a_target_that_names_another_field_is_refused(tmp_path):
    path = csv_map(tmp_path, [dict(modern(), field=2)])
    with pytest.raises(ValueError, match="Conflicting target"):
        of.enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1"]}), path)


def test_a_rename_log_restores_several_originals(tmp_path):
    path = csv_map(tmp_path, [{"Renamed TIFF": "plate1_A01_T0001F001L01A01Z01C01.tif",
                               "Original File(s)": "/raw/a.tif; /raw/b.tif"}],
                   name="rename_log.csv")
    mapped, report = of.enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1"]}), path)
    assert mapped.original_filename.iloc[0] == "a.tif; b.tif"
    assert report["matched_rows"] == 1


def test_a_channel_sorting_manifest_is_read(tmp_path):
    path = csv_map(tmp_path, [
        {"kind": "image", "original_path": "/raw/x.tif",
         "new_path": "plate1_A01_T0001F001L01A01Z01C01.tif", "status": "moved",
         "plate": "plate1", "well": "A01", "field": "1", "time": "1"},
        {"kind": "folder", "original_path": "/raw", "new_path": "plate1",
         "status": "moved", "plate": "plate1", "well": "A01", "field": "1",
         "time": "1"}], name="channel_sorting_manifest.csv")
    mapped, _ = of.enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1"]}), path)
    assert mapped.original_filename.iloc[0] == "x.tif"


@pytest.mark.parametrize("frame, column, message", [
    (pd.DataFrame([[1, 2]], columns=["prcf", "prcf"]), "original_filename",
     "unique columns"),
    (pd.DataFrame({"prcf": ["a"]}), "original_path", "distinct from original_path"),
    (pd.DataFrame({"prcf": ["a"]}), " ", "nonempty"),
])
def test_unusable_frames_and_output_names_are_refused(tmp_path, frame, column,
                                                      message):
    path = csv_map(tmp_path, [modern()])
    with pytest.raises(ValueError, match=message):
        of.enrich(frame, path, output_column=column)


def test_a_database_without_a_conversion_map_is_refused(tmp_path):
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE cell (a INTEGER)")
    with pytest.raises(ValueError, match="Cannot read a conversion_map"):
        of.enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1"]}), db)


def test_an_empty_filename_cell_is_ignored(tmp_path):
    path = csv_map(tmp_path, [modern()])
    mapped, _ = of.enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1"],
                                        "filename": [""]}), path)
    assert mapped.original_filename.iloc[0] == "sample.tif"


def test_a_filename_and_prcf_that_disagree_are_refused(tmp_path):
    path = csv_map(tmp_path, [modern(), modern(field=2, source="/raw/two.tif")])
    frame = pd.DataFrame({"prcf": ["plate1_r1_c1_f1"],
                          "filename": ["plate1_A01_T0001F002L01A01Z01C01.tif"]})
    with pytest.raises(ValueError, match="Conflicting filename"):
        of.enrich(frame, path)
