"""Original-name restoration preserves scientific rows and rejects ambiguity."""
from __future__ import annotations

import hashlib
import sqlite3

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from spacr.original_filenames import discover_maps, enrich


def modern(plate="plate1", field=1, time=1, channel=1, z=1, source="/raw/sample.tif"):
    """Return the actual converter's required map columns and output grammar."""
    return {"target": f"{plate}_A01_T{time:04d}F{field:03d}L01A01Z{z:02d}C{channel:02d}.tif",
            "source": source, "plate": plate, "well": "A01", "field": field,
            "channel": channel, "z": z, "t": time, "status": "converted"}


def csv_map(tmp_path, rows, name="conversion_map.csv"):
    """Write a tiny metadata-only fixture; no images are needed."""
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_channels_and_z_aggregate_without_row_multiplication(tmp_path):
    path = csv_map(tmp_path, [modern(source="/raw/z.tif"),
                            modern(channel=2, source="/raw/a.tif"),
                            modern(z=2, source="/raw/z.tif")])
    frame = pd.DataFrame({"prcf": ["plate1_r1_c1_f1"] * 2 + ["plate1_r1_c1_f9"],
                          "area": [1.25, 2.5, 9.0]}, index=[7, 7, 2])
    before = frame.copy(deep=True)
    mapped, report = enrich(frame, path)
    assert_frame_equal(frame, before)
    assert_frame_equal(mapped[frame.columns], before)
    assert mapped.original_filename.tolist()[:2] == ["a.tif; z.tif"] * 2
    assert mapped.original_path.iloc[0] == "/raw/a.tif; /raw/z.tif"
    assert pd.isna(mapped.original_filename.iloc[2])
    assert report["matched_rows"] == 2 and report["unmatched_rows"] == 1
    assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("column", ["filename", "file_name", "cell_filename", "pathogen_file_name"])
def test_actual_stack_names_and_prefixed_columns(tmp_path, column):
    path = csv_map(tmp_path, [modern(field=3, time=2)])
    mapped, _ = enrich(pd.DataFrame({column: [r"C:\\data\\plate1_A01_3_2.tif"]}), path)
    assert mapped.original_filename.tolist() == ["sample.tif"]


def test_canonical_composite_ids_and_multiindex_are_preserved(tmp_path):
    path = csv_map(tmp_path, [modern()])
    frame = pd.DataFrame({"cell_plateID": ["plate1"], "cell_rowID": [1],
                          "cell_columnID": ["c1"], "cell_fieldID": ["F001"],
                          "cell_timeID": ["T0001"]}, index=pd.MultiIndex.from_tuples([("x", 9)]))
    mapped, _ = enrich(frame, path, output_column="acquisition_name")
    assert_frame_equal(mapped[frame.columns], frame)
    assert mapped.acquisition_name.iloc[0] == "sample.tif"


def test_timepoints_cannot_be_silently_combined(tmp_path):
    path = csv_map(tmp_path, [modern(), modern(time=2, source="/raw/second.tif")])
    with pytest.raises(ValueError, match="Ambiguous"):
        enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1"]}), path)
    mapped, _ = enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1"],
                                     "cell_filename": ["plate1_A01_1_2.npy"]}), path)
    assert mapped.original_filename.iloc[0] == "second.tif"


def test_conflicting_time_identity_is_rejected_even_if_only_filename_matches(tmp_path):
    path = csv_map(tmp_path, [modern()])
    with pytest.raises(ValueError, match="Conflicting"):
        enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1_t2"],
                             "filename": ["plate1_A01_1_1.npy"]}), path)


def test_missing_time_cannot_borrow_an_existing_time(tmp_path):
    path = csv_map(tmp_path, [modern()])
    with pytest.raises(ValueError, match="Conflicting"):
        enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1"],
                             "filename": ["plate1_A01_1_2.npy"]}), path)


def test_duplicate_basename_across_plates_needs_explicit_plate(tmp_path):
    one, two = modern(), modern(plate="plate2", source="/raw/other.tif")
    one["target"] = two["target"] = "shared.tif"
    path = csv_map(tmp_path, [one, two])
    with pytest.raises(ValueError, match="Ambiguous"):
        enrich(pd.DataFrame({"filename": ["shared.tif"]}), path)
    mapped, _ = enrich(pd.DataFrame({"filename": ["shared.tif"], "prcf": ["plate2_r1_c1_f1"]}), path)
    assert mapped.original_filename.iloc[0] == "other.tif"


@pytest.mark.parametrize("source_column,value,expected", [
    ("Original File", "acquisition.nd2", "acquisition.nd2"),
    ("Original File(s)", "z2.tif;z1.tif;z2.tif", "z1.tif; z2.tif"),
])
def test_legacy_rename_logs(tmp_path, source_column, value, expected):
    path = csv_map(tmp_path, [{source_column: value,
                             "Renamed TIFF": "plate1_A01_T0001F001L01C01.tif"}], "rename_log.csv")
    mapped, report = enrich(pd.DataFrame({"file_name": ["plate1_A01_1_1.npy"]}), path)
    assert mapped.original_filename.iloc[0] == expected
    assert report["format"] == "rename_log"


def test_channel_manifest_ignores_masks_and_failed_images(tmp_path):
    base = {"kind": "image", "plate": "plate1", "well": "A01", "field": 1,
            "time": 1, "new_path": "/sorted/C01/plate1_A01_T0001F001L01C01.tif",
            "original_path": "/raw/image.tif", "status": "moved"}
    path = csv_map(tmp_path, [base, dict(base, kind="mask", original_path="/raw/mask.tif"),
                             dict(base, status="error", original_path="/raw/failed.tif")],
                   "channel_sorting_manifest.csv")
    mapped, _ = enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1"]}), path)
    assert mapped.original_filename.iloc[0] == "image.tif"


def test_failed_conversion_is_not_restored(tmp_path):
    path = csv_map(tmp_path, [modern(), dict(modern(field=2), status="failed")])
    mapped, report = enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1", "plate1_r1_c1_f2"]}), path)
    assert pd.isna(mapped.original_filename.iloc[1])
    assert report["matched_rows"] == report["unmatched_rows"] == 1


def test_changed_map_and_output_collisions_are_rejected(tmp_path):
    path = csv_map(tmp_path, [modern()])
    frame = pd.DataFrame({"prcf": ["plate1_r1_c1_f1"]})
    _, report = enrich(frame, path)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        enrich(frame, path, expected_sha256=report["sha256"])
    for column in ("original_filename", "original_path"):
        with pytest.raises(ValueError, match="already exist"):
            enrich(frame.assign(**{column: "keep"}), path)


def test_zero_matches_and_unknown_maps_fail_clearly(tmp_path):
    path = csv_map(tmp_path, [modern()])
    with pytest.raises(ValueError, match="No measurement rows matched"):
        enrich(pd.DataFrame({"prcf": ["other_r1_c1_f1"]}), path)
    path = csv_map(tmp_path, [{"filename": "x", "oldname": "y"}])
    with pytest.raises(ValueError, match="Not a supported"):
        enrich(pd.DataFrame({"filename": ["x"]}), path)


def test_discovery_is_bounded_and_includes_readonly_database(tmp_path):
    plate = tmp_path / "plate"
    measurements = plate / "measurements"
    measurements.mkdir(parents=True)
    path = csv_map(plate, [modern()])
    db = measurements / "measurements.db"
    with sqlite3.connect(db) as connection:
        pd.DataFrame([modern()]).to_sql("conversion_map", connection, index=False)
    assert discover_maps(db) == [path, db]
    before = db.read_bytes()
    mapped, report = enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1"]}), db)
    assert mapped.original_filename.iloc[0] == "sample.tif"
    assert report["map_table"] == "conversion_map"
    assert db.read_bytes() == before
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    _, repeated = enrich(pd.DataFrame({"prcf": ["plate1_r1_c1_f1"]}), db,
                          expected_sha256=report["sha256"])
    assert report["sha256"] == repeated["sha256"]


def test_legacy_time_alias_and_nullable_numeric_identities(tmp_path):
    path = csv_map(tmp_path, [modern(), modern(time=2, source="time2.nd2")])
    frame = pd.DataFrame({"plateID": ["plate1"], "rowID": [1.0],
                          "columnID": [1.0], "fieldID": [1.0], "time_id": [2.0]})
    mapped, _ = enrich(frame, path)
    assert mapped.original_filename.iloc[0] == "time2.nd2"
    with pytest.raises(ValueError, match="Conflicting timeID and time_id"):
        enrich(frame.assign(timeID="t1"), path)


def test_database_digest_is_stable_across_row_reordering(tmp_path):
    db = tmp_path / "measurements.db"
    rows = pd.DataFrame([modern(), modern(channel=2, source="second.nd2")])
    with sqlite3.connect(db) as connection:
        rows.to_sql("conversion_map", connection, index=False)
    frame = pd.DataFrame({"prcf": ["plate1_r1_c1_f1"]})
    before, report = enrich(frame, db)
    with sqlite3.connect(db) as connection:
        rows.iloc[::-1].to_sql("conversion_map", connection, index=False, if_exists="replace")
    after, _ = enrich(frame, db, expected_sha256=report["sha256"])
    assert_frame_equal(before, after)


def test_mismatched_plate_metadata_cannot_override_converted_filename(tmp_path):
    path = csv_map(tmp_path, [modern()])
    with pytest.raises(ValueError, match="Conflicting"):
        enrich(pd.DataFrame({"prcf": ["plate2_r1_c1_f1"],
                             "filename": [modern()["target"]]}), path)


def test_untimed_prcf_uses_separate_time_alias_without_composite_columns(tmp_path):
    path = csv_map(tmp_path, [modern(), modern(time=2, source="time2.nd2")])
    frame = pd.DataFrame({"cell_prcf": ["plate1_r1_c1_f1"], "time_id": [2]})
    mapped, _ = enrich(frame, path)
    assert mapped.original_filename.iloc[0] == "time2.nd2"
