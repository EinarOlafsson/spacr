"""The image-level annotation table in measurements.db: schema, round trip and undo."""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from spacr.measure import (
    _IMAGE_ANNOTATION_TABLE,
    _IMAGE_ANNOTATION_VERSION,
    _add_image_annotation_column,
    _image_annotation_columns,
    _image_annotation_database,
    _image_annotation_images,
    _image_measurement_frame,
    _import_overview_annotations_json,
    _read_image_annotations,
    _set_image_annotations,
    _undo_image_annotations,
    _well_image_annotations,
)


def _fields(tmp_path):
    fields = []
    for well, row, column in (("A01", "A", 1), ("A02", "A", 2), ("B01", "B", 1)):
        for field in ("1", "2"):
            files = tuple((str(channel), str(tmp_path / f"{well}_{field}_c{channel}.tif"))
                          for channel in (1, 2))
            fields.append(dict(plate="plate1", well=well, row=row, column=column,
                               field=field, files=files))
    return fields


def _table(db):
    with sqlite3.connect(db) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(f"SELECT * FROM {_IMAGE_ANNOTATION_TABLE}")]


def test_database_path_is_the_measure_database(tmp_path):
    assert _image_annotation_database(tmp_path) == tmp_path / "measurements" / "measurements.db"
    db = tmp_path / "x" / "measurements.db"
    assert _image_annotation_database(db) == db


def test_one_row_per_image_keyed_by_plate_well_field_file(tmp_path):
    db = _image_annotation_database(tmp_path)
    fields = _fields(tmp_path)
    result = _set_image_annotations(db, fields, "cell_type", "HeLa", source=tmp_path,
                                    wells=[("plate1", "A01")])
    assert result["count"] == 4 and result["operation"] == 1
    rows = _table(db)
    assert len(rows) == 4
    assert {(row["wellID"], row["fieldID"], row["file"]) for row in rows} == {
        ("A01", "f1", "A01_1_c1.tif"), ("A01", "f1", "A01_1_c2.tif"),
        ("A01", "f2", "A01_2_c1.tif"), ("A01", "f2", "A01_2_c2.tif")}
    first = next(row for row in rows if row["file"] == "A01_1_c1.tif")
    assert first["cell_type"] == "HeLa"
    assert first["prcf"] == "plate1_r1_c1_f1"
    assert first["rowID"] == "r1" and first["columnID"] == "c1" and first["fieldID"] == "f1"
    assert first["plateID"] == "plate1" and first["field_label"] == "1" and first["chanID"] == "1"
    with sqlite3.connect(db) as connection:
        info = connection.execute(f"PRAGMA index_list({_IMAGE_ANNOTATION_TABLE})").fetchall()
        assert any(entry[2] for entry in info)
        version = connection.execute("SELECT value FROM image_annotations_meta "
                                     "WHERE key='schema_version'").fetchone()[0]
    assert int(version) == _IMAGE_ANNOTATION_VERSION


def test_user_columns_listed_added_and_validated(tmp_path):
    db = tmp_path / "measurements" / "measurements.db"
    assert _image_annotation_columns(db) == []
    assert _add_image_annotation_column(db, "replicate") == ["replicate"]
    _set_image_annotations(db, _fields(tmp_path), "pathogen_type", "Toxoplasma WT")
    assert _image_annotation_columns(db) == ["replicate", "pathogen_type"]
    assert _add_image_annotation_column(db, "Replicate") == ["replicate", "pathogen_type"]
    for bad in ("well", "File", "channel", "timepoint", "plateID", "1abc", "drop table", "",
                "sqlite_x", "a" * 65):
        with pytest.raises(ValueError):
            _set_image_annotations(db, _fields(tmp_path), bad, "x")


def test_round_trip_read_and_well_summary(tmp_path):
    db = tmp_path / "measurements.db"
    fields = _fields(tmp_path)
    _set_image_annotations(db, fields, "cell_type", "HeLa", wells=[("plate1", "A01"), ("plate1", "A02")])
    _set_image_annotations(db, fields, "cell_type", "HFF", wells=[("plate1", "B01")])
    _set_image_annotations(db, fields, "replicate", "1")
    image = [field for field in fields if field["well"] == "A02" and field["field"] == "2"]
    _set_image_annotations(db, image, "replicate", "2")
    frame = _read_image_annotations(db)
    assert len(frame) == 12
    assert frame.groupby("cell_type").size().to_dict() == {"HFF": 4, "HeLa": 8}
    summary = _well_image_annotations(db)
    assert summary[("plate1", "A01")] == {"cell_type": "HeLa", "replicate": "1"}
    assert summary[("plate1", "B01")]["cell_type"] == "HFF"
    assert summary[("plate1", "A02")] == {"cell_type": "HeLa", "_mixed": ["replicate"]}
    narrowed = _read_image_annotations(db, ["replicate"])
    assert "cell_type" not in narrowed.columns and "replicate" in narrowed.columns


def test_undo_restores_values_removes_new_rows_and_stops_when_empty(tmp_path):
    db = tmp_path / "measurements.db"
    fields = _fields(tmp_path)
    _set_image_annotations(db, fields, "cell_type", "HeLa", wells=[("plate1", "A01")])
    second = _set_image_annotations(db, fields, "cell_type", "HFF")
    assert second["count"] == 12
    assert _set_image_annotations(db, fields, "cell_type", "HFF")["operation"] is None
    undone = _undo_image_annotations(db)
    assert undone == dict(operation=second["operation"], column="cell_type", count=12)
    rows = _table(db)
    assert len(rows) == 4 and {row["cell_type"] for row in rows} == {"HeLa"}
    cleared = _set_image_annotations(db, fields, "cell_type", "", wells=[("plate1", "A01")])
    assert cleared["count"] == 4
    assert {row["cell_type"] for row in _table(db)} == {None}
    _undo_image_annotations(db)
    assert {row["cell_type"] for row in _table(db)} == {"HeLa"}
    assert _undo_image_annotations(db)["count"] == 4
    assert _table(db) == []
    assert _undo_image_annotations(db) is None
    assert _undo_image_annotations(tmp_path / "missing.db") is None


def test_other_tables_are_preserved(tmp_path):
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE cell (area INTEGER)")
        connection.execute("INSERT INTO cell VALUES (7)")
    _set_image_annotations(db, _fields(tmp_path), "drug", "none")
    _undo_image_annotations(db)
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT area FROM cell").fetchall() == [(7,)]


def test_unversioned_table_with_legacy_names_is_migrated_without_losing_values(tmp_path):
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE image_annotations (plate TEXT, well TEXT, field TEXT, "
                           "file TEXT, channel TEXT, cell_type TEXT)")
        connection.execute("INSERT INTO image_annotations VALUES ('plate1','A01','f1','a.tif','1','HeLa')")
    _set_image_annotations(db, [dict(plate="plate1", well="A01", field="1", file="b.tif")],
                           "cell_type", "HFF")
    rows = {row["file"]: row for row in _table(db)}
    assert set(rows["a.tif"]) >= {"plateID", "wellID", "fieldID", "chanID", "prcf",
                                   "annotation_updated_at", "cell_type"}
    assert not {"plate", "well", "field", "channel"} & set(rows["a.tif"])
    assert rows["a.tif"]["cell_type"] == "HeLa" and rows["a.tif"]["chanID"] == "1"
    assert rows["b.tif"]["cell_type"] == "HFF" and rows["b.tif"]["fieldID"] == "f1"
    assert _image_annotation_columns(db) == ["cell_type"]
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT value FROM image_annotations_meta").fetchone()[0] == "1"


def test_a_legacy_row_column_reaches_the_rowid_alias_without_losing_values(tmp_path):
    """SQLite 3.54 refuses to rename or add a column called ``rowID``.

    The table is rebuilt instead, so ``row`` still arrives as ``rowID``.
    """
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE image_annotations (plate TEXT, well TEXT, field TEXT, "
                           "file TEXT, row TEXT, col TEXT, cell_type TEXT)")
        connection.execute("INSERT INTO image_annotations VALUES "
                           "('plate1','A01','f1','a.tif','A','1','HeLa')")
    _set_image_annotations(db, [dict(plate="plate1", well="A01", field="1", file="b.tif")],
                           "cell_type", "HFF")
    rows = {row["file"]: row for row in _table(db)}
    assert rows["a.tif"]["rowID"] == "A" and rows["a.tif"]["cell_type"] == "HeLa"
    assert "row" not in rows["a.tif"]
    assert rows["b.tif"]["cell_type"] == "HFF"
    assert _image_annotation_columns(db) == ["cell_type"]


def test_reads_never_migrate_the_database(tmp_path):
    db = tmp_path / "measurements.db"
    _set_image_annotations(db, _fields(tmp_path), "drug", "none")
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE fov (plate TEXT, well TEXT, field TEXT, fieldID TEXT, "
                           "plateID TEXT, status TEXT)")
        before = connection.execute("PRAGMA user_version").fetchone()[0]
    _read_image_annotations(db)
    _image_measurement_frame(db)
    with sqlite3.connect(db) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(fov)")]
        assert connection.execute("PRAGMA user_version").fetchone()[0] == before
    assert columns == ["plate", "well", "field", "fieldID", "plateID", "status"]


def test_newer_layout_is_refused_before_any_change(tmp_path):
    db = tmp_path / "measurements.db"
    _set_image_annotations(db, _fields(tmp_path), "drug", "none")
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE image_annotations_meta SET value='99' WHERE key='schema_version'")
    before = _table(db)
    with pytest.raises(ValueError, match="layout 99"):
        _set_image_annotations(db, _fields(tmp_path), "drug", "DMSO")
    assert _table(db) == before


def test_legacy_well_json_is_imported_per_image(tmp_path):
    db = tmp_path / "measurements.db"
    fields = _fields(tmp_path)
    legacy = tmp_path / "overview_annotations.json"
    legacy.write_text(json.dumps(dict(wells=[
        dict(plate="plate1", well="A01", metadata=dict(condition="control", replicate="1"))])))
    assert _import_overview_annotations_json(db, legacy, fields) == 8
    summary = _well_image_annotations(db)
    assert summary == {("plate1", "A01"): {"condition": "control", "replicate": "1"}}
    legacy.write_text(json.dumps(dict(wells=[dict(plate="plate9", well="A01", metadata={})])))
    with pytest.raises(ValueError):
        _import_overview_annotations_json(db, legacy, fields)


def test_unresolved_fields_are_never_annotated(tmp_path):
    fields = [dict(plate="plate1", well="A01", field="1", file="a.tif", unresolved=True)]
    assert _image_annotation_images(fields) == []


def test_measurement_frame_joins_annotations_per_field(tmp_path):
    db = tmp_path / "measurements.db"
    fields = _fields(tmp_path)
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE fov (field_uid TEXT, plateID TEXT, plate TEXT, well TEXT, "
                           "field TEXT, fieldID TEXT, time_id TEXT, z_id TEXT, status TEXT, "
                           "channel_1_intensity_mean REAL)")
        for index, field in enumerate(fields):
            connection.execute("INSERT INTO fov VALUES (?,?,?,?,?,?,?,?,?,?)",
                               (str(index), field["plate"], field["plate"], field["well"],
                                field["field"], "f" + field["field"], "", "",
                                "ok" if index else "error", float(index)))
    _set_image_annotations(db, fields, "cell_type", "HeLa")
    one_file = [dict(plate="plate1", well="B01", field="2", file=fields[-1]["files"][0][1])]
    _set_image_annotations(db, one_file, "cell_type", "HFF")
    frame = _image_measurement_frame(db)
    assert len(frame) == 5
    assert frame.attrs["annotation_columns"] == ["cell_type"]
    by_field = frame.set_index(["well", "field"])["cell_type"]
    assert by_field[("A02", "1")] == "HeLa"
    assert pd.isna(by_field[("B01", "2")])
    assert frame.attrs["annotation_conflicts"] == [("plate1", "B01", "f2", "cell_type")]


def test_tables_are_declared_as_spacr_owned():
    from spacr import schema
    from spacr.measure import _IMAGE_ANNOTATION_KEYS
    assert _IMAGE_ANNOTATION_TABLE in schema.MEASUREMENT_TABLES
    assert schema.table_key_columns(_IMAGE_ANNOTATION_TABLE) == _IMAGE_ANNOTATION_KEYS
    for table in ("image_annotations_history", "image_annotations_meta"):
        assert table in schema.OWNED_TABLES


def test_identities_outside_the_source_stay_absolute_and_duplicates_collapse(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "elsewhere.tif"
    field = dict(plate="p", well="A01", field="1", files=(("1", str(outside)), ("1", str(outside))))
    images = _image_annotation_images([field, field], source=source)
    assert len(images) == 1 and images[0]["file"] == str(outside.resolve())
    assert "prcf" not in images[0]


def test_tables_without_key_columns_are_refused(tmp_path):
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE image_annotations (note TEXT)")
    with pytest.raises(ValueError, match="key columns"):
        _set_image_annotations(db, _fields(tmp_path), "drug", "x")


def test_reads_of_missing_tables_are_empty(tmp_path):
    missing = tmp_path / "none.db"
    assert list(_read_image_annotations(missing, ["drug"]))[-1] == "drug"
    assert _image_measurement_frame(missing).empty
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE cell (area INTEGER)")
    assert _read_image_annotations(db).empty
    assert _image_measurement_frame(db).empty
    assert _well_image_annotations(db) == {}


def test_measurement_frame_without_annotations_and_with_failed_rows(tmp_path):
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE fov (plateID TEXT, well TEXT, field TEXT, fieldID TEXT, "
                           "time_id TEXT, z_id TEXT, status TEXT)")
        connection.execute("INSERT INTO fov VALUES ('plate1','A01','1','f1','3','','ok')")
        connection.execute("INSERT INTO fov VALUES ('plate1','A01','2','f2','','','error')")
    assert len(_image_measurement_frame(db, status_ok=False)) == 2
    assert _image_measurement_frame(db).attrs["annotation_columns"] == []
    assert len(_image_measurement_frame(db, annotations=False)) == 1
    _add_image_annotation_column(db, "drug")
    assert "drug" not in _image_measurement_frame(db).columns
    _set_image_annotations(db, [dict(plate="plate1", well="A01", field="1", file="a.tif",
                                     time_id="3")], "drug", "DMSO")
    assert _image_measurement_frame(db)["drug"].tolist() == ["DMSO"]


def test_wells_with_only_cleared_values_are_not_summarised(tmp_path):
    db = tmp_path / "measurements.db"
    fields = _fields(tmp_path)
    _set_image_annotations(db, fields, "drug", "x", wells=[("plate1", "A01")])
    _set_image_annotations(db, fields, "cell_type", "HeLa", wells=[("plate1", "A02")])
    _set_image_annotations(db, fields, "drug", None, wells=[("plate1", "A01")])
    assert _well_image_annotations(db) == {("plate1", "A02"): {"cell_type": "HeLa"}}
