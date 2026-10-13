"""Enforceable canonical schemas for every analysis object table."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr import schema
from spacr.measurement_schema import MEASUREMENT_STAMP_COLUMNS
from spacr.utils import _merge_and_save_to_database


def _frame(table, *, timelapse=False, parent_link=True, stamped=False):
    data = {
        "object_label": [1, 2],
        "plateID": ["plate1", "plate1"],
        "rowID": ["r1", "r1"],
        "columnID": ["c1", "c1"],
        "fieldID": ["f1", "f1"],
        "prcf": ["plate1_r1_c1_f1", "plate1_r1_c1_f1"],
        "file_name": ["plate1_A01_1", "plate1_A01_1"],
        "path_name": ["/data/plate1_A01_1.npy"] * 2,
        f"{table}_area": [10.0, 20.0],
    }
    if timelapse:
        data["timeID"] = ["t2", "t2"]
        data["prcf"] = ["plate1_r1_c1_f1_t2"] * 2
    if table in {"nucleus", "pathogen"} and parent_link:
        data["cell_id"] = [1, None]
    if stamped:
        data.update({
            "measurement_ndim": [3, 3],
            "measurement_units": ["um", "um"],
            "n_z": [5, 5],
            "voxel_size_z_um": [1.5, 1.5],
            "voxel_size_xy_um": [0.25, 0.25],
        })
    return pd.DataFrame(data)


def test_all_analysis_tables_have_declarative_schemas():
    assert schema.CANONICAL_OBJECT_TABLES == (
        "cell", "cytoplasm", "nucleus", "pathogen",
        *schema.ORGANELLE_ROLES)
    assert tuple(schema.OBJECT_TABLE_SCHEMAS) == schema.CANONICAL_OBJECT_TABLES

    for table, contract in schema.OBJECT_TABLE_SCHEMAS.items():
        assert contract.table == table
        assert contract.object_type == table
        assert set(schema.OBJECT_TABLE_REQUIRED_COLUMNS) == set(
            contract.required_columns)
        assert contract.feature_column(f"{table}_area")

    assert schema.OBJECT_TABLE_SCHEMAS["cell"].parent_column is None
    assert schema.OBJECT_TABLE_SCHEMAS["cytoplasm"].parent_column is None
    for table in ("nucleus", "pathogen", *schema.ORGANELLE_ROLES):
        assert schema.OBJECT_TABLE_SCHEMAS[table].parent_column == "cell_id"
    with pytest.raises(TypeError):
        schema.OBJECT_TABLE_SCHEMAS["other"] = object()


@pytest.mark.parametrize("table", schema.CANONICAL_OBJECT_TABLES)
def test_valid_object_frames_pass_and_preserve_dynamic_features(table):
    frame = _frame(table, stamped=True)

    validated = schema.validate_object_table_frame(frame, table)

    pd.testing.assert_frame_equal(validated, frame)
    assert validated[f"{table}_area"].tolist() == [10.0, 20.0]
    assert set(MEASUREMENT_STAMP_COLUMNS) <= set(validated.columns)


def test_legacy_metadata_spelling_is_canonicalised_on_return():
    frame = _frame("cell").rename(columns={
        "plateID": "plate_id",
        "rowID": "row_name",
        "columnID": "column_name",
        "fieldID": "field_name",
    })

    validated = schema.validate_object_table_frame(frame, "cell")

    assert set(schema.FIELD_KEY_COLUMNS) <= set(validated.columns)
    assert not {"plate_id", "row_name", "column_name", "field_name"} & set(
        validated.columns)


def test_missing_required_columns_are_named():
    frame = _frame("cell").drop(columns=["path_name", "fieldID"])

    with pytest.raises(
            schema.ObjectTableSchemaError,
            match=r"missing required canonical column.*fieldID.*path_name"):
        schema.validate_object_table_frame(frame, "cell")


@pytest.mark.parametrize("value", [None, 0, -1, 1.5, "one"])
def test_object_labels_must_be_positive_integers(value):
    frame = _frame("cell")
    frame["object_label"] = pd.Series([value, 2], dtype=object)

    with pytest.raises(
            schema.ObjectTableSchemaError,
            match="positive integer labels"):
        schema.validate_object_table_frame(frame, "cell")


def test_child_parent_links_are_optional_but_typed_when_present():
    without_link = _frame("nucleus", parent_link=False)
    validated = schema.validate_object_table_frame(without_link, "nucleus")
    assert "cell_id" not in validated.columns

    invalid_link = _frame("pathogen")
    invalid_link["cell_id"] = pd.Series(
        ["cell one", None], dtype=object)
    with pytest.raises(
            schema.ObjectTableSchemaError,
            match=r"pathogen\.cell_id.*positive integer"):
        schema.validate_object_table_frame(invalid_link, "pathogen")


def test_prcf_must_match_its_identity_components():
    frame = _frame("cell")
    frame.loc[1, "prcf"] = "plate9_r9_c9_f9"

    with pytest.raises(
            schema.ObjectTableSchemaError,
            match=r"cell\.prcf disagrees.*expected"):
        schema.validate_object_table_frame(frame, "cell")


def test_duplicate_object_keys_are_rejected_before_database_write():
    frame = _frame("cell")
    frame.loc[1, "object_label"] = 1

    with pytest.raises(
            schema.ObjectTableSchemaError,
            match=r"one-row-per-object.*duplicated keys"):
        schema.validate_object_table_frame(frame, "cell")


def test_timelapse_contract_requires_time_and_includes_it_in_the_key():
    frame = _frame("cell", timelapse=True)
    validated = schema.validate_object_table_frame(
        frame, "cell", timelapse=True)
    assert validated["timeID"].tolist() == ["t2", "t2"]
    assert schema.OBJECT_TABLE_SCHEMAS["cell"].row_key_columns(
        timelapse=True) == (
            "plateID", "rowID", "columnID", "fieldID", "timeID",
            "object_label",
        )

    with pytest.raises(
            schema.ObjectTableSchemaError,
            match="timelapse table.*no 'timeID'"):
        schema.validate_object_table_frame(
            _frame("cell"), "cell", timelapse=True)


def test_measurement_stamp_is_all_or_none():
    frame = _frame("cell")
    frame["measurement_ndim"] = 2

    with pytest.raises(
            schema.ObjectTableSchemaError,
            match="partial measurement provenance stamp"):
        schema.validate_object_table_frame(frame, "cell")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("measurement_ndim", 4, "measurement_ndim must be 2 or 3"),
        ("measurement_units", "meters", "measurement_units must be one of"),
        ("n_z", 0, "n_z must contain positive integer"),
        ("voxel_size_xy_um", -0.5, "positive numeric values or NULL"),
    ],
)
def test_measurement_stamp_values_are_typed(column, value, message):
    frame = _frame("cell", stamped=True)
    frame[column] = value

    with pytest.raises(schema.ObjectTableSchemaError, match=message):
        schema.validate_object_table_frame(frame, "cell")


def test_two_dimensional_stamp_has_canonical_units_and_depth():
    frame = _frame("cell", stamped=True)
    frame["measurement_ndim"] = 2
    frame["measurement_units"] = "um"
    frame["n_z"] = 3

    with pytest.raises(
            schema.ObjectTableSchemaError,
            match='2-D rows must use measurement_units="px" and n_z=1'):
        schema.validate_object_table_frame(frame, "cell")


def test_foreign_features_and_non_numeric_owned_features_are_rejected():
    foreign = _frame("cell")
    foreign["nucleus_area"] = [1.0, 2.0]
    with pytest.raises(
            schema.ObjectTableSchemaError,
            match="contains nucleus feature"):
        schema.validate_object_table_frame(foreign, "cell")

    text_feature = _frame("cytoplasm")
    text_feature["cytoplasm_area"] = ["large", "small"]
    with pytest.raises(
            schema.ObjectTableSchemaError,
            match=r"cytoplasm feature 'cytoplasm_area' must be numeric"):
        schema.validate_object_table_frame(text_feature, "cytoplasm")


def test_unknown_tables_receive_an_actionable_error():
    with pytest.raises(
            schema.ObjectTableSchemaError,
            match=r"no canonical object-table schema.*cell.*pathogen"):
        schema.object_table_schema("not_a_spacr_object")


def test_measurement_writer_enforces_the_schema_before_sqlite(tmp_path):
    morphology = pd.DataFrame({
        "label": [1],
        "cell_area": ["not numeric"],
    })
    intensity = pd.DataFrame({
        "label": [1],
        "cell_channel_0_mean_intensity": [3.0],
    })

    with pytest.raises(
            schema.ObjectTableSchemaError,
            match=r"cell feature 'cell_area' must be numeric"):
        _merge_and_save_to_database(
            morphology,
            intensity,
            "cell",
            str(tmp_path),
            "plate1_A01_1",
            "experiment",
        )

    db_path = tmp_path / "measurements" / "measurements.db"
    if db_path.exists():
        with sqlite3.connect(db_path) as connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        assert ("cell",) not in tables
