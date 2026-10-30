"""Every refusal a named derived-table merge can give, and why it gives it."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from spacr import derived_tables as dt
from spacr.merge_tables import MergeError
from tests.test_derived_tables_requested_merges import (
    custom_database, custom_definition, database, standard_frames)


@pytest.fixture
def standard(tmp_path):
    return database(tmp_path, standard_frames())


@pytest.fixture
def custom(tmp_path):
    path = custom_database(tmp_path)
    return path, custom_definition(path)


def test_identity_columns_are_the_ones_present():
    frame = pd.DataFrame(columns=["plateID", "rowID", "time_id", "area"])
    assert dt.identity_columns(frame) == ["plateID", "rowID", "time_id"]


def test_a_named_base_joins_the_table_list(standard):
    definition = dt.default_definition(standard, ["pathogen"], base="cell")
    assert definition["base"] == "cell"
    with pytest.raises(MergeError, match="Missing tables: nope"):
        dt.default_definition(standard, ["cell", "nope"])


def test_a_definition_from_another_version_or_with_a_repeat_is_refused(standard):
    definition = dt.default_definition(standard, ["cell", "pathogen"])
    with pytest.raises(MergeError, match="Unsupported"):
        dt.validate_source(standard, dict(definition, version=2))
    repeated = copy.deepcopy(definition)
    repeated["joins"].append(copy.deepcopy(repeated["joins"][0]))
    with pytest.raises(MergeError, match="only once"):
        dt.validate_source(standard, repeated)


def test_text_columns_aggregate_only_by_order_and_count():
    assert dt.allowed_methods(pd.Series(["a", "b"])) == (
        "first", "last", "count", "nunique", "min", "max")


@pytest.mark.parametrize("keys, message", [
    ([], "nonempty list"),
    (["image", "image"], "distinct join columns"),
])
def test_join_keys_must_be_distinct_and_present(keys, message):
    with pytest.raises(MergeError, match=message):
        dt._require_keys(pd.DataFrame({"image": [1]}), keys, "t")


def test_base_keys_must_be_complete():
    with pytest.raises(MergeError, match="missing values"):
        dt._require_keys(pd.DataFrame({"image": [1, None]}), ["image"], "t",
                         unique=True)


def _join(definition, **change):
    definition["joins"][0].update(change)
    return definition


def test_key_lists_of_different_length_are_refused(custom):
    path, definition = custom
    _join(definition, right_keys=["scene"])
    with pytest.raises(MergeError, match="same length"):
        dt.execute(path, definition)


def test_a_key_that_collides_with_another_column_is_refused(tmp_path):
    path = database(tmp_path, {
        "observations": {"image": ["a"], "id": [1], "value": [1.0]},
        "particles": {"scene": ["a"], "parent": [1], "image": ["x"],
                      "particles_image": ["y"], "value": [2.0]}})
    definition = custom_definition(path)
    with pytest.raises(MergeError, match="collides with another column"):
        dt.execute(path, definition)


def test_an_unknown_relationship_is_refused(custom):
    path, definition = custom
    _join(definition, relationship="many-to-many")
    with pytest.raises(MergeError, match="one-to-one or one-to-many"):
        dt.execute(path, definition)


def test_unknown_modes_and_unacknowledged_custom_merges_are_refused(custom):
    path, definition = custom
    with pytest.raises(MergeError, match="default or custom"):
        dt.execute(path, dict(definition, mode="other"))
    with pytest.raises(MergeError, match="Acknowledge"):
        dt.execute(path, dict(definition, acknowledged=False))


def test_metadata_mode_needs_one_table_and_a_map(custom):
    path, definition = custom
    with pytest.raises(MergeError, match="Original filename enrichment"):
        dt.execute(path, dict(definition, mode="metadata"))


def test_default_merges_are_one_row_per_cell_with_image_identity(tmp_path):
    frames = standard_frames()
    path = database(tmp_path, frames)
    definition = dt.default_definition(path, ["pathogen", "cell"], base="pathogen")
    with pytest.raises(MergeError, match="one row per cell"):
        dt.execute(path, definition)
    bare = database(tmp_path, {"cell": frames["cell"].drop(columns=["plateID"])},
                    name="bare.db")
    definition = dt.default_definition(bare, ["cell"])
    definition["base_keys"] = ["object_label", "fieldID"]
    with pytest.raises(MergeError, match="requires plateID"):
        dt.execute(bare, definition)


def test_default_mode_refuses_a_table_without_a_standard_link(tmp_path):
    frames = standard_frames()
    frames["extra"] = frames["pathogen"]
    path = database(tmp_path, frames)
    definition = dt.default_definition(path, ["cell", "extra"])
    with pytest.raises(MergeError, match="no standard spaCR relationship"):
        dt.execute(path, definition)


def test_overrides_naming_absent_columns_are_refused(custom):
    path, definition = custom
    definition["joins"][0]["overrides"]["nonexistent"] = "sum"
    with pytest.raises(MergeError, match="absent/key columns"):
        dt.execute(path, definition)


def test_a_child_column_that_clashes_with_the_output_is_refused(tmp_path):
    path = database(tmp_path, {
        "observations": {"image": ["a"], "id": [1], "particles_value": [1.0]},
        "particles": {"scene": ["a"], "parent": [1], "value": [2.0]}})
    definition = custom_definition(path)
    definition["base_keys"] = ["image", "id", "particles_value"]
    definition["joins"][0]["overrides"] = {}
    with pytest.raises(MergeError, match="output column collision"):
        dt.execute(path, definition)


def test_saved_definitions_from_another_database_are_refused(standard, tmp_path):
    sidecar = dt.sidecar_path(standard)
    sidecar.write_text(json.dumps({"source": "/elsewhere.db", "definitions": {}}), encoding="utf-8")
    with pytest.raises(MergeError, match="different database"):
        dt.load_definitions(standard)


def test_a_result_named_like_a_source_table_is_refused(standard):
    definition = dt.default_definition(standard, ["cell", "pathogen"], name="cell")
    with pytest.raises(MergeError, match="nonempty name"):
        dt.save_definition(standard, definition)


def test_a_failed_save_leaves_no_temporary_file(standard, monkeypatch):
    definition = dt.default_definition(standard, ["cell", "pathogen"], name="Merged")

    def refuse(*_args):
        raise OSError("read-only")

    monkeypatch.setattr(dt.os, "replace", refuse)
    with pytest.raises(OSError):
        dt.save_definition(standard, definition)
    sidecar = dt.sidecar_path(standard)
    assert not [p for p in sidecar.parent.iterdir()
                if p.name.startswith(sidecar.name + ".")]


def test_a_time_id_column_is_also_offered_as_time_id_for_gates(tmp_path):
    path = database(tmp_path, {
        "observations": {"image": ["a", "b"], "id": [1, 1], "time_id": [1, 2],
                         "value": [1.0, 2.0]},
        "particles": {"scene": ["a"], "parent": [1], "value": [2.0]}})
    result, _report = dt.execute(path, custom_definition(path))
    assert result["timeID"].tolist() == result["time_id"].tolist() == [1, 2]


def test_time_resolved_filename_enrichment_survives_a_saved_recipe(tmp_path):
    """Restoring acquisition names preserves time identity and source data."""
    import hashlib
    import sqlite3

    from tests.test_original_filenames import csv_map, modern

    frame = pd.DataFrame({
        "plateID": ["plate1", "plate1"], "rowID": ["r1", "r1"],
        "columnID": ["c1", "c1"], "fieldID": ["f1", "f1"],
        "object_label": [1, 1], "time_id": [1, 2], "area": [12.5, 27.0]})
    path = database(tmp_path, {"cell": frame})
    mapping = csv_map(tmp_path, [modern(time=1, source="/raw/before.tif"),
                                 modern(time=2, source="/raw/after.tif")])
    source_bytes = Path(path).read_bytes(), mapping.read_bytes()
    recipe = dt.default_definition(path, ["cell"], name="Acquisition names")
    recipe.update(mode="metadata", original_filenames={"map_path": str(mapping)})

    result, report = dt.execute(path, recipe)

    pd.testing.assert_frame_equal(result[frame.columns], frame)
    assert result.timeID.tolist() == result.time_id.tolist() == [1, 2]
    assert result.original_filename.tolist() == ["before.tif", "after.tif"]
    assert result.original_path.tolist() == ["/raw/before.tif", "/raw/after.tif"]
    assert report["base_rows"] == report["output_rows"] == 2
    assert report["image_provenance"] is True
    assert report["original_filenames"]["matched_rows"] == 2
    bound = result.attrs["merge_definition"]
    assert bound["original_filenames"]["sha256"] == hashlib.sha256(source_bytes[1]).hexdigest()
    assert "sha256" not in recipe["original_filenames"]
    assert dt.save_definition(path, bound) == "Acquisition names"
    reopened, reopened_report = dt.execute(path, dt.load_definitions(path)["Acquisition names"])
    pd.testing.assert_frame_equal(reopened, result)
    assert reopened_report == report
    with sqlite3.connect(path) as connection:
        pd.testing.assert_frame_equal(pd.read_sql_query("SELECT * FROM cell", connection), frame)
    assert (Path(path).read_bytes(), mapping.read_bytes()) == source_bytes


def test_a_named_acquisition_column_preserves_existing_filename_metadata(tmp_path):
    """A saved custom output name adds metadata without replacing source names."""
    from tests.test_original_filenames import csv_map, modern

    frame = pd.DataFrame({"file_name": ["plate1_A01_1_1.npy"],
                          "original_filename": ["reviewed-acquisition-label"],
                          "area": [42.5]})
    path = database(tmp_path, {"readings": frame})
    mapping = csv_map(tmp_path, [modern(source="/raw/acquisition.tif")])
    before = Path(path).read_bytes(), mapping.read_bytes()
    recipe = dt.default_definition(path, ["readings"], base="readings",
                                   name="Named acquisition metadata")
    recipe.update(mode="metadata", original_filenames={
        "map_path": str(mapping), "output_column": "acquisition_name"})

    result, report = dt.execute(path, recipe)

    pd.testing.assert_frame_equal(result[frame.columns], frame)
    assert result.acquisition_name.tolist() == ["acquisition.tif"]
    assert result.original_path.tolist() == ["/raw/acquisition.tif"]
    assert report["base_rows"] == report["output_rows"] == 1
    assert report["image_provenance"] is False
    assert report["original_filenames"]["output_column"] == "acquisition_name"
    bound = result.attrs["merge_definition"]
    dt.save_definition(path, bound)
    reopened, reopened_report = dt.execute(path, dt.load_definitions(path)[bound["name"]])
    pd.testing.assert_frame_equal(reopened, result)
    assert reopened_report == report
    assert (Path(path).read_bytes(), mapping.read_bytes()) == before


def test_a_conflicting_primary_policy_cannot_expand_the_named_base(tmp_path):
    """A recipe cannot replace cell observations with its more numerous children."""
    from spacr.merge_tables import MergePolicy

    identity = {"plateID": "p1", "rowID": "r1", "columnID": "c1", "fieldID": "f1"}
    frames = {
        "cell": pd.DataFrame([{**identity, "object_label": 1, "area": 100.0}]),
        "pathogen": pd.DataFrame([{**identity, "object_label": label,
                                   "cell_id": 1, "area": float(label)}
                                  for label in (1, 2, 3)])}
    path = database(tmp_path, frames)
    before = Path(path).read_bytes()
    recipe = dt.default_definition(path, ["cell", "pathogen"], base="cell",
                                   policy=MergePolicy(primary="pathogen"))
    assert recipe["base"] == "cell"
    assert recipe["policy"]["primary"] == "pathogen"

    with pytest.raises(MergeError, match="increased the base observation count"):
        dt.execute(path, recipe)

    assert Path(path).read_bytes() == before
    assert not dt.sidecar_path(path).exists()
