"""What :func:`spacr.derived_tables.execute` refuses, and what it keeps.

A named derived-table recipe is replayed against a measurement database, so
every malformed recipe has to be refused by name before a single row is
merged, and a well-formed one has to give back the same table every time it
is reopened.  One test walks every refusal with its exact message; the other
walks the successful recipes that carry time identity and acquisition
file names, and checks that a saved recipe replays to the identical result
without touching its sources.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from spacr import derived_tables as dt
from spacr.merge_tables import MergeError
from tests.test_derived_tables_requested_merges import (
    custom_database, custom_definition, database, standard_frames)
from tests.test_original_filenames import csv_map, modern


def _custom(join=None, top=None, overrides=None):
    """A custom recipe on the shared external database with ``join``/``top`` edits."""
    def build(tmp_path):
        path = custom_database(tmp_path)
        definition = custom_definition(path)
        definition["joins"][0].update(join or {})
        definition["joins"][0]["overrides"].update(overrides or {})
        definition.update(top or {})
        return path, definition
    return build


def _colliding_key(tmp_path):
    path = database(tmp_path, {
        "observations": {"image": ["a"], "id": [1], "value": [1.0]},
        "particles": {"scene": ["a"], "parent": [1], "image": ["x"],
                      "particles_image": ["y"], "value": [2.0]}})
    return path, custom_definition(path)


def _clashing_child_column(tmp_path):
    path = database(tmp_path, {
        "observations": {"image": ["a"], "id": [1], "particles_value": [1.0]},
        "particles": {"scene": ["a"], "parent": [1], "value": [2.0]}})
    definition = custom_definition(path)
    definition["base_keys"] = ["image", "id", "particles_value"]
    definition["joins"][0]["overrides"] = {}
    return path, definition


def _child_as_base(tmp_path):
    path = database(tmp_path, standard_frames())
    return path, dt.default_definition(path, ["pathogen", "cell"], base="pathogen")


def _no_plate_identity(tmp_path):
    frames = standard_frames()
    path = database(tmp_path, {"cell": frames["cell"].drop(columns=["plateID"])},
                    name="bare.db")
    definition = dt.default_definition(path, ["cell"])
    definition["base_keys"] = ["object_label", "fieldID"]
    return path, definition


def _unlinked_table(tmp_path):
    frames = standard_frames()
    frames["extra"] = frames["pathogen"]
    path = database(tmp_path, frames)
    return path, dt.default_definition(path, ["cell", "extra"])


REFUSALS = [
    ("key_lists_of_different_length", _custom(join={"right_keys": ["scene"]}),
     "particles: left and right key lists must have the same length."),
    ("unknown_relationship", _custom(join={"relationship": "many-to-many"}),
     "particles: choose one-to-one or one-to-many."),
    ("unknown_mode", _custom(top={"mode": "other"}),
     "Choose default or custom merging."),
    ("custom_merge_not_acknowledged", _custom(top={"acknowledged": False}),
     "Acknowledge the custom merging warning before previewing or applying."),
    ("metadata_mode_without_one_table_and_a_map", _custom(top={"mode": "metadata"}),
     "Original filename enrichment requires one source table and a conversion map."),
    ("override_naming_an_absent_column", _custom(overrides={"nonexistent": "sum"}),
     "particles: aggregation overrides name absent/key columns: ['nonexistent']"),
    ("key_collides_with_another_column", _colliding_key,
     "particles: key mapping collides with another column; use distinct key names."),
    ("child_column_clashes_with_the_output", _clashing_child_column,
     "particles: output column collision ['particles_value']; choose tables "
     "with unambiguous column names."),
    ("default_merge_on_a_child_base", _child_as_base,
     "Default output is one row per cell or cytoplasm. Use Customize merging "
     "for another observation level."),
    ("default_merge_without_image_identity", _no_plate_identity,
     "Default merging requires plateID, rowID, columnID and fieldID. Use "
     "Customize merging for external data without image provenance."),
    ("default_merge_of_an_unlinked_table", _unlinked_table,
     "extra: no standard spaCR relationship; use Customize merging."),
]


def test_execute_refuses_every_malformed_recipe_by_name(tmp_path, subtests):
    """Each malformed recipe raises MergeError with its exact message and leaves the database as it was."""
    for case_id, build, message in REFUSALS:
        with subtests.test(msg=case_id):
            folder = tmp_path / case_id
            folder.mkdir()
            path, definition = build(folder)
            before = Path(path).read_bytes()
            with pytest.raises(MergeError) as caught:
                dt.execute(path, definition)
            assert str(caught.value) == message
            assert Path(path).read_bytes() == before
            assert not dt.sidecar_path(path).exists()


def _time_resolved(tmp_path):
    frame = pd.DataFrame({
        "plateID": ["plate1", "plate1"], "rowID": ["r1", "r1"],
        "columnID": ["c1", "c1"], "fieldID": ["f1", "f1"],
        "object_label": [1, 1], "time_id": [1, 2], "area": [12.5, 27.0]})
    path = database(tmp_path, {"cell": frame})
    mapping = csv_map(tmp_path, [modern(time=1, source="/raw/before.tif"),
                                 modern(time=2, source="/raw/after.tif")])
    recipe = dt.default_definition(path, ["cell"], name="Acquisition names")
    recipe.update(mode="metadata", original_filenames={"map_path": str(mapping)})
    expected = {"original_filename": ["before.tif", "after.tif"],
                "original_path": ["/raw/before.tif", "/raw/after.tif"],
                "timeID": [1, 2], "time_id": [1, 2]}
    report = {"base_rows": 2, "output_rows": 2, "image_provenance": True,
              "matched_rows": 2, "output_column": "original_filename"}
    return path, mapping, frame, recipe, expected, report


def _named_output_column(tmp_path):
    frame = pd.DataFrame({"file_name": ["plate1_A01_1_1.npy"],
                          "original_filename": ["reviewed-acquisition-label"],
                          "area": [42.5]})
    path = database(tmp_path, {"readings": frame})
    mapping = csv_map(tmp_path, [modern(source="/raw/acquisition.tif")])
    recipe = dt.default_definition(path, ["readings"], base="readings",
                                   name="Named acquisition metadata")
    recipe.update(mode="metadata", original_filenames={
        "map_path": str(mapping), "output_column": "acquisition_name"})
    expected = {"acquisition_name": ["acquisition.tif"],
                "original_path": ["/raw/acquisition.tif"],
                "original_filename": ["reviewed-acquisition-label"]}
    report = {"base_rows": 1, "output_rows": 1, "image_provenance": False,
              "matched_rows": 1, "output_column": "acquisition_name"}
    return path, mapping, frame, recipe, expected, report


ENRICHMENTS = [
    ("time_resolved_names_keep_time_identity", _time_resolved),
    ("a_named_output_column_keeps_the_existing_names", _named_output_column),
]


def test_filename_enrichment_replays_identically_from_a_saved_recipe(tmp_path, subtests):
    """Enrichment keeps the source columns, adds the exact names, and a saved recipe replays it byte-for-byte."""
    for case_id, build in ENRICHMENTS:
        with subtests.test(msg=case_id):
            folder = tmp_path / case_id
            folder.mkdir()
            path, mapping, frame, recipe, expected, want = build(folder)
            sources = Path(path).read_bytes(), mapping.read_bytes()

            result, report = dt.execute(path, recipe)

            pd.testing.assert_frame_equal(result[frame.columns], frame)
            assert {column: result[column].tolist() for column in expected} == expected
            assert (report["base_rows"], report["output_rows"],
                    report["image_provenance"]) == (
                        want["base_rows"], want["output_rows"], want["image_provenance"])
            assert report["original_filenames"]["matched_rows"] == want["matched_rows"]
            assert report["original_filenames"]["output_column"] == want["output_column"]
            bound = result.attrs["merge_definition"]
            assert bound["original_filenames"]["sha256"] == hashlib.sha256(
                sources[1]).hexdigest()
            assert "sha256" not in recipe["original_filenames"]
            assert dt.save_definition(path, bound) == recipe["name"]
            reopened, reopened_report = dt.execute(
                path, dt.load_definitions(path)[recipe["name"]])
            pd.testing.assert_frame_equal(reopened, result)
            assert reopened_report == report
            with sqlite3.connect(path) as connection:
                stored = pd.read_sql_query(
                    f"SELECT * FROM {recipe['base']}", connection)
            pd.testing.assert_frame_equal(stored, frame)
            assert (Path(path).read_bytes(), mapping.read_bytes()) == sources


def test_a_time_id_column_is_also_offered_as_time_id_for_gates(tmp_path):
    """A custom merge copies ``time_id`` into ``timeID`` row for row, beside the merged children."""
    path = database(tmp_path, {
        "observations": {"image": ["a", "b"], "id": [1, 1], "time_id": [1, 2],
                         "value": [1.0, 2.0]},
        "particles": {"scene": ["a"], "parent": [1], "value": [2.0]}})
    result, report = dt.execute(path, custom_definition(path))
    assert result["timeID"].tolist() == result["time_id"].tolist() == [1, 2]
    assert result["image"].tolist() == ["a", "b"]
    assert (report["base_rows"], report["output_rows"]) == (2, 2)
