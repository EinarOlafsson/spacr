"""Scientific acceptance for the shared named/default/custom table workflow."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.derived_tables import default_definition, execute, load_definitions, save_definition
from spacr.merge_tables import MergeError, merge_tables


def database(tmp_path, frames, name="measurements.db"):
    path = str(tmp_path / name)
    with sqlite3.connect(path) as db:
        for table, frame in frames.items():
            pd.DataFrame(frame).to_sql(table, db, index=False)
    return path


def standard_frames():
    image = {"plateID": "p1", "rowID": "r1", "columnID": "c1"}
    base = pd.DataFrame([{**image, "fieldID": "f1", "object_label": 1, "area": 100},
                         {**image, "fieldID": "f2", "object_label": 1, "area": 200},
                         {**image, "fieldID": "f2", "object_label": 2, "area": 300}])
    child = pd.DataFrame([{**image, "fieldID": field, "object_label": i + 7,
                           "cell_id": 1, "area": area, "mean_intensity": area * 2}
                          for i, (field, area) in enumerate([("f1", 2), ("f1", 5), ("f2", 11)])])
    return {"cell": base, "pathogen": child, "organelle": child.assign(area=[3, 7, 17])}


def custom_database(tmp_path):
    return database(tmp_path, {
        "observations": {"image": ["a", "b", "c"], "id": [1, 1, 1], "value": [10., 20., 30.]},
        "particles": {"scene": ["a", "a", "b", "d", None], "parent": [1, 1, 1, 1, 1],
                      "particle_id": [7, 9, 7, 7, 7], "value": [2., 4., 9., 100., 900.],
                      "label": ["one", "two", "three", "four", "missing"],
                      "active": [1, 0, 1, 1, 1], "all_missing": [None] * 5}})


def custom_definition(path):
    result = default_definition(path, ["observations", "particles"], base="observations", name="External merge")
    result.update(mode="custom", acknowledged=True, base_keys=["image", "id"])
    result["joins"][0].update(left_keys=["image", "id"], right_keys=["scene", "parent"],
                              relationship="one-to-many", how="left", overrides={"value": "sum"})
    return result


def test_standard_merge_matches_shared_default_and_never_multiplies_children(tmp_path):
    path = database(tmp_path, standard_frames())
    definition = default_definition(path, ["cell", "pathogen", "organelle"])
    result, report = execute(path, definition)
    pd.testing.assert_frame_equal(result, merge_tables(path, ["cell", "pathogen", "organelle"]))
    assert result.pathogen_area.tolist()[:2] == [7, 11]
    assert result.organelle_area.tolist()[:2] == [10, 17]
    assert result.cell_area.tolist() == [100, 200, 300]
    assert len(result) == 3
    assert report["joins"][0]["unmatched_base"] == 1
    assert report["image_provenance"]


def test_custom_keys_overlap_types_nulls_and_unmatched_rows(tmp_path):
    path = custom_database(tmp_path)
    definition = custom_definition(path)
    definition["joins"][0]["overrides"].update(active="all", label="nunique", all_missing="first")
    result, report = execute(path, definition)
    assert result.particles_value.iloc[:2].tolist() == [6, 9]
    assert np.isnan(result.particles_value.iloc[2])
    assert result.observations_value.tolist() == [10, 20, 30]
    assert result.particles_label.iloc[0] == 2
    assert result.particles_particle_id.iloc[0] == 7  # kept, never averaged
    assert result.particles_active.iloc[0] == 0
    assert result.particles_all_missing.isna().all()
    assert report["joins"][0]["unmatched_child"] == 1
    assert report["joins"][0]["missing_key_rows"] == 1
    assert not report["image_provenance"]
    assert "object_label" not in result


def test_default_external_schema_requires_custom_mapping(tmp_path):
    path = custom_database(tmp_path)
    with pytest.raises(MergeError, match="missing keys|Customize"):
        execute(path, default_definition(path, ["observations", "particles"]))


@pytest.mark.parametrize("mutation, error", [
    (lambda d: d.update(acknowledged=False), "Acknowledge"),
    (lambda d: d.update(base_keys=["id"]), "duplicate observation"),
    (lambda d: d["joins"][0].update(relationship="one-to-one"), "cardinality"),
    (lambda d: d["joins"][0].update(right_keys=["absent", "parent"]), "missing keys"),
    (lambda d: d["joins"][0].update(how="outer"), "join types"),
    (lambda d: d["joins"][0]["overrides"].update(particle_id="mean"), "incompatible"),
    (lambda d: d["joins"][0]["overrides"].update(label="sum"), "incompatible"),
])
def test_invalid_custom_mapping_never_bypasses_validation(tmp_path, mutation, error):
    path = custom_database(tmp_path)
    definition = custom_definition(path)
    mutation(definition)
    with pytest.raises(MergeError, match=error):
        execute(path, definition)


def test_named_definition_roundtrip_is_source_scoped_and_preserves_source(tmp_path):
    path = custom_database(tmp_path)
    before = open(path, "rb").read()
    definition = custom_definition(path)
    save_definition(path, definition)
    loaded = load_definitions(path)[definition["name"]]
    pd.testing.assert_frame_equal(execute(path, definition)[0], execute(path, loaded)[0])
    assert open(path, "rb").read() == before
    other = database(tmp_path, {"observations": {"id": [1]}}, name="other.db")
    assert load_definitions(other) == {}
    with pytest.raises(MergeError, match="another database"):
        execute(other, loaded)
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE particles ADD COLUMN new_value REAL")
    with pytest.raises(MergeError, match="Schema changed"):
        execute(path, loaded)


def test_repeated_object_ids_in_timelapse_are_not_pooled(tmp_path):
    frames = standard_frames()
    frames = {name: pd.concat([frame.assign(timeID=1), frame.assign(timeID=2)], ignore_index=True)
              for name, frame in frames.items()}
    frames["pathogen"].loc[3:, "area"] *= 10
    path = database(tmp_path, frames)
    result, _ = execute(path, default_definition(path, ["cell", "pathogen"]))
    assert len(result) == 6
    assert result.pathogen_area.iloc[[0, 3]].tolist() == [7, 70]


def _repeated_image_frames():
    return {name: pd.concat([frame.assign(file_name="first_image"),
                            frame.assign(file_name="second_image")], ignore_index=True)
            for name, frame in standard_frames().items()}


def test_default_merge_keeps_repeated_cell_labels_in_separate_images(tmp_path):
    frames = _repeated_image_frames()
    frames["pathogen"].loc[3:, "area"] *= 10
    path = database(tmp_path, frames)
    before = open(path, "rb").read()
    definition = default_definition(path, ["cell", "pathogen"])
    assert "file_name" in definition["base_keys"]
    assert "file_name" in definition["joins"][0]["left_keys"]
    assert "file_name" in definition["joins"][0]["right_keys"]
    result, report = execute(path, definition)
    assert len(result) == 6
    assert result.file_name.tolist() == ["first_image"] * 3 + ["second_image"] * 3
    assert result.pathogen_area.iloc[[0, 3]].tolist() == [7, 70]
    assert result.cell_area.tolist() == [100, 200, 300] * 2
    assert report["joins"][0]["unmatched_base"] == 2
    assert open(path, "rb").read() == before


@pytest.mark.parametrize("defect, message", [
    ("child_missing_image", "missing keys file_name"),
    ("duplicate_in_same_image", "duplicate observation keys"),
])
def test_default_image_identity_still_refuses_ambiguous_sources(tmp_path, defect, message):
    frames = _repeated_image_frames()
    if defect == "child_missing_image":
        frames["pathogen"] = frames["pathogen"].drop(columns="file_name")
    else:
        frames["cell"] = pd.concat([frames["cell"], frames["cell"].iloc[[0]]], ignore_index=True)
    path = database(tmp_path, frames)
    before = open(path, "rb").read()
    with pytest.raises(MergeError, match=message):
        execute(path, default_definition(path, ["cell", "pathogen"]))
    assert open(path, "rb").read() == before


def test_missing_image_key_is_not_joined_by_object_label_alone(tmp_path):
    frames = standard_frames()
    frames["pathogen"] = frames["pathogen"].drop(columns="fieldID")
    path = database(tmp_path, frames)
    with pytest.raises(MergeError, match="missing keys fieldID"):
        execute(path, default_definition(path, ["cell", "pathogen"]))


def test_per_table_overrides_do_not_change_same_named_measurement_elsewhere(tmp_path):
    path = database(tmp_path, standard_frames())
    definition = default_definition(path, ["cell", "pathogen", "organelle"])
    definition["policy"]["overrides"] = {"pathogen.area": "median", "organelle.area": "max"}
    result, _ = execute(path, definition)
    assert result.pathogen_area.iloc[0] == 3.5
    assert result.organelle_area.iloc[0] == 7


def test_shared_regression_default_produces_identical_measurements(tmp_path):
    from spacr.plate_measurements import merge_plate_databases
    path = database(tmp_path, standard_frames())
    named, _ = execute(path, default_definition(path, ["cell", "pathogen", "organelle"]))
    regression = merge_plate_databases({"p1": path}, ["cell", "pathogen", "organelle"]).frame
    for column in ("cell_area", "pathogen_area", "pathogen_mean_intensity", "pathogen_count", "organelle_area"):
        pd.testing.assert_series_equal(named[column], regression[column])


def test_all_missing_numeric_sum_is_not_fabricated_zero(tmp_path):
    frames = standard_frames()
    frames["pathogen"]["area"] = float("nan")
    path = database(tmp_path, frames)
    result, _ = execute(path, default_definition(path, ["cell", "pathogen"]))
    assert result.pathogen_area.isna().all()
    assert result.pathogen_count.tolist() == [2., 1., 0.]


def test_custom_one_to_one_and_inner_join_preserve_one_base_observation(tmp_path):
    path = custom_database(tmp_path)
    definition = custom_definition(path)
    definition["joins"][0]["how"] = "inner"
    result, _ = execute(path, definition)
    assert result.image.tolist() == ["a", "b"]
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM particles WHERE particle_id=9")
    definition["joins"][0]["relationship"] = "one-to-one"
    result, _ = execute(path, definition)
    assert result.particles_value.tolist() == [2., 9.]


def test_regression_temporal_keys_match_named_default_without_pooling(tmp_path):
    from spacr.plate_measurements import merge_plate_databases
    frames = {name: pd.concat([frame.assign(timeID=1), frame.assign(timeID=2)], ignore_index=True)
              for name, frame in standard_frames().items()}
    frames["pathogen"].loc[3:, "area"] *= 10
    path = database(tmp_path, frames)
    result = merge_plate_databases({"p1": path}, ["cell", "pathogen"]).frame
    assert len(result) == 6
    assert result.pathogen_area.iloc[[0, 3]].tolist() == [7, 70]


def test_external_object_label_strings_are_not_interpreted_as_spacr_labels(tmp_path):
    path = database(tmp_path, {"samples": {"object_label": ["alpha", "beta"], "value": [1, 2]},
                               "events": {"parent": ["alpha", "beta"], "value": [11, 22]}})
    definition = default_definition(path, ["samples", "events"], base="samples")
    definition.update(mode="custom", acknowledged=True, base_keys=["object_label"])
    definition["joins"][0].update(left_keys=["object_label"], right_keys=["parent"])
    result, _ = execute(path, definition)
    assert result.object_label.tolist() == ["alpha", "beta"]
    assert result.events_value.tolist() == [11, 22]


def test_custom_one_to_one_count_override_changes_the_values(tmp_path):
    path = database(tmp_path, {"samples": {"sample": [1, 2]},
                               "events": {"parent": [1, 2], "value": [20., 40.]}})
    definition = default_definition(path, ["samples", "events"], base="samples")
    definition.update(mode="custom", acknowledged=True, base_keys=["sample"])
    definition["joins"][0].update(left_keys=["sample"], right_keys=["parent"],
                                  relationship="one-to-one", overrides={"value": "count"})
    result, _ = execute(path, definition)
    assert result.events_value.tolist() == [1, 1]


def test_existing_count_measurements_are_not_overwritten_by_row_counts(tmp_path):
    frames = standard_frames()
    frames["pathogen"]["count"] = [10, 20, 30]
    path = database(tmp_path, frames)
    result, _ = execute(path, default_definition(path, ["cell", "pathogen"]))
    assert result.pathogen_count.iloc[:2].tolist() == [30, 30]
    assert result.pathogen_source_count.iloc[:2].tolist() == [2, 1]


@pytest.mark.parametrize("reason", ["duplicate_keys", "null_keys", "missing_keys", "invalid_keys"])
def test_key_failures_keep_only_anonymous_structure(reason):
    from spacr.derived_tables import _require_keys
    from spacr.errors import ConfigurationError

    frame = pd.DataFrame({"plateID": ["secret_plate", "secret_plate"],
                          "file_name": ["secret_image.tif", "secret_image.tif"],
                          "private_person_column": [1, 1]})
    keys = ["plateID", "file_name", "private_person_column"]
    if reason == "null_keys":
        frame.loc[0, "private_person_column"] = None
    elif reason == "missing_keys":
        keys += ["timeID", "private_missing_column"]
    elif reason == "invalid_keys":
        keys += ["plateID"]
    with pytest.raises(MergeError) as caught:
        _require_keys(frame, keys, "private_person_table", unique=True)
    assert isinstance(caught.value, (ConfigurationError, ValueError))
    facts = caught.value._spacr_merge_structure
    assert facts["reason"] == reason
    assert facts["table"] == "custom"
    assert facts["present"] == ["plateID", "file_name"]
    assert facts["rows"] == 2
    if reason == "duplicate_keys":
        assert (facts["unique_keys"], facts["duplicate_rows"], facts["null_rows"]) == (1, 2, 0)
    if reason == "missing_keys":
        assert facts["missing_canonical"] == ["timeID"]
        assert facts["missing_custom_count"] == 1
    import json
    assert "secret" not in json.dumps(facts)
    assert "private_person" not in json.dumps(facts)
    assert "private_missing" not in json.dumps(facts)
