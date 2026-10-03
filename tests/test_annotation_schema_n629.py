"""Portable annotation recipes carry rules rather than source rows or values."""
import copy
import json
import os

import pandas as pd
import pytest

from spacr import condition_annotations as annotations


def recipe(frame, source):
    result = annotations.new_definition(frame, source)
    result.pop("column")
    result.pop("conditions")
    result.update(version=3, columns=[
        {"column": "cell_type", "kind": "extract", "metadata_column": "filename",
         "pattern": r"^(?P<cell>[^_]+)_rep\d+\.tif$", "group": "cell"},
        {"column": "treatment", "kind": "rules", "conditions": [
            {"name": "control α", "criteria": [
                {"metadata_column": "cell_type", "operator": "equals", "value": "HeLa"}],
             "match": "all", "manual_rows": []}]},
        {"column": "condition", "kind": "template", "parts": [
            {"kind": "column", "column": "cell_type"},
            {"kind": "text", "text": " / {literal}_"},
            {"kind": "column", "column": "treatment"}]},
    ])
    return result


def fixture_recipe(tmp_path):
    frame = pd.DataFrame({"filename": ["HeLa_rep1.tif", "U2OS_rep3.tif"],
                          "measurement": [2, 7]}, index=[9, 9])
    source = annotations.source_context(tmp_path / "source.csv")
    return frame, source, recipe(frame, source)


def test_roundtrip_rebinds_to_different_table_without_exporting_source_data(tmp_path):
    frame, source, definition = fixture_recipe(tmp_path)
    before = frame.copy(deep=True)
    path = tmp_path / "schema α.json"
    assert annotations._save_schema(path, frame, definition, source) == 0
    payload = json.loads(path.read_text())
    assert payload["format"] == "spacr.annotation-schema"
    assert set(payload) == {"format", "version", "recipe_version", "manual_rows", "columns"}
    assert "source.csv" not in path.read_text()
    assert "HeLa_rep1.tif" not in path.read_text()
    assert "control α" in path.read_text()
    other = pd.DataFrame({"filename": ["HeLa_rep9.tif", "HeLa_rep8.tif", None],
                          "measurement": [20, 30, 40]}, index=[4, 2, 4])
    context = annotations.source_context(tmp_path / "other.db", "cells")
    loaded, report = annotations._load_schema(path, other, context)
    assert loaded["source"] == context
    assert loaded["row_count"] == 3
    assert loaded["content_sha256"] != definition["content_sha256"]
    assert report.column_values["condition"].fillna("").tolist() == [
        "HeLa / {literal}_control α", "HeLa / {literal}_control α", ""]
    output = annotations.apply_conditions(other, loaded, context)
    assert list(output.columns) == list(other.columns) + ["cell_type", "treatment", "condition"]
    assert output.index.tolist() == [4, 2, 4]
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(output[other.columns], other)


def test_manual_assignments_are_not_replayed_even_on_identical_rows(tmp_path):
    frame, source, _ = fixture_recipe(tmp_path)
    definition = annotations.new_definition(frame, source)
    token = annotations.table_identity(frame)[2][1]
    definition["conditions"] = [{"name": "manual only", "metadata_column": "filename",
                                  "manual_rows": [token], "include": "", "exclude": ""}]
    snapshot = copy.deepcopy(definition)
    path = tmp_path / "manual.json"
    assert annotations._save_schema(path, frame, definition, source) == 1
    assert token not in path.read_text()
    loaded, report = annotations._load_schema(path, frame, source)
    assert loaded["version"] == 1
    assert loaded["conditions"][0]["manual_rows"] == []
    assert report.unmatched == len(frame)
    assert definition == snapshot


def test_exact_values_excludes_regex_and_legacy_combine_roundtrip(tmp_path):
    frame = pd.DataFrame({"columnID": ["c1", "c10", "c2"], "rowID": ["r1", "r2", "r1"]})
    source = annotations.source_context()
    definition = annotations.new_definition(frame, source)
    definition.pop("column")
    definition.pop("conditions")
    definition.update(version=2, columns=[
        {"column": "genotype", "kind": "rules", "conditions": [
            {"name": "WT", "metadata_column": "columnID", "match_mode": "values",
             "include_values": ["c1", "c2"], "exclude_values": ["c2"]},
            {"name": "mutant", "metadata_column": "columnID", "include": "^c10$",
             "exclude": "^c1$"}]},
        {"column": "condition", "kind": "combine", "columns": ["genotype", "rowID"],
         "separator": "_"}])
    path = tmp_path / "legacy.json"
    # Copy-on-Write exposes read-only NumPy views from pandas predicates.
    with pd.option_context("mode.copy_on_write", True):
        annotations._save_schema(path, frame, definition, source)
        loaded, report = annotations._load_schema(path, frame, source)
    assert loaded["version"] == 2
    assert report.values.fillna("").tolist() == ["WT_r1", "mutant_r2", ""]


def test_new_table_overlaps_are_available_for_review_without_applying(tmp_path):
    frame = pd.DataFrame({"metadata": ["A", "B"]})
    source = annotations.source_context()
    definition = annotations.new_definition(frame, source)
    definition["conditions"] = [
        {"name": "one", "metadata_column": "metadata", "include": "A"},
        {"name": "two", "metadata_column": "metadata", "include": "B"}]
    path = tmp_path / "rules.json"
    annotations._save_schema(path, frame, definition, source)
    new_frame = pd.DataFrame({"metadata": ["AB"]})
    loaded, report = annotations._load_schema(path, new_frame, source)
    assert report.overlaps.tolist() == [0]
    with pytest.raises(annotations.AnnotationError, match="multiple conditions"):
        annotations.apply_conditions(new_frame, loaded, source)


@pytest.mark.parametrize("mode", ["contains", "not_contains", "equals"])
def test_shorthand_predicates_become_editable_criteria_without_changing_values(tmp_path, mode):
    frame, source, definition = fixture_recipe(tmp_path)
    definition["columns"] = [{"column": "condition", "kind": "rules", "conditions": [
        {"name": "match", "metadata_column": "filename", "match_mode": mode,
         "match_text": "HeLa", "exclude": "rep3"}]}]
    expected = annotations.preview(frame, definition, source)
    path = tmp_path / "shorthand.json"
    annotations._save_schema(path, frame, definition, source)
    loaded, report = annotations._load_schema(path, frame, source)
    rule = loaded["columns"][0]["conditions"][0]
    assert rule["criteria"][0]["operator"] == mode
    assert rule["match_mode"] == "regex"
    pd.testing.assert_series_equal(report.values, expected.values)


@pytest.mark.parametrize("field,value", [("include", True), ("exclude", 2),
                                        ("metadata_column", ["filename"]), ("match_mode", {}),
                                        ("match", False), ("match_text", None),
                                        ("include_values", "HeLa"), ("exclude_values", [{}])])
def test_malformed_rule_types_fail_before_save_or_load_evaluation(tmp_path, field, value):
    frame, source, definition = fixture_recipe(tmp_path)
    definition["columns"] = [{"column": "condition", "kind": "rules", "conditions": [
        {"name": "kept", "metadata_column": "filename", "include": "HeLa", field: value}]}]
    path = tmp_path / "existing.json"
    path.write_text("previous schema")
    with pytest.raises(annotations.AnnotationError):
        annotations._save_schema(path, frame, definition, source)
    assert path.read_text() == "previous schema"
    payload = {"format": "spacr.annotation-schema", "version": 1, "recipe_version": 3,
               "manual_rows": "excluded", "columns": definition["columns"]}
    path.write_text(json.dumps(payload))
    with pytest.raises(annotations.AnnotationError):
        annotations._load_schema(path, frame, source)


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_dormant_criteria_cannot_create_an_unloadable_schema(tmp_path, version):
    frame, source, _ = fixture_recipe(tmp_path)
    definition = annotations.new_definition(frame, source)
    conditions = [{"name": "legacy", "metadata_column": "filename", "include": "HeLa",
                   "criteria": [{"metadata_column": "filename", "operator": "equals", "value": "U2OS"}]}]
    if version == 1:
        definition["conditions"] = conditions
    else:
        definition.pop("column")
        definition.pop("conditions")
        definition.update(version=2, columns=[{"column": "condition", "kind": "rules", "conditions": conditions}])
    assert annotations.preview(frame, definition, source).values.fillna("").tolist() == ["legacy", ""]
    path = tmp_path / "prior.json"
    path.write_text("previous schema")
    with pytest.raises(annotations.AnnotationError, match="version 3"):
        annotations._save_schema(path, frame, definition, source)
    assert path.read_text() == "previous schema"


@pytest.mark.parametrize("mode", ["regex", "values"])
def test_inactive_exclusion_fields_cannot_change_imported_rule_meaning(tmp_path, mode):
    frame, source, definition = fixture_recipe(tmp_path)
    definition["columns"] = [{"column": "condition", "kind": "rules", "conditions": [
        {"name": "kept", "metadata_column": "filename", "match_mode": mode,
         "criteria": [{"metadata_column": "filename", "operator": "regex", "value": ".*"}],
         "exclude": "HeLa", "exclude_values": ["U2OS_rep3.tif"]}]}]
    expected = annotations.preview(frame, definition, source)
    path = tmp_path / "active.json"
    annotations._save_schema(path, frame, definition, source)
    loaded, report = annotations._load_schema(path, frame, source)
    rule = loaded["columns"][0]["conditions"][0]
    assert ("exclude_values" if mode == "regex" else "exclude") not in rule
    pd.testing.assert_series_equal(report.values, expected.values)


@pytest.mark.parametrize("alter", [
    lambda p: p.update(version=2),
    lambda p: p.update(version=True),
    lambda p: p.update(recipe_version=True),
    lambda p: p.update(recipe_version=2),
    lambda p: p.update(format="arbitrary-json"),
    lambda p: p.update(source={"path": "old.csv"}),
    lambda p: p.update(columns=[]),
    lambda p: p["columns"][0].update(column="filename"),
    lambda p: p["columns"][0].update(column="condition"),
    lambda p: p["columns"][0].update(metadata_column="missing"),
    lambda p: p["columns"][0].update(pattern="["),
    lambda p: p["columns"][0].update(group="missing"),
    lambda p: p["columns"][0].update(kind="execute"),
    lambda p: p["columns"][1]["conditions"][0].update(manual_rows=["opaque-old-row"]),
    lambda p: p["columns"][1]["conditions"][0].update(criteria=[]),
    lambda p: p["columns"][1]["conditions"][0]["criteria"][0].update(operator="python"),
    lambda p: p["columns"][0].update(metadata_column="condition"),
    lambda p: p["columns"][2]["parts"][1].update(code="untrusted"),
])
def test_invalid_schema_never_modifies_source(tmp_path, alter):
    frame, source, definition = fixture_recipe(tmp_path)
    path = tmp_path / "recipe.json"
    annotations._save_schema(path, frame, definition, source)
    payload = json.loads(path.read_text())
    alter(payload)
    path.write_text(json.dumps(payload))
    before = frame.copy(deep=True)
    with pytest.raises(annotations.AnnotationError):
        annotations._load_schema(path, frame, source)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("raw", [b"not json", b"\xff", b"[]", b"{\"version\":1,\"version\":1}",
                                  b'{"value":NaN}', b"[" * 1500 + b"]" * 1500])
def test_malformed_or_ambiguous_json_is_rejected(tmp_path, raw):
    frame, source, _ = fixture_recipe(tmp_path)
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(annotations.AnnotationError):
        annotations._load_schema(path, frame, source)


def test_size_limit_and_non_file_inputs(tmp_path, monkeypatch):
    frame, source, definition = fixture_recipe(tmp_path)
    path = tmp_path / "large.json"
    annotations._save_schema(path, frame, definition, source)
    before = path.read_bytes()
    monkeypatch.setattr(annotations, "_SCHEMA_MAX_BYTES", 20)
    with pytest.raises(annotations.AnnotationError, match="size limit"):
        annotations._load_schema(path, frame, source)
    with pytest.raises(annotations.AnnotationError, match="size limit"):
        annotations._save_schema(path, frame, definition, source)
    assert path.read_bytes() == before
    with pytest.raises(annotations.AnnotationError, match="JSON file"):
        annotations._load_schema(tmp_path, frame, source)


def test_save_refuses_source_path_and_hardlink(tmp_path):
    frame, source, definition = fixture_recipe(tmp_path)
    source_path = tmp_path / "source.csv"
    source_path.write_text("original measurements")
    alias = tmp_path / "alias.json"
    os.link(source_path, alias)
    for path in (source_path, alias):
        with pytest.raises(annotations.AnnotationError, match="separately"):
            annotations._save_schema(path, frame, definition, source)
    assert source_path.read_text() == alias.read_text() == "original measurements"


def test_failed_atomic_replace_preserves_existing_schema(tmp_path, monkeypatch):
    from spacr import run_journal

    frame, source, definition = fixture_recipe(tmp_path)
    path = tmp_path / "schema.json"
    path.write_text("previous schema")

    def fail(*_args):
        raise OSError("disk failure")

    monkeypatch.setattr(run_journal.os, "replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        annotations._save_schema(path, frame, definition, source)
    assert path.read_text() == "previous schema"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["schema.json"]
