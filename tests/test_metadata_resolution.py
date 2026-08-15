from __future__ import annotations

import json

import pandas as pd
import pytest

from spacr.metadata_resolution import (
    MetadataDecision,
    MetadataResolutionRequired,
    clear_run_metadata_decisions,
    resolve_metadata_columns,
)


def test_known_aliases_never_prompt():
    frame = pd.DataFrame({"Plate": ["p"], "Row": ["r1"], "Column": ["c1"]})
    calls = []
    out = resolve_metadata_columns(
        frame, ["plateID", "rowID", "columnID"],
        prompt=lambda request: calls.append(request))
    assert list(out.frame[["plateID", "rowID", "columnID"]].iloc[0]) == [
        "p", "r1", "c1"]
    assert calls == []


def test_four_missing_columns_are_asked_once_and_remembered():
    clear_run_metadata_decisions()
    frame = pd.DataFrame({
        "plate_col": ["p"], "row_col": ["r1"],
        "col_col": ["c1"], "field_col": ["f1"],
    })
    calls = []

    def prompt(request):
        calls.append(request)
        assert set(request.missing) == {"plateID", "rowID", "columnID", "fieldID"}
        assert request.examples["plate_col"] == ("p",)
        return MetadataDecision(column_map={
            "plateID": "plate_col", "rowID": "row_col",
            "columnID": "col_col", "fieldID": "field_col",
        })

    first = resolve_metadata_columns(
        frame, ["plateID", "rowID", "columnID", "fieldID"],
        prompt=prompt, cache_key="plate-a")
    second = resolve_metadata_columns(
        frame, ["plateID", "rowID", "columnID", "fieldID"],
        prompt=prompt, cache_key="plate-a")
    assert calls and len(calls) == 1
    assert list(first.frame.columns) == list(second.frame.columns)


def test_well_column_derives_real_row_and_column_ids():
    frame = pd.DataFrame({"well": ["A01", "AA01"]})
    result = resolve_metadata_columns(
        frame, ["rowID", "columnID"], well_column="well")
    assert result.frame[["rowID", "columnID"]].values.tolist() == [
        ["r1", "c1"], ["r27", "c1"]]
    assert result.derived_from_well == "well"


def test_unparseable_conditions_get_distinct_audited_pseudo_wells(tmp_path):
    frame = pd.DataFrame({"condition": ["A-B", "A B", "A-B", 1, "1"]})
    audit = tmp_path / "metadata_map.json"
    result = resolve_metadata_columns(
        frame, ["rowID", "columnID"],
        pseudo_source="condition", allow_pseudo=True,
        save_path=str(audit))
    wells = list(zip(result.frame.rowID, result.frame.columnID))
    assert wells[0] != wells[1]
    assert wells[0] == wells[2]
    assert wells[3] != wells[4]
    assert len(set(wells)) == 4
    payload = json.loads(audit.read_text())
    assert len(payload["pseudo_map"]) == 4


def test_headless_missing_metadata_never_prompts_or_hangs():
    frame = pd.DataFrame({"score": [1.0]})
    with pytest.raises(MetadataResolutionRequired) as exc:
        resolve_metadata_columns(frame, ["plateID", "columnID"])
    message = str(exc.value)
    assert "plateID" in message and "columnID" in message
    assert "score" in message and "metadata_column_map" in message
    assert "headless" in message


def test_mapping_uses_case_folded_collision_guard():
    frame = pd.DataFrame({"candidate": [1], "ROWid": [2]})
    with pytest.raises(ValueError, match="collision"):
        resolve_metadata_columns(
            frame, ["rowID"], column_map={"rowID": "candidate"})


def test_request_offers_an_overridable_guess():
    from spacr.metadata_resolution import build_metadata_request

    request = build_metadata_request(
        pd.DataFrame({"plate_number": [7], "unrelated": [1]}), ["plateID"])
    assert request.guesses["plateID"] == "plate_number"


def test_object_writer_boundary_uses_the_shared_resolver():
    from spacr.schema import validate_object_table_frame

    frame = pd.DataFrame({
        "plate_number": ["p1"],
        "well": ["A01"],
        "field_number": ["f1"],
        "key": ["p1_r1_c1_f1"],
        "label": [1],
        "file_name": ["p1_A01_f1.tif"],
        "path_name": ["/tmp"],
    })
    resolved = validate_object_table_frame(
        frame,
        "cell",
        metadata_column_map={
            "plateID": "plate_number",
            "fieldID": "field_number",
            "prcf": "key",
            "object_label": "label",
        },
        metadata_well_column="well",
    )
    assert resolved.loc[0, ["plateID", "rowID", "columnID"]].tolist() == [
        "p1", "r1", "c1"]
