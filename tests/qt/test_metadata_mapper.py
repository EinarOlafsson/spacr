from __future__ import annotations

import pandas as pd

from spacr.metadata_resolution import build_metadata_request
from spacr.qt.widgets.metadata_mapper import MetadataColumnDialog


def test_dialog_resolves_multiple_columns_in_one_answer(qapp, tmp_path):
    frame = pd.DataFrame({"plate": ["p1"], "well": ["A01"], "value": [3]})
    request = build_metadata_request(
        frame, ["plateID", "rowID", "columnID", "fieldID"])
    dialog = MetadataColumnDialog(request)

    dialog._selectors["plateID"].setCurrentText("plate")
    dialog._selectors["fieldID"].setCurrentText("value")
    dialog.well_selector.setCurrentText("well")
    dialog.save_mapping.setChecked(True)
    dialog.save_path.setText(str(tmp_path / "map.json"))

    decision = dialog.decision()
    assert decision.column_map == {"plateID": "plate", "fieldID": "value"}
    assert decision.well_column == "well"
    assert decision.save_path.endswith("map.json")


def test_dialog_preselects_guess_and_previews_well_mapping(qapp):
    frame = pd.DataFrame({"plate_number": [1], "well": ["A01"]})
    dialog = MetadataColumnDialog(
        build_metadata_request(frame, ["plateID", "rowID", "columnID"]))
    assert dialog._selectors["plateID"].currentText() == "plate_number"
    dialog.well_selector.setCurrentText("well")
    assert "A01 → r1/c1" in dialog.well_preview.text()


def test_dialog_can_choose_injective_pseudo_source(qapp):
    frame = pd.DataFrame({"folder": ["control", "treated"]})
    dialog = MetadataColumnDialog(
        build_metadata_request(frame, ["rowID", "columnID"]))
    dialog.pseudo_selector.setCurrentText("folder")
    decision = dialog.decision()
    assert decision.allow_pseudo is True
    assert decision.pseudo_source == "folder"
