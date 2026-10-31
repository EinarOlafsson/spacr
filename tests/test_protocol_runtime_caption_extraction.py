"""Protocol prose is translated while workflow identities remain stable."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_i18n_catalogs as builder


def test_protocol_caption_sources_include_display_prose_without_machine_ids(tmp_path, monkeypatch):
    path = tmp_path / "spacr" / "resources" / "module_workflows.json"
    path.parent.mkdir(parents=True)
    legacy = {"pathways": {}, "modules": {}, "artifacts": {}, "connections": []}
    path.write_text(json.dumps(legacy))
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    assert builder._workflow_ui_sources() == set()
    workflow = dict(legacy, protocols={
        "protocol_id": {
            "title": "Inspect segmented cells",
            "summary": "Check the masks before measuring.",
            "pathway": "Do not translate the pathway identity",
            "inputs": ["Do not translate artifact identities"],
            "checkpoint": "Confirm that cell borders match the image.",
            "steps": [{"module": "Do not translate module identities",
                       "action": "Open a representative field.",
                       "after": ["Do not translate predecessor identities"]}],
        },
        "without_checkpoint": {"title": "Inspect a second field",
                               "summary": "Keep the original image unchanged.",
                               "pathway": "image_analysis", "inputs": [], "steps": []},
    }, protocol_choices={"data": {"data_id": "Images and masks"},
                         "goal": {"goal_id": "Measure cell features"},
                         "stage": {"stage_id": "Ready to measure"},
                         "details": {"detail_id": "One plate"},
                         "internal": {"ignore_key": "Do not expose machine metadata"}},
        protocol_tree={"id": "Hidden tree identity", "title": "Choose your starting material",
                       "description": "Select data that you already have.",
                       "children": [{"id": "Hidden branch identity",
                                     "title": "Microscopy images",
                                     "description": "Browse image analysis routes.",
                                     "children": [{"id": "Hidden leaf identity",
                                                   "title": "Review a measured field",
                                                   "description": "Inspect the measurement table.",
                                                   "protocol": "Hidden protocol identity",
                                                   "start_module": "Hidden module identity"}]}]},
        protocol_routes=[{"data": "data_id", "goal": "goal_id", "stage": "stage_id",
                          "details": "detail_id", "protocol": "protocol_id"}])
    path.write_text(json.dumps(workflow))
    assert builder._workflow_ui_sources() == {
        "Inspect segmented cells", "Check the masks before measuring.",
        "Confirm that cell borders match the image.", "Open a representative field.",
        "Inspect a second field", "Keep the original image unchanged.",
        "Images and masks", "Measure cell features", "Ready to measure", "One plate",
        "Choose your starting material", "Select data that you already have.",
        "Microscopy images", "Browse image analysis routes.",
        "Review a measured field", "Inspect the measurement table.",
    }


def test_four_family_editorial_wizard_is_available_to_runtime_translation():
    """Actual selectable wizard explanations enter localization without machine IDs."""
    workflow = json.loads((builder.ROOT / "spacr/resources/module_workflows.json").read_text())
    found = builder._workflow_ui_sources()
    for row in workflow['protocol_wizard']['families'].values():
        assert row['label'] in found and row['explanation'] in found
        for format_ in row['formats'].values():
            assert format_['label'] in found and format_['explanation'] in found
    assert set(workflow['protocol_wizard']['requirements'].values()) <= found
    assert not set(workflow['protocol_wizard']['families']).intersection(found)
