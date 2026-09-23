"""The installed workflow routes and published handoffs cannot silently drift."""
import copy
import json
from pathlib import Path

import pytest

from tools import build_module_workflows as workflow


def test_all_live_routes_and_existing_io_contracts_are_mapped():
    from spacr import ports
    from spacr.validate import APP_FUNCTIONS

    data = workflow.load()
    workflow.validate(data)
    assert set(ports.PORTS) <= set(data["modules"]), (
        "every module with a declared input/output contract needs a workflow entry")
    assert set(APP_FUNCTIONS) <= set(data["modules"]), (
        "every pipeline entry point needs a workflow entry, including API-only calls")


@pytest.mark.parametrize("key", ["endodyogeny", "cellpose_masks", "simulation"])
def test_api_only_workflows_explain_the_callable_without_inventing_a_gui_route(key):
    from spacr.validate import APP_FUNCTIONS

    data = workflow.load()
    module = data["modules"][key]
    assert module["api_entry"] == APP_FUNCTIONS[key]
    assert module["home"] is None and module["parent"] is None
    prose = workflow.module_rst(data, key)
    lesson = workflow.lesson_document(data, "79_module_inputs_outputs")
    narration = next(s["narration"] for s in lesson["scenes"]
                     if s["visual"] == "module_" + key)
    for text in (prose, narration):
        assert APP_FUNCTIONS[key] in text
        assert "no Home tile or menu entry" in text
    assert "**Open:**" not in prose
    assert not narration.startswith(("Open ", "Find "))


def test_size_proxy_inference_and_simulation_keep_their_scientific_boundaries():
    data = workflow.load()
    assert "not measured volume or a parasite count" in data["modules"]["endodyogeny"]["guidance"]
    assert "multiple vacuoles in one cell are combined" in data["modules"]["endodyogeny"]["guidance"]
    assert "inference, not training" in data["modules"]["cellpose_masks"]["guidance"]
    assert "does not produce merged arrays or measurements.db" in data["artifacts"]["cellpose_tiff_masks"]["location"]
    assert "not measured experimental hits" in data["artifacts"]["simulation_summary"]["location"]


def test_optical_sample_starts_in_mask_ops_without_mandatory_generic_alignment():
    data = workflow.load()
    route = data["pathways"]["optical_screen"]
    assert route["home_app"] == "mask"
    assert route["steps"][0]["module"] == "ops"
    assert route["steps"][0]["after"] == []
    assert all(step["module"] != "align" for step in route["steps"])
    assert "Align & Stitch is optional" in route["note"]
    assert "cannot replace the sequencing cycles" in route["note"]


@pytest.mark.parametrize("mutation,match", [
    ("port", "declared input/output ports drift"),
    ("name", "display name drift"),
    ("parent", "nested route drift"),
    ("api", "Home API target drift"),
    ("module", "workflow modules differ"),
    ("dependency", "dependency precedes"),
    ("literal", "I/O contract moved"),
    ("artifact", "invalid inputs"),
    ("api_entry", "missing API entry point"),
    ("api_home", "invalid API-only route"),
])
def test_contract_changes_require_an_editorial_update(mutation, match):
    data = copy.deepcopy(workflow.load())
    if mutation == "port":
        data["modules"]["measure"]["declared_ports"]["produces"][0]["path"] = "wrong.db"
    elif mutation == "name":
        data["modules"]["measure"]["name"] = "Old label"
    elif mutation == "parent":
        data["modules"]["ops"]["parent"] = "measure"
    elif mutation == "api":
        data["modules"]["measure"]["api_module"] = "spacr.core"
    elif mutation == "module":
        data["modules"]["unmapped_new_module"] = copy.deepcopy(data["modules"]["measure"])
    elif mutation == "dependency":
        data["pathways"]["pooled_screen"]["steps"][0]["after"] = ["measure"]
    elif mutation == "literal":
        data["source_contracts"][0]["literals"].append("REMOVED_SCORE_COLUMN")
    elif mutation == "artifact":
        data["modules"]["measure"]["inputs"] = ["unknown"]
    elif mutation == "api_entry":
        data["modules"]["endodyogeny"]["api_entry"] = "spacr.submodules.removed_entry_point"
    elif mutation == "api_home":
        data["modules"]["endodyogeny"]["home"] = "toxoplasma"
    with pytest.raises(ValueError, match=match):
        workflow.validate(data)


def test_generated_api_and_tutorial_contracts_match_the_map():
    data = workflow.load()
    for relative, content in workflow.outputs(data).items():
        assert (workflow.ROOT / relative).read_text(encoding="utf-8") == content, relative
    module_keys = set(data["modules"])
    assert set(data["tutorials"]["79_module_inputs_outputs"]["modules"]) == module_keys
    for key, lesson in data["tutorials"].items():
        contract = json.loads((workflow.ROOT / f"tools/tutorials/workflows/{key}.json").read_text())
        script = json.loads((workflow.ROOT / f"tools/tutorials/lessons/{key}.json").read_text())
        assert script["app_key"] is None
        assert script["scenes"][0]["visual"] == "home"
        assert script["scenes"][-1]["visual"] == "home_summary"
        assert [scene["visual"] for scene in script["scenes"][1:-1]] == [
            "module_" + module for module in lesson["modules"]]
        for module in lesson["modules"]:
            assert contract["module_contracts"][module] == data["modules"][module]
        for pathway in lesson["pathways"]:
            assert contract["pathway_steps"][pathway] == data["pathways"][pathway]


def test_changed_storage_contract_reaches_api_and_narration_together():
    data = copy.deepcopy(workflow.load())
    data["artifacts"]["measurements"]["location"] = "replacement/objects.db"
    generated = workflow.outputs(data)
    for path in ("docs/source/workflows.rst",
                 "docs/source/_generated/module_workflows/spacr.measure.rst",
                 "tools/tutorials/lessons/79_module_inputs_outputs.json"):
        assert "replacement/objects.db" in generated[Path(path)]


def test_screen_script_keeps_both_regression_inputs_and_classifier_alternatives():
    lesson = workflow.lesson_document(workflow.load(), "78_spacr_screens")
    scenes = {scene["visual"]: scene for scene in lesson["scenes"]}
    regression = scenes["module_regression"]["narration"]
    assert "Classify to Regression" in regression
    assert "Map Barcodes to Regression" in regression
    classifier = scenes["module_classify_merged"]["narration"]
    assert "Computer Vision" in classifier
    assert "Tabular Machine Learning" in classifier


def test_reference_lesson_explains_every_artifact_location():
    data = workflow.load()
    script = workflow.lesson_document(data, "79_module_inputs_outputs")
    narration = " ".join(scene["narration"] for scene in script["scenes"])
    for artifact in data["artifacts"].values():
        assert artifact["location"] in narration


def test_both_screen_branches_feed_regression_and_labels_feed_classification():
    data = workflow.load()
    steps = {step["module"]: step for step in data["pathways"]["pooled_screen"]["steps"]}
    assert steps["regression"]["after"] == ["classify_merged", "map_barcodes"]
    assert steps["classify_merged"]["after"] == ["annotate"]
    edges = {(edge["from"], edge["to"]): edge for edge in data["connections"]}
    assert edges["mask", "measure"]["artifacts"] == ["merged"]
    assert edges["annotate", "classify_merged"]["artifacts"] == ["labels"]
    assert edges["map_barcodes", "regression"]["artifacts"] == ["barcode_counts"]
    assert "not a direct CSV" in edges["ops", "regression"]["handoff"]


def test_api_template_includes_the_generated_module_contract():
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(workflow.ROOT / "docs/source/_autoapi_templates"))
    env.globals["spacr_nested_helpers"] = {}
    env.filters["spacr_helper_docstring"] = lambda value, app: value
    workflow.prepare_jinja(env)
    rendered = env.get_template("python/module.rst").render(
        obj={"display": True, "id": "spacr.measure", "name": "spacr.measure",
             "docstring": "", "obj": {}, "subpackages": [], "submodules": [], "children": []},
        is_own_page=True)
    assert ".. include:: /_generated/module_workflows/spacr.measure.rst" in rendered


def test_stale_generated_prose_stops_the_docs_build(tmp_path):
    from jinja2 import Environment
    data = workflow.load()
    path = tmp_path / workflow.MAP_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data))
    for relative in {m["api_module"].replace(".", "/") + ".py" for m in data["modules"].values()}:
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("")
    for source in data["source_contracts"]:
        (tmp_path / source["path"]).write_text("\n".join(source["literals"]))
    for module in data["modules"].values():
        if module.get("api_entry"):
            file = tmp_path / (module["api_module"].replace(".", "/") + ".py")
            file.write_text((workflow.ROOT / file.relative_to(tmp_path)).read_text())
    with pytest.raises(ValueError, match="stale workflow artifact"):
        workflow.prepare_jinja(Environment(), root=tmp_path)


def test_installed_resource_is_declared_for_wheels_and_source_archives():
    assert "resources/module_workflows.json" in (workflow.ROOT / "setup.py").read_text()
    assert "include spacr/resources/module_workflows.json" in (workflow.ROOT / "MANIFEST.in").read_text()
