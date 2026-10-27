"""Easy navigation retains real routes without promoting data matches to handoffs."""

import copy
import sys

from spacr.qt.widgets.workflow_diagram import _easy_workflow_data, workflow_map


APPS = (("mask", "Mask", "Generate masks", "Images"),
        ("measure", "Measure", "Measure objects", "Analysis"),
        ("make_masks", "Make Masks", "Review masks", "Images"),
        ("unmapped_plugin", "Plugin", "Registered extension", "Analysis"))


def test_easy_graph_uses_shipped_handoffs_and_keeps_unmapped_registered_tools():
    original = workflow_map()
    before = copy.deepcopy(original)
    app_module = sys.modules.get("spacr.qt.app")
    result = _easy_workflow_data(APPS, {row[0] for row in APPS}, data=original,
                                 mode_labels={("mask", "measure"): ("3D", "Time")})
    assert set(result["modules"]) == {row[0] for row in APPS}
    documented = next(edge for edge in result["easy_connections"]
                      if (edge["from"], edge["to"]) == ("mask", "measure"))
    assert documented["kind"] == "documented" and documented["strength"] == 2
    assert documented["handoff"] == next(edge["handoff"] for edge in before["connections"]
                                         if (edge["from"], edge["to"]) == ("mask", "measure"))
    assert documented["mode_labels"] == ["3D", "Time"]
    assert not {"3D", "Time", "concatenate"} & result["modules"].keys()
    assert result["modules"]["unmapped_plugin"]["inputs"] == []
    assert not any("unmapped_plugin" in (edge["from"], edge["to"])
                   for edge in result["easy_connections"])
    assert original == before
    assert sys.modules.get("spacr.qt.app") is app_module


def test_easy_graph_respects_hidden_unavailable_and_folded_route_gates():
    rows = (*APPS, ("ops", "OPS", "Optical pooled screening", "Images"))
    visible = {row[0] for row in rows} | {"train_cellpose"}
    result = _easy_workflow_data(rows, visible,
                                 {"mask": ("ops",), "make_masks": ("train_cellpose",)},
                                 folded_metadata={"train_cellpose": ("Train Cellpose", "Train a model")},
                                 available_keys=visible - {"ops"})
    assert "ops" not in result["modules"]
    assert result["modules"]["train_cellpose"]["host_key"] == "make_masks"
    assert result["modules"]["train_cellpose"]["requested_key"] == "train_cellpose"
    hidden = _easy_workflow_data(rows, visible - {"make_masks"},
                                {"make_masks": ("train_cellpose",)},
                                folded_metadata={"train_cellpose": ("Train Cellpose", "Train a model")})
    assert "train_cellpose" not in hidden["modules"]
    hidden_host = _easy_workflow_data(rows, visible - {"mask"}, {"mask": ("ops",)})
    assert "ops" not in hidden_host["modules"]
    assert all(edge["from"] in hidden["modules"] and edge["to"] in hidden["modules"]
               for edge in hidden["easy_connections"])


def test_easy_graph_matching_types_do_not_become_documented_transfers():
    result = _easy_workflow_data(APPS, {row[0] for row in APPS})
    edge = next(edge for edge in result["easy_connections"]
                if (edge["from"], edge["to"]) == ("mask", "make_masks"))
    assert edge["kind"] == "compatible" and edge["strength"] == 1
    assert edge["handoff"] == "" and edge["mode_labels"] == []


def test_easy_graph_owner_subtools_have_explicit_routes_and_no_invented_io():
    tool = dict(key="overview", host_key="mask", route="overview",
                name="Overview", guidance="Field-of-view overview")
    result = _easy_workflow_data(APPS, {"mask", "measure"}, subtools=(tool,))
    overview = result["modules"]["overview"]
    assert overview["route"] == "overview" and overview["host_key"] == "mask"
    assert overview["inputs"] == overview["outputs"] == []
    assert not any("overview" in (edge["from"], edge["to"])
                   for edge in result["easy_connections"])
    assert "overview" not in _easy_workflow_data(APPS, {"measure"}, subtools=(tool,))["modules"]
    assert "inputs" not in tool
