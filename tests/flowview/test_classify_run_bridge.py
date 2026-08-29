from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from spacr import classify, flowview
from spacr.flowview.classify_blueprint import CLASSIFY_NODE_IDS


@pytest.fixture(autouse=True)
def _restore_trace_state():
    previous_collector = flowview.get_collector()
    previous_enabled = flowview.is_enabled()
    yield
    flowview.enable(previous_collector)
    if not previous_enabled:
        flowview.disable()


def test_classify_import_and_disabled_dispatch_do_not_load_flowview():
    script = """
import os
import sys
import types

os.environ.pop("SPACR_FLOWVIEW", None)
import spacr.classify as classify
assert not any(name.startswith("spacr.flowview") for name in sys.modules)

normalizers = types.ModuleType("spacr.classify_classes")
normalizers.normalize_settings = dict
sys.modules[normalizers.__name__] = normalizers
basis = types.ModuleType("spacr.training_basis")
basis.normalize_settings = dict
sys.modules[basis.__name__] = basis
crops = types.ModuleType("spacr.crop_source")
crops.validate = lambda settings: None
sys.modules[crops.__name__] = crops
pipeline = types.ModuleType("spacr.deep_spacr")
pipeline.deep_spacr = lambda settings: ("same-result", settings)
sys.modules[pipeline.__name__] = pipeline

result, settings = classify.classify({"classifier_family": "cv"})
assert result == "same-result"
assert settings["classifier_family"] == "cv"
assert not any(name.startswith("spacr.flowview") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "SPACR_FLOWVIEW": "0"},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr


def test_each_enabled_classify_call_installs_a_fresh_eight_node_run(monkeypatch):
    import spacr.deep_spacr as deep_spacr

    sentinel = object()
    monkeypatch.setattr(deep_spacr, "deep_spacr", lambda settings: sentinel)
    flowview.enable()

    assert classify.classify({"classifier_family": "cv"}) is sentinel
    first = flowview.get_collector()
    first_graph = first.snapshot()

    assert classify.classify({"classifier_family": "cv"}) is sentinel
    second = flowview.get_collector()
    second_graph = second.snapshot()

    assert first is not second
    assert first_graph.run_id != second_graph.run_id
    assert tuple(first_graph.nodes) == CLASSIFY_NODE_IDS
    assert tuple(second_graph.nodes) == CLASSIFY_NODE_IDS
    assert len(second_graph.nodes) == 8


def test_trace_setup_fault_cannot_change_pipeline_result_or_exception(monkeypatch):
    import spacr.deep_spacr as deep_spacr

    def broken_trace(_settings):
        raise RuntimeError("trace setup failed")

    monkeypatch.setattr(classify, "_begin_flowview_run", broken_trace)
    sentinel_result = object()
    monkeypatch.setattr(
        deep_spacr,
        "deep_spacr",
        lambda settings: sentinel_result,
    )
    assert classify.classify({"classifier_family": "cv"}) is sentinel_result

    sentinel_error = LookupError("science failed")

    def failed_pipeline(_settings):
        raise sentinel_error

    monkeypatch.setattr(deep_spacr, "deep_spacr", failed_pipeline)
    with pytest.raises(LookupError) as caught:
        classify.classify({"classifier_family": "cv"})
    assert caught.value is sentinel_error
