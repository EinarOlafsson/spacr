"""Small helpers of the isolated segmentation backends, at their edges.

What spaCR's torch build is (read from torch/version.py without importing
it), the shape of an Omnipose checkpoint, the worker output relay dropping a
listener that fails or has gone, CellProfiler's Java machine started once
per worker, and the restoration plan's refusals before any worker starts.
"""
from __future__ import annotations

import gc
import importlib.util
import sys
import types

import pytest

from spacr import _segmentation_backends as SB


def _fake_torch(monkeypatch, tmp_path, text):
    folder = tmp_path / "torch"
    folder.mkdir()
    (folder / "version.py").write_text(text, encoding="utf-8")
    spec = types.SimpleNamespace(submodule_search_locations=[
        str(tmp_path / "missing"), str(folder)])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)


@pytest.mark.parametrize("text,expected", [
    ("__version__ = '2.6.0+cu124'\n", "2.6.0+cu124"),
    ("__version__ = '2.6.0'\ncuda: Optional[str] = '12.4'\n", "2.6.0+cu124"),
    ("__version__ = '2.6.0'\ncuda = None\n", ""),
])
def test_the_torch_build_is_read_from_its_version_file(monkeypatch, tmp_path,
                                                       text, expected):
    _fake_torch(monkeypatch, tmp_path, text)
    assert SB._built_torch_version() == expected


def test_no_torch_or_an_unfindable_one_has_no_build(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert SB._built_torch_version() == ""

    def broken(name):
        raise ValueError("torch.__spec__ is None")

    monkeypatch.setattr(importlib.util, "find_spec", broken)
    assert SB._built_torch_version() == ""


def test_an_omnipose_checkpoint_states_its_channels_and_classes(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "omni.pth"
    torch.save({"state_dict": {"in.weight": torch.zeros(32, 2, 3, 3),
                               "bias": torch.zeros(32),
                               "out.weight": torch.zeros(4, 32, 1, 1)}},
               path)
    assert SB._omnipose_shape(str(path)) == (2, 3)
    empty = tmp_path / "empty.pth"
    torch.save({"bias": torch.zeros(3)}, empty)
    with pytest.raises(ValueError, match="holds no Omnipose network"):
        SB._omnipose_shape(str(empty))


def test_a_listener_that_fails_or_has_gone_is_dropped(monkeypatch):
    heard = []

    class _Console:
        def hear(self, label, line):
            heard.append((label, line))

    def failing(label, line):
        raise RuntimeError("the console was closed")

    kept = _Console()
    gone = _Console()
    monkeypatch.setattr(SB, "_OUTPUT_LISTENERS", [])
    SB._listen_to_workers(kept.hear)
    SB._listen_to_workers(gone.hear)
    SB._listen_to_workers(failing)
    del gone
    gc.collect()
    SB._tell_listeners("cellpose3", "loading weights")
    assert heard == [("cellpose3", "loading weights")]
    assert len(SB._OUTPUT_LISTENERS) == 1


def test_cellprofilers_java_starts_once_per_worker(monkeypatch):
    calls = []
    preferences = types.ModuleType("cellprofiler_core.preferences")
    preferences.set_headless = lambda: calls.append("headless")
    preferences.set_allow_schema_write = lambda on: calls.append(("schema", on))
    java = types.ModuleType("cellprofiler_core.utilities.java")
    java.start_java = lambda: calls.append("java")
    java.stop_java = lambda: calls.append("stop")
    for name, module in {
            "cellprofiler_core": types.ModuleType("cellprofiler_core"),
            "cellprofiler_core.preferences": preferences,
            "cellprofiler_core.utilities":
                types.ModuleType("cellprofiler_core.utilities"),
            "cellprofiler_core.utilities.java": java}.items():
        monkeypatch.setitem(sys.modules, name, module)
    registered = []
    monkeypatch.setattr(SB.atexit, "register", registered.append)
    adapters = {}
    assert SB._cellprofiler_started(adapters) is preferences
    assert SB._cellprofiler_started(adapters) is preferences
    assert calls == ["headless", ("schema", False), "java"]
    assert registered == [java.stop_java]


@pytest.mark.parametrize("model,diameter,match", [
    ("not-a-model", 30, "unsupported same-grid restoration model"),
    (None, float("nan"), "finite and positive"),
    (None, 0, "finite and positive"),
])
def test_a_restoration_plan_refuses_before_any_worker(model, diameter, match):
    model = model or SB._RESTORATION_MODELS[0]
    with pytest.raises(ValueError, match=match):
        SB._restoration_plan(model, diameter, worker_for=lambda *a: None)


class _Measurements:
    """What CellProfiler's ``Pipeline.run()`` hands back, for two image sets."""

    def __init__(self, status="Complete"):
        self.status = status

    def has_feature(self, table, feature):
        return self.status is not None and (table, feature) == (
            "Experiment", "Exit_Status")

    def get_experiment_measurement(self, feature):
        return self.status

    def get_image_numbers(self):
        return [1, 2]

    def get_feature_names(self, table):
        return {"Image": ["FileName_DNA", "Count_Nuclei", "ObjectsFileName_N"],
                "Nuclei": ["Number_Object_Number", "AreaShape_Area",
                           "Label_Text", "Short_Column"],
                "Empty": ["Number_Object_Number", "AreaShape_Area"]}[table]

    def get_object_names(self):
        return ["Image", "Experiment", "Nuclei", "Empty"]

    def get_measurement(self, table, feature, number):
        if table == "Image":
            return f"{feature}-{number}"
        if feature == "Number_Object_Number":
            return [1, 2] if table == "Nuclei" and number == 1 else []
        return {"AreaShape_Area": [10.0, 20.0], "Label_Text": ["a", "b"],
                "Short_Column": [5.0]}[feature]


def _cellprofiler(monkeypatch, measurements):
    loaded = []

    class _Pipeline:
        def load(self, path):
            loaded.append(("load", path))

        def add_pathnames_to_file_list(self, files):
            loaded.append(("files", files))

        def modules(self):
            return []

        def run(self):
            return measurements

    pipeline = types.ModuleType("cellprofiler_core.pipeline")
    pipeline.Pipeline = _Pipeline
    monkeypatch.setitem(sys.modules, "cellprofiler_core",
                        types.ModuleType("cellprofiler_core"))
    monkeypatch.setitem(sys.modules, "cellprofiler_core.pipeline", pipeline)
    folders = []
    preferences = types.SimpleNamespace(
        set_default_output_directory=lambda path: folders.append(path),
        set_default_image_directory=lambda path: folders.append(path))
    return {"cellprofiler": preferences}, loaded, folders


def test_a_cellprofiler_run_keeps_only_numeric_object_features(
        monkeypatch, tmp_path):
    import numpy as np

    adapters, loaded, folders = _cellprofiler(monkeypatch, _Measurements())
    pipe = tmp_path / "count.cppipe"
    pipe.write_text("CellProfiler Pipeline")
    image = tmp_path / "images" / "A01.tif"
    output = tmp_path / "out"
    result = SB._worker_run_cellprofiler(
        {"pipeline": pipe, "files": [image], "output": output}, adapters)
    assert result["image_sets"] == 2
    assert result["images"]["1"] == ["FileName_DNA-1", "ObjectsFileName_N-1"]
    assert set(result["objects"]) == {"Nuclei"}
    nuclei = result["objects"]["Nuclei"]
    assert nuclei["columns"] == ["ImageNumber", "ObjectNumber",
                                 "AreaShape_Area", "Short_Column"]
    table = np.load(nuclei["path"])
    assert table.shape == (2, 4)
    assert table[:, 2].tolist() == [10.0, 20.0]
    assert np.isnan(table[:, 3]).all()
    assert loaded == [("load", str(pipe)), ("files", [str(image)])]
    assert folders == [str(output), str(image.parent)]


def test_a_cellprofiler_run_without_an_exit_status_is_read(
        monkeypatch, tmp_path):
    adapters, _loaded, _folders = _cellprofiler(monkeypatch,
                                                _Measurements(status=None))
    pipe = tmp_path / "count.cppipe"
    pipe.write_text("CellProfiler Pipeline")
    result = SB._worker_run_cellprofiler(
        {"pipeline": pipe, "files": [tmp_path / "a.tif"],
         "output": tmp_path / "out"}, adapters)
    assert result["image_sets"] == 2


@pytest.mark.parametrize("case,error,match", [
    ("no-pipeline", FileNotFoundError, "no CellProfiler pipeline at"),
    ("no-files", ValueError, "no images were given"),
    ("no-measurements", RuntimeError, "matched no image set"),
    ("aborted", RuntimeError, "pipeline stopped: Aborted"),
])
def test_a_cellprofiler_run_that_cannot_finish_says_why(
        monkeypatch, tmp_path, case, error, match):
    measurements = {"no-measurements": None,
                    "aborted": _Measurements(status="Aborted")}.get(
        case, _Measurements())
    adapters, _loaded, _folders = _cellprofiler(monkeypatch, measurements)
    pipe = tmp_path / "count.cppipe"
    if case != "no-pipeline":
        pipe.write_text("CellProfiler Pipeline")
    files = [] if case == "no-files" else [tmp_path / "a.tif"]
    with pytest.raises(error, match=match):
        SB._worker_run_cellprofiler(
            {"pipeline": pipe, "files": files, "output": tmp_path / "out"},
            adapters)
