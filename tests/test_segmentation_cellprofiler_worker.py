"""The CellProfiler worker of :mod:`spacr._segmentation_backends`, against a fake CellProfiler.

``_worker_run_cellprofiler`` loads a pipeline, hands it the images, and keeps
only numeric per-object features, each table led by ``ImageNumber`` and
``ObjectNumber``.  A run that cannot finish says why, before or after the
pipeline ran.  CellProfiler is not installed here, so ``cellprofiler_core`` is
a stand-in that records what it was asked to do.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from spacr import _segmentation_backends as SB


class _Measurements:
    """What CellProfiler's ``Pipeline.run()`` hands back, for two image sets."""

    def __init__(self, status="Complete"):
        self.status = status

    def has_feature(self, table, feature):
        return self.status is not None and (table, feature) == ("Experiment", "Exit_Status")

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
    """Install a fake ``cellprofiler_core``; return adapters and the call records."""
    calls = []

    class _Pipeline:
        def load(self, path):
            calls.append(("load", path))

        def add_pathnames_to_file_list(self, files):
            calls.append(("files", files))

        def modules(self):
            return []

        def run(self):
            return measurements

    pipeline = types.ModuleType("cellprofiler_core.pipeline")
    pipeline.Pipeline = _Pipeline
    monkeypatch.setitem(sys.modules, "cellprofiler_core", types.ModuleType("cellprofiler_core"))
    monkeypatch.setitem(sys.modules, "cellprofiler_core.pipeline", pipeline)
    preferences = types.SimpleNamespace(
        set_default_output_directory=lambda path: calls.append(("output", path)),
        set_default_image_directory=lambda path: calls.append(("images", path)))
    return {"cellprofiler": preferences}, calls


def test_a_cellprofiler_run_keeps_only_numeric_object_features(monkeypatch, tmp_path, subtests):
    """A complete (or status-less) run returns exact image names and a numeric Nuclei table only."""
    pipe = tmp_path / "count.cppipe"
    pipe.write_text("CellProfiler Pipeline", encoding="utf-8")
    image = tmp_path / "images" / "A01.tif"
    for case_id, status in [("complete", "Complete"), ("no_exit_status", None)]:
        with subtests.test(msg=case_id):
            adapters, calls = _cellprofiler(monkeypatch, _Measurements(status))
            output = tmp_path / case_id
            result = SB._worker_run_cellprofiler(
                {"pipeline": pipe, "files": [image], "output": output}, adapters)
            assert result["image_sets"] == 2
            assert result["images"] == {
                "1": ["FileName_DNA-1", "ObjectsFileName_N-1"],
                "2": ["FileName_DNA-2", "ObjectsFileName_N-2"]}
            assert list(result["objects"]) == ["Nuclei"]
            nuclei = result["objects"]["Nuclei"]
            assert nuclei == {"columns": ["ImageNumber", "ObjectNumber",
                                          "AreaShape_Area", "Short_Column"],
                              "path": str(output / "objects_2.npy")}
            np.testing.assert_array_equal(
                np.load(nuclei["path"]),
                np.array([[1.0, 1.0, 10.0, np.nan], [1.0, 2.0, 20.0, np.nan]]))
            assert calls == [("output", str(output)), ("images", str(image.parent)),
                             ("load", str(pipe)), ("files", [str(image)])]


def test_a_cellprofiler_run_that_cannot_finish_says_why(monkeypatch, tmp_path, subtests):
    """Each refusal raises its own exception type with the exact sentence."""
    pipe = tmp_path / "count.cppipe"
    missing = tmp_path / "absent.cppipe"
    pipe.write_text("CellProfiler Pipeline", encoding="utf-8")
    cases = [
        ("no_pipeline", missing, [tmp_path / "a.tif"], _Measurements(),
         FileNotFoundError, f"no CellProfiler pipeline at {missing}"),
        ("no_files", pipe, [], _Measurements(),
         ValueError, "no images were given to the CellProfiler pipeline"),
        ("no_measurements", pipe, [tmp_path / "a.tif"], None, RuntimeError,
         "the CellProfiler pipeline produced no measurements; its input modules "
         "matched no image set among the files spaCR gave it"),
        ("aborted", pipe, [tmp_path / "a.tif"], _Measurements("Aborted"),
         RuntimeError, "the CellProfiler pipeline stopped: Aborted"),
    ]
    for case_id, pipeline, files, measurements, error, message in cases:
        with subtests.test(msg=case_id):
            adapters, _calls = _cellprofiler(monkeypatch, measurements)
            with pytest.raises(error) as caught:
                SB._worker_run_cellprofiler(
                    {"pipeline": pipeline, "files": files,
                     "output": tmp_path / "out"}, adapters)
            assert type(caught.value) is error
            assert str(caught.value) == message
