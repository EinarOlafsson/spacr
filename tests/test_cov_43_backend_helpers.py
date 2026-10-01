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
