"""Reading old checkpoints and refusing the ones that cannot be read.

New spaCR artifacts are versioned mappings; older releases wrote whole
``nn.Module`` objects, and other tools write bare ``state_dict`` files. All
three are read here, and every shape that is none of them has to be refused
with a sentence naming the file rather than an attribute error two frames
later.
"""
from __future__ import annotations

import builtins
import os

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from spacr import torch_artifacts as TA  # noqa: E402


def _tiny_model():
    """A two-parameter module whose state dict is cheap to save and load."""
    model = nn.Sequential(nn.Linear(4, 2))
    model.model_name = "tiny"
    model.num_classes = 2
    return model


def test_a_missing_torchvision_is_recorded_as_unavailable(monkeypatch):
    """The artifact records what it could see, not what it wished were there."""
    real_import = builtins.__import__

    def no_torchvision(name, *args, **kwargs):
        if name == "torchvision":
            raise ImportError("No module named 'torchvision'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torchvision)
    versions = TA.dependency_versions()
    assert versions["torchvision"] == "unavailable"
    assert versions["torch"] == torch.__version__


def test_no_rng_state_at_all_is_a_no_op():
    """An artifact written with ``include_rng=False`` restores nothing."""
    before = random_snapshot = torch.get_rng_state().clone()
    TA.restore_rng_state(None)
    TA.restore_rng_state({})
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(random_snapshot, before)


def test_the_gpu_generators_travel_with_the_artifact(monkeypatch):
    """Where CUDA exists its generators are captured and put back too."""
    restored = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_rng_state_all():
            return ["device-0-state"]

        @staticmethod
        def set_rng_state_all(state):
            restored.append(state)

    monkeypatch.setattr(torch, "cuda", FakeCuda(), raising=False)
    state = TA.capture_rng_state()
    assert state["torch_cuda"] == ["device-0-state"]
    TA.restore_rng_state(state)
    assert restored == [["device-0-state"]]


def test_a_failed_save_leaves_nothing_beside_the_target(tmp_path, monkeypatch):
    """The temporary is what makes the save atomic; it must not survive."""
    target = tmp_path / "model.pth"

    def failing_replace(src, dst):
        os.unlink(src)
        raise OSError("the volume filled up")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="filled up"):
        TA.atomic_torch_save({"anything": 1}, str(target))
    assert list(tmp_path.iterdir()) == []


def test_a_saved_artifact_round_trips_into_a_model_that_was_supplied(tmp_path):
    """The module passed in is loaded in place and handed back."""
    path = str(tmp_path / "model.pth")
    TA.save_model_artifact(_tiny_model(), path)

    target = _tiny_model()
    loaded, payload = TA.load_model_artifact(path, model=target)
    assert loaded is target
    assert payload["legacy"] is False
    assert payload["model_config"]["model_name"] == "tiny"


def test_an_artifact_from_a_future_spacr_is_refused_by_version(tmp_path):
    """A version this installation does not support is named, not guessed at."""
    path = str(tmp_path / "future.pth")
    payload = TA.make_model_artifact(_tiny_model())
    payload["artifact_version"] = TA.ARTIFACT_VERSION + 7
    TA.atomic_torch_save(payload, path)

    with pytest.raises(ValueError) as excinfo:
        TA.load_model_artifact(path, model=_tiny_model())
    assert f"version {TA.ARTIFACT_VERSION + 7}" in str(excinfo.value)
    assert f"supports version {TA.ARTIFACT_VERSION}" in str(excinfo.value)


def test_a_bare_state_dict_checkpoint_is_read_as_a_legacy_artifact(tmp_path):
    """``{'state_dict': ...}`` is the shape other training tools write."""
    path = str(tmp_path / "other_tool.pth")
    TA.atomic_torch_save({"state_dict": _tiny_model().state_dict(),
                          "num_classes": 3, "image_size": 128}, path)

    target = _tiny_model()
    loaded, payload = TA.load_model_artifact(path, model=target)
    assert loaded is target
    assert payload["legacy"] is True
    assert payload["training_state"] == {}
    assert payload["model_config"]["num_classes"] == 3
    assert payload["model_config"]["image_size"] == 128


def test_a_pre_versioned_model_checkpoint_is_read_as_a_legacy_artifact(tmp_path):
    """``{'model': state_dict}`` was spaCR's own shape before versioning."""
    path = str(tmp_path / "old_spacr.pth")
    TA.atomic_torch_save({"model": _tiny_model().state_dict(),
                          "model_name": "tiny"}, path)
    loaded, payload = TA.load_model_artifact(path, model=_tiny_model())
    assert payload["legacy"] is True
    assert payload["model_config"]["model_name"] == "tiny"
    assert loaded is not None


def test_a_mapping_with_no_model_state_anywhere_is_refused(tmp_path):
    """A dict of hyperparameters is not a checkpoint, and the path is named."""
    path = str(tmp_path / "settings_only.pth")
    TA.atomic_torch_save({"learning_rate": 0.001, "epochs": 20}, path)
    with pytest.raises(ValueError) as excinfo:
        TA.load_model_artifact(path, model=_tiny_model())
    assert "no model state dictionary was found" in str(excinfo.value)
    assert path in str(excinfo.value)


def test_an_artifact_whose_model_state_is_not_a_mapping_is_refused(tmp_path):
    """The key being present is not the same as it holding weights."""
    path = str(tmp_path / "corrupt.pth")
    payload = TA.make_model_artifact(_tiny_model())
    payload["model_state_dict"] = ["not", "a", "state", "dict"]
    TA.atomic_torch_save(payload, path)
    with pytest.raises(ValueError, match="is not a state dictionary"):
        TA.load_model_artifact(path, model=_tiny_model())


def test_an_artifact_that_names_no_architecture_needs_a_model_passed_in(tmp_path):
    """Nothing in the file says what to rebuild, so it says so."""
    path = str(tmp_path / "nameless.pth")
    payload = TA.make_model_artifact(_tiny_model())
    payload["model_config"] = {}
    TA.atomic_torch_save(payload, path)
    with pytest.raises(ValueError) as excinfo:
        TA.load_model_artifact(path)
    assert "does not describe its architecture" in str(excinfo.value)


def test_something_that_is_neither_a_module_nor_a_mapping_is_refused(tmp_path):
    """A tensor file is not a model, however loadable it is."""
    path = str(tmp_path / "just_a_tensor.pth")
    TA.atomic_torch_save(torch.zeros(3), path)
    with pytest.raises(ValueError, match="expected an nn.Module"):
        TA.load_model_artifact(path)


def test_a_whole_module_file_comes_back_as_its_own_module(tmp_path):
    """The legacy full-module format ignores any model passed alongside it."""
    path = str(tmp_path / "whole_module.pth")
    TA.atomic_torch_save(_tiny_model(), path)
    other = _tiny_model()
    loaded, payload = TA.load_model_artifact(path, model=other)
    assert loaded is not other
    assert payload["artifact_role"] == "legacy_full_module"
    assert payload["training_state"] == {}
