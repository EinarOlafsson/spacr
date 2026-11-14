"""What a model artifact records about its environment, and an atomic save.

A missing torchvision is recorded as unavailable rather than guessed, and a
save that fails leaves no temporary beside the target.  Reading, refusing and
round-tripping checkpoints live in ``tests/test_torch_artifacts_checkpoints.py``.
"""
from __future__ import annotations

import builtins
import os

import pytest

torch = pytest.importorskip("torch")

from spacr import torch_artifacts as TA  # noqa: E402


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
