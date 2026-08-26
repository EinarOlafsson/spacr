"""How the live previews choose a device when torch cannot answer.

:func:`spacr.qt.widgets.preview_contract.preview_cellpose_model` asks torch
whether there is a CUDA device. On a machine with a driver mismatch or a
stripped build that question RAISES rather than returning False -- and it is
asked from a constructor the GUI thread calls while a panel is repainting.
There is a perfectly good CPU underneath, so an unanswerable question means
CPU, not an exception into a paint.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


class _RecordingCellposeModel:
    """Stand-in for ``cellpose.models.CellposeModel`` that records its kwargs."""

    calls: list = []

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)


@pytest.fixture
def fake_cellpose(monkeypatch):
    _RecordingCellposeModel.calls = []
    models = types.ModuleType("cellpose.models")
    models.CellposeModel = _RecordingCellposeModel
    package = types.ModuleType("cellpose")
    package.models = models
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    return _RecordingCellposeModel


def _build(model_name="cpsam"):
    from spacr.qt.widgets.preview_contract import preview_cellpose_model

    preview_cellpose_model(model_name)


def test_a_cuda_check_that_raises_means_cpu(fake_cellpose, monkeypatch):
    """A driver mismatch makes ``torch.cuda.is_available()`` itself raise.

    The preview must fall back to the CPU rather than let the exception out of
    a constructor called from the GUI thread.
    """
    import torch

    def _explode():
        raise RuntimeError("no CUDA driver on this machine")

    monkeypatch.setattr(torch.cuda, "is_available", _explode)

    _build()

    assert fake_cellpose.calls[-1]["gpu"] is False


def test_a_working_torch_is_believed(fake_cellpose, monkeypatch):
    """The fallback is a fallback: a torch that answers is not second-guessed."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    _build()

    assert fake_cellpose.calls[-1]["gpu"] is True
