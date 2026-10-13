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
    from cellpose import models

    _RecordingCellposeModel.calls = []
    monkeypatch.setattr(models, "CellposeModel", _RecordingCellposeModel)
    return _RecordingCellposeModel


def _build(model_name="cpsam"):
    from spacr.qt.widgets.preview_contract import preview_cellpose_model

    preview_cellpose_model(model_name)


def test_a_cuda_check_that_raises_still_lands_on_the_cpu(fake_cellpose,
                                                          monkeypatch):
    """A driver mismatch makes ``torch.cuda.is_available()`` itself raise.

    NOTE WHAT THIS DOES AND DOES NOT PROVE. It is `resolve()` that
    absorbs the raising torch, not the preview's own guard -- deleting
    the preview's try/except leaves this test passing. It is kept for the
    end-to-end behaviour, and the preview's guard is driven separately
    below so that neither one can rot behind the other.
    """
    import torch

    import spacr.accelerator as accelerator

    def _explode():
        raise RuntimeError("no CUDA driver on this machine")

    monkeypatch.setattr(torch.cuda, "is_available", _explode)

    # THE CACHE HAS TO BE CLEARED FIRST, and that is not incidental.
    #
    # `resolve()` caches, and tests/conftest.py restores the session's
    # real verdict after every test so one test's fake machine cannot
    # leak into the next. On a machine that HAS a GPU that warm cache is
    # returned before torch is consulted at all, so the raising
    # is_available above is never reached -- this test passed alone,
    # where nothing had warmed the cache yet, and failed in a full run.
    #
    # Clearing it is what makes the probe happen. monkeypatch restores
    # the attribute, and the autouse fixture puts the real verdict back
    # afterwards either way.
    monkeypatch.setattr(accelerator, "_CACHED", None, raising=False)

    _build()

    assert fake_cellpose.calls[-1]["gpu"] is False
    assert fake_cellpose.calls[-1]["use_bfloat16"] is False


def test_the_preview_survives_an_accelerator_that_raises(fake_cellpose,
                                                         monkeypatch):
    """THE PREVIEW'S OWN GUARD, driven where the test above could not.

    ``preview_cellpose_model`` is called from a constructor on the GUI
    thread while a panel repaints. An exception out of it is a traceback
    into a paint, so a broken accelerator has to mean the CPU. This makes
    `cellpose_kwargs` itself fail, which is the only thing that reaches
    the guard -- `resolve()` handles a bad torch on its own.
    """
    import spacr.accelerator as accelerator

    def _explode():
        raise RuntimeError("the accelerator module is having a bad day")

    monkeypatch.setattr(accelerator, "cellpose_kwargs", _explode)

    _build()

    assert fake_cellpose.calls[-1]["gpu"] is False
    assert fake_cellpose.calls[-1]["use_bfloat16"] is False


def test_a_working_accelerator_is_believed(fake_cellpose, monkeypatch):
    """The fallback is a fallback: an accelerator that answers is used."""
    import spacr.accelerator as accelerator

    monkeypatch.setattr(accelerator, "cellpose_kwargs",
                        lambda: {"gpu": True, "device": None})

    _build()

    assert fake_cellpose.calls[-1]["gpu"] is True


def test_an_explicit_caller_still_overrides_the_machine(fake_cellpose,
                                                        monkeypatch):
    """``gpu=`` is documented as winning, so pin it.

    Without this, the two tests above would both pass against a version
    that ignored the parameter entirely.
    """
    import spacr.accelerator as accelerator
    from spacr.qt.widgets.preview_contract import preview_cellpose_model

    monkeypatch.setattr(accelerator, "cellpose_kwargs",
                        lambda: {"gpu": True, "device": None})

    preview_cellpose_model("cpsam", gpu=False)

    assert fake_cellpose.calls[-1]["gpu"] is False
    assert fake_cellpose.calls[-1]["use_bfloat16"] is False


@pytest.mark.parametrize("machine,requested,expected_gpu,expected_bfloat16", [
    ({"gpu": True, "device": "cuda:0"}, False, False, False),
    ({"gpu": True, "device": "cuda:0", "use_bfloat16": True}, False, False, False),
    ({"gpu": True, "device": "cuda:0"}, None, True, True),
    ({"gpu": True, "device": "mps", "use_bfloat16": False}, None, True, False),
    ({"gpu": False, "device": "cpu", "use_bfloat16": False}, None, False, False),
    ({"gpu": False, "device": "cpu", "use_bfloat16": False}, True, True, False),
])
def test_preview_device_override_keeps_precision_consistent(
        fake_cellpose, monkeypatch, machine, requested, expected_gpu, expected_bfloat16):
    import spacr.accelerator as accelerator
    from spacr.qt.widgets.preview_contract import preview_cellpose_model

    monkeypatch.setattr(accelerator, "cellpose_kwargs", lambda: dict(machine))
    preview_cellpose_model("cpsam", gpu=requested)
    kwargs = fake_cellpose.calls[-1]
    assert kwargs["gpu"] is expected_gpu
    assert kwargs.get("use_bfloat16", True) is expected_bfloat16
    assert kwargs["device"] is None


def test_unrelated_checkpoint_failure_retains_original_exception(fake_cellpose, monkeypatch):
    from cellpose import models
    from spacr.qt.widgets.preview_contract import preview_cellpose_model

    failure = ValueError("checkpoint tensor dimensions are inconsistent")

    def refuse(**kwargs):
        raise failure

    monkeypatch.setattr(models, "CellposeModel", refuse)
    with pytest.raises(ValueError) as caught:
        preview_cellpose_model("cpsam", gpu=False)
    assert caught.value is failure
    assert caught.value.__cause__ is None


def test_legacy_checkpoint_failure_keeps_model_identity_and_cause(
        fake_cellpose, monkeypatch, tmp_path):
    from cellpose import models
    from spacr.submodules import Cellpose3Checkpoint
    from spacr.qt.widgets.preview_contract import preview_cellpose_model

    checkpoint = tmp_path / "old model.pth"
    checkpoint.write_bytes(b"legacy checkpoint fixture")
    failure = ValueError("This model does not appear to be a CP4 model.")

    def refuse(**kwargs):
        assert kwargs["pretrained_model"] == str(checkpoint)
        raise failure

    monkeypatch.setattr(models, "CellposeModel", refuse)
    with pytest.raises(Cellpose3Checkpoint) as caught:
        preview_cellpose_model(checkpoint, gpu=False)
    assert str(checkpoint) in str(caught.value)
    assert "Cellpose 4-compatible checkpoint" in str(caught.value)
    assert "plaque" not in str(caught.value)
    assert caught.value.__cause__ is failure
    assert checkpoint.read_bytes() == b"legacy checkpoint fixture"
