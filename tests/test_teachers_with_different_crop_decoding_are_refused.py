"""Item 288: knowledge transfer refuses teachers that decode crops differently.

A student distilled from several teachers is trained on one crop decoding.
If one teacher was trained on crops decoded one way (the declared uint8
policy) and another on the historical stored/PIL policy, there is no single
decoding under which both teachers see the images they learned from -- so
``model_knowledge_transfer`` refuses before training anything, rather than
distilling one teacher's view of the wrong pixels.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spacr.classification_pixels import DECLARED_UINT8  # noqa: E402
from tests.test_cov_deep_spacr_entry_fusion import (  # noqa: E402
    _MODEL, _const_state_dict, _tiny_loader, _torchmodel)


def _teacher(path, value, preprocessing=None):
    payload = {"model": _const_state_dict(_torchmodel(), value),
               "model_name": _MODEL, "pretrained": False,
               "dropout_rate": None, "use_checkpoint": False}
    if preprocessing is not None:
        payload["preprocessing"] = preprocessing
    torch.save(payload, str(path))
    return str(path)


def test_teachers_with_different_crop_decoding_are_refused(tmp_path):
    from spacr.deep_spacr import model_knowledge_transfer

    declared = _teacher(tmp_path / "declared.pth", 0.01,
                        {"crop_loading_policy": DECLARED_UINT8})
    legacy = _teacher(tmp_path / "legacy.pth", 0.02)
    with pytest.raises(ValueError) as excinfo:
        model_knowledge_transfer(
            teacher_paths=[declared, legacy],
            student_save_path=str(tmp_path / "student"),
            data_loader=_tiny_loader(n=2), device="cpu",
            student_model_name=_MODEL, pretrained=False, epochs=1)
    assert "must share a crop loading policy" in str(excinfo.value)
    assert not (tmp_path / "student_KD.pth").exists()
