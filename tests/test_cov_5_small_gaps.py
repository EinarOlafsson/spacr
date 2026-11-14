"""A narrow gap: a rebuilt model.

The last branch of something that works every other day -- the checkpoint
that has to reconstruct its own architecture before it can be loaded.
"""
from __future__ import annotations

from spacr import torch_artifacts as TA


# ---------------------------------------------------------------------------
# A checkpoint that has to rebuild its own architecture
# ---------------------------------------------------------------------------


def test_a_configuration_rebuilds_the_architecture_without_downloading(
        monkeypatch):
    """``pretrained=False`` is the point: loading must work offline."""
    import torchvision

    def refuse(*_args, **_kwargs):
        raise AssertionError("rebuilding a model must not fetch weights")

    monkeypatch.setattr(torchvision.models._api, "get_model_weights", refuse,
                        raising=False)

    model = TA.build_model_from_configuration({
        "model_name": "resnet18", "num_classes": 5, "image_size": 64,
        "dropout_rate": None, "use_checkpoint": False, "multilabel": False})

    assert model.model_name == "resnet18"
    assert model.num_classes == 5
    assert model.pretrained is False


