"""Two narrow gaps: a terabyte plate and a rebuilt model.

Each of these is the last branch of something that works every other day —
the biggest unit an inventory can print and the checkpoint that has to reconstruct its own architecture before it can be
loaded.
"""
from __future__ import annotations

import pytest

from spacr import torch_artifacts as TA
from spacr.workspace import _human_size


# ---------------------------------------------------------------------------
# The workspace inventory
# ---------------------------------------------------------------------------

def test_a_plate_folder_is_sized_in_the_unit_it_reaches():
    assert _human_size(0) == "0 B"
    assert _human_size(2048) == "2.0 KB"
    assert _human_size(3 * 1024 ** 3) == "3.0 GB"
    # A four-plate screen of raw images really does pass a terabyte, and the
    # top unit has to keep counting rather than roll over.
    assert _human_size(7 * 1024 ** 4) == "7.0 TB"
    assert _human_size(2500 * 1024 ** 4) == "2500.0 TB"


def test_a_size_that_is_not_a_number_is_marked_unknown():
    assert _human_size(None) == "?"
    assert _human_size("a lot") == "?"


# ---------------------------------------------------------------------------
# A checkpoint that has to rebuild its own architecture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model():
    from spacr.utils import TorchModel

    return TorchModel(model_name="resnet18", pretrained=False, num_classes=3,
                      image_size=64)


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


def test_a_saved_artifact_reloads_into_a_model_it_rebuilt_itself(tmp_path,
                                                                 small_model):
    """A collaborator with the .pth and no training script gets the model back."""
    path = TA.save_model_artifact(small_model, str(tmp_path / "model.pth"))

    restored, metadata = TA.load_model_artifact(path)

    assert metadata["legacy"] is False
    assert restored is not small_model
    assert restored.num_classes == small_model.num_classes
    original = small_model.state_dict()
    for key, value in restored.state_dict().items():
        assert value.shape == original[key].shape


def test_a_checkpoint_that_names_no_architecture_refuses_to_guess(tmp_path,
                                                                  small_model):
    """Guessing would load half the weights and report a working model."""
    import torch

    path = str(tmp_path / "headless.pth")
    torch.save({"artifact_type": TA.ARTIFACT_TYPE,
                "artifact_version": TA.ARTIFACT_VERSION,
                "model_config": {"num_classes": 3},
                "model_state_dict": small_model.state_dict()}, path)

    with pytest.raises(ValueError, match="does not describe its architecture"):
        TA.load_model_artifact(path)

    restored, _metadata = TA.load_model_artifact(path, model=small_model)
    assert restored is small_model
