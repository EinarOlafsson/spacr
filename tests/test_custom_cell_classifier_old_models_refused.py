"""The fixed custom classifier, and the clear refusal of models of the old head.

AGENTS rule 41: the head that clipped logits and fused a fixed 3 channels is
gone, and its saved models are refused with a message to retrain rather than
loaded into a head whose outputs differ.
"""

import pytest

torch = pytest.importorskip("torch")

from spacr import torch_artifacts as A  # noqa: E402
from spacr import utils as U  # noqa: E402


def test_negative_logits_survive():
    model = U.CustomCellClassifier(num_classes=3).eval()
    with torch.no_grad():
        model.fc1.bias.fill_(-5.0)
        assert (model(torch.rand(2, 3, 16, 16)) < 0).all()


def test_four_channel_input_needs_no_adapter():
    model = U.CustomCellClassifier(num_classes=2, in_channels=4).eval()
    assert model.early_fusion.conv1.in_channels == 4
    with torch.no_grad():
        assert model(torch.rand(2, 4, 16, 16)).shape == (2, 2)


def test_a_channel_count_unlike_training_is_named():
    model = U.CustomCellClassifier(num_classes=2, in_channels=4).eval()
    with pytest.raises(ValueError, match="trained on 4-channel images.*have 3"):
        model(torch.rand(1, 3, 16, 16))


def _artifact(tmp_path, model, **config_changes):
    """Save ``model`` as a current artifact with edited ``model_config``."""
    path = tmp_path / "model.pth"
    A.save_model_artifact(model, str(path))
    payload = torch.load(path, weights_only=False)
    payload["model_config"].update(config_changes)
    torch.save(payload, path)
    return path


def test_a_fixed_custom_artifact_still_loads(tmp_path):
    model = U.CustomCellClassifier(num_classes=2, in_channels=4)
    loaded, _ = A.load_model_artifact(str(_artifact(tmp_path, model)))
    assert isinstance(loaded, U.CustomCellClassifier)
    assert loaded.in_channels == 4


def test_an_old_head_artifact_is_refused_with_a_retrain_message(tmp_path):
    model = U.CustomCellClassifier(num_classes=2)
    path = _artifact(tmp_path, model, model_name="CustomCellClassifier")
    with pytest.raises(ValueError, match="old spaCR custom classifier.*Retrain"):
        A.load_model_artifact(str(path))


def test_an_old_full_module_pickle_is_refused(tmp_path):
    model = U.CustomCellClassifier(num_classes=2)
    del model.in_channels
    path = tmp_path / "old_module.pth"
    torch.save(model, path)
    with pytest.raises(ValueError, match="old spaCR custom classifier"):
        A.load_model_artifact(str(path))


def test_a_fusion_weight_unlike_the_recorded_channels_is_refused(tmp_path):
    model = U.CustomCellClassifier(num_classes=2, in_channels=3)
    path = _artifact(tmp_path, model, in_channels=4)
    with pytest.raises(ValueError, match="records 4 image channels.*take 3"):
        A.load_model_artifact(str(path))
