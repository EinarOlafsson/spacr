"""CustomCellClassifier returns raw logits for any channel count and is wired into choose_model.

Before 2026-11-07 the classifier put a ReLU on its logits, so no class score
could go below zero; its fusion convolution was fixed at 3 input channels; and
``choose_model('custom')`` raised NotImplementedError. The BMC gene-class
benchmark measured the shipped version against a fixed copy, and these tests
pin the fixed behaviour.
"""
import pytest
import torch

from spacr import utils as U
from spacr.torch_artifacts import load_model_artifact, save_model_artifact


@pytest.mark.parametrize("channels", [1, 3, 4, 5])
def test_custom_classifier_takes_the_real_channel_count(channels):
    model = U.CustomCellClassifier(num_classes=3, in_channels=channels).eval()
    with torch.no_grad():
        out = model(torch.rand(2, channels, 32, 32))
    assert out.shape == (2, 3)
    assert model.early_fusion.conv1.in_channels == channels


def test_custom_classifier_logits_can_be_negative():
    torch.manual_seed(0)
    model = U.CustomCellClassifier(num_classes=4).eval()
    with torch.no_grad():
        model.fc1.bias.fill_(-5.0)
        out = model(torch.rand(3, 3, 16, 16))
    assert (out < 0).any(), "a ReLU on the logits would clamp these to 0"


def test_custom_classifier_keeps_the_old_positional_signature():
    model = U.CustomCellClassifier(2, 1, True, False, 0.1)
    assert model.num_classes == 2 and model.in_channels == 3


@pytest.mark.parametrize("channels", [1, 3, 4, 5])
def test_choose_model_builds_custom_with_channels(channels):
    model = U.choose_model("custom", torch.device("cpu"), channels=channels,
                           height=32, width=32, num_classes=2)
    assert isinstance(model, U.CustomCellClassifier)
    assert model.in_channels == channels


@pytest.mark.parametrize("channels", [3, 5])
def test_custom_checkpoint_rebuilds_with_its_channel_count(tmp_path, channels):
    model = U.CustomCellClassifier(num_classes=2, in_channels=channels,
                                   image_size=32).eval()
    path = save_model_artifact(model, str(tmp_path / "m.pth"), include_rng=False)
    loaded, meta = load_model_artifact(path)
    assert isinstance(loaded, U.CustomCellClassifier)
    assert loaded.in_channels == channels
    x = torch.rand(1, channels, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded.eval()(x))


@pytest.mark.parametrize("channels", [1, 4, 5])
def test_torchmodel_stem_adapts_extra_channels_from_the_rgb_mean(channels):
    torch.manual_seed(0)
    ref = U.TorchModel("resnet18", pretrained=False, num_classes=2, image_size=32)
    torch.manual_seed(0)
    model = U.TorchModel("resnet18", pretrained=False, num_classes=2,
                         image_size=32, in_channels=channels).eval()
    w_ref = ref.base_model.conv1.weight
    w = model.base_model.conv1.weight
    assert w.shape[1] == channels
    keep = min(3, channels)
    assert torch.allclose(w[:, :keep], w_ref[:, :keep])
    for extra in range(3, channels):
        assert torch.allclose(w[:, extra], w_ref.mean(dim=1))
    with torch.no_grad():
        assert model(torch.rand(1, channels, 32, 32)).shape == (1, 2)


def test_maxvit_refuses_an_input_size_that_is_not_a_multiple_of_224():
    with pytest.raises(ValueError, match="multiple of 224"):
        U.TorchModel("maxvit_t", pretrained=False, num_classes=2, image_size=128)
