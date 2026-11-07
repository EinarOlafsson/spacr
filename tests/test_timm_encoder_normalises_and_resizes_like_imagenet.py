"""The timm encoder feeds its backbone what the backbone was trained on.

A timm backbone pretrained on ImageNet expects inputs resized to its training
size and standardised by the ImageNet mean and standard deviation. The
encoder reads both from the model's own pretrained data configuration by
default; the spec can override the mean, the standard deviation and the
size, or switch the transform off and feed the [0, 1] crops as they are.
These tests use a tiny randomly initialised timm network (no download) or a
recording stand-in, on the CPU.
"""
from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

import spacr.embeddings as emb

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")


class _Recorder(torch.nn.Module):
    """Returns each image's per-channel mean and remembers what it saw."""

    def __init__(self, cfg=None):
        super().__init__()
        self.seen = []
        if cfg is not None:
            self.pretrained_cfg = cfg

    def forward(self, tensor):
        self.seen.append(tensor.detach().clone())
        return tensor.mean(dim=(2, 3))


_CFG = {"input_size": (3, 16, 16), "interpolation": "bilinear",
        "mean": (0.5, 0.25, 0.125), "std": (0.5, 0.25, 0.5)}


def _encoder(monkeypatch, model, **spec):
    monkeypatch.setattr(timm, "create_model", Mock(return_value=model))
    return emb._timm_encoder(emb.EmbeddingSpec(
        backbone="recorded", device="cpu", batch_size=8, **spec))


def _crops():
    return np.random.default_rng(0).random((3, 8, 12, 3)).astype(np.float32)


def test_the_default_uses_the_models_mean_std_and_size(monkeypatch):
    model = _Recorder(_CFG)
    run = _encoder(monkeypatch, model)
    crops = _crops()
    run(crops)
    seen = model.seen[0]
    assert tuple(seen.shape) == (3, 3, 16, 16)
    resized = torch.nn.functional.interpolate(
        torch.from_numpy(crops.transpose(0, 3, 1, 2).copy()), size=(16, 16),
        mode="bilinear", align_corners=False, antialias=False).clamp(0, 1)
    mean = torch.tensor(_CFG["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(_CFG["std"]).view(1, 3, 1, 1)
    torch.testing.assert_close(seen, (resized - mean) / std)


def test_overrides_replace_the_pretrained_mean_std_and_size(monkeypatch):
    model = _Recorder(_CFG)
    run = _encoder(monkeypatch, model, input_mean=(0.0, 0.0, 0.0),
                   input_std=(2.0, 2.0, 2.0), input_size=10)
    crops = _crops()
    run(crops)
    seen = model.seen[0]
    assert tuple(seen.shape) == (3, 3, 10, 10)
    resized = torch.nn.functional.interpolate(
        torch.from_numpy(crops.transpose(0, 3, 1, 2).copy()), size=(10, 10),
        mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)
    torch.testing.assert_close(seen, resized / 2.0)


def test_size_zero_keeps_the_crop_size_and_still_standardises(monkeypatch):
    model = _Recorder(_CFG)
    run = _encoder(monkeypatch, model, input_size=0)
    crops = _crops()
    run(crops)
    seen = model.seen[0]
    assert tuple(seen.shape) == (3, 3, 8, 12)
    expected = (torch.from_numpy(crops.transpose(0, 3, 1, 2).copy())
                - torch.tensor(_CFG["mean"]).view(1, 3, 1, 1)) / torch.tensor(
                    _CFG["std"]).view(1, 3, 1, 1)
    torch.testing.assert_close(seen, expected)


def test_the_transform_can_be_switched_off(monkeypatch):
    model = _Recorder(_CFG)
    run = _encoder(monkeypatch, model, input_transform="none")
    crops = _crops()
    values = run(crops)
    torch.testing.assert_close(
        model.seen[0], torch.from_numpy(crops.transpose(0, 3, 1, 2).copy()))
    np.testing.assert_allclose(values, crops.mean(axis=(1, 2)), rtol=1e-6)


def test_a_real_timm_network_gets_imagenet_statistics(monkeypatch):
    model = timm.create_model("resnet18", pretrained=False, num_classes=0)
    seen = []
    model.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    monkeypatch.setattr(timm, "create_model", Mock(return_value=model))
    run = emb._timm_encoder(emb.EmbeddingSpec(device="cpu", batch_size=4))
    crops = np.full((2, 32, 32, 3), 0.485, dtype=np.float32)
    values = run(crops)
    assert values.shape == (2, 512)
    assert tuple(seen[0].shape) == (2, 3, 224, 224)
    torch.testing.assert_close(seen[0][:, 0], torch.zeros(2, 224, 224),
                               atol=1e-5, rtol=0)
    torch.testing.assert_close(
        seen[0][:, 1], torch.full((2, 224, 224), (0.485 - 0.456) / 0.224),
        atol=1e-5, rtol=0)


def test_a_strict_patch_size_wins_over_the_configured_size(monkeypatch):
    from timm.layers import PatchEmbed

    class Strict(_Recorder):
        def __init__(self):
            super().__init__(dict(_CFG, input_size=(3, 32, 32)))
            self.patch_embed = PatchEmbed(img_size=(28, 42), patch_size=14,
                                          in_chans=3, embed_dim=6)

    model = Strict()
    run = _encoder(monkeypatch, model)
    run(_crops())
    assert tuple(model.seen[0].shape) == (3, 3, 28, 42)


@pytest.mark.parametrize("kwargs, match", [
    (dict(input_transform="imagenet"), "input_transform"),
    (dict(input_mean=(0.5, 0.5)), "three"),
    (dict(input_std=(0.5, 0.0, 0.5)), "positive"),
    (dict(input_size=-1), "input_size"),
    (dict(input_transform="none", input_size=64), "none"),
])
def test_an_impossible_transform_is_refused_when_the_spec_is_built(kwargs, match):
    with pytest.raises(emb.EmbeddingError, match=match):
        emb.EmbeddingSpec(**kwargs)


def test_the_fingerprint_records_the_transform():
    plain = emb.EmbeddingSpec(input_transform="none")
    default = emb.EmbeddingSpec()
    assert plain.fingerprint() == "ee13fbde21d1062c"
    assert default.fingerprint() != plain.fingerprint()
    assert emb.EmbeddingSpec(input_size=128).fingerprint() != default.fingerprint()
    assert emb.EmbeddingSpec(input_mean=(0.1, 0.1, 0.1)).fingerprint() != \
        default.fingerprint()
    assert emb.EmbeddingSpec(backbone="openphenom").fingerprint() == \
        emb.EmbeddingSpec(backbone="openphenom", input_size=64).fingerprint()


def test_overview_settings_choose_the_transform():
    from spacr import measure
    from spacr.settings import _overview_measurement_defaults

    defaults = _overview_measurement_defaults({})
    assert defaults["overview_embedding_transform"] == "pretrained"
    assert measure._image_advanced_options(defaults)["embedding_transform"] == {
        "input_transform": "pretrained"}
    chosen = measure._image_embedding_transform({
        "overview_embedding_mean": "0.5, 0.5,0.5",
        "overview_embedding_std": "1;1;1",
        "overview_embedding_input_size": 0})
    assert chosen == {"input_transform": "pretrained",
                      "input_mean": (0.5, 0.5, 0.5),
                      "input_std": (1.0, 1.0, 1.0), "input_size": 0}
    assert emb.EmbeddingSpec(**chosen).input_size == 0
    assert measure._image_embedding_transform({
        "overview_embedding_transform": "none",
        "overview_embedding_mean": "1, 1, 1"}) == {"input_transform": "none"}


def test_the_overview_embedding_passes_the_transform_to_its_spec(monkeypatch):
    from spacr import measure

    specs = []

    def recording(spec):
        specs.append(spec)
        return lambda stack: stack.mean(axis=(1, 2)).astype(np.float32)

    monkeypatch.setattr(measure, "_IMAGE_ADVANCED_CACHE", {})
    monkeypatch.setattr(emb, "_backbone_encoder", recording)
    planes = {0: np.random.default_rng(0).random((40, 40)).astype(np.float32)}
    record = measure._image_embedding(planes, "resnet18", 32,
                                      transform={"input_transform": "none"})
    assert specs[0].input_transform == "none"
    assert record["embedding_dims"] == 3
