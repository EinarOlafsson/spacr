"""The foundation-model loaders, with stand-ins for the downloaded networks.

OpenPhenom, ChAda-ViT and SubCell are fetched from the Hugging Face hub or
an S3 bucket the first time they are chosen, which a test run must not do.
These tests stand in for exactly the two seams the loaders call --
``transformers`` (``AutoModel``, ``ViTConfig``, ``ViTModel``) and
``torch.hub`` -- and check what spaCR does around them: the pinned revision
it asks for, the resize to each model's training size, the batches, the
checkpoint key renaming SubCell needs, and the sentence a missing package
gets.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import spacr.embeddings as emb

torch = pytest.importorskip("torch")


def _spec(backbone, batch_size=2):
    return emb.EmbeddingSpec(backbone=backbone, device="cpu",
                             batch_size=batch_size)


def _transformers(monkeypatch, net=None, vit=None):
    asked = []
    module = types.ModuleType("transformers")

    class AutoModel:
        @staticmethod
        def from_pretrained(repo, revision, trust_remote_code):
            asked.append((repo, revision, trust_remote_code))
            return net

    module.AutoModel = AutoModel
    if vit is not None:
        module.ViTConfig = lambda **config: types.SimpleNamespace(**config)
        module.ViTModel = vit
    monkeypatch.setitem(sys.modules, "transformers", module)
    return asked


class _OpenPhenom(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def predict(self, x):
        self.seen.append(tuple(x.shape))
        return x.mean(dim=(2, 3)) / 255.0


def test_openphenom_is_asked_for_by_revision_and_fed_its_own_size(monkeypatch):
    net = _OpenPhenom()
    asked = _transformers(monkeypatch, net)
    encode = emb._foundation_encoder(_spec("openphenom"))
    info = emb._FOUNDATION_MODELS["openphenom"]
    assert asked == [(info["repo"], info["revision"], True)]
    assert net.return_channelwise_embeddings is False
    assert encode.in_channels is None
    stack = np.full((3, 10, 12, 2), 0.5, dtype=np.float32)
    features = encode(stack)
    assert features.shape == (3, 2)
    assert np.allclose(features, 0.5)
    assert net.seen == [(2, 2, 256, 256), (1, 2, 256, 256)]


class _ChAda(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, flat, index, list_num_channels):
        self.calls.append((tuple(flat.shape), index, list_num_channels))
        return flat.reshape(len(list_num_channels[0]), -1).mean(
            dim=1, keepdim=True)


def test_chada_vit_sees_each_channel_as_one_token_sequence(monkeypatch):
    net = _ChAda()
    _transformers(monkeypatch, net)
    encode = emb._foundation_encoder(_spec("chada_vit", batch_size=4))
    assert net.return_all_tokens is False
    stack = np.random.default_rng(0).random((2, 224, 224, 3)).astype(np.float32)
    features = encode(stack)
    assert features.shape == (2, 1)
    assert net.calls == [((6, 1, 224, 224), 0, [[3, 3]])]


class _ViT(torch.nn.Module):
    """A token encoder without weights: 4 tokens of the crop's mean."""

    def __init__(self, config, add_pooling_layer):
        super().__init__()
        assert add_pooling_layer is False
        self.config = config

    def forward(self, x, interpolate_pos_encoding):
        n = x.shape[0]
        tokens = x.mean(dim=(1, 2, 3)).reshape(n, 1, 1).expand(n, 4, 768)
        return types.SimpleNamespace(last_hidden_state=tokens.contiguous())


def _subcell_checkpoint():
    """SubCell's published pooler keys, before spaCR renames them."""
    generator = torch.Generator().manual_seed(0)

    def weights(*shape):
        return torch.randn(*shape, generator=generator) * 0.01

    return {"pool_model.attention_v.1.weight": weights(512, 768),
            "pool_model.attention_v.1.bias": weights(512),
            "pool_model.attention_u.1.weight": weights(512, 768),
            "pool_model.attention_u.1.bias": weights(512),
            "pool_model.attention.weight": weights(2, 512),
            "pool_model.attention.bias": weights(2)}


def test_subcell_is_downloaded_once_renamed_and_min_max_scaled(
        monkeypatch, tmp_path):
    _transformers(monkeypatch, vit=_ViT)
    downloads = []

    def download(url, target):
        downloads.append(url)
        torch.save(_subcell_checkpoint(), target)

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    encode = emb._foundation_encoder(_spec("subcell"))
    assert encode.in_channels == 2
    assert downloads == [emb._FOUNDATION_MODELS["subcell"]["url"]]
    stack = np.random.default_rng(1).random((1, 20, 20, 2)).astype(np.float32)
    features = encode(stack)
    assert features.shape == (1, 1536)
    assert np.isfinite(features).all()
    emb._foundation_encoder(_spec("subcell"))
    assert len(downloads) == 1


@pytest.mark.parametrize("policy,channels", [
    (emb.CHANNEL_PROJECT, None),
    (emb.CHANNEL_PROJECT, (0, 1, 2)),
    (emb.CHANNEL_PROJECT, (0, 1, 2, 2)),
    (emb.CHANNEL_PROJECT, (0, 1, 2, 4)),
    (emb.CHANNEL_PROJECT, (0, 1, 2, -1)),
    (emb.CHANNEL_PROJECT, (0, 1, 2, True)),
    (emb.CHANNEL_PROJECT, (0, 1, 2, 3.0)),
    (emb.CHANNEL_PROJECT, (0, 1, 2, "3")),
    (emb.CHANNEL_PER_CHANNEL, (0, 1, 2, 3)),
])
def test_subcell_rybg_refuses_ambiguous_planes_before_model_load(
        monkeypatch, policy, channels):
    def unexpected_model_load(_spec):
        raise AssertionError("model load must not begin")

    monkeypatch.setattr(emb, "_backbone_encoder", unexpected_model_load)
    spec = emb.EmbeddingSpec(backbone="subcell_rybg", channel_policy=policy,
                             channels=channels, normalize=False)
    with pytest.raises(emb.EmbeddingError, match="subcell_rybg|channel 4"):
        emb.embed_array(np.zeros((1, 20, 20, 4), dtype=np.float32), spec)


@pytest.mark.parametrize("width", [3, None, "missing"])
def test_subcell_rybg_refuses_an_encoder_that_would_drop_a_plane(width):
    calls = []

    def wrong_encoder(stack):
        calls.append(stack.shape)
        raise AssertionError("a wrong-width encoder must never be called")

    if width != "missing":
        wrong_encoder.in_channels = width
    spec = emb.EmbeddingSpec(
        backbone="subcell_rybg", channel_policy=emb.CHANNEL_PROJECT,
        channels=(0, 1, 2, 3), normalize=False)
    with pytest.raises(emb.EmbeddingError, match="exactly four planes"):
        emb.embed_array(np.ones((1, 20, 20, 4), dtype=np.float32), spec,
                        encoder=wrong_encoder)
    assert calls == []


def test_subcell_rybg_uses_explicit_plane_order_and_whole_crop_scale(
        monkeypatch, tmp_path):
    seen = []
    configurations = []

    class RecordingViT(_ViT):
        """Capture the actual model input after SubCell normalization."""

        def __init__(self, config, add_pooling_layer):
            super().__init__(config, add_pooling_layer)
            configurations.append(config.num_channels)

        def forward(self, x, interpolate_pos_encoding):
            seen.append(x.detach().cpu().numpy().copy())
            return super().forward(x, interpolate_pos_encoding)

    _transformers(monkeypatch, vit=RecordingViT)
    downloads = []

    def download(url, target):
        downloads.append((url, target))
        torch.save(_subcell_checkpoint(), target)

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    crops = np.broadcast_to(np.array([1.0, 3.0, 0.0, 2.0], dtype=np.float32),
                            (2, 20, 21, 4)).copy()
    spec = emb.EmbeddingSpec(
        backbone="subcell_rybg", channel_policy=emb.CHANNEL_PROJECT,
        channels=(2, 0, 3, 1), normalize=False, device="cpu", batch_size=1)
    result = emb.embed_array(crops, spec)
    assert result.values.shape == (2, 1536)
    assert result.n_channels == 4 and configurations == [4]
    assert len(downloads) == 1
    assert downloads[0][0] == emb._FOUNDATION_MODELS["subcell_rybg"]["url"]
    assert downloads[0][1].endswith("all_channels_ViT-ProtS-Pool.pth")
    assert len(seen) == 2 and all(x.shape == (1, 4, 20, 21) for x in seen)
    np.testing.assert_allclose(seen[0].mean(axis=(2, 3))[0],
                               [0.0, 1 / 3, 2 / 3, 1.0], atol=1e-6)
    assert result.spec.fingerprint() != emb.EmbeddingSpec(
        backbone="subcell_rybg", channel_policy=emb.CHANNEL_PROJECT,
        channels=(0, 1, 2, 3), normalize=False).fingerprint()


def test_subcell_rybg_preserves_native_large_crop_and_authors_normalizer(
        monkeypatch, tmp_path):
    seen = []

    class RecordingViT(_ViT):
        """Record the whole native crop the model receives."""

        def forward(self, x, interpolate_pos_encoding):
            seen.append(x.detach().cpu().numpy().copy())
            return super().forward(x, interpolate_pos_encoding)

    _transformers(monkeypatch, vit=RecordingViT)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    monkeypatch.setattr(torch.hub, "download_url_to_file",
                        lambda _url, target: torch.save(_subcell_checkpoint(), target))
    crops = np.full((1, 512, 544, 4), 100.0, dtype=np.float32)
    crops[0, 2, 3, 0] = 0.0
    crops[0, 500, 501, 3] = 1000.0
    spec = emb.EmbeddingSpec(
        backbone="subcell_rybg", channel_policy=emb.CHANNEL_PROJECT,
        channels=(0, 1, 2, 3), normalize=False, device="cpu")
    result = emb.embed_array(crops, spec)
    original = crops[0].transpose(2, 0, 1)
    minimum = np.amin(original, keepdims=True)
    maximum = np.amax(original, keepdims=True)
    authors_input = (original - minimum) / (maximum - minimum + 1e-8)
    assert result.values.shape == (1, 1536)
    assert len(seen) == 1 and seen[0].shape == (1, 4, 512, 544)
    np.testing.assert_array_equal(seen[0][0], authors_input)
    assert seen[0][0, 0, 2, 3] == 0.0
    assert seen[0][0, 3, 500, 501] == 1.0


@pytest.mark.parametrize("shape", [(15, 20), (20, 15)])
def test_subcell_rybg_refuses_a_crop_smaller_than_one_patch(
        monkeypatch, tmp_path, shape):
    calls = []

    class RecordingViT(_ViT):
        """Reject a short input before patch embedding runs."""

        def forward(self, x, interpolate_pos_encoding):
            calls.append(tuple(x.shape))
            return super().forward(x, interpolate_pos_encoding)

    _transformers(monkeypatch, vit=RecordingViT)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    monkeypatch.setattr(torch.hub, "download_url_to_file",
                        lambda _url, target: torch.save(_subcell_checkpoint(), target))
    spec = emb.EmbeddingSpec(
        backbone="subcell_rybg", channel_policy=emb.CHANNEL_PROJECT,
        channels=(0, 1, 2, 3), normalize=False, device="cpu")
    encoder = emb._foundation_encoder(spec)
    with pytest.raises(emb.EmbeddingError, match="at least 16 pixels"):
        encoder(np.zeros((1, *shape, 4), dtype=np.float32))
    assert calls == []


@pytest.mark.parametrize("shape", [(15, 20), (20, 15)])
@pytest.mark.parametrize("entry", ["array", "plate"])
def test_subcell_rybg_small_crops_fail_before_scale_or_checkpoint(
        monkeypatch, shape, entry):
    calls = []

    def unexpected(*_args, **_kwargs):
        calls.append("scale or model")
        raise AssertionError("small crops must fail in input preflight")

    monkeypatch.setattr(emb, "_estimate_channel_scale", unexpected)
    monkeypatch.setattr(emb, "_backbone_encoder", unexpected)
    spec = emb.EmbeddingSpec(
        backbone="subcell_rybg", channel_policy=emb.CHANNEL_PROJECT,
        channels=(0, 1, 2, 3), normalize=True)
    crops = np.zeros((2, *shape, 4), dtype=np.float32)
    record = {}
    with pytest.raises(emb.EmbeddingError, match="at least 16 pixels"):
        if entry == "plate":
            emb._embed_plate(crops, spec, record=record)
        else:
            emb.embed_array(crops, spec)
    assert calls == [] and record == {}


def test_subcell_rybg_refuses_a_two_plane_checkpoint(
        monkeypatch, tmp_path):
    class PatchViT(_ViT):
        """Carry a real channel-dependent parameter for strict loading."""

        def __init__(self, config, add_pooling_layer):
            super().__init__(config, add_pooling_layer)
            self.patch = torch.nn.Conv2d(config.num_channels, 1, 1)

    _transformers(monkeypatch, vit=PatchViT)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))

    def wrong_checkpoint(_url, target):
        state = _subcell_checkpoint()
        state["encoder.patch.weight"] = torch.zeros((1, 2, 1, 1))
        state["encoder.patch.bias"] = torch.zeros(1)
        torch.save(state, target)

    monkeypatch.setattr(torch.hub, "download_url_to_file", wrong_checkpoint)
    with pytest.raises(RuntimeError, match="size mismatch for encoder.patch.weight"):
        emb._foundation_encoder(emb.EmbeddingSpec(
            backbone="subcell_rybg", channel_policy=emb.CHANNEL_PROJECT,
            channels=(0, 1, 2, 3), device="cpu"))


@pytest.mark.parametrize("backbone", ["openphenom", "subcell"])
def test_a_missing_transformers_package_is_named(monkeypatch, backbone):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(emb.EmbeddingError, match="transformers"):
        emb._foundation_encoder(_spec(backbone))


def test_foundation_models_without_torch_name_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(emb.EmbeddingError, match=r"spacr\[embeddings\]"):
        emb._foundation_encoder(_spec("openphenom"))
