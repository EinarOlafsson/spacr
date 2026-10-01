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


@pytest.mark.parametrize("backbone", ["openphenom", "subcell"])
def test_a_missing_transformers_package_is_named(monkeypatch, backbone):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(emb.EmbeddingError, match="transformers"):
        emb._foundation_encoder(_spec(backbone))


def test_foundation_models_without_torch_name_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(emb.EmbeddingError, match=r"spacr\[embeddings\]"):
        emb._foundation_encoder(_spec("openphenom"))
