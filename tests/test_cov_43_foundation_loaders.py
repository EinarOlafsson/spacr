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
