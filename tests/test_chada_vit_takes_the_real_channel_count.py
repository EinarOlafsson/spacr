"""ChAda-ViT runs each crop with its own channel count, not ten padded slots.

The published forward pads every crop's tokens to the model's ten channel
slots and masks the padding out of attention. The class token cannot see
masked padding, so the padded and the unpadded sequences give the same
embedding, but the padded one costs ten slots of attention for a
one-channel crop. These tests build a small randomly initialised model with
the published layout (no weights are downloaded), run the published padded
forward as the reference, and check that spaCR's path gives the same class
token from a sequence of ``1 + k * tokens`` instead of ``1 + 10 * tokens``.
"""
from __future__ import annotations

import importlib
import os
import sys
import types

import numpy as np
import pytest

import spacr.embeddings as emb

torch = pytest.importorskip("torch")


class _TokenLearner(torch.nn.Module):
    def __init__(self, size, patch, dims):
        super().__init__()
        self.patch_size = patch
        self.num_patches = (size // patch) ** 2
        self.proj = torch.nn.Conv2d(1, dims, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _ChAda(torch.nn.Module):
    """The published ChAda-ViT layout at toy size, with its padded forward."""

    def __init__(self, slots=10, size=32, patch=8, dims=16, depth=2):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.max_channels = slots
        self.token_learner = _TokenLearner(size, patch, dims)
        count = self.token_learner.num_patches

        def weights(*shape):
            return torch.nn.Parameter(torch.randn(*shape, generator=generator) * 0.5)

        self.cls_token = weights(1, 1, dims)
        self.channel_token = weights(1, slots, 1, dims)
        self.pos_embed = weights(1, 1, count + 1, dims)
        self.pos_drop = torch.nn.Dropout(p=0.0)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(dims, 2, 32, dropout=0.0,
                                             batch_first=True)
            for _ in range(depth)])
        self.norm = torch.nn.LayerNorm(dims)
        self.lengths = []
        self.learner_inputs = []
        self.blocks[0].register_forward_pre_hook(
            lambda module, args, kwargs: self.lengths.append(int(args[0].shape[1])),
            with_kwargs=True)
        self.token_learner.register_forward_pre_hook(
            lambda module, args: self.learner_inputs.append(tuple(args[0].shape)))

    def add_pos_encoding_per_channel(self, x, w, h, class_pos_embed=False):
        if class_pos_embed:
            return self.pos_embed[:, :, 0]
        return self.pos_embed[:, :, 1:]

    def forward(self, x, index, list_num_channels):
        """The published path: pad to every slot, mask the padding."""
        slots = self.max_channels
        _, _, w, h = x.shape
        tokens = self.token_learner(x)
        chunks = torch.split(tokens, list_num_channels[index], dim=0)
        padded = torch.stack([
            torch.cat([chunk, torch.zeros(slots - chunk.size(0), chunk.size(1),
                                          chunk.size(2))], dim=0)
            for chunk in chunks], dim=0)
        count = padded.size(2)
        mask = torch.all(padded.reshape(padded.size(0), -1, padded.size(3)) == 0,
                         dim=-1)
        padded = padded + self.add_pos_encoding_per_channel(padded, w, h)
        padded = padded + self.channel_token.expand(padded.shape[0], -1, count, -1)
        sequence = padded.reshape(padded.size(0), -1, padded.size(3))
        head = self.cls_token.expand(sequence.shape[0], -1, -1) + \
            self.add_pos_encoding_per_channel(sequence, w, h, class_pos_embed=True)
        sequence = torch.cat([head, sequence], dim=1)
        mask = torch.cat([torch.zeros(mask.size(0), 1, dtype=torch.bool), mask], dim=1)
        for block in self.blocks:
            sequence = block(sequence, src_key_padding_mask=mask)
        return self.norm(sequence)[:, 0]


def _padded(net, x):
    n, k, h, w = x.shape
    flat = torch.nn.functional.instance_norm(x).reshape(n * k, 1, h, w)
    return net(flat, index=0, list_num_channels=[[k] * n])


@pytest.mark.parametrize("channels", [1, 3])
def test_the_real_channel_count_gives_the_padded_class_token(channels):
    net = _ChAda().eval()
    x = torch.rand(4, channels, 32, 32, generator=torch.Generator().manual_seed(channels))
    with torch.no_grad():
        reference = _padded(net, x)
        padded_length = net.lengths[-1]
        actual = emb._chada_class_token(net, x, torch)
        actual_length = net.lengths[-1]
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
    tokens = net.token_learner.num_patches
    assert padded_length == 1 + 10 * tokens
    assert actual_length == 1 + channels * tokens


def test_channel_tokens_still_tell_the_channels_apart():
    net = _ChAda().eval()
    plane = torch.rand(2, 1, 32, 32, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        one = emb._chada_class_token(net, plane, torch)
        swapped = emb._chada_class_token(
            net, torch.cat([plane, torch.zeros_like(plane) + 0.5], dim=1), torch)
    assert one.shape == swapped.shape == (2, 16)
    assert not torch.allclose(one, swapped)


def test_more_channels_than_slots_is_refused():
    net = _ChAda(slots=2).eval()
    with pytest.raises(emb.EmbeddingError, match="at most 2 channels"):
        emb._chada_class_token(net, torch.rand(1, 3, 32, 32), torch)


def test_the_loaded_encoder_feeds_each_crop_once_at_its_training_size(monkeypatch):
    net = _ChAda(size=224, patch=16).eval()
    module = types.ModuleType("transformers")

    class AutoModel:
        @staticmethod
        def from_pretrained(repo, revision, trust_remote_code):
            return net

    module.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", module)
    encode = emb._foundation_encoder(emb.EmbeddingSpec(
        backbone="chada_vit", device="cpu", batch_size=4))
    assert net.return_all_tokens is False and encode.in_channels is None
    stack = np.random.default_rng(0).random((3, 40, 40, 2)).astype(np.float32)
    features = encode(stack)
    assert features.shape == (3, 16)
    assert net.learner_inputs == [(6, 1, 224, 224)]
    assert net.lengths == [1 + 2 * 196]


_CACHED = os.path.expanduser(
    "~/.cache/huggingface/modules/transformers_modules/nicoboou/"
    "chadavit16-moyen/" + emb._FOUNDATION_MODELS["chada_vit"]["revision"])


@pytest.mark.skipif(not os.path.isdir(_CACHED),
                    reason="the published ChAda-ViT code is not cached here")
def test_the_published_model_code_gives_the_same_class_token(monkeypatch):
    pytest.importorskip("transformers")
    monkeypatch.syspath_prepend(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(_CACHED)))))
    base = ("transformers_modules.nicoboou.chadavit16-moyen."
            + os.path.basename(_CACHED))
    config = importlib.import_module(base + ".config_chada_vit")
    modeling = importlib.import_module(base + ".modeling_chada_vit")
    torch.manual_seed(0)
    net = modeling.ChAdaViTModel(config.ChAdaViTConfig(
        img_size=[32], embed_dim=24, depth=2, num_heads=4,
        return_all_tokens=False)).eval()
    for channels in (1, 3):
        x = torch.rand(3, channels, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(emb._chada_class_token(net, x, torch),
                                       _padded(net, x), rtol=1e-4, atol=1e-5)
