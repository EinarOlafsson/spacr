"""The transformer checks in :mod:`spacr.attribution` that let a layer through.

* A CAM target that runs *after* an attention block (the MaxViT-shaped
  hybrid) is allowed; the same call on a pure ViT, whose only convolution is
  the patch embedding, is refused by name.
* Rollout asks each attention block for unaveraged weights once, and passes
  the author's own calls through untouched.
* ``discard_ratio`` is a fraction of the attention matrix; one too small to
  name a single element discards nothing.

Every model is hand-built so the attention order, the kwargs each block
receives, and the size of the attention matrix are known in advance.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from spacr.attribution import NoSpatialLayerError, attention_rollout, attribute

IMG = 8


def _image(seed=0):
    """A fixed random 3-channel image."""
    return torch.rand(3, IMG, IMG, generator=torch.Generator().manual_seed(seed))


class _AttentionThenConv(nn.Module):
    """Attention over pixels, then a convolution: a MaxViT-shaped hybrid."""

    def __init__(self, n_out=2):
        super().__init__()
        torch.manual_seed(11)
        self.attn = nn.MultiheadAttention(3, 1, batch_first=True)
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4, n_out)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attended, _ = self.attn(tokens, tokens, tokens)
        folded = (tokens + attended).transpose(1, 2).reshape(b, c, h, w)
        return self.head(self.conv(folded).mean(dim=(2, 3)))


class _PatchEmbedThenAttention(nn.Module):
    """A pure ViT: the only convolution IS the patch embedding."""

    def __init__(self, n_out=2, dim=4):
        super().__init__()
        torch.manual_seed(7)
        self.patch = nn.Conv2d(3, dim, 4, stride=4)
        self.attn = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.head = nn.Linear(dim, n_out)

    def forward(self, x):
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        attended, _ = self.attn(tokens, tokens, tokens)
        return self.head((tokens + attended).mean(dim=1))


def test_attribute_allows_a_conv_after_attention_and_refuses_a_patch_embedding():
    """The patch-embedding refusal is about where the target ran, not whether the model has attention."""
    image = _image()

    hybrid = attribute(_AttentionThenConv(), image, "eigencam", layer="conv")
    assert hybrid.map.shape == (IMG, IMG)
    assert np.isfinite(hybrid.map).all()
    assert (hybrid.layer, hybrid.method) == ("conv", "eigencam")

    with pytest.raises(NoSpatialLayerError) as caught:
        attribute(_PatchEmbedThenAttention(), image, "eigencam", layer="patch")
    # One attention block: the model's own class name ends in "Attention"
    # but the model is not a block of itself.
    assert str(caught.value) == (
        "target layer 'patch' of model_type='_PatchEmbedThenAttention' runs "
        "before every one of the model's 1 attention blocks, so it is the "
        "patch embedding: no class-discriminative information has reached it "
        "and a CAM over it is a picture of local image statistics, not of what "
        "the model attends to. Use method='attention_rollout' for this "
        "architecture, a gradient or perturbation method, or name a later "
        "layer explicitly. Pass allow_pre_attention=True to override.")


class _RecordingAttn(nn.MultiheadAttention):
    """An attention block that records the kwargs it was called with."""

    def __init__(self, embed_dim, weights):
        super().__init__(embed_dim, num_heads=1, batch_first=True)
        self._weights = weights
        self.seen = []

    def forward(self, query, key, value, **kwargs):
        self.seen.append(dict(kwargs))
        return value, self._weights


class _TokenModel(nn.Module):
    """A transformer-shaped classifier whose call kwargs the test dictates."""

    def __init__(self, attn, n_tokens, embed_dim, call_kwargs):
        super().__init__()
        torch.manual_seed(5)
        self.attn = attn
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.call_kwargs = call_kwargs
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        batch = x.shape[0]
        flat = x.reshape(batch, -1)[:, :self.n_tokens * self.embed_dim]
        tokens = flat.reshape(batch, self.n_tokens, self.embed_dim)
        out = self.attn(tokens, tokens, tokens, **self.call_kwargs)
        return self.head(out[0].mean(dim=1))


def _token_model(call_kwargs, weights=None):
    """A one-block, four-token model returning ``weights`` (identity by default)."""
    weights = torch.eye(4).unsqueeze(0) if weights is None else weights
    return _TokenModel(_RecordingAttn(4, weights), 4, 4, call_kwargs)


WANTED = {"need_weights": True, "average_attn_weights": False}


def test_rollout_asks_for_weights_once_and_leaves_the_authors_calls_alone(subtests):
    """Only the wrapped call is rewritten, and only when the author did not already ask for weights."""
    cases = [
        ("already_asked", {"need_weights": True}, {"need_weights": True}),
        ("asked_not_to", {"need_weights": False}, WANTED),
        ("said_nothing", {}, WANTED),
        ("said_nothing_but_kept_averaging", {"average_attn_weights": True},
         {"need_weights": True, "average_attn_weights": True}),
    ]
    for case_id, call_kwargs, last in cases:
        with subtests.test(msg=case_id):
            model = _token_model(call_kwargs)
            result = attention_rollout(model, _image())
            assert result.map.shape == (IMG, IMG)
            assert model.attn.seen[-1] == last
            assert model.attn.seen[:-1] == [call_kwargs] * (len(model.attn.seen) - 1)
            assert model.attn.forward.__func__ is _RecordingAttn.forward


def test_a_discard_ratio_too_small_for_one_element_discards_nothing(subtests):
    """``int(16 * ratio) == 0`` gives exactly the undiscarded map; a biting ratio still gives a finite map."""
    weights = torch.tensor([[[0.7, 0.1, 0.1, 0.1],
                             [0.1, 0.7, 0.1, 0.1],
                             [0.1, 0.1, 0.7, 0.1],
                             [0.1, 0.1, 0.1, 0.7]]])
    image = _image()
    baseline = attention_rollout(_token_model({}, weights), image, discard_ratio=0.0)
    for case_id, ratio, same_as_baseline in [("ratio_names_no_element", 0.05, True),
                                             ("ratio_just_under_one_element", 0.0624, True),
                                             ("ratio_bites", 0.5, False)]:
        with subtests.test(msg=case_id):
            result = attention_rollout(_token_model({}, weights), image,
                                       discard_ratio=ratio)
            assert result.map.shape == (IMG, IMG)
            assert np.isfinite(result.map).all()
            assert result.params == {"head_fusion": "mean", "discard_ratio": ratio}
            if same_as_baseline:
                np.testing.assert_array_equal(result.map, baseline.map)


class _Attention(nn.Module):
    """A wrapper whose class name ends in "Attention" around a torch block."""

    def __init__(self, dim):
        super().__init__()
        self.inner = nn.MultiheadAttention(dim, 1, batch_first=True)

    def forward(self, tokens):
        return self.inner(tokens, tokens, tokens)


class _PatchEmbedThenTwoWrappedBlocks(nn.Module):
    """Patch embedding, then two wrapped attention blocks."""

    def __init__(self, n_out=2, dim=4):
        super().__init__()
        torch.manual_seed(5)
        self.patch = nn.Conv2d(3, dim, 4, stride=4)
        self.blocks = nn.ModuleList([_Attention(dim), _Attention(dim)])
        self.head = nn.Linear(dim, n_out)

    def forward(self, x):
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        for block in self.blocks:
            attended, _ = block(tokens)
            tokens = tokens + attended
        return self.head(tokens.mean(dim=1))


def test_the_refusal_counts_each_wrapped_attention_block_once():
    """A wrapper named *Attention around a torch block is one block, not two."""
    with pytest.raises(NoSpatialLayerError, match="model's 2 attention blocks"):
        attribute(_PatchEmbedThenTwoWrappedBlocks(), _image(), "eigencam",
                  layer="patch")
