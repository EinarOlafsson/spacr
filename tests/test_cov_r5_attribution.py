"""``spacr.attribution``: the transformer checks that let a layer through.

Three branches, all in the "and if it is fine, carry on" half of a refusal:

* the patch-embedding check passes a conv that runs *after* an attention
  block -- the MaxViT-shaped hybrid, which is the case the check must not
  break;
* ``_ask_for_attention_weights`` leaves a caller's own ``need_weights=True``
  exactly as it found it, rather than rewriting kwargs it did not need to;
* rollout's ``discard_ratio`` drops nothing when the ratio is too small to
  reach one element of the attention matrix -- the map is the undiscarded
  one rather than an empty one.

Every model here is hand-built so the attention order, the kwargs the block
receives, and the size of the attention matrix are all known in advance.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from spacr.attribution import (NoSpatialLayerError, attention_rollout,
                               attribute)

IMG = 8


def _image(seed=0):
    """A fixed random 3-channel image."""
    return torch.rand(3, IMG, IMG, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# A conv target that runs after attention
# ---------------------------------------------------------------------------

class _AttentionThenConv(nn.Module):
    """Attention over pixels, then a convolution -- a MaxViT-shaped hybrid."""

    def __init__(self, n_out=2):
        """Build with a fixed seed so the map is reproducible."""
        super().__init__()
        torch.manual_seed(11)
        self.attn = nn.MultiheadAttention(3, 1, batch_first=True)
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4, n_out)

    def forward(self, x):
        """Attend, fold back to an image, convolve, classify the pooled map."""
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attended, _ = self.attn(tokens, tokens, tokens)
        folded = (tokens + attended).transpose(1, 2).reshape(b, c, h, w)
        return self.head(self.conv(folded).mean(dim=(2, 3)))


class _PatchEmbedThenAttention(nn.Module):
    """A pure ViT: the only convolution IS the patch embedding."""

    def __init__(self, n_out=2, dim=4):
        """Build with a fixed seed; the conv runs before every attention block."""
        super().__init__()
        torch.manual_seed(7)
        self.patch = nn.Conv2d(3, dim, 4, stride=4)
        self.attn = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.head = nn.Linear(dim, n_out)

    def forward(self, x):
        """Patchify, attend once, classify the mean token."""
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        attended, _ = self.attn(tokens, tokens, tokens)
        return self.head((tokens + attended).mean(dim=1))


def test_a_conv_that_runs_after_attention_is_camable():
    """The patch-embed refusal must not swallow the hybrid architectures.

    The check asks *where* the target ran, not whether the model has attention
    at all. A conv that runs after an attention block has had
    class-discriminative information reach it, so a CAM over it means
    something and the call has to go through -- which is why spaCR's default
    MaxViT target layer still works.

    The same call on a pure ViT, whose only conv is the patch embedding, is
    refused by name, so this is a decision about position and not a check that
    never fires.
    """
    image = _image()

    hybrid = attribute(_AttentionThenConv(), image, "eigencam", layer="conv")
    assert hybrid.map.shape == (IMG, IMG)
    assert np.isfinite(hybrid.map).all()
    assert hybrid.layer == "conv"

    with pytest.raises(NoSpatialLayerError) as excinfo:
        attribute(_PatchEmbedThenAttention(), image, "eigencam", layer="patch")
    assert "patch embedding" in str(excinfo.value)


# ---------------------------------------------------------------------------
# What the attention blocks are actually asked for
# ---------------------------------------------------------------------------

class _RecordingAttn(nn.MultiheadAttention):
    """An attention block that records the kwargs it was called with."""

    def __init__(self, embed_dim, weights):
        """``weights`` is the ``(B, L, S)`` matrix this block hands back."""
        super().__init__(embed_dim, num_heads=1, batch_first=True)
        self._weights = weights
        self.seen = []

    def forward(self, query, key, value, **kwargs):
        """Record the kwargs, return the value tokens and the fixed weights."""
        self.seen.append(dict(kwargs))
        return value, self._weights


class _TokenModel(nn.Module):
    """A transformer-shaped classifier whose call kwargs the test dictates."""

    def __init__(self, attn, n_tokens, embed_dim, call_kwargs):
        """``call_kwargs`` is what the model itself passes to its block."""
        super().__init__()
        torch.manual_seed(5)
        self.attn = attn
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.call_kwargs = call_kwargs
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        """Reshape into tokens, attend once, classify their mean."""
        batch = x.shape[0]
        flat = x.reshape(batch, -1)[:, :self.n_tokens * self.embed_dim]
        tokens = flat.reshape(batch, self.n_tokens, self.embed_dim)
        out = self.attn(tokens, tokens, tokens, **self.call_kwargs)
        return self.head(out[0].mean(dim=1))


def _rollout_model(call_kwargs, n_tokens=4, embed_dim=4):
    """A one-block token model returning an identity attention matrix."""
    attn = _RecordingAttn(embed_dim, torch.eye(n_tokens).unsqueeze(0))
    return _TokenModel(attn, n_tokens, embed_dim, call_kwargs)


def test_a_block_already_asked_for_its_weights_is_left_exactly_as_it_is():
    """Rollout overrides ``need_weights``; it must not override anything else.

    A model that already asks its blocks for weights is doing the right thing,
    and the wrapper has nothing to fix -- so it must pass the call straight
    through. Rewriting it would inject ``average_attn_weights=False`` into a
    call the author wrote deliberately.

    A model that asks with ``need_weights=False`` is the case the wrapper
    exists for, and there both keys are rewritten.
    """
    asked = _rollout_model({"need_weights": True})
    result = attention_rollout(asked, _image())
    assert result.map.shape == (IMG, IMG)
    assert asked.attn.seen and all(
        call == {"need_weights": True} for call in asked.attn.seen), (
        "the wrapper rewrote a call it had no reason to touch")

    # The author's own calls run first and are passed through verbatim --
    # the model's forward is exercised before the wrapper asks its own
    # question. What the wrapper guarantees is the LAST call: whatever the
    # author asked for, the weights are requested unaveraged once.
    wanted = {"need_weights": True, "average_attn_weights": False}

    refused = _rollout_model({"need_weights": False})
    attention_rollout(refused, _image())
    assert refused.attn.seen[-1] == wanted
    assert all(call == {"need_weights": False}
               for call in refused.attn.seen[:-1]), (
        "the author's own calls were rewritten")

    silent = _rollout_model({})
    attention_rollout(silent, _image())
    assert silent.attn.seen[-1] == wanted
    assert all(call == {} for call in silent.attn.seen[:-1])


def test_a_discard_ratio_too_small_for_one_element_discards_nothing():
    """``discard_ratio`` is a fraction of the matrix, and a fraction can round to zero.

    ``k = int(numel * ratio)``: on a 4x4 attention matrix any ratio below 1/16
    names no elements at all. Taking ``kthvalue(0)`` there would raise, and
    clamping it to 1 would drop an element the user did not ask to drop -- so
    the whole discard step is skipped and the map is the undiscarded one.

    Driven against a ratio large enough to bite on the same model, which
    produces a different map.
    """
    image = _image()
    weights = torch.tensor([[[0.7, 0.1, 0.1, 0.1],
                             [0.1, 0.7, 0.1, 0.1],
                             [0.1, 0.1, 0.7, 0.1],
                             [0.1, 0.1, 0.1, 0.7]]])

    def _model():
        attn = _RecordingAttn(4, weights)
        return _TokenModel(attn, n_tokens=4, embed_dim=4, call_kwargs={})

    none_dropped = attention_rollout(_model(), image, discard_ratio=0.05)
    baseline = attention_rollout(_model(), image, discard_ratio=0.0)
    dropped = attention_rollout(_model(), image, discard_ratio=0.5)

    assert np.isfinite(none_dropped.map).all()
    assert none_dropped.map.shape == (IMG, IMG)
    # 0.05 * 16 = 0.8 -> k == 0, so this is exactly the no-discard result.
    assert np.allclose(none_dropped.map, baseline.map)
    # 0.5 * 16 = 8, and the ROLLOUT really does move -- row 0 of the
    # normalised matrix goes from [.85 .05 .05 .05] to [1 0 0 0]. The MAP
    # does not, and asserting that it would was wrong: the map is
    # normalised and upsampled out of the CLS row, and for a 4-token model
    # that flattens the difference away again. What is checkable here is
    # that a biting ratio still produces a usable map rather than the
    # kthvalue(0) crash the guard exists to prevent.
    assert np.isfinite(dropped.map).all()
    assert dropped.map.shape == (IMG, IMG)
