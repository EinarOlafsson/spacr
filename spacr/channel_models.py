"""Channel-adaptive models for any stain panel: a CNN, a ViT and a well set transformer.

All three models take any number of channels (1 to more than 10) with the
same weights. Every channel is processed on its own and tagged with a learned
stain embedding, looked up by stain name rather than by slot index, so
BBBC021 "DNA" and an OPS "DAPI" share one vector and an unseen panel still
runs.

* :class:`SpacrChannelCNN` is a channel-adaptive ConvNeXt-style CNN: a
  HyperNet stem per channel, ConvNeXt blocks shared across channels and
  FiLM-conditioned on the stain, cross-channel attention at every spatial
  location (one feature map per channel is kept) and channel attention
  pooling at the end, which records the stain that drove the call.
* :class:`SpacrChannelViT` is a channel-aware ViT: per-channel patch tokens
  plus a stain embedding and a 2-D sin-cos position, an optional per-sample
  channel-availability mask for missing stains, and Hierarchical Channel
  Sampling (channel dropout, Bao et al. 2024) in training. It is pre-trained
  with :class:`ChannelMAE` (He et al. 2022) or :class:`ChannelDINO` (Caron et
  al. 2021).
* :class:`SpacrWellTransformer` is a set model over the single-cell
  embeddings of a well. Mode ``'arrayed'``: an ISAB set encoder (Lee et al.
  2019) and gated multi-head attention-MIL pooling (Ilse et al. 2018) give
  the well logits, the attention weights and per-cell instance logits (the
  penetrant fraction, :func:`penetrant_fraction`). Mode ``'pooled'``: per-cell
  logits conditioned on a context set of other cells of the same well; an
  empty context is a learned null token, so the no-context ablation is the
  same model with set size 0.

:func:`build_channel_model` is the registry: it builds any of the three by
the name in :data:`CHANNEL_ARCHITECTURES`, which is also what the train
setting ``train_arch`` and :func:`spacr.utils.choose_model` accept.
:class:`ChannelClassifier` gives the two image models the ``model(x) ->
logits`` interface of the rest of spaCR's train path.

Inputs are expected already normalised by the benchmark transform,
``((x / scale).clip(0, 1.5) - 0.5) / 0.25``. The models use plain torch
modules and no BatchNorm, so a step with a zero learning rate leaves every
weight unchanged.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: The architectures :func:`build_channel_model` builds, by ``train_arch`` name.
CHANNEL_ARCHITECTURES = ('spacr_channel_cnn', 'spacr_channel_vit', 'spacr_well_transformer')

#: The two image architectures, which :func:`spacr.utils.choose_model` also builds.
CHANNEL_IMAGE_ARCHITECTURES = ('spacr_channel_cnn', 'spacr_channel_vit')

#: One learned embedding per stain or biological target, shared by every dataset.
#: Unknown stains map to the ``other_<k>`` slots (``k`` = slot index).
STAIN_VOCAB: List[str] = [
    "dna", "tubulin", "actin", "gh2ax", "er", "rna", "agp", "mito", "parasite", "gbp1", "ubiquitin",
    "brightfield", "membrane", "golgi", "lysosome", "protein_of_interest",
] + [f"other_{k}" for k in range(16)]

#: Common stain names and the vocabulary entry they mean.
STAIN_ALIASES: Dict[str, str] = {
    "dapi": "dna", "hoechst": "dna", "nuclei": "dna", "nucleus": "dna", "gfp_tubulin": "tubulin", "gfp": "tubulin",
    "phalloidin": "actin", "af750": "actin", "f-actin": "actin", "a594": "gh2ax", "gamma-h2ax": "gh2ax",
    "concanavalin": "er", "syto14": "rna", "wga": "agp", "mitotracker": "mito", "pathogen": "parasite",
}

#: The channel order of each benchmark dataset as stored on disk.
DATASET_STAINS: Dict[str, List[str]] = {
    "ops": ["dna", "tubulin", "gh2ax", "actin"],
    "bbbc021": ["dna", "tubulin", "actin"],
    "jump": ["dna", "er", "rna", "agp", "mito"],
    "hrman": ["dna", "parasite", "gbp1"],
}

#: ViT sizes: (embedding dimension, depth, heads).
VIT_SIZES = {"tiny": (192, 12, 3), "small": (384, 12, 6), "mini": (96, 4, 3)}


def stain_ids(names: Sequence[str]) -> torch.Tensor:
    """Vocabulary indices of a list of stain names.

    :param names: stain names in channel order; aliases such as ``'dapi'``
        are resolved, and an unknown name in slot ``k`` takes ``other_<k % 16>``.
    :returns: a long tensor with one index per name.
    """
    out = []
    for k, n in enumerate(names):
        n = STAIN_ALIASES.get(str(n).lower(), str(n).lower())
        out.append(STAIN_VOCAB.index(n) if n in STAIN_VOCAB else STAIN_VOCAB.index(f"other_{k % 16}"))
    return torch.tensor(out, dtype=torch.long)


def hierarchical_channel_sample(C: int, generator: Optional[torch.Generator] = None, min_keep: int = 1,
                                always: Sequence[int] = ()) -> torch.Tensor:
    """Hierarchical Channel Sampling (Bao et al. 2024): draw a count, then that many channels.

    :param C: channel count.
    :param generator: torch generator; None uses the global one.
    :param min_keep: the smallest count drawn; the count is uniform on ``min_keep..C``.
    :param always: channel slots that are always kept (such as the DNA slot).
    :returns: the kept channel indices, sorted.
    """
    k = int(torch.randint(min_keep, C + 1, (1,), generator=generator))
    always = [a for a in always if a < C]
    rest = [c for c in range(C) if c not in always]
    k_rest = max(0, k - len(always))
    pick = torch.randperm(len(rest), generator=generator)[:k_rest].tolist()
    return torch.tensor(sorted(always + [rest[i] for i in pick]), dtype=torch.long)


class DropPath(nn.Module):
    """Stochastic depth: drop a whole residual branch per sample in training.

    :param p: drop probability.
    """

    def __init__(self, p=0.0):
        """Store the drop probability."""
        super().__init__()
        self.p = p

    def forward(self, x):
        """Return ``x`` with whole samples zeroed at rate ``p`` and the rest rescaled.

        :param x: the branch output; the first axis is the sample axis.
        :returns: a tensor shaped like ``x``.
        """
        if not self.training or self.p == 0:
            return x
        keep = (torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device) > self.p).to(x.dtype)
        return x * keep / (1 - self.p)


class GatedAttentionPool(nn.Module):
    """Gated multi-head attention pooling (Ilse et al. 2018).

    :param dim: token dimension.
    :param hidden: hidden size of the gate.
    :param heads: pooling heads.
    :param dropout: dropout on the gate.
    """

    def __init__(self, dim, hidden=128, heads=1, dropout=0.0):
        """Build the two gate projections and the head projection."""
        super().__init__()
        self.V, self.U = nn.Linear(dim, hidden), nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, heads)
        self.drop = nn.Dropout(dropout)
        self.heads = heads

    def forward(self, x, mask=None):
        """Pool a set of tokens.

        :param x: tokens shaped ``(B, N, D)``.
        :param mask: ``(B, N)`` booleans, True for a valid token; None keeps all.
        :returns: the pooled ``(B, heads * D)`` tensor and the ``(B, heads, N)`` weights;
            a row with no valid token gets zero weights.
        """
        a = self.w(self.drop(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))))
        if mask is not None:
            a = a.masked_fill(~mask[..., None], float("-inf"))
        a = torch.softmax(a, 1).transpose(1, 2)
        a = torch.nan_to_num(a, 0.0)
        return (a @ x).flatten(1), a


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel axis of a ``(B, D, H, W)`` feature map."""

    def forward(self, x):
        """Normalise every pixel's feature vector.

        :param x: feature map shaped ``(B, D, H, W)``.
        :returns: the normalised map, same shape.
        """
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (Liu et al. 2022) with FiLM from the stain embedding.

    Depth-wise 7x7 convolution, LayerNorm, FiLM, a 4x MLP with GELU and a
    layer scale, with a residual connection. The FiLM lets the channel-shared
    weights specialise per stain; it starts at zero, so a new block is a
    plain ConvNeXt block.

    :param dim: feature dimension.
    :param stain_dim: stain embedding dimension.
    :param drop_path: stochastic depth rate.
    :param ls_init: initial layer scale.
    """

    def __init__(self, dim, stain_dim, drop_path=0.0, ls_init=1e-6):
        """Build the convolution, the FiLM projection and the MLP."""
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Linear(stain_dim, 2 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.pw1, self.pw2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(ls_init * torch.ones(dim))
        self.dp = DropPath(drop_path)

    def forward(self, x, s):
        """Apply the block.

        :param x: per-channel feature maps shaped ``(B * C, D, h, w)``.
        :param s: the stain embedding of each map, shaped ``(B * C, stain_dim)``.
        :returns: the updated maps, same shape as ``x``.
        """
        h = self.dw(x).permute(0, 2, 3, 1)
        g, b = self.film(s).chunk(2, -1)
        h = self.norm(h) * (1 + g[:, None, None]) + b[:, None, None]
        h = self.pw2(F.gelu(self.pw1(h))) * self.gamma
        return x + self.dp(h.permute(0, 3, 1, 2))


class CrossChannelMix(nn.Module):
    """Self-attention across the channel tokens at every spatial location.

    The feature maps stay per channel, so channel identity is kept. The cost
    is linear in pixels and quadratic only in the channel count.

    :param dim: feature dimension.
    :param stain_dim: stain embedding dimension.
    :param heads: attention heads.
    :param ls_init: initial layer scale.
    """

    def __init__(self, dim, stain_dim, heads=4, ls_init=1e-4):
        """Build the projections."""
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.s = nn.Linear(stain_dim, dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.heads = heads
        self.gamma = nn.Parameter(ls_init * torch.ones(dim))

    def forward(self, x, s, B, C, avail=None):
        """Mix the channels.

        :param x: per-channel feature maps shaped ``(B * C, D, h, w)``.
        :param s: the stain embeddings, shaped ``(C, stain_dim)``.
        :param B: batch size.
        :param C: channel count.
        :param avail: ``(B, C)`` booleans, True for a present channel; None treats all as present.
        :returns: the mixed maps, same shape as ``x``.
        """
        BC, D, h, w = x.shape
        t = x.reshape(B, C, D, h * w).permute(0, 3, 1, 2).reshape(B * h * w, C, D)
        se = self.s(s)[None]
        q, k, v = self.qkv(self.norm(t) + se).reshape(B * h * w, C, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        am = None
        if avail is not None:
            am = avail[:, None, None, None, :].expand(B, h * w, 1, 1, C).reshape(B * h * w, 1, 1, C)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am).transpose(1, 2).reshape(B * h * w, C, D)
        t = t + self.proj(o) * self.gamma
        return t.reshape(B, h * w, C, D).permute(0, 2, 3, 1).reshape(BC, D, h, w)


class HyperStem(nn.Module):
    """Per-channel patchify stem whose one-channel kernel is generated from the stain.

    ``W_c = W_0 + MLP(e_c)`` (HyperNet, the best from-scratch adaptive CNN on
    CHAMMI, Chen et al. 2023). The MLP starts at zero, so every stain starts
    with the shared kernel.

    :param dim: output feature dimension.
    :param stain_dim: stain embedding dimension.
    :param k: kernel size and stride.
    """

    def __init__(self, dim, stain_dim, k=4):
        """Build the shared kernel, its bias and the HyperNet."""
        super().__init__()
        self.k, self.dim = k, dim
        self.w0 = nn.Parameter(torch.randn(dim, 1, k, k) * (2.0 / (k * k)) ** 0.5)
        self.b0 = nn.Parameter(torch.zeros(dim))
        self.hyper = nn.Sequential(nn.Linear(stain_dim, 64), nn.GELU(), nn.Linear(64, dim * k * k + dim))
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)
        self.norm = LayerNorm2d(dim)

    def forward(self, x, s):
        """Patchify every channel with its own kernel.

        :param x: images shaped ``(B, C, H, W)``.
        :param s: the stain embeddings, shaped ``(C, stain_dim)``.
        :returns: feature maps shaped ``(B * C, dim, H / k, W / k)``.
        """
        B, C, H, W = x.shape
        d = self.hyper(s)
        w = self.w0[None] + d[:, : self.dim * self.k * self.k].reshape(C, self.dim, 1, self.k, self.k)
        b = self.b0[None] + d[:, self.dim * self.k * self.k:]
        y = F.conv2d(x, w.reshape(C * self.dim, 1, self.k, self.k), b.reshape(-1), stride=self.k, groups=C)
        y = y.reshape(B, C, self.dim, H // self.k, W // self.k).reshape(B * C, self.dim, H // self.k, W // self.k)
        return self.norm(y)


class SpacrChannelCNN(nn.Module):
    """spaCR channel-adaptive CNN.

    Default size: dims (32, 64, 128, 256), depths (2, 2, 4, 2), about 2.4 M
    parameters. Height and width must be divisible by 32 (96, 128 and 224 all
    work). In training, with ``channel_dropout``, a random channel subset is
    kept per batch (the ``always_keep`` stains are never dropped). At test
    time ``eval_channel_subset`` keeps only the named stains, the
    missing-stain evaluation.

    :param stains: stain names of the input channels in order.
    :param dims: feature dimension per stage.
    :param depths: blocks per stage.
    :param stain_dim: stain embedding dimension.
    :param drop_path: largest stochastic depth rate.
    :param mix_heads: heads of the cross-channel mixing.
    :param channel_dropout: Hierarchical Channel Sampling in training.
    :param always_keep: stains never dropped.
    :param num_classes: classes of the attached head; 0 attaches none.
    :param head_dropout: dropout before the head.
    """

    def __init__(self, stains: Sequence[str] = DATASET_STAINS["ops"], dims=(32, 64, 128, 256), depths=(2, 2, 4, 2),
                 stain_dim=32, drop_path=0.1, mix_heads=4, channel_dropout=True, always_keep=("dna",),
                 num_classes: int = 0, head_dropout=0.1):
        """Build the stem, the stages, the mixing layers and the channel pooling."""
        super().__init__()
        self.stains = list(stains)
        self.stain_embed = nn.Embedding(len(STAIN_VOCAB), stain_dim)
        nn.init.trunc_normal_(self.stain_embed.weight, std=0.5)
        self.stem = HyperStem(dims[0], stain_dim)
        dpr = np.linspace(0, drop_path, sum(depths)).tolist()
        self.stages, self.mixes, self.downs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        i = 0
        for si, (d, n) in enumerate(zip(dims, depths)):
            self.stages.append(nn.ModuleList([ConvNeXtBlock(d, stain_dim, dpr[i + j]) for j in range(n)]))
            i += n
            self.mixes.append(CrossChannelMix(d, stain_dim, heads=mix_heads))
            if si < len(dims) - 1:
                self.downs.append(nn.Sequential(LayerNorm2d(d), nn.Conv2d(d, dims[si + 1], 2, 2)))
        self.final_norm = nn.LayerNorm(dims[-1])
        self.ch_pool = GatedAttentionPool(dims[-1], hidden=64, heads=1)
        self.ch_s = nn.Linear(stain_dim, dims[-1])
        self.num_features = dims[-1]
        self.channel_dropout, self.always_keep = channel_dropout, always_keep
        self.eval_channel_subset: Optional[list] = None
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(dims[-1], num_classes)) if num_classes else None
        self.last_channel_weights = None

    def _select(self, x, stains):
        """Apply channel dropout in training or ``eval_channel_subset`` at test time.

        :param x: images shaped ``(B, C, H, W)``.
        :param stains: their stain names.
        :returns: the kept images and their stain names.
        """
        idx = None
        if self.training and self.channel_dropout and x.shape[1] > 1:
            keep = [k for k, s in enumerate(stains) if STAIN_ALIASES.get(s, s) in self.always_keep]
            idx = hierarchical_channel_sample(x.shape[1], always=keep)
        elif not self.training and self.eval_channel_subset:
            idx = torch.tensor([k for k, s in enumerate(stains) if s in self.eval_channel_subset])
        if idx is not None:
            x, stains = x[:, idx], [stains[k] for k in idx.tolist()]
        return x, stains

    def forward(self, x, stains: Optional[Sequence[str]] = None, avail: Optional[torch.Tensor] = None):
        """Embed a batch of images.

        :param x: images shaped ``(B, C, H, W)``.
        :param stains: stain names of the channels; None uses the model's.
        :param avail: ``(B, C)`` booleans, True for a present channel. With a
            mask, no channel is dropped or selected, and the absent ones are zeroed.
        :returns: the ``(B, dims[-1])`` embedding. ``last_channel_weights``
            holds the channel attention, which stain drove the embedding.
        """
        stains = list(stains or self.stains)
        if len(stains) != x.shape[1]:
            raise ValueError(f"{len(stains)} stain names for {x.shape[1]} channels")
        if avail is None:
            x, stains = self._select(x, stains)
        B, C = x.shape[:2]
        s = self.stain_embed(stain_ids(stains).to(x.device))
        if avail is not None:
            x = x * avail[:, :, None, None].to(x.dtype)
        h = self.stem(x, s)
        sb = s[None].expand(B, -1, -1).reshape(B * C, -1)
        for si, blocks in enumerate(self.stages):
            for blk in blocks:
                h = blk(h, sb)
            h = self.mixes[si](h, s, B, C, avail)
            if si < len(self.downs):
                h = self.downs[si](h)
        t = self.final_norm(h.mean((2, 3))).reshape(B, C, -1) + self.ch_s(s)[None]
        z, a = self.ch_pool(t, avail)
        self.last_channel_weights = a.detach()
        return z

    def classify(self, x, stains=None, avail=None):
        """Class logits from the attached head.

        :param x: images shaped ``(B, C, H, W)``.
        :param stains: stain names of the channels; None uses the model's.
        :param avail: optional ``(B, C)`` channel-availability mask.
        :returns: ``(B, num_classes)`` logits.
        """
        return self.head(self.forward(x, stains, avail))


def sincos_2d(n_side, dim):
    """2-D sin-cos position embedding of an ``n_side`` x ``n_side`` patch grid.

    :param n_side: patches per side.
    :param dim: embedding dimension, divisible by 4.
    :returns: a float tensor shaped ``(n_side ** 2, dim)``.
    """
    gy, gx = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
    omega = 1.0 / 10000 ** (np.arange(dim // 4) / (dim / 4))
    out = [np.sin(gy.reshape(-1, 1) * omega), np.cos(gy.reshape(-1, 1) * omega),
           np.sin(gx.reshape(-1, 1) * omega), np.cos(gx.reshape(-1, 1) * omega)]
    return torch.tensor(np.concatenate(out, 1), dtype=torch.float32)


class Block(nn.Module):
    """Pre-norm transformer block with a key-padding mask and layer scale.

    With ``kv`` given it is a cross-attention block (queries ``x``, keys and
    values ``kv``).

    :param dim: token dimension.
    :param heads: attention heads.
    :param mlp: MLP expansion ratio.
    :param drop_path: stochastic depth rate.
    :param ls_init: initial layer scale.
    :param dropout: attention and MLP dropout.
    :param cross: add a LayerNorm on the keys and values.
    """

    def __init__(self, dim, heads, mlp=4.0, drop_path=0.0, ls_init=1e-4, dropout=0.0, cross=False):
        """Build the norms, the projections and the MLP."""
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.nkv = nn.LayerNorm(dim) if cross else None
        self.q, self.kv, self.proj = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(int(dim * mlp), dim))
        self.g1 = nn.Parameter(ls_init * torch.ones(dim))
        self.g2 = nn.Parameter(ls_init * torch.ones(dim))
        self.heads, self.dp, self.drop = heads, DropPath(drop_path), dropout

    def attn(self, x, kv, mask, need_weights=False):
        """Multi-head attention of ``x`` over ``kv``.

        :param x: queries shaped ``(B, N, D)``.
        :param kv: keys and values shaped ``(B, M, D)``.
        :param mask: ``(B, M)`` booleans, True to attend; a row with nothing
            to attend to gives a zero output instead of NaN.
        :param need_weights: also return the attention weights.
        :returns: the projected output and the ``(B, heads, N, M)`` weights, or None.
        """
        B, N, D = x.shape
        M = kv.shape[1]
        h = self.heads
        q = self.q(x).reshape(B, N, h, D // h).transpose(1, 2)
        k, v = self.kv(kv).reshape(B, M, 2, h, D // h).permute(2, 0, 3, 1, 4)
        am = None if mask is None else mask[:, None, None, :]
        if need_weights:
            w = (q @ k.transpose(-1, -2)) / math.sqrt(D // h)
            if am is not None:
                w = w.masked_fill(~am, float("-inf"))
            w = torch.nan_to_num(torch.softmax(w, -1), 0.0)
            return self.proj((w @ v).transpose(1, 2).reshape(B, N, D)), w
        if am is not None:
            empty = ~mask.any(1)
            if empty.any():
                am = am.clone()
                am[empty] = True
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=self.drop if self.training else 0.0)
        o = o.transpose(1, 2).reshape(B, N, D)
        if mask is not None and (~mask.any(1)).any():
            o = o * mask.any(1)[:, None, None].to(o.dtype)
        return self.proj(o), None

    def forward(self, x, kv=None, mask=None, need_weights=False):
        """Apply the block.

        :param x: tokens shaped ``(B, N, D)``.
        :param kv: keys and values for cross-attention; None attends to ``x``.
        :param mask: key-padding mask, True to attend.
        :param need_weights: also return the attention weights.
        :returns: the updated tokens, or the tokens and weights with ``need_weights``.
        """
        h = self.n1(x)
        kvn = h if kv is None else self.nkv(kv) if self.nkv is not None else kv
        a, w = self.attn(h, kvn, mask, need_weights)
        x = x + self.dp(a * self.g1)
        x = x + self.dp(self.mlp(self.n2(x)) * self.g2)
        return (x, w) if need_weights else x


class SpacrChannelViT(nn.Module):
    """spaCR channel-aware ViT.

    Tokens: one shared one-channel patch embedding per channel, plus the
    stain embedding and a 2-D sin-cos position shared over channels, so
    ``C * (img / patch) ** 2`` tokens and a CLS token. Default img 224 and
    patch 32 (49 tokens per channel). The native-resolution arm is img 96 /
    patch 12 (OPS) or img 128 / patch 16 (BBBC021). With
    ``bag_of_channels`` every channel is a separate sequence and the
    per-channel CLS vectors are fused late, which is cheaper and the better
    self-supervised setting on CHAMMI-75.

    :param stains: stain names of the input channels in order.
    :param img: input size; other sizes are resized to it.
    :param patch: patch size; ``img`` must be divisible by it.
    :param size: ``'tiny'``, ``'small'`` or ``'mini'`` (:data:`VIT_SIZES`).
    :param dim: embedding dimension; None takes the size's.
    :param depth: blocks; None takes the size's.
    :param heads: heads; None takes the size's.
    :param drop_path: largest stochastic depth rate.
    :param channel_dropout: Hierarchical Channel Sampling in training.
    :param always_keep: stains never dropped.
    :param num_classes: classes of the attached head; 0 attaches none.
    :param head_dropout: dropout before the head.
    :param bag_of_channels: encode every channel on its own.
    """

    def __init__(self, stains: Sequence[str] = DATASET_STAINS["ops"], img=224, patch=32, size="small", dim=None,
                 depth=None, heads=None, drop_path=0.1, channel_dropout=True, always_keep=("dna",),
                 num_classes=0, head_dropout=0.1, bag_of_channels=False):
        """Build the patch and stain embeddings, the blocks and the channel pooling."""
        super().__init__()
        self.boc = bag_of_channels
        d0, dp0, h0 = VIT_SIZES[size]
        dim, depth, heads = dim or d0, depth or dp0, heads or h0
        if img % patch:
            raise ValueError(f"img {img} is not divisible by patch {patch}")
        self.stains, self.img, self.patch, self.dim, self.n_side = list(stains), img, patch, dim, img // patch
        self.patch_embed = nn.Conv2d(1, dim, patch, patch)
        self.stain_embed = nn.Embedding(len(STAIN_VOCAB), dim)
        self.register_buffer("pos", sincos_2d(self.n_side, dim), persistent=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, drop_path=drop_path * i / max(depth - 1, 1)) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.ch_pool = GatedAttentionPool(dim, hidden=64)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.stain_embed.weight, std=0.02)
        self.num_features = 2 * dim
        self.channel_dropout, self.always_keep = channel_dropout, always_keep
        self.eval_channel_subset: Optional[list] = None
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(2 * dim, num_classes)) if num_classes else None
        self.last_channel_weights = None

    _select = SpacrChannelCNN._select

    def tokens(self, x, stains):
        """Per-channel patch tokens with position and stain embeddings.

        :param x: images shaped ``(B, C, H, W)``; resized to ``img`` when needed.
        :param stains: stain names of the channels.
        :returns: tokens shaped ``(B, C, N, D)``.
        """
        B, C, H, W = x.shape
        if (H, W) != (self.img, self.img):
            x = F.interpolate(x, size=(self.img, self.img), mode="bilinear", align_corners=False)
        t = self.patch_embed(x.reshape(B * C, 1, self.img, self.img)).flatten(2).transpose(1, 2)
        t = t.reshape(B, C, -1, self.dim)
        t = t + self.pos[None, None] + self.stain_embed(stain_ids(stains).to(x.device))[None, :, None]
        return t

    def encode(self, t, mask=None):
        """Run the blocks over a token sequence with a CLS token in front.

        :param t: tokens shaped ``(B, L, D)``.
        :param mask: ``(B, L)`` booleans, True for a valid token.
        :returns: the normalised ``(B, L + 1, D)`` sequence, CLS first.
        """
        x = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1)
        m = None if mask is None else torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
        for b in self.blocks:
            x = b(x, mask=m)
        return self.norm(x)

    def forward(self, x, stains: Optional[Sequence[str]] = None, avail: Optional[torch.Tensor] = None):
        """Embed a batch of images.

        :param x: images shaped ``(B, C, H, W)``.
        :param stains: stain names of the channels; None uses the model's.
        :param avail: ``(B, C)`` booleans, True for a present channel; the
            tokens of absent channels are masked out.
        :returns: the ``(B, 2 * dim)`` embedding: CLS (or the channel mean
            with ``bag_of_channels``) and the channel attention pool.
        """
        stains = list(stains or self.stains)
        if len(stains) != x.shape[1]:
            raise ValueError(f"{len(stains)} stain names for {x.shape[1]} channels")
        if avail is None:
            x, stains = self._select(x, stains)
        t = self.tokens(x, stains)
        B, C, N, D = t.shape
        if self.boc:
            z = self.encode(t.reshape(B * C, N, D))[:, 0].reshape(B, C, D)
            pooled, a = self.ch_pool(z, avail)
            self.last_channel_weights = a.detach()
            m = torch.ones(B, C, 1, device=z.device) if avail is None else avail[..., None].to(z.dtype)
            return torch.cat([(z * m).sum(1) / m.sum(1).clamp_min(1), pooled], 1)
        mask = None if avail is None else avail[:, :, None].expand(B, C, N).reshape(B, C * N)
        z = self.encode(t.reshape(B, C * N, D), mask)
        per_ch = z[:, 1:].reshape(B, C, N, D).mean(2)
        pooled, a = self.ch_pool(per_ch, avail)
        self.last_channel_weights = a.detach()
        return torch.cat([z[:, 0], pooled], 1)

    def classify(self, x, stains=None, avail=None):
        """Class logits from the attached head.

        :param x: images shaped ``(B, C, H, W)``.
        :param stains: stain names of the channels; None uses the model's.
        :param avail: optional ``(B, C)`` channel-availability mask.
        :returns: ``(B, num_classes)`` logits.
        """
        return self.head(self.forward(x, stains, avail))


class ChannelMAE(nn.Module):
    """Masked autoencoder (He et al. 2022) on channel-patch tokens.

    ``mask_ratio`` of all ``C * N`` tokens is masked at random, and with
    probability ``p_whole_channel`` per sample one whole channel is masked in
    addition, so the model learns to predict one stain from the others. One
    decoder is shared by all channels; the stain embedding tells it which to
    paint. The loss is the mean squared error on the masked patches, each
    patch normalised to zero mean and unit variance.

    :param enc: the encoder to pre-train.
    :param dec_dim: decoder dimension.
    :param dec_depth: decoder blocks.
    :param dec_heads: decoder heads.
    :param mask_ratio: share of tokens masked.
    :param p_whole_channel: probability of masking one whole channel too.
    """

    def __init__(self, enc: SpacrChannelViT, dec_dim=192, dec_depth=4, dec_heads=6, mask_ratio=0.75,
                 p_whole_channel=0.5):
        """Build the decoder around ``enc``."""
        super().__init__()
        self.enc, self.mask_ratio, self.p_whole = enc, mask_ratio, p_whole_channel
        self.to_dec = nn.Linear(enc.dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_stain = nn.Embedding(len(STAIN_VOCAB), dec_dim)
        self.register_buffer("dec_pos", sincos_2d(enc.n_side, dec_dim), persistent=False)
        self.dec = nn.ModuleList([Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, enc.patch ** 2)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, stains=None):
        """The reconstruction loss of one batch.

        :param x: images shaped ``(B, C, H, W)``.
        :param stains: stain names of the channels; None uses the encoder's.
        :returns: the scalar loss.
        """
        stains = list(stains or self.enc.stains)
        if x.shape[-1] != self.enc.img:
            x = F.interpolate(x, size=(self.enc.img, self.enc.img), mode="bilinear", align_corners=False)
        t = self.enc.tokens(x, stains)
        B, C, N, D = t.shape
        L = C * N
        noise = torch.rand(B, C, N, device=x.device)
        if C > 1:
            whole = torch.rand(B, device=x.device) < self.p_whole
            ch = torch.randint(0, C, (B,), device=x.device)
            noise[whole, ch[whole]] += 2.0
        noise = noise.reshape(B, L)
        keep = max(1, int(round(L * (1 - self.mask_ratio))))
        vis = noise.argsort(1)[:, :keep]
        tv = torch.gather(t.reshape(B, L, D), 1, vis[..., None].expand(-1, -1, D))
        z = self.enc.encode(tv)
        d = self.to_dec(z[:, 1:])
        full = self.mask_token.expand(B, L, -1).clone()
        full.scatter_(1, vis[..., None].expand(-1, -1, d.shape[2]), d)
        ids = stain_ids(stains).to(x.device)
        full = full + (self.dec_pos[None, None] + self.dec_stain(ids)[None, :, None]).reshape(1, L, -1)
        h = torch.cat([self.to_dec(z[:, :1]), full], 1)
        for b in self.dec:
            h = b(h)
        pred = self.pred(self.dec_norm(h)[:, 1:])
        p, n = self.enc.patch, self.enc.n_side
        tgt = x.reshape(B, C, n, p, n, p).permute(0, 1, 2, 4, 3, 5).reshape(B, L, p * p)
        tgt = (tgt - tgt.mean(-1, keepdim=True)) / (tgt.std(-1, keepdim=True) + 1e-6)
        m = torch.ones(B, L, device=x.device)
        m.scatter_(1, vis, 0)
        return (((pred - tgt) ** 2).mean(-1) * m).sum() / m.sum()


class DINOHead(nn.Module):
    """DINO projection head: a 3-layer MLP, L2 normalisation and a weight-normalised prototype layer.

    :param dim: input dimension.
    :param out: prototypes.
    :param hidden: MLP hidden size.
    :param bottleneck: MLP output size.
    """

    def __init__(self, dim, out=4096, hidden=1024, bottleneck=256):
        """Build the MLP and the prototype layer."""
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck, out, bias=False))

    def forward(self, x):
        """Prototype scores of a batch of embeddings.

        :param x: embeddings shaped ``(B, dim)``.
        :returns: ``(B, out)`` scores.
        """
        return self.last(F.normalize(self.mlp(x), dim=-1))


class ChannelDINO(nn.Module):
    """DINO self-distillation (Caron et al. 2021) for the channel ViT.

    Student and EMA teacher see two augmented views of the same cell; the
    student, in training mode, also sees a random channel subset of its view,
    so the embedding learns to stay the same when stains are missing. The
    loss is the cross-entropy between the sharpened, centred teacher and the
    student prototype distributions.

    :param enc_student: the encoder trained.
    :param enc_teacher: an encoder of the same shape; it starts as a copy.
    :param out: prototypes.
    :param t_student: student temperature.
    :param t_teacher: teacher temperature.
    :param center_m: momentum of the teacher centre.
    """

    def __init__(self, enc_student: SpacrChannelViT, enc_teacher: SpacrChannelViT, out=4096, t_student=0.1,
                 t_teacher=0.04, center_m=0.9):
        """Build the heads and copy the student into the frozen teacher."""
        super().__init__()
        self.s_enc, self.t_enc = enc_student, enc_teacher
        self.s_head, self.t_head = DINOHead(enc_student.num_features, out), DINOHead(enc_student.num_features, out)
        self.t_enc.load_state_dict(self.s_enc.state_dict())
        self.t_head.load_state_dict(self.s_head.state_dict())
        for p in list(self.t_enc.parameters()) + list(self.t_head.parameters()):
            p.requires_grad = False
        self.register_buffer("center", torch.zeros(1, out))
        self.ts, self.tt, self.cm = t_student, t_teacher, center_m

    @torch.no_grad()
    def ema(self, m):
        """Move the teacher towards the student.

        :param m: momentum; the teacher keeps ``m`` of itself.
        """
        for ps, pt in zip(list(self.s_enc.parameters()) + list(self.s_head.parameters()),
                          list(self.t_enc.parameters()) + list(self.t_head.parameters())):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)

    def forward(self, v1, v2, stains=None):
        """The self-distillation loss of two views.

        :param v1: the first view, images shaped ``(B, C, H, W)``.
        :param v2: the second view, same shape.
        :param stains: stain names of the channels; None uses the student's.
        :returns: the scalar loss.
        """
        stains = list(stains or self.s_enc.stains)
        self.t_enc.eval()
        with torch.no_grad():
            t = torch.cat([self.t_head(self.t_enc(v, stains)) for v in (v1, v2)])
            tp = F.softmax((t - self.center) / self.tt, -1).chunk(2)
            self.center.mul_(self.cm).add_(t.mean(0, keepdim=True), alpha=1 - self.cm)
        s = [self.s_head(self.s_enc(v, stains)) / self.ts for v in (v1, v2)]
        loss = 0.0
        for i, ti in enumerate(tp):
            for j, sj in enumerate(s):
                if i != j:
                    loss = loss + (-ti * F.log_softmax(sj, -1)).sum(-1).mean()
        return loss / 2


class ISAB(nn.Module):
    """Induced set attention block (Lee et al. 2019, Set Transformer).

    ``m`` inducing points attend to the set and the set attends back, so the
    cost is ``O(N m)`` instead of ``O(N ** 2)`` and a well of 1,000 cells is cheap.

    :param dim: token dimension.
    :param heads: attention heads.
    :param m: inducing points.
    :param dropout: attention and MLP dropout.
    :param drop_path: stochastic depth rate.
    """

    def __init__(self, dim, heads, m=16, dropout=0.1, drop_path=0.0):
        """Build the inducing points and the two cross-attention blocks."""
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, dim) * 0.02)
        self.b1 = Block(dim, heads, dropout=dropout, cross=True, drop_path=drop_path)
        self.b2 = Block(dim, heads, dropout=dropout, cross=True, drop_path=drop_path)

    def forward(self, x, mask=None):
        """Encode a set.

        :param x: tokens shaped ``(B, N, D)``.
        :param mask: ``(B, N)`` booleans, True for a valid token.
        :returns: the encoded set, same shape as ``x``.
        """
        h = self.b1(self.I.expand(x.shape[0], -1, -1), kv=x, mask=mask)
        return self.b2(x, kv=h)


class SpacrWellTransformer(nn.Module):
    """spaCR well transformer over single-cell embeddings.

    * ``'arrayed'``: ``forward(cells, mask)`` returns a dict with ``logits``
      ``(B, K)``, ``attn`` ``(B, pool_heads, N)``, ``cell_logits`` ``(B, N, K)``
      and ``well_emb``. A set size of 1 is the no-context ablation (a
      single-cell model); ``depth=0`` is plain attention-MIL.
    * ``'pooled'``: ``forward(targets, context, ctx_mask)`` returns ``logits``
      ``(B, T, K)``, ``ctx_attn`` and ``cell_emb``. Targets attend only to the
      context, never to each other or to themselves; the context must come
      from other cells of the same well. An empty context is the no-context
      ablation (a learned null token).

    :param in_dim: dimension of the cell embeddings.
    :param num_classes: classes.
    :param mode: ``'arrayed'`` or ``'pooled'``.
    :param dim: model dimension.
    :param depth: ISAB blocks (and, pooled, decoder blocks).
    :param heads: attention heads.
    :param inducing: ISAB inducing points.
    :param pool_heads: attention-MIL heads (arrayed).
    :param dropout: attention and head dropout.
    :param feat_dropout: dropout on the input embeddings.
    :param drop_path: stochastic depth rate.
    """

    def __init__(self, in_dim, num_classes, mode="arrayed", dim=128, depth=2, heads=4, inducing=16, pool_heads=4,
                 dropout=0.1, feat_dropout=0.1, drop_path=0.05):
        """Build the input projection, the set encoder and the mode's head."""
        super().__init__()
        if mode not in ("arrayed", "pooled"):
            raise ValueError(f"mode must be 'arrayed' or 'pooled', got {mode!r}")
        self.mode = mode
        self.inp = nn.Sequential(nn.Dropout(feat_dropout), nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.GELU(),
                                 nn.Linear(dim, dim))
        self.enc = nn.ModuleList([ISAB(dim, heads, inducing, dropout, drop_path) for _ in range(depth)])
        if mode == "arrayed":
            self.pool = GatedAttentionPool(dim, hidden=dim, heads=pool_heads, dropout=dropout)
            self.head = nn.Sequential(nn.LayerNorm(pool_heads * dim), nn.Dropout(dropout),
                                      nn.Linear(pool_heads * dim, num_classes))
            self.cell_head = nn.Linear(dim, num_classes)
        else:
            self.null = nn.Parameter(torch.zeros(1, 1, dim))
            self.tgt = nn.Sequential(nn.Dropout(feat_dropout), nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.GELU(),
                                     nn.Linear(dim, dim))
            self.dec = nn.ModuleList([Block(dim, heads, dropout=dropout, cross=True, drop_path=drop_path)
                                      for _ in range(depth)])
            self.film = nn.Linear(2 * dim, 2 * dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, num_classes))
        self.num_features = dim

    def forward_arrayed(self, cells, mask=None):
        """Well logits from all cells of a well.

        :param cells: cell embeddings shaped ``(B, N, in_dim)``.
        :param mask: ``(B, N)`` booleans, True for a real cell (bags are padded).
        :returns: dict with ``logits``, ``attn``, ``cell_logits`` and ``well_emb``.
        """
        h = self.inp(cells)
        for blk in self.enc:
            h = blk(h, mask)
        z, a = self.pool(h, mask)
        return {"logits": self.head(z), "attn": a, "cell_logits": self.cell_head(h), "well_emb": z}

    def forward_pooled(self, targets, context=None, ctx_mask=None, return_attn=False):
        """Per-cell logits of target cells, conditioned on a context set.

        :param targets: target cell embeddings shaped ``(B, T, in_dim)``.
        :param context: context cell embeddings shaped ``(B, M, in_dim)``; None or ``M == 0`` is no context.
        :param ctx_mask: ``(B, M)`` booleans, True for a real context cell.
        :param return_attn: also return the last block's attention weights.
        :returns: dict with ``logits``, ``ctx_attn`` (None unless asked) and ``cell_emb``.
        """
        B = targets.shape[0]
        if context is None or context.shape[1] == 0:
            c = self.null.expand(B, 1, -1)
            m = torch.ones(B, 1, dtype=torch.bool, device=targets.device)
        else:
            c = self.inp(context)
            for blk in self.enc:
                c = blk(c, ctx_mask)
            if ctx_mask is None:
                ctx_mask = torch.ones(c.shape[:2], dtype=torch.bool, device=c.device)
            c = torch.cat([self.null.expand(B, 1, -1), c], 1)
            m = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=c.device), ctx_mask], 1)
        mf = m[..., None].to(c.dtype)
        mu = (c * mf).sum(1) / mf.sum(1)
        sd = (((c - mu[:, None]) ** 2) * mf).sum(1).div(mf.sum(1)).clamp_min(1e-6).sqrt()
        g, b = self.film(torch.cat([mu, sd], -1)).chunk(2, -1)
        t = self.tgt(targets) * (1 + g[:, None]) + b[:, None]
        w = None
        for blk in self.dec:
            if return_attn:
                t, w = blk(t, kv=c, mask=m, need_weights=True)
            else:
                t = blk(t, kv=c, mask=m)
        return {"logits": self.head(t), "ctx_attn": w, "cell_emb": t}

    def forward(self, *a, **k):
        """Dispatch to :meth:`forward_arrayed` or :meth:`forward_pooled` by mode.

        :param a: positional arguments of the mode's forward.
        :param k: keyword arguments of the mode's forward.
        :returns: the mode's output dict.
        """
        return self.forward_arrayed(*a, **k) if self.mode == "arrayed" else self.forward_pooled(*a, **k)


def penetrant_fraction(cell_probs: torch.Tensor, called: torch.Tensor, thresh: torch.Tensor, mask=None):
    """Fraction of a well's cells called for the well's class.

    A cell counts when its instance probability for the well's called class
    exceeds that class's threshold, the 95th percentile of the same score in
    control (DMSO) cells.

    :param cell_probs: instance probabilities shaped ``(B, N, K)``.
    :param called: the called class of each well, shaped ``(B,)``.
    :param thresh: one threshold per class, shaped ``(K,)``.
    :param mask: ``(B, N)`` validity mask of padded bags; None counts all cells.
    :returns: the ``(B,)`` penetrant fractions.
    """
    p = cell_probs.gather(2, called[:, None, None].expand(-1, cell_probs.shape[1], 1))[..., 0]
    hit = (p > thresh[called][:, None]).float()
    if mask is not None:
        return (hit * mask).sum(1) / mask.sum(1).clamp_min(1)
    return hit.mean(1)


def count_params(m):
    """Number of parameters of a module.

    :param m: the module.
    :returns: the parameter count.
    """
    return sum(p.numel() for p in m.parameters())


class ChannelClassifier(nn.Module):
    """An image model of this module with a linear head: ``model(x) -> logits``.

    This is the interface the rest of spaCR's train path expects. The channel
    subsets of the missing-stain evaluation are applied here too:
    ``train_subset`` in training, ``eval_subset`` at test time (falling back
    to ``train_subset``).

    :param enc: a :class:`SpacrChannelCNN` or :class:`SpacrChannelViT`.
    :param num_classes: classes of the head.
    :param stains: stain names of the input channels; None uses the encoder's.
    :param train_subset: stains kept in training; None keeps all.
    :param eval_subset: stains kept at test time; None keeps ``train_subset``.
    :param head_dropout: dropout before the head.
    """

    def __init__(self, enc: nn.Module, num_classes: int, stains: Optional[Sequence[str]] = None,
                 train_subset: Optional[Sequence[str]] = None, eval_subset: Optional[Sequence[str]] = None,
                 head_dropout: float = 0.1):
        """Build the head and resolve the channel subsets to slot indices."""
        super().__init__()
        self.enc = enc
        self.stains = list(stains or enc.stains)
        self.num_classes = int(num_classes)
        self.num_features = enc.num_features
        self.train_subset = None
        self.eval_subset = None
        self.set_subsets(train_subset, eval_subset)
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(enc.num_features, self.num_classes))

    def set_subsets(self, train_subset=None, eval_subset=None):
        """Set the stains kept in training and at test time.

        :param train_subset: stain names kept in training; None keeps all.
        :param eval_subset: stain names kept at test time; None keeps ``train_subset``.
        :raises ValueError: a subset names a stain the model's channels do not have.
        """
        for subset in (train_subset, eval_subset):
            missing = [s for s in (subset or ()) if s not in self.stains]
            if missing:
                raise ValueError(f"stains {missing} are not among the channels {self.stains}")
        self.train_subset = [self.stains.index(s) for s in train_subset] if train_subset else None
        self.eval_subset = [self.stains.index(s) for s in eval_subset] if eval_subset else None

    def embed(self, x):
        """The encoder embedding of a batch, after the channel subset.

        :param x: images shaped ``(B, C, H, W)``.
        :returns: the ``(B, num_features)`` embedding.
        """
        idx = self.train_subset if self.training else (
            self.eval_subset if self.eval_subset is not None else self.train_subset)
        if idx is not None:
            return self.enc(x[:, idx], [self.stains[i] for i in idx])
        return self.enc(x, self.stains)

    def forward(self, x):
        """Class logits of a batch.

        :param x: images shaped ``(B, C, H, W)``.
        :returns: ``(B, num_classes)`` logits.
        """
        return self.head(self.embed(x))


def default_stains(channels: int, stains: Optional[Sequence[str]] = None,
                   dataset: Optional[str] = None) -> List[str]:
    """The stain names of a model's input channels.

    :param channels: channel count.
    :param stains: explicit names; used when given.
    :param dataset: a key of :data:`DATASET_STAINS`; used when its panel has ``channels`` stains.
    :returns: one name per channel; without names, ``other_0``, ``other_1``...
    :raises ValueError: ``stains`` does not have ``channels`` names.
    """
    if stains:
        names = [str(s).strip() for s in stains]
        if len(names) != int(channels):
            raise ValueError(f"{len(names)} stain names for {channels} channels")
        return names
    if dataset and len(DATASET_STAINS.get(dataset, ())) == int(channels):
        return list(DATASET_STAINS[dataset])
    return [f"other_{k}" for k in range(int(channels))]


def build_channel_model(arch: str, *, num_classes: int, channels: int = 3,
                        stains: Optional[Sequence[str]] = None, in_dim: Optional[int] = None,
                        image_size: int = 224, options: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Build one of :data:`CHANNEL_ARCHITECTURES` by name.

    :param arch: ``'spacr_channel_cnn'``, ``'spacr_channel_vit'`` or ``'spacr_well_transformer'``.
    :param num_classes: classes.
    :param channels: input channels of the image models.
    :param stains: stain names of the channels (see :func:`default_stains`).
    :param in_dim: embedding dimension of the well transformer's input cells.
    :param image_size: input size of the ViT (the CNN takes any multiple of 32).
    :param options: keyword arguments for the model class (such as ``dims``,
        ``size``, ``patch``, ``mode``, ``depth``); ``train_subset``,
        ``eval_subset`` and ``head_dropout`` go to :class:`ChannelClassifier`.
    :returns: a :class:`ChannelClassifier` for the image models, a
        :class:`SpacrWellTransformer` for the well model.
    :raises ValueError: ``arch`` is not one of :data:`CHANNEL_ARCHITECTURES`,
        or the well transformer has no ``in_dim``.
    """
    opts = dict(options or {})
    if arch not in CHANNEL_ARCHITECTURES:
        raise ValueError(f"train_arch={arch!r} is not one of {list(CHANNEL_ARCHITECTURES)}")
    if arch == 'spacr_well_transformer':
        if not in_dim:
            raise ValueError("spacr_well_transformer needs in_dim, the dimension of the cell embeddings")
        return SpacrWellTransformer(int(in_dim), int(num_classes), **opts)
    names = default_stains(channels, stains, opts.pop('dataset', None))
    train_subset = opts.pop('train_subset', None)
    eval_subset = opts.pop('eval_subset', None)
    head_dropout = float(opts.pop('head_dropout', 0.1))
    if arch == 'spacr_channel_cnn':
        enc = SpacrChannelCNN(stains=names, **opts)
    else:
        opts.setdefault('img', int(image_size))
        enc = SpacrChannelViT(stains=names, **opts)
    return ChannelClassifier(enc, num_classes, names, train_subset, eval_subset, head_dropout)
