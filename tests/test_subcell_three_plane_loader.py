"""The three-plane SubCell models and the shared SubCell loader, ``spacr.embeddings``.

``subcell_rbg`` and ``subcell_ybg`` encode two reference planes plus one
protein plane per pass and join the passes. These tests check what spaCR
does around the network without downloading weights: the plane order of
every pass, the authors' whole-pass min-max or the fixed scale, the resize
to the Human Protein Atlas pixel size, the pinned-digest and plain-tensor
checks of the checkpoint, the random-weights twin and the layer read-out.
"""
from __future__ import annotations

import hashlib
import sys
import types

import numpy as np
import pytest

import spacr.embeddings as emb

torch = pytest.importorskip("torch")


class _ViT(torch.nn.Module):
    """A weightless token encoder that records every input it sees."""

    seen: list = []

    def __init__(self, config, add_pooling_layer):
        super().__init__()
        assert add_pooling_layer is False

    def forward(self, x, interpolate_pos_encoding):
        type(self).seen.append(x.detach().cpu().numpy().copy())
        n = x.shape[0]
        tokens = x.mean(dim=(1, 2, 3)).reshape(n, 1, 1).expand(n, 4, 768)
        return types.SimpleNamespace(last_hidden_state=tokens.contiguous())


def _pooler_state():
    generator = torch.Generator().manual_seed(0)

    def weights(*shape):
        return torch.randn(*shape, generator=generator) * 0.01

    return {"pool_model.attention_v.1.weight": weights(512, 768),
            "pool_model.attention_v.1.bias": weights(512),
            "pool_model.attention_u.1.weight": weights(512, 768),
            "pool_model.attention_u.1.bias": weights(512),
            "pool_model.attention.weight": weights(2, 512),
            "pool_model.attention.bias": weights(2)}


def _fake_transformers(monkeypatch):
    module = types.ModuleType("transformers")
    module.ViTConfig = lambda **config: types.SimpleNamespace(**config)
    module.ViTModel = _ViT
    monkeypatch.setitem(sys.modules, "transformers", module)
    _ViT.seen = []


def _local_checkpoint(tmp_path, state=None):
    path = tmp_path / "MT-DNA-Protein_MAE-CellS-ProtS-Pool.pth"
    torch.save(_pooler_state() if state is None else state, path)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _pin(monkeypatch, digest):
    name = "MT-DNA-Protein_MAE-CellS-ProtS-Pool.pth"
    weights = dict(emb._SUBCELL_WEIGHTS)
    weights[name] = (weights[name][0], digest)
    monkeypatch.setattr(emb, "_SUBCELL_WEIGHTS", weights)


def _spec(path, **kwargs):
    base = dict(backbone="subcell_rbg", channel_policy=emb.CHANNEL_PROJECT,
                channels=(1, 0, 2, 3), normalize=False, device="cpu",
                batch_size=4, checkpoint_path=path)
    base.update(kwargs)
    return emb.EmbeddingSpec(**base)


def test_each_protein_plane_is_one_pass_with_the_two_references(
        monkeypatch, tmp_path):
    """Planes (ref0, ref1, protein) per pass, min-max as a whole, joined."""
    _fake_transformers(monkeypatch)
    path, digest = _local_checkpoint(tmp_path)
    _pin(monkeypatch, digest)
    crops = np.random.default_rng(3).random((2, 20, 20, 4)).astype(
        np.float32) * 1000
    result = emb.embed_array(crops, _spec(path))
    assert result.values.shape == (2, 2 * 1536)
    assert len(_ViT.seen) == 2
    for seen, protein in zip(_ViT.seen, (2, 3)):
        assert seen.shape == (2, 3, 20, 20)
        for cell in range(2):
            raw = crops[cell][..., [1, 0, protein]].transpose(2, 0, 1)
            expected = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
            np.testing.assert_allclose(seen[cell], expected, rtol=1e-5,
                                       atol=1e-6)


def test_fixed_scale_and_pixel_size_rescale(monkeypatch, tmp_path):
    """With normalize the fixed scale is kept; pixel size resizes to HPA."""
    _fake_transformers(monkeypatch)
    path, digest = _local_checkpoint(tmp_path)
    _pin(monkeypatch, digest)
    crops = np.full((1, 96, 96, 4), 50.0, dtype=np.float32)
    spec = _spec(path, normalize=True, channel_scale=(100.0, 100.0, 100.0,
                                                      100.0),
                 pixel_size_um=0.2125, channels=(1, 0, 2, 3))
    emb.embed_array(crops, spec)
    side = 16 * int(round(96 * 0.2125 / emb._SUBCELL_HPA_PIXEL_UM / 16))
    assert side == 256
    for seen in _ViT.seen:
        assert seen.shape == (1, 3, side, side)
        np.testing.assert_allclose(seen, 0.5, atol=1e-6)


def test_the_checkpoint_is_checked_before_it_is_used(monkeypatch, tmp_path,
                                                     subtests):
    """Pinned digest, explicit digest, missing file and non-tensor states."""
    _fake_transformers(monkeypatch)
    path, digest = _local_checkpoint(tmp_path)
    info = emb._SUBCELL_VARIANTS["subcell_rbg"]
    with subtests.test(msg="pinned_digest_mismatch_refused"):
        _pin(monkeypatch, "0" * 64)
        with pytest.raises(emb.EmbeddingError, match="has SHA-256"):
            emb._subcell_model(info, torch, checkpoint_path=path)
    with subtests.test(msg="explicit_digest_used_when_none_pinned"):
        _pin(monkeypatch, None)
        emb._subcell_model(info, torch, checkpoint_path=path,
                           checkpoint_sha256=digest)
        with pytest.raises(emb.EmbeddingError, match="has SHA-256"):
            emb._subcell_model(info, torch, checkpoint_path=path,
                               checkpoint_sha256="f" * 64)
    with subtests.test(msg="missing_local_file"):
        with pytest.raises(emb.EmbeddingError, match="is not a file"):
            emb._subcell_model(info, torch,
                               checkpoint_path=str(tmp_path / "absent.pth"))
    with subtests.test(msg="non_tensor_state_refused"):
        _pin(monkeypatch, None)
        bad = tmp_path / "bad"
        bad.mkdir()
        bad_path, _ = _local_checkpoint(bad, {"step": 3,
                                              **_pooler_state()})
        with pytest.raises(emb.EmbeddingError, match="plain model state"):
            emb._subcell_model(info, torch, checkpoint_path=bad_path)
    with subtests.test(msg="unknown_init"):
        with pytest.raises(emb.EmbeddingError, match="unknown SubCell init"):
            emb._subcell_model(info, torch, init="imagenet")


def test_the_three_plane_spec_is_refused_before_loading(subtests):
    """Too few, repeated or per-channel planes, and a bad pixel size."""
    crops = np.zeros((1, 20, 20, 4), dtype=np.float32)
    for name, kwargs in (
            ("two_planes", dict(channels=(0, 1))),
            ("repeated", dict(channels=(0, 1, 1))),
            ("per_channel", dict(channel_policy=emb.CHANNEL_PER_CHANNEL)),
            ("no_channels", dict(channels=None))):
        with subtests.test(msg=name):
            with pytest.raises(emb.EmbeddingError, match="subcell_rbg needs"):
                emb.embed_array(crops, _spec(None, **kwargs),
                                encoder=lambda a: a[:, 0, 0, :1])
    for bad in (0.0, -1.0, float("nan"), True):
        with subtests.test(msg=f"pixel_size_{bad!r}"):
            with pytest.raises(emb.EmbeddingError, match="pixel_size_um"):
                _spec(None, pixel_size_um=bad)


def test_random_twin_and_layer_read_out_with_the_real_vit():
    """The untrained twin is seed-stable; layer blocks have the ViT width."""
    pytest.importorskip("transformers")
    info = emb._SUBCELL_VARIANTS["subcell_rbg"]
    first, _ = emb._subcell_model(info, torch, init="random", seed=20261116)
    again, _ = emb._subcell_model(info, torch, init="random", seed=20261116)
    other, _ = emb._subcell_model(info, torch, init="random", seed=1)
    a, b, c = (net.state_dict() for net in (first, again, other))
    assert all(torch.equal(a[key], b[key]) for key in a)
    assert any(not torch.equal(a[key], c[key]) for key in a)
    first.eval()
    x = torch.rand(2, 4, 32, 32)
    with torch.no_grad():
        pooled, blocks = emb._subcell_passes(first, x, torch=torch,
                                             normalize=False,
                                             layers=(6, 12))
        plain = emb._subcell_passes(first, x, torch=torch, normalize=False)
    assert tuple(pooled.shape) == (2, 2 * 1536)
    torch.testing.assert_close(pooled, plain)
    assert sorted(blocks) == sorted(
        f"p{p}_L{layer}_{token}" for p in (0, 1) for layer in (6, 12)
        for token in ("cls", "meanpatch"))
    assert all(tuple(value.shape) == (2, 768) for value in blocks.values())
