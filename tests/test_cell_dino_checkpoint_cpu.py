"""Cell-DINO local-checkpoint contract without claiming official weight parity."""
from __future__ import annotations

import hashlib
from dataclasses import fields
from dataclasses import replace

import numpy as np
import pytest

import spacr.embeddings as emb


def test_cell_dino_spec_fields_are_only_three_optional_additions():
    names = [field.name for field in fields(emb.EmbeddingSpec)]
    assert names == [
        "backbone", "channel_policy", "channels", "batch_size", "device",
        "normalize", "channel_scale", "cell_dino_factory",
        "checkpoint_path", "checkpoint_sha256",
    ]
    assert all(field.default is None for field in fields(emb.EmbeddingSpec)[-3:])


def _checkpoint(tmp_path, torch, width):
    class SmallDino(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.patch_embed = torch.nn.Module()
            self.patch_embed.proj = torch.nn.Conv2d(channels, 1, 1,
                                                    bias=False)

        def forward(self, x):
            return self.patch_embed.proj(x).mean(dim=(2, 3))

    model = SmallDino(width)
    with torch.no_grad():
        model.patch_embed.proj.weight.copy_(
            torch.arange(1, width + 1, dtype=torch.float32).view(1, width, 1, 1))
    path = tmp_path / "official-shaped-fixture.pth"
    torch.save(model.state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest, SmallDino


@pytest.mark.parametrize("factory,width,size", [
    ("cell_dino_hpa_vitl16", 4, 224),
    ("cell_dino_hpa_vitl14", 4, 518),
    ("cell_dino_cp_vits8", 5, 128),
])
def test_local_digest_strict_load_and_ordered_planes(
        tmp_path, monkeypatch, factory, width, size):
    torch = pytest.importorskip("torch")
    path, digest, model_type = _checkpoint(tmp_path, torch, width)
    calls = []

    def hub(repo, name, *, pretrained, trust_repo):
        calls.append((repo, name, pretrained, trust_repo))
        return model_type(width)

    monkeypatch.setattr(torch.hub, "load", hub)
    channels = tuple(reversed(range(width)))
    spec = emb.EmbeddingSpec(
        backbone="cell_dino", channel_policy=emb.CHANNEL_PROJECT,
        channels=channels, cell_dino_factory=factory,
        checkpoint_path=str(path), checkpoint_sha256=digest,
        normalize=False, device="cpu")
    height, image_width = ((128, 128) if factory == "cell_dino_cp_vits8"
                           else (20, 22))
    crops = np.zeros((2, height, image_width, width), dtype=np.float32)
    for channel in range(width):
        crops[..., channel] = channel + 1
    result = emb.embed_array(crops, spec)
    expected = sum((i + 1) * (channels[i] + 1) for i in range(width))
    assert result.values.shape == (2, 1)
    np.testing.assert_allclose(result.values[:, 0], expected)
    assert calls == [("facebookresearch/dinov2:" + emb._CELL_DINO_REVISION,
                      factory, False, True)]
    assert emb._CELL_DINO_FACTORIES[factory] == (width, size)


def test_missing_or_wrong_digest_fails_before_official_code_load(
        tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path, digest, _ = _checkpoint(tmp_path, torch, 4)
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: pytest.fail(
        "official model code should not load for invalid weights"))
    base = emb.EmbeddingSpec(
        backbone="cell_dino", channel_policy=emb.CHANNEL_PROJECT,
        channels=(0, 1, 2, 3), cell_dino_factory="cell_dino_hpa_vitl16",
        checkpoint_path=str(path), checkpoint_sha256=digest, normalize=False)
    with pytest.raises(emb.EmbeddingError, match="local official"):
        emb.embed_array(np.ones((1, 20, 20, 4), dtype=np.float32),
                        replace(base, checkpoint_path=None))
    with pytest.raises(emb.EmbeddingError, match="SHA-256 does not match"):
        emb.embed_array(np.ones((1, 20, 20, 4), dtype=np.float32),
                        replace(base, checkpoint_sha256="0" * 64))
    assert base.fingerprint() != replace(
        base, checkpoint_sha256="0" * 64).fingerprint()
    assert base.fingerprint() != replace(
        base, cell_dino_factory="cell_dino_hpa_vitl14").fingerprint()
    assert base.fingerprint() != replace(
        base, checkpoint_path=str(tmp_path / "other.pth")).fingerprint()
    with pytest.raises(emb.EmbeddingError, match="SHA-256 digest"):
        emb.embed_array(np.ones((1, 20, 20, 4), dtype=np.float32),
                        replace(base, checkpoint_sha256="invalid"))
    with pytest.raises(emb.EmbeddingError, match="official Cell-DINO factory"):
        emb.embed_array(np.ones((1, 20, 20, 4), dtype=np.float32),
                        replace(base, cell_dino_factory="unknown"))


@pytest.mark.parametrize("channels", [None, 3, (0, 1, 2), (0, 1, 1, 3),
                                       (0, 1, 2, True), (0, 1, 2, 9)])
def test_bad_plane_mapping_refused_before_model_load(
        tmp_path, monkeypatch, channels):
    torch = pytest.importorskip("torch")
    path, digest, _ = _checkpoint(tmp_path, torch, 4)
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: pytest.fail(
        "invalid mapping reached official code"))
    spec = emb.EmbeddingSpec(
        backbone="cell_dino", channel_policy=emb.CHANNEL_PROJECT,
        channels=channels, cell_dino_factory="cell_dino_hpa_vitl16",
        checkpoint_path=str(path), checkpoint_sha256=digest, normalize=False)
    with pytest.raises(emb.EmbeddingError, match="channel|indices"):
        emb.embed_array(np.ones((1, 20, 20, 4), dtype=np.float32), spec)


def test_wrong_factory_state_is_rejected_by_strict_load(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path, digest, model_type = _checkpoint(tmp_path, torch, 4)
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: model_type(5))
    spec = emb.EmbeddingSpec(
        backbone="cell_dino", channel_policy=emb.CHANNEL_PROJECT,
        channels=(0, 1, 2, 3), cell_dino_factory="cell_dino_hpa_vitl16",
        checkpoint_path=str(path), checkpoint_sha256=digest, normalize=False)
    with pytest.raises(emb.EmbeddingError, match="cannot be loaded strictly"):
        emb.embed_array(np.ones((1, 20, 20, 4), dtype=np.float32), spec)


def test_nonregular_or_missing_checkpoint_never_reaches_hub(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: pytest.fail(
        "invalid local file reached official code"))
    spec = emb.EmbeddingSpec(
        backbone="cell_dino", channel_policy=emb.CHANNEL_PROJECT,
        channels=(0, 1, 2, 3), cell_dino_factory="cell_dino_hpa_vitl16",
        checkpoint_path=str(tmp_path), checkpoint_sha256="a" * 64,
        normalize=False)
    with pytest.raises(emb.EmbeddingError, match="regular file"):
        emb.embed_array(np.ones((1, 20, 20, 4), dtype=np.float32), spec)
    with pytest.raises(emb.EmbeddingError, match="Cannot read"):
        emb.embed_array(np.ones((1, 20, 20, 4), dtype=np.float32),
                        replace(spec, checkpoint_path=str(tmp_path / "missing")))


def test_checkpoint_change_during_safe_load_is_refused(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path, digest, _ = _checkpoint(tmp_path, torch, 4)
    spec = emb.EmbeddingSpec(
        backbone="cell_dino", channel_policy=emb.CHANNEL_PROJECT,
        channels=(0, 1, 2, 3), cell_dino_factory="cell_dino_hpa_vitl16",
        checkpoint_path=str(path), checkpoint_sha256=digest, normalize=False)
    original_load = torch.load

    def altered_load(handle, **kwargs):
        state = original_load(handle, **kwargs)
        with open(path, "ab") as appended:
            appended.write(b"changed")
        return state

    monkeypatch.setattr(torch, "load", altered_load)
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: pytest.fail(
        "changed file reached official code"))
    with pytest.raises(emb.EmbeddingError, match="changed during loading"):
        emb.embed_array(np.ones((1, 20, 20, 4), dtype=np.float32), spec)


def test_nonplain_state_and_wrong_factory_width_are_refused(
        tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path, digest, model_type = _checkpoint(tmp_path, torch, 4)
    spec = emb.EmbeddingSpec(
        backbone="cell_dino", channel_policy=emb.CHANNEL_PROJECT,
        channels=(0, 1, 2, 3), cell_dino_factory="cell_dino_hpa_vitl16",
        checkpoint_path=str(path), checkpoint_sha256=digest, normalize=False)
    torch.save({"model": model_type(4).state_dict()}, path)
    nested = replace(spec, checkpoint_sha256=hashlib.sha256(
        path.read_bytes()).hexdigest())
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: pytest.fail(
        "nested checkpoint reached official code"))
    with pytest.raises(emb.EmbeddingError, match="plain model state"):
        emb.embed_array(np.ones((1, 20, 20, 4), dtype=np.float32), nested)

    torch.save(model_type(4).state_dict(), path)
    valid = replace(spec, checkpoint_sha256=hashlib.sha256(
        path.read_bytes()).hexdigest())

    def wrong_width(*args, **kwargs):
        model = model_type(4)
        model.patch_embed.proj.in_channels = 3
        return model

    monkeypatch.setattr(torch.hub, "load", wrong_width)
    with pytest.raises(emb.EmbeddingError, match="unexpected input width"):
        emb.embed_array(np.ones((1, 20, 20, 4), dtype=np.float32), valid)


def test_pinned_source_failure_is_actionable(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path, digest, _ = _checkpoint(tmp_path, torch, 4)
    spec = emb.EmbeddingSpec(
        backbone="cell_dino", channel_policy=emb.CHANNEL_PROJECT,
        channels=(0, 1, 2, 3), cell_dino_factory="cell_dino_hpa_vitl16",
        checkpoint_path=str(path), checkpoint_sha256=digest, normalize=False)

    def unavailable(*args, **kwargs):
        raise OSError("offline and source not cached")

    monkeypatch.setattr(torch.hub, "load", unavailable)
    with pytest.raises(emb.EmbeddingError, match="pinned official dinov2 source"):
        emb.embed_array(np.ones((1, 20, 20, 4), dtype=np.float32), spec)


def test_injected_encoder_cannot_silently_drop_a_declared_plane(tmp_path):
    path = tmp_path / "official.pth"
    path.write_bytes(b"a checkpoint fixture is not used by injected encoder")
    spec = emb.EmbeddingSpec(
        backbone="cell_dino", channel_policy=emb.CHANNEL_PROJECT,
        channels=(0, 1, 2, 3, 4), cell_dino_factory="cell_dino_cp_vits8",
        checkpoint_path=str(path), checkpoint_sha256="a" * 64,
        normalize=False)

    def wrong(stack):
        pytest.fail("wrong-width encoder received crop data")

    wrong.in_channels = 4
    with pytest.raises(emb.EmbeddingError, match="input width differs"):
        emb.embed_array(np.ones((1, 20, 20, 5), dtype=np.float32), spec,
                        encoder=wrong)
