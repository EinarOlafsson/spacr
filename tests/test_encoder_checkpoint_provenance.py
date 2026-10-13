"""Offline provenance describes the same checkpoints the real loaders use."""

import builtins
import hashlib
import sys
from types import SimpleNamespace

import pytest

from spacr import embeddings as emb
from spacr.model_zoo import UNKNOWN


def _refuse_models(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "timm", "transformers"}:
            raise AssertionError("provenance must not load a model")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize("backbone", ["openphenom", "chada_vit"])
@pytest.mark.parametrize("filename", ["model.safetensors", "pytorch_model.bin"])
def test_foundation_cache_pins_the_loader_revision(
        tmp_path, monkeypatch, backbone, filename):
    import huggingface_hub

    checkpoint = tmp_path / filename
    data = b"checkpoint bytes at the pinned revision"
    checkpoint.write_bytes(data)
    calls = []
    info = emb._FOUNDATION_MODELS[backbone]

    def cached(repo, name, *, revision):
        calls.append((repo, name, revision))
        return str(checkpoint) if name == filename else object()

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    _refuse_models(monkeypatch)
    entry = emb.encoder_entry(emb.EmbeddingSpec(backbone=backbone))
    assert entry.path == str(checkpoint)
    assert entry.sha256 == hashlib.sha256(data).hexdigest()
    assert entry.size_bytes == len(data)
    assert entry.uri == (f"https://huggingface.co/{info['repo']}/resolve/"
                         f"{info['revision']}/{filename}")
    assert all(repo == info["repo"] and revision == info["revision"]
               for repo, _, revision in calls)
    assert entry.trained_by == info["repo"]
    assert entry.trained_on == UNKNOWN and not entry.verified


@pytest.mark.parametrize("location", ["loaded_hub", "torch_home", "xdg_home"])
def test_subcell_resolves_the_real_hub_directory_without_importing_torch(
        tmp_path, monkeypatch, location):
    monkeypatch.delenv("TORCH_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delitem(sys.modules, "torch.hub", raising=False)
    if location == "loaded_hub":
        hub = tmp_path / "custom-hub"
        monkeypatch.setitem(sys.modules, "torch.hub",
                            SimpleNamespace(get_dir=lambda: str(hub)))
    elif location == "torch_home":
        monkeypatch.setenv("TORCH_HOME", str(tmp_path))
        hub = tmp_path / "hub"
    else:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        hub = tmp_path / "torch" / "hub"
    checkpoint = hub / "checkpoints" / "DNA-Protein_ViT-ProtS-Pool.pth"
    checkpoint.parent.mkdir(parents=True)
    data = b"SubCell checkpoint bytes"
    checkpoint.write_bytes(data)
    _refuse_models(monkeypatch)
    entry = emb.encoder_entry(emb.EmbeddingSpec(backbone="subcell"))
    assert entry.path == str(checkpoint)
    assert entry.sha256 == hashlib.sha256(data).hexdigest()
    assert entry.size_bytes == len(data)
    assert entry.uri == emb._FOUNDATION_MODELS["subcell"]["url"]
    assert entry.trained_by == emb._FOUNDATION_MODELS["subcell"]["label"]
    assert entry.trained_on == UNKNOWN and not entry.verified


def test_local_dino_checkpoint_is_hashed_without_deserialization(
        tmp_path, monkeypatch):
    checkpoint = tmp_path / "teacher.pth"
    data = b"local teacher bytes"
    checkpoint.write_bytes(data)
    _refuse_models(monkeypatch)
    entry = emb.encoder_entry(emb.EmbeddingSpec(backbone=f"dino:{checkpoint}"))
    assert entry.path == str(checkpoint) and entry.uri == str(checkpoint)
    assert entry.sha256 == hashlib.sha256(data).hexdigest()
    assert entry.size_bytes == len(data)
    assert entry.trained_by == UNKNOWN and not entry.verified


def test_unreadable_checkpoint_cannot_claim_a_digest(tmp_path, monkeypatch):
    checkpoint = tmp_path / "teacher.pth"
    checkpoint.write_bytes(b"unreadable")
    original = builtins.open

    def unreadable(path, *args, **kwargs):
        if str(path) == str(checkpoint):
            raise PermissionError("not readable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", unreadable)
    entry = emb.encoder_entry(emb.EmbeddingSpec(backbone=f"dino:{checkpoint}"))
    assert entry.path == "" and entry.sha256 == "" and entry.size_bytes == 0
    assert any("No checksum" in note for note in entry.notes)


def test_unsupported_cell_dino_has_no_invented_weight_provider(monkeypatch):
    _refuse_models(monkeypatch)
    entry = emb.encoder_entry(emb.EmbeddingSpec(backbone="cell_dino"))
    assert entry.path == "" and entry.sha256 == "" and entry.uri == ""
    assert entry.trained_by == UNKNOWN and not entry.verified


@pytest.mark.parametrize("factory,channels", [
    ("cell_dino_hpa_vitl16", (2, 0, 3, 1)),
    ("cell_dino_hpa_vitl14", (2, 0, 3, 1)),
    ("cell_dino_cp_vits8", (4, 2, 0, 3, 1)),
])
def test_cell_dino_entry_verifies_declared_bytes_without_loading_a_model(
        tmp_path, monkeypatch, factory, channels):
    checkpoint = tmp_path / "local.pth"
    data = b"local checkpoint fixture\x00" * 60000
    checkpoint.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    alias = tmp_path / "alias.pth"
    alias.symlink_to(checkpoint)
    spec = emb.EmbeddingSpec(
        backbone="cell_dino", channel_policy=emb.CHANNEL_PROJECT,
        channels=channels, cell_dino_factory=factory,
        checkpoint_path=str(alias), checkpoint_sha256=digest.upper())
    _refuse_models(monkeypatch)
    entry = emb.encoder_entry(spec, scorecard={"mAP": 0.5})
    assert entry.path == str(checkpoint.resolve())
    assert entry.sha256 == digest and entry.size_bytes == len(data)
    assert entry.source == "local" and entry.uri == ""
    assert entry.trained_by == entry.trained_on == UNKNOWN
    assert entry.metrics == {"mAP": 0.5} and not entry.verified
    assert not any("No checksum" in note for note in entry.notes)


@pytest.mark.parametrize("condition", [
    "missing_path", "missing_digest", "bad_digest", "wrong_digest",
    "missing_file", "directory", "fifo", "unreadable",
])
def test_cell_dino_invalid_checkpoint_has_no_claimed_provenance(
        tmp_path, monkeypatch, condition):
    import os

    checkpoint = tmp_path / "local.pth"
    data = b"local checkpoint fixture"
    checkpoint.write_bytes(data)
    path = str(checkpoint)
    digest = hashlib.sha256(data).hexdigest()
    if condition == "missing_path":
        path = None
    elif condition == "missing_digest":
        digest = None
    elif condition == "bad_digest":
        digest = "not a digest"
    elif condition == "wrong_digest":
        digest = "0" * 64
    elif condition == "missing_file":
        path = str(tmp_path / "missing.pth")
    elif condition == "directory":
        path = str(tmp_path)
    elif condition == "fifo":
        checkpoint.unlink()
        os.mkfifo(checkpoint)
    else:
        original = os.open

        def unreadable(candidate, *args, **kwargs):
            if candidate == path:
                raise PermissionError("checkpoint not readable")
            return original(candidate, *args, **kwargs)

        monkeypatch.setattr(os, "open", unreadable)
    _refuse_models(monkeypatch)
    spec = emb.EmbeddingSpec(backbone="cell_dino", checkpoint_path=path,
                             checkpoint_sha256=digest)
    entry = emb.encoder_entry(spec)
    assert entry.path == entry.sha256 == entry.uri == ""
    assert entry.size_bytes == 0 and entry.source == "remote"
    assert entry.trained_by == UNKNOWN and not entry.verified
    assert any("No checksum" in note for note in entry.notes)


@pytest.mark.parametrize("change", ["append", "replace", "remove"])
def test_cell_dino_checkpoint_changed_while_hashing_cannot_be_saved(
        tmp_path, monkeypatch, change):
    checkpoint = tmp_path / "local.pth"
    data = b"local checkpoint fixture\x00" * 60000
    checkpoint.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    original_hash = hashlib.sha256
    changed = []

    class ChangingDigest:
        def __init__(self):
            self.digest = original_hash()

        def update(self, chunk):
            self.digest.update(chunk)
            if changed:
                return
            changed.append(change)
            if change == "append":
                with checkpoint.open("ab") as handle:
                    handle.write(b"changed checkpoint")
            elif change == "replace":
                replacement = tmp_path / "replacement.pth"
                replacement.write_bytes(data)
                replacement.replace(checkpoint)
            else:
                checkpoint.unlink()

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(emb.hashlib, "sha256", ChangingDigest)
    _refuse_models(monkeypatch)
    entry = emb.encoder_entry(emb.EmbeddingSpec(
        backbone="cell_dino", checkpoint_path=str(checkpoint),
        checkpoint_sha256=expected))
    assert changed == [change]
    assert entry.path == entry.sha256 == "" and entry.size_bytes == 0
    assert not entry.verified
