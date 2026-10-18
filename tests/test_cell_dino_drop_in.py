"""Installed Cell-DINO is a drop-in upgrade; open DINOv2 runs otherwise."""
from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")

from spacr import embeddings
from spacr.embeddings import _OPEN_DINOV2, _dino_choice, _find_cell_dino


@pytest.fixture(autouse=True)
def model_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_MODEL_CACHE", str(tmp_path / "models"))
    monkeypatch.delenv("SPACR_CELL_DINO_CHECKPOINT", raising=False)
    monkeypatch.setattr(embeddings, "_CELL_DINO_SEEN", {})
    return tmp_path / "models" / "cell_dino"


def _install(folder, planes, patch, name="weights.pth"):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    torch.save({"patch_embed.proj.weight": torch.zeros(8, planes, patch, patch),
                "cls_token": torch.zeros(1, 1, 8)}, path)
    return path


def test_without_a_checkpoint_open_dinov2_runs_with_the_chosen_policy(model_cache):
    spec, note = _dino_choice(None, 3, channel_policy="per_channel",
                              batch_size=8, device="cpu")

    assert (spec.backbone, spec.channel_policy, spec.batch_size, spec.device) == (
        _OPEN_DINOV2, "per_channel", 8, "cpu")
    assert note["model"] == "dinov2" and note["reason"] == "absent"
    assert note["folder"] == str(model_cache)
    assert _find_cell_dino() == {}


@pytest.mark.parametrize("planes, patch, factory", [
    (5, 8, "cell_dino_cp_vits8"),
    (4, 16, "cell_dino_hpa_vitl16"),
    (4, 14, "cell_dino_hpa_vitl14"),
])
def test_an_installed_checkpoint_is_recognised_by_its_patch_embedding(
        model_cache, planes, patch, factory):
    path = _install(model_cache, planes, patch)

    found = _find_cell_dino(verify=True)

    assert found == {"path": str(path.resolve()), "factory": factory,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_unmapped_planes_keep_dinov2_and_say_why(model_cache):
    _install(model_cache, 5, 8)

    spec, note = _dino_choice((0, 1, 2), 3)

    assert spec.backbone == _OPEN_DINOV2
    assert (note["model"], note["reason"], note["planes"]) == (
        "dinov2", "unmapped", "5")
    assert note["factory"] == "cell_dino_cp_vits8"


def test_mapped_planes_run_the_official_checkpoint_with_its_digest(model_cache):
    path = _install(model_cache, 4, 16)

    spec, note = _dino_choice((3, 2, 1, 0), 4, channel_policy="per_channel")

    assert note["model"] == "cell_dino"
    assert spec.backbone == "cell_dino"
    assert spec.channel_policy == "project"
    assert spec.channels == (3, 2, 1, 0)
    assert spec.cell_dino_factory == "cell_dino_hpa_vitl16"
    assert spec.checkpoint_path == str(path.resolve())
    assert spec.checkpoint_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert embeddings._cell_dino_spec(spec) == (4, 224)


@pytest.mark.parametrize("channels", [(0, 1, 2, 2), (0, 1, 2, 4), (0, 1, True, 3)])
def test_invalid_mappings_fall_back(model_cache, channels):
    _install(model_cache, 4, 16)

    spec, note = _dino_choice(channels, 4)

    assert spec.backbone == _OPEN_DINOV2 and note["reason"] == "unmapped"


def test_the_environment_names_a_checkpoint_outside_the_cache(
        model_cache, tmp_path, monkeypatch):
    _install(model_cache, 4, 16)
    chosen = _install(tmp_path / "elsewhere", 5, 8, name="cp.pt")
    monkeypatch.setenv("SPACR_CELL_DINO_CHECKPOINT", str(chosen))

    assert _find_cell_dino()["factory"] == "cell_dino_cp_vits8"


def test_a_file_that_is_not_a_cell_dino_state_is_reported(model_cache):
    model_cache.mkdir(parents=True)
    (model_cache / "broken.pth").write_bytes(b"not a checkpoint")

    spec, note = _dino_choice((0, 1, 2, 3), 4)

    assert spec.backbone == _OPEN_DINOV2
    assert (note["model"], note["reason"]) == ("dinov2", "unreadable")
    assert note["path"].endswith("broken.pth")
