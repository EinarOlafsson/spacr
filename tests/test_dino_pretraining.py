"""Self-supervised (DINO) pretraining on a screen's own crops.

A tiny backbone is trained for one or two epochs on synthetic crops on the
CPU: enough to check that the checkpoint resumes, refuses a mismatched run,
and loads back as an embedding backbone that can be scored.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from spacr import embeddings as emb


def _crops(n=24, size=16):
    rng = np.random.default_rng(0)
    crops = rng.random((n, size, size, 2)).astype(np.float32) * 0.2
    labels = np.arange(n) % 2
    crops[labels == 1, 4:12, 4:12, 0] += 1.0
    return crops, labels


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.set_num_threads(2)
    crops, _labels = _crops()
    path = str(tmp_path_factory.mktemp("dino") / "own.pt")
    first = emb._dino_pretrain(crops, path, arch="resnet10t", size=16,
                               epochs=1, batch_size=16, out_dim=32,
                               n_local=2, device="cpu")
    return path, first


def test_views_are_the_requested_size_and_stay_in_range():
    gen = torch.Generator().manual_seed(0)
    batch = torch.rand(3, 2, 20, 20)
    view = emb._dino_views(batch, 12, (0.1, 0.4), gen, torch)
    assert tuple(view.shape) == (3, 2, 12, 12)
    assert float(view.min()) >= 0.0 and float(view.max()) <= 1.0


def test_per_channel_training_images_are_one_stain_each():
    crops, _labels = _crops(n=5)
    images, scale = emb._dino_planes(crops, emb.CHANNEL_PER_CHANNEL)
    assert images.shape == (10, 1, 16, 16) and len(scale) == 2
    images, _scale = emb._dino_planes(crops, emb.CHANNEL_PROJECT)
    assert images.shape == (5, 2, 16, 16)


def test_training_writes_a_checkpoint_that_resumes(checkpoint):
    path, first = checkpoint
    assert first["epochs"] == 1 and len(first["loss"]) == 1
    assert np.isfinite(first["loss"][0])
    crops, _labels = _crops()
    second = emb._dino_pretrain(crops, path, arch="resnet10t", size=16,
                                epochs=2, batch_size=16, out_dim=32,
                                n_local=2, device="cpu")
    assert second["epochs"] == 2 and len(second["loss"]) == 2
    assert second["loss"][0] == first["loss"][0]
    again = emb._dino_pretrain(crops, path, arch="resnet10t", size=16,
                               epochs=2, out_dim=32, device="cpu")
    assert again["seconds"] == 0.0 and again["loss"] == second["loss"]


def test_a_checkpoint_from_other_settings_is_refused(checkpoint):
    path, _first = checkpoint
    crops, _labels = _crops()
    with pytest.raises(emb.EmbeddingError, match="different settings"):
        emb._dino_pretrain(crops, path, arch="resnet10t", size=24, epochs=3,
                           out_dim=32, device="cpu")


def test_the_checkpoint_embeds_and_is_scored(checkpoint):
    path, _first = checkpoint
    crops, labels = _crops()
    spec = emb.EmbeddingSpec(backbone=emb._DINO_PREFIX + path, device="cpu")
    result = emb.embed_array(crops, spec)
    assert result.values.shape[0] == len(crops)
    assert result.n_channels == 2 and np.isfinite(result.values).all()
    table = emb._backbone_scorecards(crops, labels, {"own": spec}, k=3)
    assert list(table.index) == ["own"]
    assert 0.0 <= table.loc["own", "map"] <= 1.0
    assert table.loc["own", "chance_map"] > 0


def test_the_other_policy_is_refused(checkpoint):
    path, _first = checkpoint
    crops, _labels = _crops(n=4)
    spec = emb.EmbeddingSpec(backbone=emb._DINO_PREFIX + path,
                             channel_policy=emb.CHANNEL_PROJECT, device="cpu")
    with pytest.raises(emb.EmbeddingError, match="per_channel"):
        emb.embed_array(crops, spec)
