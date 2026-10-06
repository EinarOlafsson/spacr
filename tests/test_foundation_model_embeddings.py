"""Single-cell foundation models as embedding backbones, and their scorecard.

The models themselves are downloaded weights, so these tests pin the wiring
around them with injected encoders: how many planes a channel-adaptive or
fixed-width encoder is handed under each channel policy, which loader a
backbone name reaches, the refusal for unsupported checkpoints, and the kNN
accuracy / mAP scorecard that compares backbones on a labelled set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spacr.embeddings as emb


def _recording(width):
    seen = []

    def run(stack):
        seen.append(stack.shape[-1])
        return stack.mean(axis=(1, 2))[:, :1].repeat(4, axis=1)

    run.in_channels = width
    return run, seen


def _crops(n=6, channels=3):
    return np.random.default_rng(1).random((n, 8, 8, channels),
                                           dtype=np.float32)


def test_a_channel_adaptive_encoder_sees_one_plane_per_channel():
    run, seen = _recording(None)
    result = emb.embed_array(_crops(), emb.EmbeddingSpec(), encoder=run)
    assert seen == [1, 1, 1]
    assert result.values.shape == (6, 12)
    assert [emb.channel_of_column(c) for c in result.columns[::4]] == [0, 1, 2]


def test_a_channel_adaptive_encoder_sees_every_channel_at_once_when_projected():
    run, seen = _recording(None)
    spec = emb.EmbeddingSpec(backbone="openphenom",
                             channel_policy=emb.CHANNEL_PROJECT)
    emb.embed_array(_crops(channels=5), spec, encoder=run)
    assert seen == [5]
    with pytest.raises(emb.EmbeddingError, match="cannot be projected"):
        emb.embed_array(_crops(channels=5), emb.EmbeddingSpec(
            channel_policy=emb.CHANNEL_PROJECT), encoder=run)


def test_a_two_plane_encoder_gets_the_first_two_or_a_zero_pad():
    run, seen = _recording(2)
    spec = emb.EmbeddingSpec(channel_policy=emb.CHANNEL_PROJECT)
    emb.embed_array(_crops(channels=3), spec, encoder=run)
    emb.embed_array(_crops(channels=1), spec, encoder=run)
    emb.embed_array(_crops(channels=2), emb.EmbeddingSpec(), encoder=run)
    assert seen == [2, 2, 2, 2]


def test_a_plain_encoder_still_gets_three_planes():
    seen = []

    def run(stack):
        seen.append(stack.shape[-1])
        return stack.mean(axis=(1, 2))

    emb.embed_array(_crops(channels=2), emb.EmbeddingSpec(), encoder=run)
    assert seen == [3, 3]


def test_a_backbone_name_reaches_its_loader(monkeypatch):
    calls = []
    monkeypatch.setattr(emb, "_foundation_encoder",
                        lambda spec: calls.append(("foundation", spec.backbone)))
    monkeypatch.setattr(emb, "_timm_encoder",
                        lambda spec: calls.append(("timm", spec.backbone)))
    for name in emb._foundation_names():
        emb._backbone_encoder(emb.EmbeddingSpec(backbone=name))
    emb._backbone_encoder(emb.EmbeddingSpec(backbone="resnet18"))
    assert calls == [("foundation", n) for n in emb._foundation_names()] + [
        ("timm", "resnet18")]
    assert set(emb._foundation_names()) == {
        "openphenom", "chada_vit", "subcell", "subcell_rybg", "cell_dino"}


def test_cell_dino_is_refused_with_a_reason_before_any_download():
    pytest.importorskip("torch")
    with pytest.raises(emb.EmbeddingError, match="not yet supported"):
        emb._foundation_encoder(emb.EmbeddingSpec(backbone="cell_dino"))


def test_every_foundation_model_says_how_many_channels_it_takes():
    for info in emb._FOUNDATION_MODELS.values():
        assert info["in_channels"] in (None, 2, 4)
        assert info["size"] > 0 and info["label"]


def _clusters(separation, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(["a", "b", "c"], 20)
    values = rng.normal(scale=0.05, size=(60, 3))
    for column, name in enumerate("abc"):
        values[:, column] += separation * (labels == name)
    keys = [f"crop{i}" for i in range(60)]
    return pd.DataFrame(values, index=keys), dict(zip(keys, labels))


def test_the_scorecard_is_perfect_on_separated_classes():
    frame, labels = _clusters(1.0)
    card = emb._retrieval_scorecard(frame, labels, k=5)
    assert card["knn_accuracy"] == 1.0
    assert card["map"] == pytest.approx(1.0)
    assert card["precision_at_k"] == 1.0
    assert card["n"] == 60 and card["classes"] == 3
    assert card["chance_map"] == pytest.approx(19 / 59)


def test_the_scorecard_is_near_chance_on_noise():
    frame, labels = _clusters(0.0)
    card = emb._retrieval_scorecard(frame, labels, k=5)
    assert card["map"] < 0.5
    assert card["precision_at_k"] < 0.6
    assert abs(card["map"] - card["chance_map"]) < 0.1


def test_the_scorecard_drops_blanks_and_refuses_one_class():
    frame, labels = _clusters(1.0)
    labels = {k: ("" if i % 2 else "a") for i, k in enumerate(labels)}
    with pytest.raises(ValueError, match="two classes"):
        emb._retrieval_scorecard(frame, labels)


def test_all_cells_are_scored_and_singletons_do_not_lower_average_precision():
    values = np.concatenate((np.tile([1., 0.], (260, 1)),
                             np.tile([0., 1.], (260, 1)), [[.5, .5]]))
    keys = [f"crop{i}" for i in range(len(values))]
    labels = dict(zip(keys, ["a"] * 260 + ["b"] * 260 + ["singleton"]))
    card = emb._retrieval_scorecard(pd.DataFrame(values, index=keys), labels,
                                  k=300)
    assert card["n"] == 521 and card["classes"] == 3
    assert card["knn_accuracy"] == pytest.approx(520 / 521)
    assert card["map"] == pytest.approx(1.0)
    assert card["precision_at_k"] == pytest.approx(520 * 259 / (521 * 300))
    assert card["chance_map"] == pytest.approx(259 / 520)
    assert card["chance_precision"] == pytest.approx(259 / 521)
