"""Virtual staining: a small U-Net predicts one channel from another.

Synthetic paired fields stand in for a plate: the target channel holds
round blobs and the input channel holds the same blobs inverted, so the
input alone correlates negatively with the stain and a model that learned
anything turns that around.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spacr import deep_spacr as ds


def _field(seed, size=64, blobs=5):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    stain = np.zeros((size, size), np.float32)
    for _ in range(blobs):
        cy, cx = rng.integers(8, size - 8, 2)
        stain[(yy - cy) ** 2 + (xx - cx) ** 2 < 25] = 1.0
    other = 1.0 - stain + rng.normal(0, 0.05, stain.shape).astype(np.float32)
    return np.stack([stain * 1000, other * 1000], axis=-1).astype(np.float32)


def _train(fields, epochs=15):
    return ds._train_virtual_stain(fields, [1], 0, scale=1, crop=32,
                                   per_field=16, epochs=epochs, batch_size=8,
                                   base=8, depth=2, lr=5e-3, seed=0)


def test_normalize_maps_percentiles_to_zero_and_one():
    plane = np.arange(10000, dtype=np.float32).reshape(100, 100)
    out = ds._vs_normalize(plane)
    assert out.min() == 0.0 and out.max() == 1.0
    assert not ds._vs_normalize(np.ones((5, 5))).any()


def test_the_target_cannot_also_be_an_input():
    with pytest.raises(ValueError):
        ds._train_virtual_stain([_field(0)], [0], 0, epochs=1)


def test_the_model_learns_the_stain_and_beats_its_input():
    fitted = _train([_field(s) for s in range(4)])
    assert fitted["losses"][-1] < fitted["losses"][0]
    scores = ds._virtual_stain_scorecard(
        fitted, [_field(10)], ["held"],
        segment=lambda p: ds._vs_segment(p, sigma=1, min_size=10,
                                         min_distance=4))
    pred = scores.set_index("kind").loc["predicted"]
    base = scores.set_index("kind").loc["input_baseline"]
    assert pred["pearson"] > 0.5 > base["pearson"]
    assert {"ssim", "psnr", "nrmse", "f1_50", "f1_75", "n_real"} <= set(scores)


def test_identical_planes_score_perfectly():
    plane = ds._vs_normalize(_field(3)[..., 0])
    metrics = ds._vs_pixel_metrics(plane, plane)
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["ssim"] == pytest.approx(1.0)
    from spacr.scorecard import match_objects
    labels = ds._vs_segment(plane, sigma=1, min_size=10, min_distance=4)
    assert labels.max() > 0
    assert match_objects(labels, labels, 0.5).f1 == pytest.approx(1.0)


def test_a_saved_model_predicts_the_same(tmp_path):
    fitted = _train([_field(1)], epochs=1)
    ds._save_virtual_stain(fitted, tmp_path / "vs.pt")
    loaded = ds._load_virtual_stain(tmp_path / "vs.pt")
    field = _field(7, size=60)
    first = ds._predict_virtual_stain(fitted, field)
    assert first.shape == (60, 60)
    np.testing.assert_allclose(first, ds._predict_virtual_stain(loaded, field),
                               atol=1e-6)


def test_a_folder_run_writes_model_predictions_and_scores(tmp_path):
    from spacr.tabular import read_table

    for i in range(4):
        np.save(tmp_path / f"f{i}.npy", _field(i))
    scores, summary = ds._virtual_stain_from_folder(
        str(tmp_path), [1], 0, scale=1, crop=32, per_field=4, epochs=1,
        base=4, depth=2)
    out = tmp_path / "virtual_stain"
    assert summary["train_fields"] == 3 and summary["test_fields"] == 1
    assert (out / "virtual_stain_c0.pt").exists()
    assert np.load(out / "f3_virtual_c0.npy").shape == (64, 64)
    table = read_table(out / "virtual_stain_scores.csv", report=None)
    assert sorted(table["kind"]) == ["input_baseline", "predicted"]
    assert "predicted_f1_50" in summary
