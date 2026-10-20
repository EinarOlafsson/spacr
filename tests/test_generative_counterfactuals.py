"""Counterfactual sequences morph a crop toward the other class.

A tiny classifier learns a known synthetic phenotype (bright puncta inside a
smooth cell); the generator trained against it must flip the classifier on
held-out crops, raise its score monotonically, beat the class-mean shift, and
grow the puncta on control crops, while keeping the edit small.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spacr.attribution import (_CounterfactualDiffusion, _cf_features,
                               _cf_frechet, _counterfactual_report,
                               _counterfactual_sequences,
                               _train_counterfactual_diffusion,
                               _train_counterfactual_generator)


def _crops(n, side=32, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:side, :side].astype(np.float32)
    images, labels = [], []
    for i in range(n):
        label = i % 2
        cy, cx = rng.uniform(side * .35, side * .65, 2)
        r = rng.uniform(side * .18, side * .28)
        img = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
        img = img * rng.uniform(.5, .9)
        if label:
            for _ in range(rng.integers(3, 6)):
                a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, r * .9)
                py, px = cy + d * np.sin(a), cx + d * np.cos(a)
                img = img + .8 * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / 2)
        images.append((img + rng.normal(0, .03, img.shape))[None])
        labels.append(label)
    return (torch.tensor(np.stack(images), dtype=torch.float32),
            torch.tensor(labels))


def _puncta(x):
    blur = F.conv2d(x, torch.ones(1, 1, 5, 5) / 25, padding=2)
    return F.relu(x - blur).sum(dim=(1, 2, 3))


@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    x, y = _crops(160)
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(16, 2))
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    for _ in range(30):
        for s in range(0, len(x), 32):
            opt.zero_grad()
            F.cross_entropy(model(x[s:s + 32]), y[s:s + 32]).backward()
            opt.step()
    return model.eval(), x


def test_the_report_flips_the_classifier_with_a_small_edit(trained, tmp_path):
    model, x = trained
    before = [p.detach().clone() for p in model.parameters()]
    summary, rows, frames = _counterfactual_report(
        model, x, epochs=20, show=3, out_dir=str(tmp_path))
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert summary["heldout_crops"] == len(rows) == 40
    assert summary["flip_rate"] >= 0.8
    assert summary["flip_rate"] > summary["baseline_flip_rate"]
    assert summary["monotone_fraction"] >= 0.8
    assert summary["mean_spearman"] >= 0.8
    assert summary["median_edit_l1"] < 0.15
    assert frames.shape == (3, 7, 1, 32, 32)
    for name in ("counterfactual_cells.csv", "counterfactual_summary.csv",
                 "counterfactual_sequences.pdf"):
        assert (tmp_path / name).exists()


def test_control_crops_gain_the_hit_phenotype(trained):
    model, x = trained
    generator, history = _train_counterfactual_generator(model, x, epochs=20)
    assert history[-1]["reconstruction"] < history[0]["reconstruction"]
    controls, labels = _crops(40, seed=1)
    controls = controls[labels == 0]
    rows, frames = _counterfactual_sequences(
        model, generator, controls, targets=[1] * len(controls),
        keep=len(controls))
    frames = torch.tensor(frames)
    assert torch.allclose(frames[:, 0], controls, atol=1e-5)
    puncta = torch.stack([_puncta(frames[:, k]) for k in range(7)], dim=1)
    assert float((puncta[:, -1] > puncta[:, 0]).float().mean()) >= 0.9
    assert np.mean([r["flipped"] for r in rows]) >= 0.8


def test_too_few_crops_are_refused(trained):
    model, x = trained
    with pytest.raises(ValueError, match="at least 4"):
        _counterfactual_report(model, x[:3], epochs=1)


def test_the_activation_run_skips_counterfactuals_on_too_few_crops(trained,
                                                                   tmp_path,
                                                                   capsys):
    from spacr.deep_spacr import _run_counterfactuals

    model, x = trained
    assert _run_counterfactuals({}, model, [x[:2]], ["a", "b"],
                                str(tmp_path / "cf"), "cpu") is None
    assert "at least 4" in capsys.readouterr().out
    summary = _run_counterfactuals({"counterfactual_epochs": 2}, model,
                                   [x[:6], x[6:12]], [str(i) for i in range(12)],
                                   str(tmp_path / "cf"), "cpu")
    assert summary["train_crops"] + summary["heldout_crops"] == 12
    assert (tmp_path / "cf" / "counterfactual_summary.csv").exists()


# ---------------------------------------------------------------------------
# Helpers at their edges (coverage ratchet, dispatch 36739819315)
# ---------------------------------------------------------------------------

def test_crops_of_another_size_are_resized_and_the_right_size_is_kept():
    from spacr.attribution import _cf_resize

    batch = torch.rand(2, 1, 20, 20)
    assert _cf_resize(batch, 20) is batch
    resized = _cf_resize(batch, 16)
    assert tuple(resized.shape) == (2, 1, 16, 16)


def test_a_constant_ranking_has_no_spearman_correlation():
    from spacr.attribution import _spearman

    assert np.isnan(_spearman(np.ones(5), np.arange(5.0)))
    assert np.isnan(_spearman(np.arange(5.0), np.full(5, 2.0)))
    assert np.isnan(_spearman(np.ones(1), np.ones(1)))
    assert _spearman(np.arange(5.0), np.arange(5.0) * 3) == pytest.approx(1.0)


def test_the_class_mean_baseline_needs_a_training_crop_of_every_class():
    """A model that calls every crop class 0 leaves class 1 with no mean, so
    the naive counterfactual has nothing to shift towards."""
    from spacr.attribution import _class_mean_baseline

    class _AlwaysZero(nn.Module):
        n_classes = 2

        def forward(self, x):
            return torch.stack([torch.ones(len(x)), torch.zeros(len(x))], 1)

    x, _y = _crops(6, side=16)
    flip, edit = _class_mean_baseline(_AlwaysZero(), x[:4], x[4:])
    assert np.isnan(flip) and np.isnan(edit)


def test_without_a_sequence_only_the_tables_are_written(tmp_path):
    from spacr.attribution import _write_counterfactual_outputs
    from spacr.tabular import read_table

    _write_counterfactual_outputs(
        str(tmp_path), {"n_cells": 1, "flip_rate": 1.0},
        [{"cell": 0, "flipped": True}], np.zeros((0,)))
    assert read_table(tmp_path / "counterfactual_cells.csv",
                      report=None)["flipped"].tolist() == [True]
    assert (tmp_path / "counterfactual_summary.csv").is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "counterfactual_cells.csv", "counterfactual_summary.csv"]


def test_a_report_without_an_output_folder_writes_nothing(trained, tmp_path,
                                                          monkeypatch):
    """The summary, rows and frames come back; no folder is created."""
    monkeypatch.chdir(tmp_path)
    model, x = trained
    summary, rows, frames = _counterfactual_report(model, x[:8], epochs=1,
                                                   show=1)
    assert summary["heldout_crops"] == len(rows) == 2
    assert frames.shape[0] == 1
    assert list(tmp_path.iterdir()) == []


def test_condition_names_come_from_the_crop_file_name():
    from spacr.attribution import _cf_condition_of

    name = "plate2_B03_1_7.png"
    assert _cf_condition_of(name, "plate") == "plate2"
    assert _cf_condition_of(name, "well") == "plate2_B03"
    assert _cf_condition_of(name, "row") == "r2"
    assert _cf_condition_of(name, "column") == "c3"
    assert _cf_condition_of("nowell", "column") == ""


def test_a_control_well_morphs_toward_the_hit_well(trained, tmp_path):
    model, x = trained
    _x, y = _crops(160)
    wells = ["control" if int(label) == 0 else "hit" for label in y]
    summary, rows, frames = _counterfactual_report(
        model, x, epochs=20, show=2, out_dir=str(tmp_path),
        conditions=wells)
    assert summary["conditions"] == 2
    assert {(r["source_condition"], r["target_condition"]) for r in rows} == {
        ("control", "hit"), ("hit", "control")}
    assert summary["flip_rate"] >= 0.7, summary
    assert summary["flip_rate"] > summary["baseline_flip_rate"], summary
    assert summary["monotone_fraction"] >= 0.8, summary
    assert (tmp_path / "counterfactual_frames.npy").exists()
    with pytest.raises(ValueError, match="at least 2"):
        _counterfactual_report(model, x[:8], epochs=1, conditions=["a"] * 8)


def test_the_activation_run_uses_the_condition_setting(trained, tmp_path,
                                                       capsys):
    from spacr.deep_spacr import _run_counterfactuals

    model, x = trained
    names = [f"p1_A0{1 + i % 2}_1_{i}.png" for i in range(12)]
    summary = _run_counterfactuals(
        {"counterfactual_epochs": 2, "counterfactual_condition": "column"},
        model, [x[:12]], names, str(tmp_path / "cf"), "cpu")
    assert summary["conditions"] == 2
    one = [f"p1_A01_1_{i}.png" for i in range(12)]
    assert _run_counterfactuals(
        {"counterfactual_epochs": 1, "counterfactual_condition": "well"},
        model, [x[:12]], one, str(tmp_path / "cf2"), "cpu") is None
    assert "at least 2" in capsys.readouterr().out


def test_every_crop_morphs_toward_the_chosen_condition(trained):
    model, x = trained
    _x, y = _crops(160)
    wells = ["control" if int(label) == 0 else "hit" for label in y]
    summary, rows, _frames = _counterfactual_report(
        model, x, epochs=10, show=0, conditions=wells, target="hit")
    assert summary["target"] == "hit" and rows
    assert {(r["source_condition"], r["target_condition"]) for r in rows} == {
        ("control", "hit")}
    assert summary["flip_rate"] >= 0.7, summary
    with pytest.raises(ValueError, match="not among"):
        _counterfactual_report(model, x[:16], epochs=1,
                               conditions=wells[:16], target="nowhere")


def test_a_class_target_is_an_index_and_skips_crops_already_there(trained):
    from spacr.attribution import _cf_target_index

    model, x = trained
    summary, rows, _frames = _counterfactual_report(
        model, x[:40], epochs=2, show=0, target="0")
    assert summary["target"] == "0"
    assert rows and all(r["source_class"] == 1 and r["target_class"] == 0
                        for r in rows)
    assert _cf_target_index("1", [], 2) == 1
    with pytest.raises(ValueError, match="class index"):
        _cf_target_index("2", [], 2)
    with pytest.raises(ValueError, match="class index"):
        _cf_target_index("hit", [], 2)


def test_the_activation_run_passes_the_target_setting(trained, tmp_path,
                                                      capsys):
    from spacr.deep_spacr import _run_counterfactuals

    model, x = trained
    names = [f"p1_A0{1 + i % 2}_1_{i}.png" for i in range(12)]
    summary = _run_counterfactuals(
        {"counterfactual_epochs": 1, "counterfactual_condition": "column",
         "counterfactual_target": "c2"},
        model, [x[:12]], names, str(tmp_path / "cf"), "cpu")
    assert summary["target"] == "c2"
    assert _run_counterfactuals(
        {"counterfactual_epochs": 1, "counterfactual_condition": "column",
         "counterfactual_target": "c9"},
        model, [x[:12]], names, str(tmp_path / "cf2"), "cpu") is None
    assert "not among" in capsys.readouterr().out


# A diffusion model this small learns little in two epochs; these tests pin
# the plumbing (interface, exact zero edit at the own class, outputs, saved
# weights, the setting), not the quality, which the GPU run in the item file
# measures on real crops.
TINY = {"width": 8, "timesteps": 50, "sample_steps": 6}


def test_the_diffusion_generator_runs_through_the_report(trained, tmp_path):
    model, x = trained
    summary, rows, frames = _counterfactual_report(
        model, x[:48], epochs=2, show=2, out_dir=str(tmp_path),
        generator="diffusion", generator_options=TINY)
    assert summary["generator"] == "diffusion"
    assert summary["heldout_crops"] == len(rows) == 12
    for key in ("flip_rate", "monotone_fraction", "median_edit_l1",
                "reconstruction_mse"):
        assert np.isfinite(summary[key])
    assert frames.shape == (2, 7, 1, 32, 32)
    saved = torch.load(tmp_path / "counterfactual_diffusion.pt")
    fresh = _CounterfactualDiffusion(**saved["config"])
    fresh.load_state_dict(saved["state_dict"])
    assert (tmp_path / "counterfactual_summary.csv").exists()


def test_the_first_diffusion_frame_is_the_unedited_crop(trained):
    model, x = trained
    generator, history = _train_counterfactual_diffusion(model, x[:32],
                                                         epochs=1, **TINY)
    assert set(history[-1]) == {"denoising", "reconstruction"}
    rows, frames = _counterfactual_sequences(model, generator, x[:3], keep=3)
    # Batched against single decodes differ only by float rounding.
    assert np.allclose(frames[:, 0], x[:3].numpy(), atol=1e-4)
    assert all(r["score_path"].count(";") == 6 for r in rows)


def test_diffusion_inversion_to_no_noise_is_the_identity(trained):
    model, x = trained
    generator, _ = _train_counterfactual_diffusion(
        model, x[:16], epochs=1, strength=0.0, **TINY)
    with torch.no_grad():
        back = generator.decode(generator.encode(x[:4]),
                                F.one_hot(torch.ones(4).long(), 2).float())
    assert torch.allclose(back, x[:4], atol=1e-5)


def test_an_unknown_generator_is_refused(trained):
    model, x = trained
    with pytest.raises(ValueError, match="not one of"):
        _counterfactual_report(model, x[:8], epochs=1, generator="stylegan")


def test_the_realism_distance_grows_with_the_gap(trained):
    _model, x = trained
    a, b = _cf_features(x[:80:2]), _cf_features(x[1:80:2])
    assert _cf_frechet(a, a) == pytest.approx(0.0, abs=1e-6)
    assert _cf_frechet(_cf_features(x[:80:2] * 1.5), a) > _cf_frechet(b, a)
    assert _cf_features(x[:0]).shape == (0, 4)
    assert np.isnan(_cf_frechet(a[:1], a))


def test_the_activation_run_passes_the_generator_setting(trained, tmp_path):
    from spacr.deep_spacr import _run_counterfactuals

    model, x = trained
    names = [f"p1_A0{1 + i % 2}_1_{i}.png" for i in range(12)]
    summary = _run_counterfactuals(
        {"counterfactual_epochs": 1, "counterfactual_generator": "diffusion"},
        model, [x[:12]], names, str(tmp_path / "cf"), "cpu")
    assert summary["generator"] == "diffusion"
    assert (tmp_path / "cf" / "counterfactual_diffusion.pt").exists()


class _ReferenceScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, image):
        self.seen.append(image.detach().clone())
        score = image[:, 0].mean(dim=(1, 2))
        return torch.stack([-score, score], dim=1)


class _BothChannelEdit(nn.Module):
    side = 8
    n_classes = 2

    def encode(self, image):
        return image.clone()

    def decode(self, latent, code):
        return latent + code[:, 1, None, None, None]


def test_preserved_reference_is_exact_in_frames_finals_and_actual_scorer():
    model = _ReferenceScore()
    generator = _BothChannelEdit()
    crops = torch.stack([torch.full((8, 8), 0.2), torch.full((8, 8), 0.3)])[None]
    before = crops.clone()
    finals = []
    rows, frames = _counterfactual_sequences(model, generator, crops, keep=1, finals=finals, preserve_channels=[0])
    np.testing.assert_array_equal(frames[0, :, 0], np.repeat(crops.numpy()[0:1, 0], 7, axis=0))
    assert not np.allclose(frames[0, -1, 1], crops.numpy()[0, 1])
    assert torch.equal(model.seen[-1], torch.from_numpy(frames[0]))
    assert torch.equal(finals[0], torch.from_numpy(frames[0, -1:]))
    assert rows[0]['score_end'] == rows[0]['score_start'] and not rows[0]['flipped']
    assert rows[0]['preserved_channels'] == '0'
    assert torch.equal(crops, before)
    free_rows, free_frames = _counterfactual_sequences(_ReferenceScore(), generator, crops, keep=1)
    assert free_rows[0]['flipped'] and not np.allclose(free_frames[0, -1, 0], crops[0, 0].numpy())
    assert 'preserved_channels' not in free_rows[0]


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize('batch', [1, 3])
def test_reference_copy_preserves_dtype_shape_and_inputs(dtype, batch):
    from spacr.attribution import _cf_preserve_reference_channels
    edited = torch.arange(3 * 2 * 4 * 5, dtype=dtype).reshape(3, 2, 4, 5)
    source = torch.full((batch, 2, 4, 5), 0.125, dtype=dtype)
    before, original = edited.clone(), source.clone()
    result = _cf_preserve_reference_channels(edited, source, [1])
    assert result.dtype == dtype and result.shape == edited.shape
    assert torch.equal(result[:, 1], source[:, 1].expand(3, -1, -1))
    assert torch.equal(result[:, 0], edited[:, 0])
    assert torch.equal(source, original) and torch.equal(edited, before)
    assert result.data_ptr() != edited.data_ptr()


@pytest.mark.parametrize('channels', [[-1], [2], [0.5], [True], [float('nan')], [0, 0], '0'])
def test_invalid_reference_indices_refuse_before_any_classifier_call(channels):
    model = _ReferenceScore()
    with pytest.raises(ValueError, match='channel'):
        _counterfactual_sequences(model, _BothChannelEdit(), torch.ones(1, 2, 8, 8), preserve_channels=channels)
    assert model.seen == []


@pytest.mark.parametrize('kind', ['dtype', 'shape', 'batch', 'integer', 'nan', 'empty'])
def test_reference_copy_rejects_invalid_tensors(kind):
    from spacr.attribution import _cf_preserve_reference_channels
    edited, source = torch.ones(3, 2, 4, 5), torch.zeros(1, 2, 4, 5)
    if kind == 'dtype':
        source = source.double()
    elif kind == 'shape':
        source = torch.zeros(1, 2, 5, 4)
    elif kind == 'batch':
        source = source.expand(2, -1, -1, -1)
    elif kind == 'integer':
        edited, source = edited.long(), source.long()
    elif kind == 'nan':
        source[0, 0, 0, 0] = float('nan')
    else:
        edited, source = edited[:, :, :0], source[:, :, :0]
    with pytest.raises(ValueError):
        _cf_preserve_reference_channels(edited, source, [0])


def test_reference_constraint_also_applies_to_class_mean_baseline():
    from spacr.attribution import _class_mean_baseline
    train = torch.stack([torch.full((2, 8, 8), -1.0), torch.full((2, 8, 8), 1.0)])
    test = torch.full((1, 2, 8, 8), 0.2)
    original = test.clone()
    unconstrained = _class_mean_baseline(_ReferenceScore(), train, test)
    constrained = _class_mean_baseline(_ReferenceScore(), train, test, preserve_channels=[0])
    assert unconstrained[0] == 1 and constrained[0] == 0
    assert constrained[1] < unconstrained[1]
    assert torch.equal(test, original)


def test_reference_report_passes_constraint_to_outputs_and_baseline(monkeypatch, tmp_path):
    import spacr.attribution as attribution
    model = _ReferenceScore()
    crops = torch.stack([torch.full((8, 8), 0.2), torch.full((8, 8), 0.3)])[None].repeat(8, 1, 1, 1)
    crops[::2, 0] *= -1
    monkeypatch.setattr(attribution, '_train_counterfactual_generator', lambda *a, **kw: (_BothChannelEdit(), []))
    monkeypatch.setattr(attribution, '_cf_realism', lambda *a, **kw: (0.0, 0.0))
    summary, rows, frames = attribution._counterfactual_report(model, crops, epochs=0, show=2, preserve_channels=[0], out_dir=str(tmp_path))
    assert summary['preserved_channels'] == '0'
    assert all(row['preserved_channels'] == '0' for row in rows)
    assert summary['flip_rate'] == 0 and summary['baseline_flip_rate'] == 0
    np.testing.assert_array_equal(frames[:, :, 0], np.repeat(frames[:, :1, 0], 7, axis=1))
    assert 'preserved_channels' in (tmp_path / 'counterfactual_summary.csv').read_text(encoding="utf-8")


@pytest.mark.parametrize('entry', ['sequence', 'report'])
def test_nonfinite_reference_refuses_before_scoring_or_training(entry, monkeypatch):
    import spacr.attribution as attribution
    model = _ReferenceScore()
    crops = torch.ones(8, 2, 8, 8)
    crops[0, 0, 0, 0] = float('nan')
    monkeypatch.setattr(attribution, '_train_counterfactual_generator', lambda *a, **kw: pytest.fail('nonfinite reference reached training'))
    with pytest.raises(ValueError, match='finite pixels'):
        if entry == 'sequence':
            attribution._counterfactual_sequences(model, _BothChannelEdit(), crops, preserve_channels=[0])
        else:
            attribution._counterfactual_report(model, crops, preserve_channels=[0])
    assert model.seen == []


def test_reference_overflow_during_existing_float32_conversion_refuses_before_scoring():
    model = _ReferenceScore()
    crops = torch.full((1, 2, 8, 8), 1e300, dtype=torch.float64)
    with pytest.raises(ValueError, match='finite pixels'):
        _counterfactual_sequences(model, _BothChannelEdit(), crops, preserve_channels=[0])
    assert model.seen == []
