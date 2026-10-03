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

from spacr.attribution import (_counterfactual_report,
                               _counterfactual_sequences,
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
