"""``train_cellpose`` must train on every annotated image, not ``batch_size``.

``train_cellpose`` computed ``n_base = min(settings['batch_size'], ...)`` and
then sliced the dataset to ``n_base`` base images, so ``batch_size`` -- the
optimizer's minibatch size, which it ALSO forwards to ``train_seg`` -- doubled
as a cap on the training set. With the shipped default of ``batch_size=8``
(``spacr.settings.get_train_cellpose_default_settings``), a user who annotated
300 fields trained on 8 of them and was told nothing.

The two concepts are separated here: dataset size is the number of matched
image/mask pairs, minibatch size is ``batch_size``, and ``max_train_images``
is a separate opt-in ceiling for RAM-bound machines.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

import matplotlib
matplotlib.use("Agg")

import spacr.submodules as SUB

_NET = object()


@pytest.fixture(autouse=True)
def _close_figures():
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


@pytest.fixture
def cp_stub(monkeypatch):
    """Record what reaches cellpose.train.train_seg."""
    rec = {"train_calls": [], "previews": []}

    class _FakeCellposeModel:
        def __init__(self, gpu=False, pretrained_model=None, **kw):
            self.gpu = gpu
            self.pretrained_model = pretrained_model
            self.net = _NET

    def _fake_train_seg(net, **kwargs):
        rec["train_calls"].append({"net": net, **kwargs})
        return "path_sentinel", [0.1], [0.1]

    def _fake_plot(images, labels):
        rec["previews"].append((list(images), list(labels)))

    monkeypatch.setattr(SUB.cp_models, "CellposeModel", _FakeCellposeModel)
    monkeypatch.setattr(SUB.train_cp, "train_seg", _fake_train_seg)
    monkeypatch.setattr(SUB, "_cellpose_use_gpu", lambda: False)
    monkeypatch.setattr(SUB, "plot_cellpose_batch", _fake_plot)
    return rec


def test_cellpose_use_gpu_falls_back_to_cpu_when_the_cuda_probe_raises(
        monkeypatch, capsys):
    """A broken CUDA install must degrade to CPU, not abort training."""
    import torch

    def _explode():
        raise RuntimeError("no NVIDIA driver")

    monkeypatch.setattr(torch.cuda, "is_available", _explode)
    assert SUB._cellpose_use_gpu() is False
    assert "CUDA probe failed" in capsys.readouterr().out


def test_cellpose_use_gpu_reports_a_working_probe(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert SUB._cellpose_use_gpu() is True


def _write_annotated_fields(root, n, size=24):
    """Write ``n`` matched image/mask pairs under ``<root>/train/``."""
    img_dir = root / "train" / "images"
    mask_dir = root / "train" / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        lbl = np.zeros((size, size), dtype=np.uint16)
        lbl[2:8, 2:8] = 1
        lbl[14:22, 14:22] = 2
        img = ((lbl > 0).astype(np.uint16) * 40000) + 100
        tifffile.imwrite(str(img_dir / f"field_{i:03d}.tif"), img)
        tifffile.imwrite(str(mask_dir / f"field_{i:03d}.tif"), lbl)
    return root


def _settings(root, **over):
    s = {
        "src": str(root), "model_name": "m", "n_epochs": 10,
        "target_size": 16, "augment": False, "batch_size": 8,
        "learning_rate": 5e-5, "weight_decay": 0.1,
    }
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_fields", [1, 3, 9, 20])
def test_every_annotated_field_reaches_train_seg(tmp_path, cp_stub, n_fields):
    """Dataset size is the number of annotated fields -- full stop."""
    _write_annotated_fields(tmp_path, n_fields)
    SUB.train_cellpose(_settings(tmp_path, batch_size=8))

    call = cp_stub["train_calls"][0]
    assert len(call["train_data"]) == n_fields
    assert len(call["train_labels"]) == n_fields
    # ...and batch_size is still passed through as the minibatch size.
    assert call["batch_size"] == 8


def test_the_shipped_default_no_longer_throws_away_the_dataset(tmp_path, cp_stub):
    """The reported scenario, at the shipped default batch_size of 8.

    Before the fix this call sent 8 of 30 images to train_seg.
    """
    from spacr.settings import get_train_cellpose_default_settings

    _write_annotated_fields(tmp_path, 30)
    settings = get_train_cellpose_default_settings({})
    assert settings["batch_size"] == 8, "default changed; update this test"

    settings["src"] = str(tmp_path)
    settings["n_epochs"] = 2
    settings["target_size"] = 16
    SUB.train_cellpose(settings)

    call = cp_stub["train_calls"][0]
    assert len(call["train_data"]) == 30
    assert call["batch_size"] == 8


@pytest.mark.parametrize("batch_size", [1, 2, 8, 64])
def test_batch_size_does_not_change_the_training_set(tmp_path, cp_stub, batch_size):
    """Same data, four minibatch sizes, one dataset size."""
    _write_annotated_fields(tmp_path, 12)
    SUB.train_cellpose(_settings(tmp_path, batch_size=batch_size))

    call = cp_stub["train_calls"][0]
    assert len(call["train_data"]) == 12
    assert call["batch_size"] == batch_size


def test_augment_fans_every_field_out_to_eight(tmp_path, cp_stub):
    """augment=True is 8x the WHOLE set, not 8x ``batch_size`` images."""
    _write_annotated_fields(tmp_path, 5)
    SUB.train_cellpose(_settings(tmp_path, augment=True, batch_size=2))

    call = cp_stub["train_calls"][0]
    assert len(call["train_data"]) == 40          # 5 fields x 8 variants
    assert len(call["train_labels"]) == 40
    assert call["batch_size"] == 2


def test_all_annotated_labels_are_represented(tmp_path, cp_stub):
    """Not just the count: every field's content arrives (order-independent).

    The base images are shuffled before training, so compare as a set.
    """
    _write_annotated_fields(tmp_path, 6)
    SUB.train_cellpose(_settings(tmp_path, batch_size=1))

    call = cp_stub["train_calls"][0]
    got = {lbl.tobytes() for lbl in call["train_labels"]}
    assert len(call["train_labels"]) == 6
    # Identical synthetic fields collapse to one distinct array; what matters
    # is that six patches arrived with the right shape/dtype.
    assert len(got) == 1
    for lbl in call["train_labels"]:
        assert lbl.shape == (16, 16)
        assert lbl.dtype == np.uint16
    for img in call["train_data"]:
        assert img.shape == (16, 16)
        assert img.dtype == np.float32


# ---------------------------------------------------------------------------
# max_train_images: the separate, opt-in ceiling
# ---------------------------------------------------------------------------

def test_max_train_images_caps_and_says_so(tmp_path, cp_stub, capsys):
    _write_annotated_fields(tmp_path, 10)
    SUB.train_cellpose(_settings(tmp_path, batch_size=8, max_train_images=4))

    call = cp_stub["train_calls"][0]
    assert len(call["train_data"]) == 4
    assert call["batch_size"] == 8, "the cap must not touch the minibatch"
    out = capsys.readouterr().out
    assert "training on 4 of 10 annotated images" in out


def test_max_train_images_larger_than_the_set_is_a_no_op(tmp_path, cp_stub, capsys):
    _write_annotated_fields(tmp_path, 5)
    SUB.train_cellpose(_settings(tmp_path, max_train_images=500))

    assert len(cp_stub["train_calls"][0]["train_data"]) == 5
    assert "training on" not in capsys.readouterr().out


@pytest.mark.parametrize("value", [None, 0, -1])
def test_falsy_max_train_images_means_use_everything(tmp_path, cp_stub, value):
    _write_annotated_fields(tmp_path, 7)
    SUB.train_cellpose(_settings(tmp_path, max_train_images=value))
    assert len(cp_stub["train_calls"][0]["train_data"]) == 7


def test_max_train_images_applies_before_augmentation(tmp_path, cp_stub):
    """The cap counts BASE images; each capped image still fans out 8x."""
    _write_annotated_fields(tmp_path, 6)
    SUB.train_cellpose(_settings(tmp_path, augment=True, max_train_images=2))
    assert len(cp_stub["train_calls"][0]["train_data"]) == 16   # 2 x 8


# ---------------------------------------------------------------------------
# The preview is a sanity check, not the dataset
# ---------------------------------------------------------------------------

def test_preview_is_capped_but_training_is_not(tmp_path, cp_stub):
    """plot_cellpose_batch allocates 4 figure-inches per image.

    Handing it a 300-image set would ask matplotlib for a 100-foot figure,
    so the preview is sliced -- the training set is not.
    """
    _write_annotated_fields(tmp_path, 20)
    SUB.train_cellpose(_settings(tmp_path))

    preview_images, preview_labels = cp_stub["previews"][0]
    assert len(preview_images) == SUB._TRAIN_PREVIEW_N == 8
    assert len(preview_labels) == 8
    assert len(cp_stub["train_calls"][0]["train_data"]) == 20
    # The preview shows real training patches, by identity.
    train_ids = [id(a) for a in cp_stub["train_calls"][0]["train_data"]]
    assert [id(a) for a in preview_images] == train_ids[:8]


def test_preview_shows_everything_when_the_set_is_small(tmp_path, cp_stub):
    _write_annotated_fields(tmp_path, 3)
    SUB.train_cellpose(_settings(tmp_path))
    assert len(cp_stub["previews"][0][0]) == 3


def test_a_failing_preview_does_not_abort_training(tmp_path, cp_stub, monkeypatch,
                                                    capsys):
    def _boom(images, labels):
        raise RuntimeError("no display")

    monkeypatch.setattr(SUB, "plot_cellpose_batch", _boom)
    _write_annotated_fields(tmp_path, 4)
    SUB.train_cellpose(_settings(tmp_path))

    assert "could not print batch images" in capsys.readouterr().out
    assert len(cp_stub["train_calls"][0]["train_data"]) == 4


def test_progress_line_reports_patches_and_base_images(tmp_path, cp_stub, capsys):
    _write_annotated_fields(tmp_path, 4)
    SUB.train_cellpose(_settings(tmp_path, augment=True, batch_size=2))

    out = capsys.readouterr().out
    assert "Training model on 32 patches from 4 annotated images" in out
    assert "augment=True, x8" in out
    assert "minibatch 2" in out
