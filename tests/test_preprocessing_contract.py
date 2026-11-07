"""The input transform travels with the model, and fixed scaling uses no batch statistics.

Pinned here:

* fixed per-plate, per-channel scales come from object pixels of raw crops;
* the batch transform gives a crop the same input alone or in any batch,
  divides by its own plate's scale, clips, resizes to the model input and
  applies the recorded mean and std;
* ``generate_loaders(crop_scaling='fixed')`` records the whole contract,
  including the scales file and its md5;
* a checkpoint carries the contract through save and load; testing refuses a
  run that would transform crops differently, warns when asked to, and keeps
  old checkpoints without a contract working with a warning;
* scoring paths that only know the 8-bit transform refuse a fixed-scale model;
* the validation split stays grouped by well, and the shuffled-label control
  permutes training labels only.
"""
from __future__ import annotations

import hashlib
import os

import cv2
import numpy as np
import pytest
import torch

from spacr import classification_pixels as cp
from spacr.crops import _CROP_FORMAT_DECLARED_RAW, write_crop_folder_marker


def _write_raw(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(str(path), np.ascontiguousarray(arr[:, :, ::-1]))


def _raw_crop(level, size=16, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.zeros((size, size, 3), np.uint16)
    arr[4:12, 4:12] = rng.integers(level // 2, level, size=(8, 8, 3))
    return arr


def _raw_dataset(root, *, classes=("nc", "pc"), wells=("A01", "A02", "B01", "B02"),
                 plates=("plate1", "plate2"), per_well=3):
    for cls_i, cls in enumerate(classes):
        folder = root / "train" / cls
        folder.mkdir(parents=True)
        write_crop_folder_marker(str(folder), _CROP_FORMAT_DECLARED_RAW)
        k = 0
        for p_i, plate in enumerate(plates):
            for well in wells:
                for obj in range(per_well):
                    level = 1000 * (p_i + 1) * (cls_i + 1)
                    _write_raw(folder / f"{plate}_{well}_1_{obj + 10 * cls_i}.png",
                               _raw_crop(level, seed=k))
                    k += 1
    return root


def test_scales_are_the_object_pixel_percentile_per_plate(tmp_path):
    a = tmp_path / "plateA_A01_1_1.png"
    b = tmp_path / "plateB_A01_1_1.png"
    crop_a, crop_b = _raw_crop(1000), _raw_crop(8000, seed=3)
    _write_raw(a, crop_a)
    _write_raw(b, crop_b)
    for folder in {str(tmp_path)}:
        write_crop_folder_marker(folder, _CROP_FORMAT_DECLARED_RAW)
    scales = cp._estimate_crop_scales([str(a), str(b)])
    for name, crop in (("plateA", crop_a), ("plateB", crop_b)):
        for c in range(3):
            plane = crop[:, :, c].astype(float)
            assert scales[name][c] == pytest.approx(np.percentile(plane[plane > 0], 99))
    assert scales["plateB"][0] > scales["plateA"][0]


def test_stretched_crops_are_refused_for_fixed_scaling(tmp_path):
    path = tmp_path / "plate1_A01_1_1.png"
    _write_raw(path, _raw_crop(1000))
    write_crop_folder_marker(str(tmp_path), 3)
    with pytest.raises(ValueError, match="raw crops"):
        cp._read_raw_classification_array(str(path))


def _contract(**over):
    contract = {"contract_version": 1, "crop_scaling": "fixed",
                "crop_loading_policy": cp.RAW_FIXED_SCALE,
                "scales": {"p1": [100.0, 200.0, 400.0], "p2": [10.0, 20.0, 40.0]},
                "clip": 1.5, "image_size": 32, "interpolation": "bilinear",
                "mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25],
                "selected_channels": [1, 2, 3], "augment": False}
    contract.update(over)
    return contract


def test_batch_transform_uses_no_batch_statistics():
    t = cp._FixedScaleInput(_contract())
    x = torch.rand(4, 3, 16, 16) * 300
    names = ["p1_A01_1_1.png", "p2_A01_1_2.png", "p1_B01_1_3.png", "p2_B02_1_4.png"]
    together = t(x, names)
    for i in range(4):
        alone = t(x[i:i + 1], names[i:i + 1])
        assert torch.allclose(alone, together[i:i + 1], atol=1e-6)
    assert together.shape == (4, 3, 32, 32)
    one = torch.full((1, 3, 8, 8), 100.0)
    out = cp._FixedScaleInput(_contract(image_size=8))(one, ["p1_A01_1_1.png"])
    want = (torch.tensor([1.0, 0.5, 0.25]) - 0.5) / 0.25
    assert torch.allclose(out[0, :, 0, 0], want)
    clipped = cp._FixedScaleInput(_contract(image_size=8))(one * 100, ["p1_A01_1_1.png"])
    assert torch.allclose(clipped[0, :, 0, 0], torch.full((3,), (1.5 - 0.5) / 0.25))


def test_augmentation_keeps_intensities_within_the_gain():
    t = cp._FixedScaleInput(_contract(image_size=16, augment=True, mean=None, std=None,
                                      augment_shift_px=0))
    x = torch.full((8, 3, 16, 16), 100.0)
    out = t(x, ["p1_A01_1_1.png"] * 8, train=True)
    scaled = torch.tensor([1.0, 0.5, 0.25])[None, :, None, None]
    ratio = out / scaled
    assert float(ratio.min()) >= 0.9 - 1e-6 and float(ratio.max()) <= 1.1 + 1e-6


def test_fixed_loaders_record_the_contract_and_scales_file(tmp_path):
    from spacr.io import dataset_filenames, generate_loaders

    root = _raw_dataset(tmp_path / "ds")
    train, val, _ = generate_loaders(
        str(root), mode="train", image_size=32, batch_size=4,
        classes=["nc", "pc"], validation_split=0.25, group_by="well",
        crop_scaling="fixed", input_statistics="symmetric", augment=True)
    contract = train.preprocessing_contract
    assert contract is val.preprocessing_contract
    assert contract["crop_loading_policy"] == cp.RAW_FIXED_SCALE
    path = contract["scales_file"]
    with open(path, "rb") as handle:
        assert hashlib.md5(handle.read()).hexdigest() == contract["scales_md5"]
    assert set(contract["scales"]) == {"plate1", "plate2"}
    assert contract["mean"] == [0.5, 0.5, 0.5] and contract["image_size"] == 32
    batch, labels, names = next(iter(train))
    out = cp._prepare_batch(train, batch, names, torch.device("cpu"), train=True)
    assert out.shape[1:] == (3, 32, 32)
    wells = lambda loader: {os.path.basename(n).split("_")[1] + os.path.basename(n).split("_")[0]
                            for n in dataset_filenames(loader.dataset)}
    assert not wells(train) & wells(val), "validation must be grouped by well"


def test_checkpoint_round_trip_and_mismatch_refusal(tmp_path, capsys):
    from spacr.deep_spacr import _inference_contract_check
    from spacr.torch_artifacts import load_model_artifact, save_model_artifact
    from spacr.utils import CustomCellClassifier

    contract = _contract(scales_md5="abc", mask_mode="masked")
    model = CustomCellClassifier(num_classes=2, image_size=32)
    path = save_model_artifact(model, str(tmp_path / "m.pth"),
                               preprocessing=contract, include_rng=False)
    _, meta = load_model_artifact(path)
    assert meta["preprocessing"] == contract
    assert cp._check_preprocessing(meta, dict(contract)) == []
    with pytest.raises(ValueError, match="image_size"):
        cp._check_preprocessing(meta, dict(contract, image_size=64))
    with pytest.raises(ValueError, match="mask_mode"):
        cp._check_preprocessing(meta, dict(contract, mask_mode="bounding_box"))
    found = cp._check_preprocessing(meta, dict(contract, scales_md5="def"),
                                    on_mismatch="warn")
    assert [k for k, _, _ in found] == ["scales_md5"]
    with pytest.raises(ValueError, match="crop_scaling='fixed'"):
        _inference_contract_check(meta, image_size=32, stats=None, on_mismatch="warn")

    legacy = {"preprocessing": {"crop_loading_policy": cp.DECLARED_UINT8,
                                "image_size": 224, "normalize": True}}
    assert cp._check_preprocessing(legacy, {"image_size": 64}) == []
    assert "no full preprocessing record" in capsys.readouterr().out
    assert _inference_contract_check(
        legacy, image_size=64, stats=((0.5,) * 3, (0.5,) * 3), on_mismatch="refuse") == []

    recorded = {"preprocessing": {"contract_version": 1, "crop_scaling": "legacy",
                                  "image_size": 224, "resize": "center_crop",
                                  "mean": [0.5] * 3, "std": [0.5] * 3}}
    with pytest.raises(ValueError, match="image_size"):
        _inference_contract_check(recorded, image_size=128,
                                  stats=((0.5,) * 3, (0.5,) * 3), on_mismatch="refuse")
    assert _inference_contract_check(recorded, image_size=224,
                                     stats=((0.5,) * 3, (0.5,) * 3),
                                     on_mismatch="refuse") == []


def test_shuffled_label_control_permutes_training_labels_only(tmp_path):
    from spacr.io import dataset_labels, generate_loaders

    root = _raw_dataset(tmp_path / "ds", per_well=4)
    kwargs = dict(mode="train", image_size=16, batch_size=4, classes=["nc", "pc"],
                  validation_split=0.25, group_by="well", crop_scaling="fixed", seed=3)
    import random
    random.seed(0)
    plain_t, plain_v, _ = generate_loaders(str(root), **kwargs)
    random.seed(0)
    ctrl_t, ctrl_v, _ = generate_loaders(str(root), shuffle_labels=True, **kwargs)
    assert dataset_labels(plain_v.dataset) == dataset_labels(ctrl_v.dataset)
    assert sorted(dataset_labels(plain_t.dataset)) == sorted(dataset_labels(ctrl_t.dataset))
    assert dataset_labels(plain_t.dataset) != dataset_labels(ctrl_t.dataset)
