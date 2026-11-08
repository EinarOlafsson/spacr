"""Coverage for the fixed-scale contract options, the leave-one-group-out guard, fixed loaders without a split and an empty cellpose dataset."""
from __future__ import annotations

import os

import pytest

import cv2
import numpy as np

from spacr.crops import _CROP_FORMAT_DECLARED_RAW, write_crop_folder_marker


def _raw_dataset(root, *, classes=("nc", "pc"), wells=("A01", "A02", "B01", "B02"),
                 plates=("plate1", "plate2"), per_well=1):
    for cls_i, cls in enumerate(classes):
        folder = root / "train" / cls
        folder.mkdir(parents=True)
        write_crop_folder_marker(str(folder), _CROP_FORMAT_DECLARED_RAW)
        k = 0
        for p_i, plate in enumerate(plates):
            for well in wells:
                for obj in range(per_well):
                    level = 1000 * (p_i + 1) * (cls_i + 1)
                    rng = np.random.default_rng(k)
                    arr = np.zeros((16, 16, 3), np.uint16)
                    arr[4:12, 4:12] = rng.integers(level // 2, level, size=(8, 8, 3))
                    name = folder / f"{plate}_{well}_1_{obj + 10 * cls_i}.png"
                    cv2.imwrite(str(name), np.ascontiguousarray(arr[:, :, ::-1]))
                    k += 1
    return root


def _files(root):
    return sorted(str(p) for p in (root / "train").rglob("*.png"))


def test_an_existing_scale_file_is_read_not_rewritten(tmp_path):
    """A supplied scales file is reused and its md5 comes back in the contract."""
    from spacr import classification_pixels as cp
    from spacr.io import _fixed_scale_contract

    root = _raw_dataset(tmp_path / "ds", per_well=1)
    files = _files(root)
    first = _fixed_scale_contract(files, src=str(root), image_size=32,
                                  channels=[1, 2, 3], augment=False)
    again = _fixed_scale_contract(files, src=str(root), image_size=32,
                                  channels=[1, 2, 3], augment=False,
                                  scale_file=first["scales_file"])
    assert again["scales_file"] == first["scales_file"]
    assert again["scales_md5"] == first["scales_md5"]
    assert again["scales"] == first["scales"]
    assert cp._read_crop_scales(first["scales_file"])[1] == first["scales_md5"]


def test_dataset_statistics_sample_at_most_512_crops(tmp_path):
    """A dataset larger than the sample still gives per-channel dataset statistics."""
    from spacr.io import _fixed_scale_contract

    root = _raw_dataset(tmp_path / "ds", per_well=1)
    files = _files(root)
    many = [files[i % len(files)] for i in range(513)]
    contract = _fixed_scale_contract(many, src=str(root), image_size=32,
                                     channels=[1, 2, 3], augment=False,
                                     input_statistics="dataset")
    assert len(contract["mean"]) == 3
    assert all(0 < m < 1.5 for m in contract["mean"])
    assert all(s > 0 for s in contract["std"])


def test_imagenet_statistics_for_a_model_without_weights_fall_back(tmp_path):
    """A model name with no torchvision weights uses the imagenet statistics from the settings."""
    from spacr.io import _fixed_scale_contract
    from spacr.normalization import IMAGENET_MEAN

    root = _raw_dataset(tmp_path / "ds", per_well=1)
    contract = _fixed_scale_contract(
        _files(root), src=str(root), image_size=32, channels=[1, 2, 3],
        augment=False, input_statistics="imagenet", model_type="custom")
    assert contract["mean"] == pytest.approx(list(IMAGENET_MEAN))


def test_no_statistics_leave_the_contract_without_mean_and_std(tmp_path):
    """The none mode records no mean and no std."""
    from spacr.io import _fixed_scale_contract

    root = _raw_dataset(tmp_path / "ds", per_well=1)
    contract = _fixed_scale_contract(
        _files(root), src=str(root), image_size=32, channels=[1, 2, 3],
        augment=False, input_statistics="none")
    assert contract["mean"] is None
    assert contract["std"] is None


def test_leave_one_group_out_needs_two_groups():
    """A single distinct group value cannot be held out."""
    from spacr.io import _leave_one_group_out_folds

    with pytest.raises(ValueError, match="at least two plate values"):
        _leave_one_group_out_folds(["p1", "p1", "p1"], "plate")


def test_fixed_loaders_without_a_split_carry_the_contract(tmp_path):
    """Without a validation split the single loader still gets the fixed-scale contract."""
    from spacr import classification_pixels as cp
    from spacr.io import generate_loaders

    root = _raw_dataset(tmp_path / "ds", per_well=1)
    train, val, fig = generate_loaders(
        str(root), mode="train", image_size=32, batch_size=4,
        classes=["nc", "pc"], validation_split=0.0, crop_scaling="fixed",
        input_statistics="symmetric")
    assert train.preprocessing_contract["crop_loading_policy"] == cp.RAW_FIXED_SCALE
    assert os.path.isfile(train.preprocessing_contract["scales_file"])
    assert fig is None
