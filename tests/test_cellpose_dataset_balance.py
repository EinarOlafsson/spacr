"""The "balanced" Cellpose training split was not balanced.

`prepare_cellpose_dataset` brings every under-sized folder up to
``target_size`` by adding ``needed = target_size - dataset_len`` augmented
pairs. It used to do that by zipping ``pairs`` (length ``dataset_len``)
against ``aug_methods * (dataset_len // len(aug_methods))`` -- a list
truncated to a multiple of five -- inside a loop running ``needed // 5``
times. The number added therefore depended on ``dataset_len`` rather than on
``needed``, and was right only for ``5 <= dataset_len <= 9``.

Measured on folders of 12, 20 and 29 pairs against a target of 29:

    12 -> 44 pairs   (32 added where 17 were needed)
    20 -> 44 pairs   (24 added where 9 were needed)
    29 -> 29 pairs

The smallest folder became the LARGEST, so a model trained on this saw the
scarcest condition most often -- the exact bias the balancing exists to
remove. Below five pairs the multiplier is 0, the augmentation list is empty
and the zip yields nothing, so that folder stayed short instead.
"""

import os

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from spacr.io import prepare_cellpose_dataset


def _dataset(root, name, n):
    """One folder of n image/mask pairs, in the layout the scanner expects."""
    images = os.path.join(root, name)
    masks = os.path.join(images, "masks")
    os.makedirs(masks, exist_ok=True)
    rng = np.random.default_rng(abs(hash(name)) % 2**31)
    for i in range(n):
        tifffile.imwrite(os.path.join(images, f"{name}_{i}.tif"),
                         rng.integers(0, 4000, (24, 24)).astype(np.uint16))
        mask = np.zeros((24, 24), dtype=np.uint16)
        mask[4:10, 4:10] = 1
        tifffile.imwrite(os.path.join(masks, f"{name}_{i}.tif"), mask)
    return images


def _written(output_root):
    """{split: number of image files written}."""
    counts = {}
    for split in ("train", "test"):
        folder = os.path.join(output_root, split, "images")
        counts[split] = len(os.listdir(folder)) if os.path.isdir(folder) else 0
    return counts


@pytest.mark.parametrize("sizes", [
    (12, 20, 29),      # the reported case
    (3, 29),           # below one round of augmentations -- used to add none
    (5, 9, 29),        # the only window the old arithmetic got right
])
def test_every_folder_contributes_the_same_number(tmp_path, sizes):
    root = tmp_path / "in"
    root.mkdir()
    for index, n in enumerate(sizes):
        _dataset(str(root), f"ds{index}", n)
    prepare_cellpose_dataset(str(root), augment_data=True,
                             train_fraction=0.8, n_jobs=1)

    counts = _written(os.path.join(str(root), "cellpose_dataset"))
    target = max(sizes)
    total = counts["train"] + counts["test"]
    assert total == target * len(sizes), (
        f"expected {len(sizes)} folders of {target}, got {total} files "
        f"across {counts}")


def test_without_augmentation_every_folder_is_cut_to_the_smallest(tmp_path):
    """The other branch, pinned so the fix did not move it."""
    root = tmp_path / "in"
    root.mkdir()
    for index, n in enumerate((7, 15, 22)):
        _dataset(str(root), f"ds{index}", n)
    prepare_cellpose_dataset(str(root), augment_data=False,
                             train_fraction=0.8, n_jobs=1)

    counts = _written(os.path.join(str(root), "cellpose_dataset"))
    assert counts["train"] + counts["test"] == 7 * 3


def test_a_pair_is_re_augmented_before_a_combination_repeats(tmp_path):
    """Sampling from the (pair x augmentation) product, not from pairs alone.

    With 2 pairs and 5 augmentations there are 10 distinct combinations, so
    reaching 10 additions must use each exactly once rather than repeating a
    handful.
    """
    root = tmp_path / "in"
    root.mkdir()
    _dataset(str(root), "small", 2)
    _dataset(str(root), "big", 12)
    prepare_cellpose_dataset(str(root), augment_data=True,
                             train_fraction=0.5, n_jobs=1)

    counts = _written(os.path.join(str(root), "cellpose_dataset"))
    assert counts["train"] + counts["test"] == 12 * 2
