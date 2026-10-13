"""A mask must be written under the name of the image it segments.

`_load_images_and_labels` built its name list as
``sorted(basename(f) for f in image_files)`` while filling ``images`` in the
caller's order -- and the caller shuffles: `spacr_cellpose.py:169` and `:296`
both call ``random.shuffle(all_image_files)`` first. `identify_masks_finetune`
then writes each mask as ``os.path.join(dst, image_names[file_index])`` over
``enumerate(images)``.

So on any run with more than one file, masks were saved under other images'
filenames. Nothing raises, nothing is logged, every file that should exist
exists, and the segmentation of well A01 is on disk called A05. Every
downstream measurement then joins the wrong mask to the wrong image.

Sorting was only half. Each loop `continue`s past a file that will not read,
which shortened ``images`` while the precomputed names kept every entry, so
one unreadable file misnamed every mask after it even on sorted input.
"""

import os

import numpy as np
import pytest

from spacr.io import _load_images_and_labels


@pytest.fixture
def tiffs(tmp_path):
    """Six images whose pixel values encode their own filename."""
    tifffile = pytest.importorskip("tifffile")
    paths = {}
    for index, stem in enumerate(["e", "c", "a", "f", "b", "d"]):
        path = tmp_path / f"{stem}.tif"
        # A constant plane per file, so a loaded image identifies itself.
        tifffile.imwrite(str(path),
                         np.full((8, 8), (index + 1) * 10, dtype=np.uint16))
        paths[f"{stem}.tif"] = str(path)
    return paths


def _identify(image):
    """Recover which file an image came from, from its constant value."""
    return int(round(float(np.max(image)) * 10)) // 10 if image.max() <= 1 \
        else int(np.max(image))


def test_names_follow_the_pixels_when_the_caller_shuffles(tiffs):
    """The regression, in the order the real caller produces.

    `random.shuffle` is what `identify_masks_finetune` does to its file list,
    so an unsorted input is the NORMAL case here, not an edge case.
    """
    order = ["f.tif", "b.tif", "e.tif", "a.tif", "d.tif", "c.tif"]
    files = [tiffs[name] for name in order]

    images, _labels, names, _label_names = _load_images_and_labels(
        image_files=files, label_files=None)

    assert names == order, (
        "names came back in a different order than the images; a mask written "
        "as dst/names[i] would carry another image's filename")
    assert len(images) == len(names)

    # And prove it at the pixel level, not just by list equality.
    for image, name in zip(images, names):
        expected = _load_expected(tiffs[name])
        assert np.array_equal(np.squeeze(image), expected), name


def _load_expected(path):
    import tifffile
    raw = tifffile.imread(path).astype(float)
    return raw / raw.max() if raw.max() > 1 else raw


def test_a_skipped_file_does_not_misname_everything_after_it(tiffs, tmp_path,
                                                             monkeypatch):
    """The second half of the bug, which survives even sorted input.

    The loop `continue`s past a file that reads as None. With the names
    computed up front, `images` got shorter and the name list did not, so
    every mask after the skipped file was written under its neighbour's name.

    `imread` is patched rather than a bad file written, because cellpose
    distinguishes the two failures and only one reaches this branch: a
    non-image extension returns None (this path), while a `.tif` that is not
    a TIFF RAISES and takes the whole run down. Patching tests the branch the
    production code actually has instead of the one it does not.
    """
    from spacr import io as spacr_io

    order = ["a.tif", "b.tif", "c.tif", "d.tif"]
    files = [tiffs[name] for name in order]
    skipped = tiffs["b.tif"]

    real = None

    def fake_imread(path):
        if path == skipped:
            return None
        return real(path)

    import cellpose.io as cellpose_io
    real = cellpose_io.imread
    monkeypatch.setattr(cellpose_io, "imread", fake_imread)

    images, _labels, names, _ = spacr_io._load_images_and_labels(
        image_files=files, label_files=None)

    assert "b.tif" not in names, "a file that did not load kept its name"
    assert names == ["a.tif", "c.tif", "d.tif"]
    assert len(images) == len(names)
    for image, name in zip(images, names):
        assert np.array_equal(np.squeeze(image), _load_expected(tiffs[name])), name


def test_paired_images_and_labels_stay_aligned_with_both_name_lists(tiffs,
                                                                    tmp_path):
    """The paired branch has to keep four lists in step, not two."""
    tifffile = pytest.importorskip("tifffile")
    labels = {}
    for index, stem in enumerate(["e", "c", "a"]):
        path = tmp_path / f"{stem}_mask.tif"
        tifffile.imwrite(str(path), np.full((8, 8), index + 1, dtype=np.uint16))
        labels[f"{stem}_mask.tif"] = str(path)

    image_order = ["e.tif", "c.tif", "a.tif"]
    label_order = ["e_mask.tif", "c_mask.tif", "a_mask.tif"]

    images, masks, names, mask_names = _load_images_and_labels(
        image_files=[tiffs[n] for n in image_order],
        label_files=[labels[n] for n in label_order])

    assert names == image_order
    assert mask_names == label_order
    assert len(images) == len(masks) == len(names) == len(mask_names)


def test_the_label_only_branch_names_its_labels(tmp_path):
    """It used to return sorted names for labels too, and the same shuffle
    applies -- there is no branch where a precomputed list was safe."""
    tifffile = pytest.importorskip("tifffile")
    order = ["z.tif", "m.tif", "b.tif"]
    files = []
    for index, name in enumerate(order):
        path = tmp_path / name
        tifffile.imwrite(str(path), np.full((4, 4), index + 1, dtype=np.uint16))
        files.append(str(path))

    images, masks, names, mask_names = _load_images_and_labels(
        image_files=None, label_files=files)

    assert images == [] and names == []
    assert mask_names == order
    assert len(masks) == len(mask_names)


def test_no_files_at_all_returns_four_empty_lists(tmp_path):
    assert _load_images_and_labels(image_files=None, label_files=None) == \
        ([], [], [], [])
