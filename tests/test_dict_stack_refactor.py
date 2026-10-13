"""Regression tests for the dict-based stack generation in
``_rename_and_organize_image_files`` (no per-channel sub-folders).

The refactor must:
  1. produce stacks byte-identical to the old folder + ``_merge_file`` merge,
     including correct multi-channel concatenation order,
  2. never create the intermediate per-channel folders,
  3. honour ``save_original_images`` (backup to orig/ vs delete raws),
  4. return the channel count.
"""
from __future__ import annotations

import os
import numpy as np
from PIL import Image

from spacr.io import _rename_and_organize_image_files, _merge_file
from spacr.utils import _get_regex


def _write_src(root, n_channels=3):
    """3-channel FOV grid using C10/C20/C30 tokens (which parse to distinct
    channels 1/2/3) so the merge order is actually exercised."""
    src = os.path.join(root, "cv")
    os.makedirs(src)
    tokens = [f"C{c}0" for c in range(1, n_channels + 1)]  # C10,C20,C30
    for wi, well in enumerate(("A01", "A02")):
        for fi, field in enumerate((1, 2)):
            for ci, tok in enumerate(tokens):
                fname = f"plate1_{well}_T01F0{field}L01A01Z01{tok}.tif"
                # Distinct per-channel constant so a wrong merge order is visible.
                val = (wi * 1000 + fi * 100 + (ci + 1))
                img = np.full((12, 10), val, dtype=np.uint16)
                Image.fromarray(img).save(os.path.join(src, fname))
    return src


def _reference_stack(src_imgs, channels_sorted):
    """Build the expected stack the OLD way: lay each channel MIP into a
    per-channel folder, then merge with the unchanged _merge_file."""
    import tempfile
    ref = tempfile.mkdtemp()
    chan_dirs = []
    for chan in channels_sorted:
        cdir = os.path.join(ref, chan)
        os.makedirs(cdir)
        Image.fromarray(src_imgs[chan]).save(os.path.join(cdir, "fov.tif"))
        chan_dirs.append(cdir)
    stackdir = os.path.join(ref, "stack")
    _merge_file(chan_dirs, stackdir, "fov.tif")
    return np.load(os.path.join(stackdir, "fov.npy"))


def test_multichannel_stack_matches_merge_file(tmp_path):
    src = _write_src(str(tmp_path))
    regex = _get_regex("cellvoyager", ".tif", None)
    n = _rename_and_organize_image_files(
        src, regex, batch_size=100, metadata_type="cellvoyager",
        img_format=[".tif"], save_original_images=True)
    assert n == 3, f"expected 3 channels, got {n}"

    stackdir = os.path.join(src, "stack")
    stacks = sorted(f for f in os.listdir(stackdir) if f.endswith(".npy"))
    assert len(stacks) == 4  # 2 wells x 2 fields

    # Every stack must be 3-channel and match a from-scratch _merge_file merge
    # of the same per-channel constants, proving concatenation order is right.
    for name in stacks:
        arr = np.load(os.path.join(stackdir, name))
        assert arr.shape == (12, 10, 3), arr.shape
        # channel c (1-based sorted) should hold constant (c) offset by fov base
        base = int(arr[0, 0, 0]) - 1
        expected = {f"{c}": np.full((12, 10), base + c, dtype=np.uint16)
                    for c in range(1, 4)}
        ref = _reference_stack(expected, ["1", "2", "3"])
        assert np.array_equal(arr, ref), f"{name} differs from _merge_file merge"


def test_no_channel_folders_created(tmp_path):
    src = _write_src(str(tmp_path))
    regex = _get_regex("cellvoyager", ".tif", None)
    _rename_and_organize_image_files(
        src, regex, 100, "cellvoyager", [".tif"], save_original_images=True)
    leftover = [d for d in os.listdir(src)
                if os.path.isdir(os.path.join(src, d)) and d in {"1", "2", "3", "0"}]
    assert leftover == [], f"channel folders must not be created, found {leftover}"


def test_save_original_images_true_backs_up(tmp_path):
    src = _write_src(str(tmp_path))
    regex = _get_regex("cellvoyager", ".tif", None)
    _rename_and_organize_image_files(
        src, regex, 100, "cellvoyager", [".tif"], save_original_images=True)
    orig = os.path.join(src, "orig")
    assert os.path.isdir(orig)
    assert len([f for f in os.listdir(orig) if f.endswith(".tif")]) == 12
    # raws moved out of src root
    assert not [f for f in os.listdir(src) if f.endswith(".tif")]


def test_save_original_images_false_deletes(tmp_path):
    src = _write_src(str(tmp_path))
    regex = _get_regex("cellvoyager", ".tif", None)
    _rename_and_organize_image_files(
        src, regex, 100, "cellvoyager", [".tif"], save_original_images=False)
    assert not os.path.isdir(os.path.join(src, "orig"))
    assert not [f for f in os.listdir(src) if f.endswith(".tif")]
    # stack still produced
    assert len(os.listdir(os.path.join(src, "stack"))) == 4


def test_setting_default_present():
    from spacr.settings import set_default_settings_preprocess_generate_masks
    out = set_default_settings_preprocess_generate_masks({})
    assert out.get("save_original_images") is True
