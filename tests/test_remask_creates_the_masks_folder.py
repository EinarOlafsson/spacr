"""Re-masking a measured plate died on a folder it was about to fill.

Reported as issue #13. Re-masking means deleting `masks/` and running again,
and it means unticking `preprocess` -- the plate is already converted, so
there is nothing to preprocess. But `preprocess_img_data` is the ONLY thing
that creates `masks/`, so unticking it left the directory absent and the run
failed on a missing path.

The reported workaround was to `mkdir masks` by hand, which is the clearest
possible statement that the code should have done it.
"""

import ast
import inspect
import os

import pytest


def _masks_block():
    """The `if settings['masks']:` block of preprocess_generate_masks."""
    from spacr import core

    source = inspect.getsource(core.preprocess_generate_masks)
    marker = "mask_src = os.path.join(src, 'masks')"
    assert marker in source, "the mask source is built differently now"
    start = source.index(marker)
    return source[start:start + 1500]


def test_the_masks_folder_is_created_before_it_is_used():
    """The fix, pinned at the source, because reaching this block needs a
    converted plate and a Cellpose model."""
    block = _masks_block()
    assert "os.makedirs(mask_src, exist_ok=True)" in block, (
        "mask_src is used without being created; re-masking with preprocess "
        "off fails on a missing directory again")


def test_it_is_created_before_the_first_generator_runs():
    """Creating it after the first mask call would fix nothing."""
    block = _masks_block()
    made = block.index("os.makedirs(mask_src")
    used = block.index("generate_cellpose_masks_sam")
    assert made < used, (
        "the folder is created after the first generator is called")


def test_exist_ok_so_the_normal_path_is_untouched():
    """When preprocessing DID run, the folder is already there."""
    block = _masks_block()
    line = next(l for l in block.splitlines() if "os.makedirs(mask_src" in l)
    assert "exist_ok=True" in line, (
        "a second run would now fail on an existing directory")


def test_makedirs_with_exist_ok_is_idempotent(tmp_path):
    """The property the fix relies on, stated once."""
    target = tmp_path / "masks"
    os.makedirs(str(target), exist_ok=True)
    os.makedirs(str(target), exist_ok=True)
    assert target.is_dir()
