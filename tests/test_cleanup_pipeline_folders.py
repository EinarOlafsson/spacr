"""Tests for utils.cleanup_pipeline_folders — the default 'keep only merged'
post-processing cleanup, and its guards against destroying un-merged data."""
from __future__ import annotations

import os
import numpy as np


def _mk(root, merged_files, stack_files, with_masks=True, with_orig=True):
    for sub, files in (("merged", merged_files), ("stack", stack_files)):
        d = os.path.join(root, sub)
        os.makedirs(d, exist_ok=True)
        for f in files:
            np.save(os.path.join(d, f), np.zeros((2, 2), np.uint16))
    if with_masks:
        md = os.path.join(root, "masks", "cell_mask_stack")
        os.makedirs(md, exist_ok=True)
        for f in stack_files:
            np.save(os.path.join(md, f), np.zeros((2, 2), np.uint16))
    if with_orig:
        od = os.path.join(root, "orig")
        os.makedirs(od, exist_ok=True)
        open(os.path.join(od, "raw.tif"), "w").close()


def test_default_keeps_only_merged(tmp_path):
    from spacr.utils import cleanup_pipeline_folders
    root = str(tmp_path)
    _mk(root, ["a.npy", "b.npy"], ["a.npy", "b.npy"])
    cleanup_pipeline_folders(root, verbose=False)
    assert os.path.isdir(os.path.join(root, "merged"))
    assert not os.path.exists(os.path.join(root, "stack"))
    assert not os.path.exists(os.path.join(root, "masks"))
    assert not os.path.exists(os.path.join(root, "orig"))


def test_keep_flags_preserve_folders(tmp_path):
    from spacr.utils import cleanup_pipeline_folders
    root = str(tmp_path)
    _mk(root, ["a.npy"], ["a.npy"])
    cleanup_pipeline_folders(root, keep_intermediate=True, keep_original=True,
                             verbose=False)
    assert os.path.isdir(os.path.join(root, "stack"))
    assert os.path.isdir(os.path.join(root, "masks"))
    assert os.path.isdir(os.path.join(root, "orig"))


def test_guard_keeps_intermediates_when_merge_incomplete(tmp_path):
    # stack has b.npy that never made it into merged -> must NOT delete stack.
    from spacr.utils import cleanup_pipeline_folders
    root = str(tmp_path)
    _mk(root, ["a.npy"], ["a.npy", "b.npy"])
    cleanup_pipeline_folders(root, verbose=False)
    assert os.path.isdir(os.path.join(root, "stack"))
    assert os.path.isdir(os.path.join(root, "masks"))
    # orig is independent of the stack guard and still removed by default
    assert not os.path.exists(os.path.join(root, "orig"))


def test_guard_empty_merged_deletes_nothing(tmp_path):
    from spacr.utils import cleanup_pipeline_folders
    root = str(tmp_path)
    _mk(root, [], ["a.npy"])   # merged/ exists but empty
    cleanup_pipeline_folders(root, verbose=False)
    assert os.path.isdir(os.path.join(root, "stack"))
    assert os.path.isdir(os.path.join(root, "orig"))


def test_no_merged_folder_is_noop(tmp_path):
    from spacr.utils import cleanup_pipeline_folders
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "stack"))
    deleted = cleanup_pipeline_folders(root, verbose=False)
    assert deleted == []
    assert os.path.isdir(os.path.join(root, "stack"))
