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
    # b.npy was never merged, so its raw images in orig/ are the only copy.
    assert os.path.isfile(os.path.join(root, "orig", "raw.tif"))


def test_orig_is_kept_and_the_unmerged_fields_named(tmp_path, capsys):
    from spacr.utils import cleanup_pipeline_folders
    root = str(tmp_path)
    _mk(root, ["a.npy"], ["a.npy", "b.npy", "c.npy"])
    deleted = cleanup_pipeline_folders(root, verbose=False)
    assert os.path.join(root, "orig") not in deleted
    assert os.path.isfile(os.path.join(root, "orig", "raw.tif"))
    out = capsys.readouterr().out
    assert "2 field(s) have no complete merged/ output: b.npy, c.npy" in out


def test_orig_is_kept_when_a_merged_array_is_truncated(tmp_path, capsys):
    from spacr.utils import cleanup_pipeline_folders
    root = str(tmp_path)
    _mk(root, ["a.npy", "b.npy"], ["a.npy", "b.npy"])
    path = os.path.join(root, "merged", "b.npy")
    np.save(path, np.zeros((64, 64), np.uint16))
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 100)
    cleanup_pipeline_folders(root, verbose=False)
    assert os.path.isfile(os.path.join(root, "orig", "raw.tif"))
    assert "1 field(s) have no complete merged/ output: b.npy" in (
        capsys.readouterr().out)


def test_orig_is_kept_when_no_stack_list_can_verify_merged(tmp_path, capsys):
    from spacr.utils import cleanup_pipeline_folders
    root = str(tmp_path)
    _mk(root, ["a.npy"], [], with_masks=False)
    cleanup_pipeline_folders(root, verbose=False)
    assert os.path.isfile(os.path.join(root, "orig", "raw.tif"))
    assert "cannot be verified" in capsys.readouterr().out


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
