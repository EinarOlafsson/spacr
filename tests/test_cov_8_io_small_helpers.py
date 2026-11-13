"""Small, load-bearing helpers in :mod:`spacr.io` and what they do with bad input.

Every function here sits between a user's folder and a stage that will run for
hours, so each of them has a "cannot answer that" path that must not be an
exception thrown three stages later:

* ``select_fields`` re-runs one field of a plate rather than all of them, so a
  file name it cannot parse has to be skipped, not crashed on;
* ``crop_png_name`` has to reproduce, from a database row alone, the exact name
  the PNG folder would have held -- the parsers downstream key on it;
* ``crop_rows_from_object_table`` is the path for a project that never wrote a
  PNG folder, so it reads whatever schema is actually in the database;
* ``LazyCropPNG`` stands where a path string used to, which means it has to
  satisfy the parts of the file protocol ``PIL.Image.open`` uses;
* the cross-validation helpers refuse splits that would leak or that cannot be
  scored, and say which;
* ``_save_array_atomic`` exists so a killed run never leaves a truncated array
  at the final name.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from spacr import io


# ---------------------------------------------------------------------------
# select_fields
# ---------------------------------------------------------------------------

_STACKS = ["plate1_A01_f1.npy", "plate1_A01_f2.npy", "plate1_A02_f10.npy",
           "not_a_field_stem.npy"]


# ---------------------------------------------------------------------------
# _merge_key_details
# ---------------------------------------------------------------------------

def test_duplicate_index_entries_are_reported_by_index():
    """A merge on the index has to name the index, not a column."""
    frame = pd.DataFrame({"v": [1, 2, 3]}, index=["a", "a", "b"])

    key, examples = io._merge_key_details(frame, use_index=True)

    assert key == "index"
    assert examples == ["a"]


def test_a_merge_with_no_declared_key_says_so_rather_than_guessing():
    """"unspecified keys" is an honest answer; a guessed column is not."""
    frame = pd.DataFrame({"v": [1, 2]})

    assert io._merge_key_details(frame, columns=[]) == ("unspecified keys", [])
    assert io._merge_key_details(frame) == ("unspecified keys", [])


# ---------------------------------------------------------------------------
# crop_png_name / crop_rows_from_object_table
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LazyCropPNG
# ---------------------------------------------------------------------------

class _Source:
    """A crop source that cuts a constant 4x4 RGB square."""

    kind = "merged"

    def get(self, _row):
        return np.full((4, 4, 3), 200, dtype=np.uint8)


class _PngSource(_Source):
    """A PNG-folder source whose files are not where it says they are."""

    kind = "png"

    def resolve(self, row):
        return row["png_path"]


def test_a_lazy_crop_satisfies_the_file_protocol_pillow_uses():
    """It stands where a path string used to, so it has to behave like one."""
    crop = io.LazyCropPNG(_Source(), {"object_label": 1}, name="c.png")

    with crop as handle:
        assert handle is crop
        assert crop.readable() is True
        assert crop.seekable() is True
        assert crop.writable() is False
        assert crop.closed is False
        assert crop.png_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    assert crop.closed is False, "closing releases bytes, it does not end the object"
    assert crop.png_bytes()[:4] == b"\x89PNG", "the bytes are produced again"
    assert "c.png" in repr(crop)


def test_a_missing_png_file_yields_no_raw_bytes_rather_than_raising():
    """The on-disk shortcut is an optimisation; its absence is not a failure."""
    crop = io.LazyCropPNG(_PngSource(), {"png_path": "/no/such/crop.png"},
                          name="c.png")

    assert crop._raw_bytes() is None
    assert crop.png_bytes()[:4] == b"\x89PNG", "it falls back to cutting one"


# ---------------------------------------------------------------------------
# cross-validation splits
# ---------------------------------------------------------------------------


def test_a_class_with_no_samples_does_not_break_the_fold_deal():
    """Label 1 is absent; the folds still partition every sample exactly once."""
    labels = [0, 0, 0, 2, 2, 2]

    folds = io.make_cv_folds(labels, 3)

    seen = np.concatenate([val for _train, val in folds])
    assert sorted(seen.tolist()) == list(range(len(labels)))


def test_fold_columns_are_named_for_the_classes_when_none_are_supplied():
    """Without names the table still has to say which class each column is."""
    labels = [0, 1, 0, 1]
    folds = [(np.array([0, 1]), np.array([2, 3])),
             (np.array([2, 3]), np.array([0, 1]))]

    table = io.summarize_cv_folds(labels, folds)

    assert "val_class_0" in table.columns
    assert "val_class_1" in table.columns


def test_a_fold_with_an_empty_validation_set_earns_a_warning():
    """A fold that scores nothing would otherwise show as a perfect fold."""
    labels = [0, 1, 0, 1]
    folds = [(np.array([0, 1, 2, 3]), np.array([], dtype=int)),
             (np.array([0, 1]), np.array([2, 3]))]

    _table, warnings_out = io.report_cv_folds(labels, folds, verbose=False)

    assert any("empty validation set" in w for w in warnings_out)
    assert any("[1]" in w for w in warnings_out)


def test_a_split_with_an_empty_class_is_told_it_cannot_be_scored():
    """A class with no samples cannot be learned; the report has to say so.

    Sampling is already being corrected here, so the balance advice is
    silent -- which is exactly when the empty class has to be the sentence
    the user is given.
    """
    summary = io.report_class_balance([0, 0, 0, 1, 1, 1],
                                      classes=["nc", "pc", "unused"],
                                      class_balance="weighted_sampler",
                                      split_name="train", verbose=False)

    assert summary["empty_classes"]
    assert "cannot be learned or scored" in summary["recommendation"]


# ---------------------------------------------------------------------------
# atomic array writes
# ---------------------------------------------------------------------------

def test_a_failed_write_leaves_no_array_at_the_final_name(tmp_path,
                                                          monkeypatch):
    """A truncated ``.npy`` at the final name is what this function prevents."""
    def full_disk(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(np, "save", full_disk)
    out = tmp_path / "field.npy"

    with pytest.raises(OSError, match="No space left on device"):
        io._save_array_atomic(str(out), np.zeros((2, 2)))

    assert not out.exists()


def test_a_cleanup_that_also_fails_does_not_replace_the_real_error(
        tmp_path, monkeypatch):
    """The user needs the disk error, not a failure to tidy up after it."""
    def full_disk(*_args, **_kwargs):
        raise OSError("No space left on device")

    def cannot_remove(_path):
        raise OSError("Permission denied")

    monkeypatch.setattr(np, "save", full_disk)
    monkeypatch.setattr(os, "remove", cannot_remove)
    out = tmp_path / "field.npy"

    with pytest.raises(OSError, match="No space left on device"):
        io._save_array_atomic(str(out), np.zeros((2, 2)))

    assert not out.exists()


# ---------------------------------------------------------------------------
# the optional CZI backend
# ---------------------------------------------------------------------------

def test_the_czi_backend_is_returned_when_the_wheel_is_installed(monkeypatch):
    """The import guard must not stand between a working wheel and the reader."""
    package = types.ModuleType("pylibCZIrw")
    reader = types.ModuleType("pylibCZIrw.czi")
    package.czi = reader
    monkeypatch.setitem(sys.modules, "pylibCZIrw", package)
    monkeypatch.setitem(sys.modules, "pylibCZIrw.czi", reader)

    assert io._load_pylibczi() is reader


def test_a_missing_czi_wheel_names_the_extra_that_installs_it(monkeypatch):
    """"No module named pylibCZIrw" is not a next step; the extra is."""
    monkeypatch.setitem(sys.modules, "pylibCZIrw", None)

    with pytest.raises(ImportError, match=r"spacr\[czi\]"):
        io._load_pylibczi()


# ---------------------------------------------------------------------------
# tar archives and their crop-format sidecar
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# inheriting a crop format from a folder that cannot be read
# ---------------------------------------------------------------------------

def test_an_unreadable_source_folder_is_assumed_to_hold_legacy_crops(
        tmp_path, monkeypatch):
    """Guessing "current" would reverse every channel name on a legacy copy."""
    import spacr.crops as crops

    def unreadable(*_args, **_kwargs):
        raise OSError("the source folder is gone")

    monkeypatch.setattr(crops, "crop_folder_format", unreadable)
    destination = tmp_path / "train" / "nc"
    destination.mkdir(parents=True)

    sidecar = io.mark_crop_output_folder(
        str(destination), source_folder=str(tmp_path / "source"))

    import json
    assert sidecar is not None
    written = json.loads(open(sidecar, encoding="utf-8").read())
    assert written["spacr_crop_format"] == crops.CROP_FORMAT_LEGACY_BGR


def test_items_from_an_unreadable_folder_refuse_to_guess_channels(tmp_path,
                                                               monkeypatch):
    """Do not silently label unreadable source metadata as legacy BGR."""
    import spacr.crops as crops

    def unreadable(*_args, **_kwargs):
        raise OSError("the crop folder is gone")

    monkeypatch.setattr(crops, "crop_folder_format", unreadable)

    with pytest.raises(OSError, match='the crop folder is gone'):
        io._crop_format_of_items([str(tmp_path / "cell_png" / "a.png")])


# ---------------------------------------------------------------------------
# writing a crop tar from mixed sources
# ---------------------------------------------------------------------------


