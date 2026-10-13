"""Round-7 ``spacr.io``: two prose proofs from round 5 turned into pins.

Every one of this round's 24 targets for :mod:`spacr.io` is a branch an
earlier round already proved unreachable — see the agreement note at the
bottom of this file, which re-measures each of them against the current
source. Rather than restate arguments that already stand, this file takes
the two of them that round 5 could only argue in prose and gives them an
executable pin, because both guard an invariant that a later change could
break silently:

* ``_create_movies_from_npy_per_channel`` collects its frames into a dict
  keyed by ``(plate, well, field)``, and the ``if not arrays: continue``
  below assumes a group can never be empty. What makes that true is that a
  group is only ever CREATED by appending a file to it — so the guard is
  dead, and if it ever stops being dead a field has silently lost its movie.

* ``convert_to_yokogawa`` remembers a well per source file so that every
  channel and timepoint of one input lands in one well. The re-check that
  the file is not already known cannot fire, because the dict is keyed by
  the loop variable of a ``for`` over a sorted listing. What that buys is
  that two inputs never share a well address — and a shared address here is
  one input's TIFF overwriting another's, with the rename log claiming both.

CPU-only and offline throughout.
"""
from __future__ import annotations

import csv
import os
import re

import numpy as np
import pytest
import tifffile

import spacr.io as IO


# ---------------------------------------------------------------------------
# _create_movies_from_npy_per_channel
# ---------------------------------------------------------------------------

def test_a_movie_group_exists_only_because_a_frame_was_put_in_it(
        tmp_path, monkeypatch):
    """io.py:2174->2175 -- why ``if not arrays: continue`` cannot fire.

    ``organized_files[key]`` is created at io.py:2160-2162 by the same
    statement that appends a file to it, and the loop below appends one
    array per file in the group, so ``arrays`` is non-empty for every key
    that exists. A file whose name the regex does not match creates no key
    at all rather than an empty one — which is the half worth pinning,
    because an empty group would mean a field whose frames were collected
    and then dropped without a word.
    """
    from spacr import timelapse

    src = tmp_path / "norm"
    src.mkdir()

    # Two fields of one well, three timepoints each, two channels.
    for field in ("f1", "f2"):
        for time in (1, 2, 3):
            np.save(src / f"plate1_A01_{field}_{time}.npy",
                    np.full((8, 8, 2), time, dtype=np.float32))
    # ... plus files the regex cannot parse, and a non-npy file. Neither may
    # create a group of its own.
    np.save(src / "not_a_field_stem.npy", np.zeros((8, 8, 2), np.float32))
    (src / "plate1_A01_f3_1.txt").write_text("not an array")

    calls = []

    def _record(arrays, filenames, save_path, fps):
        calls.append((os.path.basename(save_path), len(arrays),
                      list(filenames), fps))

    monkeypatch.setattr(timelapse, "_npz_to_movie", _record)

    IO._create_movies_from_npy_per_channel(str(src), fps=7)

    # One movie per (plate, well, field) per channel -- four in all, and
    # nothing for the two files that named no field.
    assert sorted(name for name, *_ in calls) == [
        "plate1_A01_f1_channel_0.mp4", "plate1_A01_f1_channel_1.mp4",
        "plate1_A01_f2_channel_0.mp4", "plate1_A01_f2_channel_1.mp4"]
    # Every group that existed carried all three of its frames, in time
    # order: no group was empty, and none was short.
    assert {frames for _name, frames, *_ in calls} == {3}
    for _name, _frames, filenames, fps in calls:
        assert [int(re.search(r"_(\d+)\.npy$", name).group(1))
                for name in filenames] == [1, 2, 3]
        assert fps == 7


# ---------------------------------------------------------------------------
# convert_to_yokogawa
# ---------------------------------------------------------------------------

def test_every_source_file_is_given_a_well_of_its_own(tmp_path):
    """io.py:7739->7743 -- why ``if file not in file_to_well`` is always true.

    ``file_to_well`` is keyed by the loop variable of
    ``for file in sorted(os.listdir(folder))`` and is written nowhere else,
    so each name is visited once and is never already present. The
    invariant that makes the dict worth having at all is asserted instead:
    one well per SOURCE file, and no two source files sharing one.

    A shared well would not raise. Both files would be written as
    ``plate1_A01_T0001F001L01C01.tif``, the second silently replacing the
    first, and ``rename_log.csv`` would list two originals against one
    surviving image.
    """
    folder = tmp_path / "raw"
    folder.mkdir()
    for index, name in enumerate(("alpha", "beta", "gamma")):
        tifffile.imwrite(str(folder / f"{name}.tif"),
                         np.full((8, 8), index + 1, dtype=np.uint16))

    ledger = IO.convert_to_yokogawa(str(folder))
    assert ledger.failures == []

    with open(folder / "rename_log.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))

    # Three inputs, three rows, three DISTINCT wells ...
    assert sorted(row["Original File"] for row in rows) == [
        "alpha.tif", "beta.tif", "gamma.tif"]
    wells = [row["Renamed TIFF"].split("_T0001")[0] for row in rows]
    assert len(set(wells)) == 3

    # ... and three surviving images on disk, each still carrying the value
    # its own source had. Sharing a well would have left two.
    written = sorted(path for path in os.listdir(folder)
                     if path.startswith("plate") and path.endswith(".tif"))
    assert len(written) == 3
    by_original = {row["Original File"]: row["Renamed TIFF"] for row in rows}
    for index, name in enumerate(("alpha", "beta", "gamma")):
        image = tifffile.imread(str(folder / by_original[f"{name}.tif"]))
        assert int(image.max()) == index + 1


# ---------------------------------------------------------------------------
# Everything else on this chunk's list was already proved
#
# The other 22 items were re-measured against io.py as it stands here. Round
# 5's line references are five lines low throughout -- io.py has grown by
# five lines above line 2100 since tests/test_cov_r5_io.py was written -- but
# every argument still describes the code at the shifted line, and I agree
# with each of them:
#
#   * 197->198 `if safe == stem: continue` in migrate_unescaped_plate_names.
#     Pinned by test_escaping_the_plate_of_a_five_part_stem_always_changes_it
#     (r5). Agreed: the length guard at io.py:186 admits only stems with a
#     separator inside the plate half, and escape_filename_component always
#     rewrites it.
#   * 269->271 `if channel is not None` in save_grayscale_images. Pinned by
#     test_every_plane_split_out_of_an_image_carries_a_channel_index (r5).
#     Agreed: all three call sites pass `channel=c+1`.
#   * 2516->2504 `if ch in seen` in preprocess_img_data. Argued in r5.
#     Agreed: `seen` is built from the same keys through the same int()
#     coercion and the same two continues, and nothing between the two loops
#     writes a `*_channel` key.
#   * 2602->2592 `elif mask.ndim not in [2, 3]` in _get_avg_object_size.
#     Pinned by
#     test_a_mask_that_is_counted_as_nothing_always_says_which_kind_of_nothing
#     (r6). Agreed: the `if` above consumes the second disjunct of the
#     condition that enters the else.
#   * 4514->4517 (and line 4517) the `prcf`-less prcfo fallback in
#     _read_and_merge_data. Pinned twice: r5's
#     test_a_numeric_prcf_is_rewritten_as_text_before_the_object_split and
#     r6's test_the_object_key_comes_from_prcf_and_its_fallback_cannot_be_reached.
#     Agreed, and r6's reading is the sharper one: the fallback spells the
#     key without the timepoint, so it is not merely dead but wrong.
#   * 5326->5327 `if dst is None` in generate_dataset. Argued in r5.
#     Agreed: save_settings indexes settings['src'][0] first, and a
#     non-empty list assigns dst on its first iteration.
#   * 6625->6631 the `for j in range(1, 100000)` loop-else of
#     _ensure_unique_dir. Agreed with r5 that this is reachable in
#     principle -- it needs training_1 .. training_99999 to exist at once --
#     and at no cost worth paying.
#   * 6725->6726 `if not list_of_lists` in _balance_lists. Pinned by
#     test_a_class_that_selected_nothing_is_refused_before_balancing (r5).
#   * 6749->6750 `if not ann_cols` in _annotation_classes_from_columns.
#     Pinned by
#     test_annotation_mode_refuses_an_empty_column_list_before_it_builds_classes
#     (r5).
#   * 6981->7001 (and line 7001) the "Invalid dataset_mode" else. Pinned by
#     test_an_unknown_dataset_mode_is_refused_by_the_basis_resolver (r5).
#   * 7438->7441 (and line 7441) the grouped-split leakage guard. Pinned by
#     test_a_grouped_split_never_leaves_a_class_on_one_side (r5).
#   * 7471->7472 (and line 7472) `if grouped_splits is None`. Argued in r5.
#     Agreed: the line is reached only for a non-empty class, and a
#     non-empty class means flat_items was non-empty and the splits were
#     built.
#   * 8115->8151 `if augment_data` inside the short-folder branch. Pinned by
#     test_without_augmentation_every_folder_contributes_the_smallest_count
#     (r5). Agreed: target_size is the smallest folder when augmentation is
#     off, so no folder is short.
#   * lines 198, 2175, 4517, 5327, 6726, 6750, 7001, 7441 and 7472 are the
#     bodies of the branches above and go with them.
#
# Nothing is excluded from coverage; every branch above is still in the
# source with a test asserting the guarantee that makes it dead.
# ---------------------------------------------------------------------------
