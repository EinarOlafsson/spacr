"""A crop is cut from a plane and a colour slot; both have to be right.

Nothing here fails loudly in production. A wrong mask plane cuts a nucleus
and files it as a cell; a wrong colour slot puts the DNA stain in the green
channel of every crop in the run; a folder marked with the wrong format
reads back with its channels reversed forever. These drive the refusals that
stand between a malformed manifest, a malformed settings value and those
outcomes.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from spacr.crops import (CROP_FORMAT_LEGACY_BGR, CROP_FORMAT_RGB,
                         CropError, CorruptMergedFile, MERGED_LAYOUT_SIDECAR,
                         PlaneLayoutConflict, build_png_channels,
                         crop_spec_from_settings, migrate_crop_folder,
                         path_components, picture_source_label,
                         png_dims_to_channel_mapping,
                         read_merged_plane_layout, reconcile_merged_mask_dims,
                         resolve_png_channel_mapping,
                         write_crop_folder_marker)


def _manifest(folder, **payload):
    """Write a plane-layout sidecar into ``folder`` and return the folder."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MERGED_LAYOUT_SIDECAR).write_text(json.dumps(payload))
    return str(folder)


# --------------------------------------------------------------------------- #
#  The plane manifest
# --------------------------------------------------------------------------- #

def test_a_manifest_that_is_not_json_stops_the_run(tmp_path):
    """A sidecar that cannot be parsed raises instead of being ignored.

    Ignoring it falls back to the caller's own mask indices, which is exactly
    the wrong-plane failure the manifest was added to prevent -- and the run
    would finish, with measurements taken off the wrong object.
    """
    folder = tmp_path / "merged"
    folder.mkdir()
    (folder / MERGED_LAYOUT_SIDECAR).write_text("{not json at all")

    with pytest.raises(CorruptMergedFile, match="cannot read merged plane"):
        read_merged_plane_layout(str(folder))


def test_a_manifest_from_a_future_version_is_not_guessed_at(tmp_path):
    """An unknown layout version raises rather than being read as version 1.

    The version is the only promise about what the other keys mean. Reading a
    version 2 document with version 1 rules is how a plane index silently
    becomes the wrong plane.
    """
    folder = _manifest(tmp_path / "a", version=2, mask_plane_order=["cell"],
                       mask_dims={"cell": 2}, intensity_channels=[0, 1])
    with pytest.raises(CorruptMergedFile, match="version 1"):
        read_merged_plane_layout(folder)

    folder = _manifest(tmp_path / "b", version=1)
    with pytest.raises(CorruptMergedFile, match="expected"):
        read_merged_plane_layout(folder)


def test_a_manifest_naming_an_unknown_object_is_refused(tmp_path):
    """A plane order with a repeat or an unknown role raises.

    The order is what maps a role onto an index. A duplicate makes two roles
    claim one plane; a role spaCR does not segment means the file was written
    by something else entirely.
    """
    folder = _manifest(tmp_path / "dup", version=1,
                       mask_plane_order=["cell", "cell"],
                       mask_dims={"cell": 2}, intensity_channels=[0, 1])
    with pytest.raises(CorruptMergedFile, match="mask_plane_order"):
        read_merged_plane_layout(folder)

    folder = _manifest(tmp_path / "unknown", version=1,
                       mask_plane_order=["spaceship"],
                       mask_dims={"spaceship": 2}, intensity_channels=[0, 1])
    with pytest.raises(CorruptMergedFile, match="mask_plane_order"):
        read_merged_plane_layout(folder)


def test_a_plane_index_that_is_not_a_number_is_refused(tmp_path):
    """``True`` and ``"two"`` are not plane indices, and neither is accepted.

    ``int(True)`` is 1, a perfectly plausible plane, so a boolean has to be
    caught before the coercion rather than after it.
    """
    folder = _manifest(tmp_path / "bool", version=1,
                       mask_plane_order=["cell"],
                       mask_dims={"cell": True}, intensity_channels=[0, 1])
    with pytest.raises(CorruptMergedFile, match="invalid mask dim"):
        read_merged_plane_layout(folder)

    folder = _manifest(tmp_path / "text", version=1,
                       mask_plane_order=["cell"],
                       mask_dims={"cell": "two"}, intensity_channels=[0, 1])
    with pytest.raises(CorruptMergedFile, match="invalid mask dim"):
        read_merged_plane_layout(folder)


def test_plane_indices_must_follow_from_the_channel_count(tmp_path):
    """Mask dims that do not sit after the intensity channels are refused.

    The masks are stacked after the channels, so the arithmetic is fixed:
    a manifest whose recorded index disagrees with its own channel count
    describes an array that cannot exist, and trusting either half would
    read a channel as a mask.
    """
    folder = _manifest(tmp_path / "off", version=1,
                       mask_plane_order=["cell"],
                       mask_dims={"cell": 7}, intensity_channels=[0, 1])
    with pytest.raises(CorruptMergedFile, match="inconsistent mask dims"):
        read_merged_plane_layout(folder)


def test_a_mask_index_that_is_not_an_index_conflicts(tmp_path):
    """An explicit ``cell_mask_dim`` that is not a number stops the run.

    The value came from a settings file a person edited. Coercing it to a
    default would measure a plane nobody asked for; the manifest exists so
    that disagreement is fatal rather than silent.
    """
    folder = _manifest(tmp_path / "merged", version=1,
                       mask_plane_order=["cell"],
                       mask_dims={"cell": 2}, intensity_channels=[0, 1])

    with pytest.raises(PlaneLayoutConflict, match="not a plane index"):
        reconcile_merged_mask_dims({"cell_mask_dim": "second"}, folder,
                                   explicit_keys=["cell_mask_dim"])

    # The same key, agreeing, is accepted and returned as the manifest's.
    out = reconcile_merged_mask_dims({"cell_mask_dim": 2}, folder,
                                     explicit_keys=["cell_mask_dim"])
    assert out["cell_mask_dim"] == 2


# --------------------------------------------------------------------------- #
#  Which source channel lands in which colour
# --------------------------------------------------------------------------- #

def test_a_crop_needs_at_least_one_channel_and_at_most_three():
    """An empty or over-long ``png_dims`` says so instead of cutting.

    A PNG holds three colour planes. Silently truncating a four-entry list
    would drop one stain from every crop in the run, and an empty list would
    produce a black image the classifier would happily train on.
    """
    with pytest.raises(CropError, match="png_dims is empty"):
        png_dims_to_channel_mapping([])

    with pytest.raises(CropError, match="at most 3"):
        png_dims_to_channel_mapping([0, 1, 2, 3])


def test_a_channel_mapping_has_to_be_a_mapping():
    """``png_channel_mapping`` given as a list is refused with its type.

    The legacy key is a list and the new one is a dict; a settings file that
    carries the old shape under the new name has to be told so, not indexed
    into and mis-read.
    """
    with pytest.raises(CropError, match="must be a dict"):
        resolve_png_channel_mapping({"png_channel_mapping": [0, 1, 2]})


def test_a_channel_index_off_the_end_of_the_array_is_refused():
    """A colour pointing past the last source plane raises with the count.

    Numpy would wrap a negative index and raise an opaque IndexError for a
    positive one, deep inside a worker; naming the array's channel count is
    what makes the settings error findable.
    """
    data = np.zeros((4, 4, 2), np.uint16)

    with pytest.raises(CropError, match="out of range"):
        build_png_channels(data, {"r": 5, "g": 1, "b": 0})

    with pytest.raises(CropError, match="\\(H, W, C\\)"):
        build_png_channels(np.zeros((4, 4), np.uint16), {"r": 0})


def test_one_channel_three_times_is_a_single_grey_plane():
    """A mapping that names one source for all three colours stays 1-plane.

    Writing the same plane three times triples the file size of every crop in
    a dataset that routinely has hundreds of thousands of them, and a
    greyscale crop is what a single-stain run should produce.
    """
    data = np.arange(16, dtype=np.uint16).reshape(4, 4, 1)

    out = build_png_channels(data, {"r": 0, "g": 0, "b": 0})

    assert out.shape == (4, 4, 1)
    assert np.array_equal(out[:, :, 0], data[:, :, 0])


def test_an_unused_colour_slot_is_one_shared_blank_plane():
    """Colours with no source become zeros, and the blank is made once.

    A two-stain crop leaves one slot empty. It has to be black rather than a
    repeat of another stain, or the reader sees a colour that was never
    measured.
    """
    data = np.stack([np.full((4, 4), 7, np.uint16),
                     np.full((4, 4), 9, np.uint16)], axis=2)

    out = build_png_channels(data, {"r": None, "g": 1, "b": 0})

    assert out.shape == (4, 4, 3)
    assert not out[:, :, 0].any()
    assert (out[:, :, 1] == 9).all()
    assert (out[:, :, 2] == 7).all()
    assert out.dtype == data.dtype


# --------------------------------------------------------------------------- #
#  Settings shapes and recorded paths
# --------------------------------------------------------------------------- #

def test_a_single_crop_size_means_a_square():
    """``png_size`` given as one number becomes width and height.

    The annotator's image-size control is a single spin box, and a settings
    CSV can carry the scalar too. Passing it through unchanged raises
    ``'int' object is not subscriptable`` inside the montage worker, which
    surfaces only as "the montage load failed".
    """
    spec = crop_spec_from_settings({"crop_mode": ["cell"], "png_size": 96,
                                    "png_dims": [0, 1, 2]})

    assert tuple(spec.size) == (96, 96)


def test_a_doubled_separator_is_not_an_empty_folder_name():
    """Repeated and trailing separators collapse, and the root survives.

    Recorded paths are compared component by component to decide whether a
    crop lies under a dataset root. An empty component from ``a//b`` would
    make two spellings of one location compare unequal, and the crop would
    be treated as belonging to no dataset at all.
    """
    assert path_components("a//b/") == ("a", "b")
    assert path_components("/a//b//") == ("", "a", "b")
    assert path_components("a/./b/../c") == ("a", "c")


def test_an_unrecognised_crop_source_shows_what_was_stored():
    """An unknown stored value is shown as itself; a blank one as the default.

    The label goes on a panel next to the run. Showing a stored value spaCR
    no longer offers is how a user finds out their settings file names a mode
    that is gone, instead of seeing the default and assuming it was used.
    """
    assert picture_source_label("png") == "load images"
    assert picture_source_label("some_retired_mode") == "some_retired_mode"
    assert picture_source_label("") == "load images"
    assert picture_source_label(None) == "load images"


# --------------------------------------------------------------------------- #
#  The folder format marker
# --------------------------------------------------------------------------- #

def test_marking_a_corrected_folder_as_legacy_is_refused(tmp_path):
    """A folder already in corrected order cannot be marked legacy.

    The marker is what tells the loader whether to reverse the channels. Put
    "legacy" on a folder whose pixels are already correct and every crop in
    it reads back with red and blue swapped, permanently and silently.
    """
    folder = tmp_path / "cell_png"
    folder.mkdir()
    write_crop_folder_marker(str(folder), CROP_FORMAT_RGB)

    with pytest.raises(CropError, match="marking it"):
        migrate_crop_folder(str(folder), mode="mark")

    # The same call on a folder that IS legacy is a no-op, not an error.
    other = tmp_path / "nucleus_png"
    other.mkdir()
    write_crop_folder_marker(str(other), CROP_FORMAT_LEGACY_BGR)
    assert migrate_crop_folder(str(other), mode="mark").already is True


def test_the_module_is_the_migration_command(tmp_path, capsys):
    """``python -m spacr.crops <path>`` runs the migration and reports.

    The migration exists so an old dataset can be corrected with a command
    rather than a Python snippet a user has to be told how to write. A folder
    with nothing to migrate must exit non-zero and say why, so a script that
    points at the wrong directory does not report success.
    """
    import runpy
    import sys

    argv = sys.argv
    sys.argv = ["python -m spacr.crops", str(tmp_path), "--dry-run"]
    try:
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("spacr.crops", run_name="__main__")
    finally:
        sys.argv = argv

    assert excinfo.value.code != 0
    assert "no '*_png' crop folders found" in capsys.readouterr().err
