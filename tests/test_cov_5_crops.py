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

import pytest

from spacr.crops import (CROP_FORMAT_LEGACY_BGR, CROP_FORMAT_RGB,
                         CropError, MERGED_LAYOUT_SIDECAR,
                         PlaneLayoutConflict,
                         crop_spec_from_settings, migrate_crop_folder,
                         path_components, picture_source_label,
                         png_dims_to_channel_mapping,
                         reconcile_merged_mask_dims,
                         resolve_png_channel_mapping,
                         write_crop_folder_marker)


def _manifest(folder, **payload):
    """Write a plane-layout sidecar into ``folder`` and return the folder."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MERGED_LAYOUT_SIDECAR).write_text(json.dumps(payload), encoding="utf-8")
    return str(folder)


# --------------------------------------------------------------------------- #
#  The plane manifest
# --------------------------------------------------------------------------- #

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
