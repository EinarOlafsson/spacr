"""Branch coverage for the raw-image ingest helpers in :mod:`spacr.io`.

Covers the defensive / rarely-taken paths of

    load_images_from_paths
    _rename_and_organize_image_files
    _merge_file
    _move_to_chan_folder

i.e. unreadable input images, ``img_format`` given as a bare string, timelapse
file naming, an already-populated ``stack/``, FOVs missing a channel, a MIP that
cannot be computed, duplicate names in ``orig/``, un-deletable raws, a channel
merge with no readable channel, and the CQ1 / no-plateID / unparsable-filename
paths of the per-channel sorter.

Everything is synthetic uint16 TIFF data written into ``tmp_path`` — no network,
no GPU, no external data.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pytest
import tifffile

# Regexes with explicitly named groups (spacr's `custom` metadata mode) so the
# parsing here is deterministic and independent of _get_regex's quirks.
CV_REGEX = (r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d+)_T(?P<timeID>\d+)"
            r"F(?P<fieldID>\d+)L(?P<laserID>\d+)A(?P<AID>\d+)"
            r"Z(?P<sliceID>\d+)C(?P<chanID>\d+)\.tif")

CQ1_REGEX = (r"W(?P<wellID>\d+)F(?P<fieldID>\d+)T(?P<timeID>\d+)"
             r"Z(?P<sliceID>\d+)C(?P<chanID>\d+)\.tif")


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _img(value, shape=(6, 8)):
    """Constant-valued uint16 field so channel identity is checkable."""
    return np.full(shape, value, dtype=np.uint16)


def _write(path, value, shape=(6, 8)):
    tifffile.imwrite(str(path), _img(value, shape))
    return _img(value, shape)


def _cv_name(well="A01", time="0001", field="001", chan="01", ext=".tif"):
    return f"plate1_{well}_T{time}F{field}L01A01Z01C{chan}{ext}"


# ---------------------------------------------------------------------------
# load_images_from_paths
# ---------------------------------------------------------------------------

def test_load_images_from_paths_skips_unreadable_paths(tmp_path, capsys):
    """Unopenable / missing paths are reported and dropped, good ones kept."""
    from spacr.io import load_images_from_paths

    good = tmp_path / "good.tif"
    expected = _write(good, 4242, shape=(3, 5))
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_text("this is definitely not a TIFF")
    missing = tmp_path / "not_there.tif"

    out = load_images_from_paths({"fov": [str(good), str(corrupt), str(missing)],
                                  "empty": []})

    assert set(out) == {"fov", "empty"}
    assert out["empty"] == []
    assert len(out["fov"]) == 1
    arr = out["fov"][0]
    assert arr.shape == (3, 5)
    assert arr.dtype == np.uint16
    assert np.array_equal(arr, expected)

    err = capsys.readouterr().out
    assert err.count("Error loading image from") == 2
    assert "corrupt.tif" in err and "not_there.tif" in err


def test_rename_survives_one_corrupt_raw_image(tmp_path):
    """A single corrupt TIFF must not abort ingest of the other FOVs."""
    from spacr.io import _rename_and_organize_image_files

    src = tmp_path / "plate1"
    src.mkdir()
    _write(src / _cv_name(well="A01", chan="01"), 11)
    _write(src / _cv_name(well="A01", chan="02"), 12)
    (src / _cv_name(well="A02", chan="01")).write_text("truncated file")
    _write(src / _cv_name(well="A02", chan="02"), 22)

    _rename_and_organize_image_files(
        str(src), CV_REGEX, batch_size=10, metadata_type="custom",
        img_format=[".tif"], save_original_images=False)

    stacked = sorted(os.listdir(src / "stack"))
    assert "plate1_A01_1_1.npy" in stacked


# ---------------------------------------------------------------------------
# _rename_and_organize_image_files
# ---------------------------------------------------------------------------

def test_img_format_string_is_wrapped_in_list(tmp_path):
    """A bare ``img_format`` string must behave like a one-element list.

    Without the wrap the ``any(f.endswith(ext) for ext in img_format)`` filter
    iterates the *characters* of ``'.tif'``, so a ``.tiff`` decoy (ending in
    'f') would be ingested as a second FOV.
    """
    from spacr.io import _rename_and_organize_image_files

    src = tmp_path / "plate1"
    src.mkdir()
    _write(src / _cv_name(well="A01", chan="01"), 5)
    _write(src / _cv_name(well="A01", chan="02"), 6)
    decoy = src / _cv_name(well="A02", chan="01", ext=".tiff")
    _write(decoy, 7)

    n_channels = _rename_and_organize_image_files(
        str(src), CV_REGEX, batch_size=10, metadata_type="custom",
        img_format=".tif", save_original_images=True)

    assert n_channels == 2
    stacks = sorted(os.listdir(src / "stack"))
    assert stacks == ["plate1_A01_1_1.npy"], stacks
    arr = np.load(src / "stack" / "plate1_A01_1_1.npy")
    assert arr.shape == (6, 8, 2)
    assert arr[..., 0].max() == 5 and arr[..., 1].max() == 6
    # the .tiff decoy is not an ingest format, so it is left where it was
    assert decoy.exists()


def test_timelapse_naming_keeps_one_stack_per_timepoint(tmp_path):
    """``timelapse=True`` must write one stack PER TIMEPOINT, timeID included.

    This test used to assert the opposite — that all timepoints of a FOV share
    one ``<plate>_<well>_<field>.npy`` — and so pinned a bug that made the whole
    Timelapse module unusable, in two ways at once:

    * the ``np.maximum`` combine in ``_rename_and_organize_image_files`` folded
      every frame of a field into a single max projection, so the movie was
      destroyed before anything downstream saw it; and
    * ``_generate_time_lists`` (the grouper both ``_concatenate_channel`` and
      ``concatenate_and_normalize`` build their timelapse stacks from) skips any
      name with fewer than four underscore-separated parts, so it returned an
      empty list for the whole plate. No ``*_norm_timelapse.npz`` was written,
      no masks were generated, and ``preprocess_generate_masks`` died far away
      in ``_pivot_counts_table`` on ``no such table: object_counts``.

    ``plate_well_field_time`` is the spelling ``_generate_time_lists`` parses,
    and it is what the non-timelapse branch already produced, so there was
    nothing for the timelapse branch to spell differently.
    """
    from spacr.io import _generate_time_lists, _rename_and_organize_image_files

    src = tmp_path / "plate1"
    src.mkdir()
    for t, base in (("0001", 10), ("0002", 20)):
        _write(src / _cv_name(time=t, chan="01"), base + 1)
        _write(src / _cv_name(time=t, chan="02"), base + 2)

    n_channels = _rename_and_organize_image_files(
        str(src), CV_REGEX, batch_size=10, metadata_type="custom",
        img_format=[".tif"], timelapse=True, save_original_images=False)

    assert n_channels == 2
    stacks = sorted(os.listdir(src / "stack"))
    assert stacks == ["plate1_A01_1_1.npy", "plate1_A01_1_2.npy"], stacks

    # Each frame keeps its own pixels rather than being max-projected with the
    # other: T0001 carries 11/12, T0002 carries 21/22.
    first = np.load(src / "stack" / "plate1_A01_1_1.npy")
    second = np.load(src / "stack" / "plate1_A01_1_2.npy")
    assert first.shape == second.shape == (6, 8, 2)
    assert (first[..., 0].max(), first[..., 1].max()) == (11, 12)
    assert (second[..., 0].max(), second[..., 1].max()) == (21, 22)

    # And the grouper downstream can actually see them as one time series.
    groups = _generate_time_lists(stacks)
    assert groups == [["plate1_A01_1_1.npy", "plate1_A01_1_2.npy"]], groups


def test_existing_stack_file_is_not_overwritten(tmp_path, monkeypatch, capsys):
    """A stack that already exists on disk is warned about and left alone.

    ``stack/`` is reported empty on the entry check (as after a crashed run
    whose directory listing was still cached) but already holds the target
    ``.npy`` when the writer gets there.
    """
    from spacr.io import _rename_and_organize_image_files

    src = tmp_path / "plate1"
    src.mkdir()
    _write(src / _cv_name(chan="01"), 3)
    _write(src / _cv_name(chan="02"), 4)

    stack_dir = src / "stack"
    stack_dir.mkdir()
    sentinel = np.zeros((2, 2), dtype=np.uint8)
    np.save(stack_dir / "plate1_A01_1_1.npy", sentinel)

    real_listdir = os.listdir
    seen = {"n": 0}

    def fake_listdir(path):
        if os.path.abspath(str(path)) == os.path.abspath(str(stack_dir)):
            seen["n"] += 1
            if seen["n"] == 1:
                return []          # pretend the stack dir is still empty
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", fake_listdir)

    n_channels = _rename_and_organize_image_files(
        str(src), CV_REGEX, batch_size=10, metadata_type="custom",
        img_format=[".tif"], save_original_images=True)

    assert n_channels == 2
    kept = np.load(stack_dir / "plate1_A01_1_1.npy")
    assert kept.shape == (2, 2)
    assert np.array_equal(kept, sentinel)     # untouched
    out = capsys.readouterr().out
    assert "A file with the same name already exists" in out
    # processing carried on: the raws were still archived
    assert sorted(os.listdir(src / "orig")) == sorted(
        [_cv_name(chan="01"), _cv_name(chan="02")])


def test_fov_missing_a_channel_is_stacked_with_what_is_there(tmp_path, capsys):
    """A FOV lacking one of the plate-wide channels warns and keeps the rest."""
    from spacr.io import _rename_and_organize_image_files

    src = tmp_path / "plate1"
    src.mkdir()
    _write(src / _cv_name(well="A01", chan="01"), 101)
    _write(src / _cv_name(well="A01", chan="02"), 102)
    _write(src / _cv_name(well="A02", chan="01"), 201)   # no C02 for A02

    n_channels = _rename_and_organize_image_files(
        str(src), CV_REGEX, batch_size=10, metadata_type="custom",
        img_format=[".tif"], save_original_images=False)

    assert n_channels == 2
    full = np.load(src / "stack" / "plate1_A01_1_1.npy")
    partial = np.load(src / "stack" / "plate1_A02_1_1.npy")
    assert full.shape == (6, 8, 2)
    assert partial.shape == (6, 8, 1)
    assert partial[..., 0].max() == 201
    out = capsys.readouterr().out
    assert "is missing channel 2" in out
    assert "plate1_A02_1_1.tif" in out


def test_fov_with_no_usable_mip_writes_nothing(tmp_path, monkeypatch, capsys):
    """If every channel MIP comes back empty, no stack is written."""
    from spacr.io import _rename_and_organize_image_files

    src = tmp_path / "plate1"
    src.mkdir()
    _write(src / _cv_name(chan="01"), 9)

    real_stack = np.stack

    class _NoMip:
        """Stands in for a stacked z-series whose reduction yields nothing."""

        def max(self, axis=None, out=None, **kwargs):
            return None

    def fake_stack(arrays, *args, **kwargs):
        if isinstance(arrays, list) and arrays and isinstance(arrays[0], np.ndarray):
            return _NoMip()
        return real_stack(arrays, *args, **kwargs)

    monkeypatch.setattr(np, "stack", fake_stack)

    n_channels = _rename_and_organize_image_files(
        str(src), CV_REGEX, batch_size=10, metadata_type="custom",
        img_format=[".tif"], save_original_images=False)

    assert n_channels == 1
    assert [f for f in os.listdir(src / "stack") if f.endswith(".npy")] == []
    out = capsys.readouterr().out
    assert "is missing channel 1" in out
    assert "No valid channels to merge for file plate1_A01_1_1.tif" in out


def test_orig_backup_collision_leaves_raw_in_place(tmp_path, capsys):
    """A name clash in ``orig/`` warns and neither file is clobbered."""
    from spacr.io import _rename_and_organize_image_files

    src = tmp_path / "plate1"
    src.mkdir()
    _write(src / _cv_name(chan="01"), 31)
    _write(src / _cv_name(chan="02"), 32)

    orig = src / "orig"
    orig.mkdir()
    previous = _write(orig / _cv_name(chan="01"), 999)   # older backup

    _rename_and_organize_image_files(
        str(src), CV_REGEX, batch_size=10, metadata_type="custom",
        img_format=[".tif"], save_original_images=True)

    out = capsys.readouterr().out
    assert "A file with the same name already exists" in out
    # the pre-existing backup is intact ...
    assert np.array_equal(tifffile.imread(str(orig / _cv_name(chan="01"))), previous)
    # ... and the raw that could not be archived stayed in src
    assert (src / _cv_name(chan="01")).exists()
    assert not (src / _cv_name(chan="02")).exists()      # this one moved
    assert (orig / _cv_name(chan="02")).exists()
    # the stack was still produced from both channels
    assert np.load(src / "stack" / "plate1_A01_1_1.npy").shape == (6, 8, 2)


def test_undeletable_raw_image_is_reported_not_fatal(tmp_path, monkeypatch, capsys):
    """save_original_images=False keeps going when a raw cannot be removed."""
    from spacr.io import _rename_and_organize_image_files

    src = tmp_path / "plate1"
    src.mkdir()
    _write(src / _cv_name(chan="01"), 41)
    _write(src / _cv_name(chan="02"), 42)

    real_remove = os.remove

    def stubborn_remove(path, *args, **kwargs):
        if str(path).endswith(".tif"):
            raise OSError(13, "Permission denied")
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(os, "remove", stubborn_remove)

    n_channels = _rename_and_organize_image_files(
        str(src), CV_REGEX, batch_size=10, metadata_type="custom",
        img_format=[".tif"], save_original_images=False)

    assert n_channels == 2
    out = capsys.readouterr().out
    assert out.count("could not delete original image") == 2
    assert "Permission denied" in out
    # raws are still there, and the stack was written anyway
    assert (src / _cv_name(chan="01")).exists()
    assert (src / _cv_name(chan="02")).exists()
    assert not (src / "orig").exists()
    assert np.load(src / "stack" / "plate1_A01_1_1.npy").shape == (6, 8, 2)


# ---------------------------------------------------------------------------
# _merge_file
# ---------------------------------------------------------------------------

def test_merge_file_with_no_readable_channel_writes_nothing(tmp_path, capsys):
    from spacr.io import _merge_file

    chan_dirs = []
    for c in ("1", "2"):
        d = tmp_path / c
        d.mkdir()
        chan_dirs.append(str(d))          # neither holds fov.tif
    stack_dir = tmp_path / "stack"

    _merge_file(chan_dirs, str(stack_dir), "fov.tif")

    assert os.listdir(stack_dir) == []
    assert not (stack_dir / "fov.npy").exists()
    out = capsys.readouterr().out
    assert out.count("Warning: Failed to read image") == 2
    assert "No valid channels to merge for file fov.tif" in out


# ---------------------------------------------------------------------------
# _move_to_chan_folder
# ---------------------------------------------------------------------------

def test_move_to_chan_folder_cq1_ids_and_plate_fallback(tmp_path, capsys):
    """CQ1 names: numeric well ids are converted and plateID falls back to src."""
    from spacr.io import _move_to_chan_folder

    src = tmp_path / "cq1plate"
    src.mkdir()
    a01 = _write(src / "W1F001T0001Z01C1.tif", 71)
    b01 = _write(src / "W25F002T0001Z01C2.tif", 72)

    _move_to_chan_folder(str(src), CQ1_REGEX, timelapse=False,
                         metadata_type="cq1")

    # plateID is not a group in the CQ1 regex -> falls back to the folder name;
    # wellID 1 -> A01 and 25 -> B01 via the CQ1 well conversion.
    copy_a = src / "1" / "cq1plate_A01_1_.tif"
    copy_b = src / "2" / "cq1plate_B01_2_.tif"
    assert copy_a.exists(), sorted(p.name for p in (src / "1").iterdir())
    assert copy_b.exists()
    assert np.array_equal(tifffile.imread(str(copy_a)), a01)
    assert np.array_equal(tifffile.imread(str(copy_b)), b01)

    out = capsys.readouterr().out
    assert "Converted Well ID: 1 to A01" in out
    assert "Converted Well ID: 25 to B01" in out

    # originals archived under orig/
    assert sorted(os.listdir(src / "orig")) == ["W1F001T0001Z01C1.tif",
                                                "W25F002T0001Z01C2.tif"]
    assert not [f for f in os.listdir(src) if f.endswith(".tif")]


def test_move_to_chan_folder_reports_unparsable_filename(tmp_path, capsys):
    """A file the regex cannot parse is reported and no channel copy is made."""
    from spacr.io import _move_to_chan_folder

    src = tmp_path / "plate1"
    src.mkdir()
    _write(src / _cv_name(chan="01"), 81)
    _write(src / "garbage_image.tif", 82)

    _move_to_chan_folder(str(src), CV_REGEX, timelapse=False,
                         metadata_type="cellvoyager")

    out = capsys.readouterr().out
    assert "Could not extract information from filename garbage_image.tif" in out
    # only the parsable file produced a channel folder copy
    chan_dirs = sorted(p.name for p in src.iterdir()
                       if p.is_dir() and p.name != "orig")
    assert chan_dirs == ["1"]
    assert [p.name for p in (src / "1").iterdir()] == ["plate1_A01_1_.tif"]
    # both raws still end up archived
    assert sorted(os.listdir(src / "orig")) == sorted(
        [_cv_name(chan="01"), "garbage_image.tif"])


def test_move_to_chan_folder_duplicate_target_name_warns(tmp_path, capsys):
    """Two z-slices of one FOV collapse to the same name -> only one copy."""
    from spacr.io import _move_to_chan_folder

    src = tmp_path / "plate1"
    src.mkdir()
    # identical plate/well/field/channel, different sliceID -> same new name
    _write(src / "plate1_A01_T0001F001L01A01Z01C01.tif", 55)
    _write(src / "plate1_A01_T0001F001L01A01Z02C01.tif", 56)

    _move_to_chan_folder(str(src), CV_REGEX, timelapse=False,
                         metadata_type="cellvoyager")

    out = capsys.readouterr().out
    assert "A file with the same name already exists" in out
    copies = sorted(p.name for p in (src / "1").iterdir())
    assert copies == ["plate1_A01_1_.tif"]
    assert int(tifffile.imread(str(src / "1" / "plate1_A01_1_.tif")).max()) in (55, 56)
    assert len(os.listdir(src / "orig")) == 2


def test_move_to_chan_folder_orig_collision_keeps_both(tmp_path, capsys):
    """An existing orig/<name> blocks the archive move and warns."""
    from spacr.io import _move_to_chan_folder

    src = tmp_path / "plate1"
    src.mkdir()
    keep = _cv_name(well="A01", chan="01")
    move = _cv_name(well="A02", chan="01")
    _write(src / keep, 91)
    _write(src / move, 92)
    orig = src / "orig"
    orig.mkdir()
    previous = _write(orig / keep, 500)

    _move_to_chan_folder(str(src), CV_REGEX, timelapse=False,
                         metadata_type="cellvoyager")

    out = capsys.readouterr().out
    assert "A file with the same name already exists" in out
    assert np.array_equal(tifffile.imread(str(orig / keep)), previous)
    assert (src / keep).exists()          # blocked, stays put
    assert not (src / move).exists()      # archived
    assert (orig / move).exists()
    # both were copied into the channel folder before archiving
    assert sorted(p.name for p in (src / "1").iterdir()) == [
        "plate1_A01_1_.tif", "plate1_A02_1_.tif"]


def test_move_to_chan_folder_noop_when_stack_exists(tmp_path):
    """An existing stack/ short-circuits the whole sorter."""
    from spacr.io import _move_to_chan_folder

    src = tmp_path / "plate1"
    src.mkdir()
    (src / "stack").mkdir()
    _write(src / _cv_name(chan="01"), 61)

    assert _move_to_chan_folder(str(src), CV_REGEX) is None
    assert sorted(p.name for p in src.iterdir()) == [_cv_name(chan="01"), "stack"]
    assert not (src / "orig").exists()


# ---------------------------------------------------------------------------
# _generate_time_lists / _is_dir_empty (guards around the ingest)
# ---------------------------------------------------------------------------

def test_generate_time_lists_skips_malformed_names(tmp_path):
    from spacr.io import _generate_time_lists

    files = [
        "plate1_A01_1_2.npy", "plate1_A01_1_10.npy", "plate1_A01_1_1.npy",
        "plate1_A02_1_1.npy",
        "plate1_A01_1_x.npy",     # timepoint not an int -> skipped
        "plate1_A01.npy",         # too few parts -> skipped
        "plate1_A01_1_1.tif",     # not .npy -> skipped
    ]
    groups = _generate_time_lists(files)

    assert sorted(len(g) for g in groups) == [1, 3]
    by_key = {g[0].rsplit("_", 1)[0]: g for g in groups}
    assert by_key["plate1_A01_1"] == [
        "plate1_A01_1_1.npy", "plate1_A01_1_2.npy", "plate1_A01_1_10.npy"]
    assert by_key["plate1_A02_1"] == ["plate1_A02_1_1.npy"]
    assert all(re.match(r".*_\d+\.npy$", f) for g in groups for f in g)


# ---------------------------------------------------------------------------
# the writer and the reader, pinned to each other
# ---------------------------------------------------------------------------

def test_measure_can_read_back_the_stack_names_this_ingest_writes(tmp_path):
    """The ingest names every stack ``plate_well_field_TIME`` -- always.

    ``timelapse`` decides whether the timepoint is carried into the keys, not
    whether it is written into the file name, so an ordinary plate's
    ``merged/*.npy`` has four components and is read back with
    ``timelapse=False``. When the reader refused that fourth component, every
    field of every ordinary run came back as the literal string ``'error'`` in
    all five slots; ``_merge_and_save_to_database`` then rejected the frame
    for a prcf disagreeing with its identity columns, and ``measure_crop``
    wrote no measurement tables at all.

    The writer and the reader live in two modules and only their agreement
    matters, so this drives the real ingest and feeds its own output back.
    The demo-pipeline test that showed the same failure is ``slow``-marked
    and therefore deselected in CI, which is how it went unnoticed.
    """
    from spacr.io import _rename_and_organize_image_files
    from spacr.utils import _map_wells

    src = tmp_path / "plate1"
    src.mkdir()
    _write(src / _cv_name(well="A01", field="002", chan="01"), 5)
    _write(src / _cv_name(well="A01", field="002", chan="02"), 6)

    _rename_and_organize_image_files(
        str(src), CV_REGEX, batch_size=10, metadata_type="custom",
        img_format=[".tif"], save_original_images=False)

    written = sorted(os.listdir(src / "stack"))
    assert written == ["plate1_A01_2_1.npy"], written

    plate, row, column, field, prcf = _map_wells(written[0], timelapse=False)
    assert (plate, row, column, field) == ("plate1", "r1", "c1", "f2")
    assert prcf == "plate1_r1_c1_f2"
