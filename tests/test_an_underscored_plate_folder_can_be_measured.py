"""A plate folder whose name holds an underscore is an ordinary thing to have.

``spacr.io`` composes every stack name as ``plate_well_field_time`` with the
plate taken from a regex group or, far more often, from
``os.path.basename(src)`` -- a folder name.  A folder called ``exp_1``
therefore produced ``exp_1_A01_1_1.npy``, five separator-delimited components
for a four-component grammar, and the reader could not tell the plate from the
well::

    _map_wells('exp_1_A01_1_1.npy')  ->  ('error',) * 5

Every field of the plate.  ``utils._merge_and_save_to_database`` then refused
each frame for a ``prcf`` disagreeing with its identity columns, so nothing was
measured at all.  That is loud rather than quiet, which is an improvement on
what it did before the identity keys were escaped -- the same name parsed as
plate ``exp``, well ``1``, field ``A01`` -- but a plate a user is entitled to
have still could not be run.

The plate is now escaped at the writer, where the four components are still
separate and there is nothing to split.  These cases drive the real ingest and
feed its own output back to the reader, which is the only way to keep the two
halves of the grammar pinned to each other.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile

from spacr import schema as S
from spacr.io import (_escaped_field_stem, _rename_and_organize_image_files,
                      migrate_unescaped_plate_names)
from spacr.utils import _map_wells

#: No plateID group, so the plate falls back to the source FOLDER name -- the
#: path that puts a user's underscore into the stack name.
NO_PLATE_REGEX = (r"(?P<wellID>[A-Z]\d+)_T(?P<timeID>\d+)F(?P<fieldID>\d+)"
                  r"C(?P<chanID>\d+)\.tif")


def _plate_folder(tmp_path, name, wells=("A01",), fields=("001",)):
    src = tmp_path / name
    src.mkdir(parents=True)
    for well in wells:
        for field in fields:
            for chan in ("01", "02"):
                tifffile.imwrite(
                    str(src / f"{well}_T0001F{field}C{chan}.tif"),
                    np.full((6, 8), int(chan), dtype=np.uint16))
    return src


# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------

def test_the_ingest_writes_a_name_the_reader_can_split(tmp_path):
    """The measurement this fix exists for, taken through the real ingest."""
    src = _plate_folder(tmp_path, "exp_1")

    _rename_and_organize_image_files(str(src), NO_PLATE_REGEX, batch_size=10,
                                     metadata_type="custom")

    names = sorted(p.name for p in (src / "stack").glob("*.npy"))
    assert names == ["exp%5F1_A01_1_1.npy"]
    assert _map_wells(names[0]) == ("exp_1", "r1", "c1", "f1",
                                    "exp%5F1_r1_c1_f1")


def test_an_ordinary_plate_folder_is_not_renamed(tmp_path):
    """The escape has to be a no-op for the names everyone already has."""
    src = _plate_folder(tmp_path, "plate1")

    _rename_and_organize_image_files(str(src), NO_PLATE_REGEX, batch_size=10,
                                     metadata_type="custom")

    names = sorted(p.name for p in (src / "stack").glob("*.npy"))
    assert names == ["plate1_A01_1_1.npy"]


def test_the_plate_survives_the_round_trip_character_for_character(tmp_path):
    """`exp_1` back out as `exp_1`, not as `exp` or as `exp%5F1`."""
    src = _plate_folder(tmp_path, "two_word_plate")

    _rename_and_organize_image_files(str(src), NO_PLATE_REGEX, batch_size=10,
                                     metadata_type="custom")

    stem = next((src / "stack").glob("*.npy")).stem
    assert S.parse_field_stem(stem).plateID == "two_word_plate"


def test_a_plate_holding_a_percent_sign_is_not_confused_with_an_escape():
    """`p%5Fx` and `p_x` must stay two plates, not merge into one."""
    escaped = _escaped_field_stem("p%5Fx", "A01", 1, 1)
    raw = _escaped_field_stem("p_x", "A01", 1, 1)

    assert escaped != raw
    assert S.parse_field_stem(escaped).plateID == "p%5Fx"
    assert S.parse_field_stem(raw).plateID == "p_x"


def test_a_plate_of_none_keeps_its_historical_spelling():
    """A regex group that did not participate gave the literal 'None'.

    That is a bad plate id and always was, but changing it here would rename
    every array such a run has already written, for no gain.
    """
    assert _escaped_field_stem(None, "A01", 1, 1) == "None_A01_1_1"


def test_the_channel_folder_layout_escapes_the_same_way(tmp_path):
    """`_move_to_chan_folder` is the older two-step ingest and the same rule."""
    from spacr.io import _move_to_chan_folder

    src = _plate_folder(tmp_path, "exp_1")
    _move_to_chan_folder(str(src), NO_PLATE_REGEX, timelapse=False,
                         metadata_type="custom")

    channel_folders = [d for d in src.iterdir()
                       if d.is_dir() and d.name != "orig"]
    written = sorted(name for folder in channel_folders
                     for name in os.listdir(folder))
    assert written and all(name.startswith("exp%5F1_") for name in written), \
        written


# ---------------------------------------------------------------------------
# the migration
# ---------------------------------------------------------------------------

def _legacy_plate(tmp_path):
    """A plate as a previous release left it: raw underscore in every stem."""
    src = tmp_path / "exp_1"
    for folder in ("stack", "merged", "masks/cell_mask_stack",
                   "masks/nucleus_mask_stack"):
        (src / folder).mkdir(parents=True)
    for folder in ("stack", "merged", "masks/cell_mask_stack",
                   "masks/nucleus_mask_stack"):
        for field in (1, 2):
            np.save(src / folder / f"exp_1_A01_{field}_1.npy",
                    np.zeros((4, 4), dtype=np.uint16))
    return src


def test_the_migration_moves_the_arrays_and_the_masks_with_them(tmp_path):
    """Masks are hours to days of segmentation; the plate must not be re-run."""
    src = _legacy_plate(tmp_path)

    moved = migrate_unescaped_plate_names(str(src))

    assert len(moved) == 8
    for folder in ("stack", "merged", "masks/cell_mask_stack",
                   "masks/nucleus_mask_stack"):
        names = sorted(os.listdir(src / folder))
        assert names == ["exp%5F1_A01_1_1.npy", "exp%5F1_A01_2_1.npy"], folder
    assert _map_wells("exp%5F1_A01_1_1.npy")[0] == "exp_1"


def test_a_dry_run_reports_and_moves_nothing(tmp_path):
    """The user gets to see what a migration would do before it does it."""
    src = _legacy_plate(tmp_path)

    planned = migrate_unescaped_plate_names(str(src), dry_run=True)

    assert len(planned) == 8
    assert all(os.path.exists(old) for old, _new in planned)
    assert not any(os.path.exists(new) for _old, new in planned)


def test_migrating_an_ordinary_plate_twice_does_nothing_either_time(tmp_path):
    """Running it on a folder that does not need it must be free and safe."""
    src = tmp_path / "plate1"
    (src / "merged").mkdir(parents=True)
    np.save(src / "merged" / "plate1_A01_1_1.npy", np.zeros((4, 4), np.uint16))

    assert migrate_unescaped_plate_names(str(src)) == []
    assert migrate_unescaped_plate_names(str(src)) == []
    assert os.listdir(src / "merged") == ["plate1_A01_1_1.npy"]


def test_a_migrated_plate_is_not_migrated_again(tmp_path):
    """The escaped name is already correct; a second pass must not re-escape."""
    src = _legacy_plate(tmp_path)
    migrate_unescaped_plate_names(str(src))

    assert migrate_unescaped_plate_names(str(src)) == []
    assert sorted(os.listdir(src / "merged")) == ["exp%5F1_A01_1_1.npy",
                                                  "exp%5F1_A01_2_1.npy"]


def test_the_migration_refuses_before_it_clobbers_anything(tmp_path):
    """A half-applied rename leaves a plate neither reader can read."""
    src = _legacy_plate(tmp_path)
    np.save(src / "merged" / "exp%5F1_A01_1_1.npy",
            np.zeros((4, 4), dtype=np.uint16))

    with pytest.raises(FileExistsError) as exc:
        migrate_unescaped_plate_names(str(src))

    assert "already exist" in str(exc.value)
    # and nothing moved: the raw names are all still there
    assert "exp_1_A01_2_1.npy" in os.listdir(src / "merged")


def test_the_migration_leaves_names_that_are_not_field_stems_alone(tmp_path):
    """A sidecar or a hand-dropped array must not be renamed on a guess."""
    src = tmp_path / "exp_1"
    (src / "merged").mkdir(parents=True)
    np.save(src / "merged" / "exp_1_A01_1_1.npy", np.zeros((4, 4), np.uint16))
    np.save(src / "merged" / "notes.npy", np.zeros(2))
    (src / "merged" / "layout.json").write_text("{}")

    migrate_unescaped_plate_names(str(src))

    assert sorted(os.listdir(src / "merged")) == [
        "exp%5F1_A01_1_1.npy", "layout.json", "notes.npy"]


def test_the_raw_drop_is_never_touched(tmp_path):
    """`orig/` and the source folder hold the vendor's names, not spaCR's."""
    src = tmp_path / "exp_1"
    (src / "orig").mkdir(parents=True)
    (src / "merged").mkdir()
    raw = src / "orig" / "exp_1_A01_T0001_F001.tif"
    tifffile.imwrite(str(raw), np.zeros((4, 4), dtype=np.uint16))

    migrate_unescaped_plate_names(str(src))

    assert raw.exists()
