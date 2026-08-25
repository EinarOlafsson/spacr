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
import sqlite3
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


def test_a_single_field_can_be_named_in_any_spelling():
    """``3``, ``'f3'`` and ``'F003'`` are one field, as the writer spells it."""
    names = ["plate1_A01_f3.npy", "plate1_A01_f4.npy"]

    for spelling in (3, "3", "f3", "F003"):
        assert io.select_fields(names, spelling) == ["plate1_A01_f3.npy"]


def test_a_comma_separated_field_list_keeps_the_order_given():
    """Typed into a settings CSV, a list arrives as one string."""
    assert io.select_fields(_STACKS, "f1, f2") == [
        "plate1_A01_f1.npy", "plate1_A01_f2.npy"]


def test_a_field_list_of_only_separators_keeps_everything():
    """An empty request is "no filter", not "no fields"."""
    assert io.select_fields(_STACKS, " , , ") == list(_STACKS)
    assert io.select_fields(_STACKS, []) == list(_STACKS)


def test_a_name_that_is_not_a_field_stem_is_skipped_not_crashed_on():
    """A sidecar or hand-dropped array in the folder must not stop the run."""
    kept = io.select_fields(_STACKS, "f1*")

    assert kept == ["plate1_A01_f1.npy", "plate1_A02_f10.npy"]
    assert "not_a_field_stem.npy" not in kept


def test_a_field_token_that_cannot_be_normalised_is_matched_literally():
    """An unparseable request matches nothing rather than everything."""
    assert io.select_fields(_STACKS, "banana") == []


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

def test_a_nucleus_crop_carries_its_parent_cell_in_its_name():
    """Nucleus and pathogen crops are named ``<stem>_<cell>_<label>.png``."""
    assert io.crop_png_name("plate1_A01_1.npy", "nucleus", 7, 3) == \
        "plate1_A01_1_3_7.png"
    assert io.crop_png_name("plate1_A01_1", "pathogen", 2, "o11") == \
        "plate1_A01_1_11_2.png"


def test_a_nucleus_crop_with_no_resolvable_parent_says_none():
    """"none" is a name a parser can read; an empty gap is not."""
    assert io.crop_png_name("plate1_A01_1", "nucleus", 4, "omulti") == \
        "plate1_A01_1_none_4.png"
    assert io.crop_png_name("plate1_A01_1", "nucleus", 4, None) == \
        "plate1_A01_1_none_4.png"


def test_a_cell_crop_is_named_without_a_parent():
    """The contrast that makes the parent token meaningful."""
    assert io.crop_png_name("plate1_A01_1", "cell", 5) == "plate1_A01_1_5.png"


def _object_db(path, table, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute('CREATE TABLE "%s" (%s)'
                    % (table, ", ".join(f'"{c}"' for c in columns)))
        con.executemany(
            'INSERT INTO "%s" VALUES (%s)' % (table, ", ".join("?" * len(columns))),
            rows)
        con.commit()
    finally:
        con.close()
    return str(path)


def test_a_database_without_the_object_table_answers_an_empty_frame(
        tmp_path, capsys):
    """A project with no nucleus table asked for nuclei gets nothing, said out loud."""
    db = _object_db(tmp_path / "m.db", "cell",
                    ["object_label", "plateID", "rowID", "columnID",
                     "fieldID", "prcf", "file_name", "path_name"],
                    [(1, "plate1", "A", "01", "1", "plate1_A_01_1",
                      "plate1_A01_1.npy", "/data/plate1_A01_1.npy")])

    frame = io.crop_rows_from_object_table(db, object_type="nucleus")

    assert frame.empty
    assert "no 'nucleus' table" in capsys.readouterr().out


def test_a_nucleus_table_without_parent_links_still_yields_crop_rows(
        tmp_path, capsys):
    """No ``cell_id`` column means no parent, not no crops."""
    db = _object_db(tmp_path / "m.db", "nucleus",
                    ["object_label", "plateID", "rowID", "columnID",
                     "fieldID", "prcf", "file_name", "path_name"],
                    [(4, "plate1", "A", "01", "1", "plate1_A_01_1",
                      "plate1_A01_1.npy", "/data/plate1_A01_1.npy")])

    frame = io.crop_rows_from_object_table(db, object_type="nucleus")

    assert list(frame["png_name"]) == ["plate1_A01_1_none_4.png"]
    assert frame["png_path"].iloc[0].endswith(
        os.path.join("nucleus_png", "plate1_A01_1_none_4.png"))
    assert "1 'nucleus' objects" in capsys.readouterr().out


def test_a_nucleus_table_with_parent_links_names_the_parent_cell(tmp_path):
    """The link is what makes the on-demand name match the folder's name."""
    db = _object_db(tmp_path / "m.db", "nucleus",
                    ["object_label", "plateID", "rowID", "columnID",
                     "fieldID", "prcf", "file_name", "path_name", "cell_id"],
                    [(4, "plate1", "A", "01", "1", "plate1_A_01_1",
                      "plate1_A01_1.npy", "/data/plate1_A01_1.npy", 9)])

    frame = io.crop_rows_from_object_table(db, object_type="nucleus",
                                           verbose=False)

    assert list(frame["png_name"]) == ["plate1_A01_1_9_4.png"]


def test_an_object_table_with_no_rows_answers_an_empty_frame(tmp_path):
    """An empty table is not an error, and gains no derived columns."""
    db = _object_db(tmp_path / "m.db", "cell",
                    ["object_label", "plateID", "rowID", "columnID",
                     "fieldID", "prcf", "file_name", "path_name"], [])

    frame = io.crop_rows_from_object_table(db, object_type="cell",
                                           verbose=False)

    assert frame.empty
    assert "png_name" not in frame.columns


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

def test_a_validation_fraction_outside_the_open_unit_interval_is_refused():
    """0 and 1 are not splits, and neither is 1.5."""
    labels = [0, 1, 0, 1]
    groups = ["a", "a", "b", "b"]

    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            io.make_validation_holdout(labels, bad, groups)


def test_a_group_aware_holdout_needs_one_group_per_sample():
    """A group list of the wrong length would silently mis-assign crops."""
    with pytest.raises(ValueError, match="one group per sample"):
        io.make_validation_holdout([0, 1, 0, 1], 0.5, ["a", "b"])

    with pytest.raises(ValueError, match="one group per sample"):
        io.make_validation_holdout([0, 1, 0, 1], 0.5, None)


def test_a_holdout_needs_at_least_two_distinct_groups():
    """One group cannot be on both sides of a leakage-safe line."""
    with pytest.raises(ValueError, match="two distinct groups"):
        io.make_validation_holdout([0, 1, 0, 1], 0.5, ["a", "a", "a", "a"])


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

def _tar_with(tmp_path, members, sidecar_bytes=None, add_directory=False):
    """Build a tar holding ``members`` (name -> PNG bytes) plus extras."""
    import io as _io
    import tarfile

    from PIL import Image

    from spacr.crops import CROP_FORMAT_SIDECAR

    path = tmp_path / "dataset.tar"
    with tarfile.open(path, "w") as tar:
        if add_directory:
            info = tarfile.TarInfo("subdir")
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        if sidecar_bytes is not None:
            info = tarfile.TarInfo(CROP_FORMAT_SIDECAR)
            info.size = len(sidecar_bytes)
            tar.addfile(info, _io.BytesIO(sidecar_bytes))
        for name in members:
            buf = _io.BytesIO()
            Image.fromarray(
                np.full((4, 4, 3), 128, dtype=np.uint8)).save(buf, format="PNG")
            payload = buf.getvalue()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, _io.BytesIO(payload))
    return str(path)


def test_a_directory_entry_in_a_tar_is_not_a_sample(tmp_path):
    """Only files are crops; a directory member would decode to nothing."""
    tar = _tar_with(tmp_path, ["plate1_A01_1_1.png"], add_directory=True)

    dataset = io.TarImageDataset(tar)

    assert len(dataset) == 1
    assert [m.name for m in dataset.members] == ["plate1_A01_1_1.png"]


def test_an_unreadable_crop_format_sidecar_reports_an_unknown_format(tmp_path):
    """An unreadable marker means "not stated", which is not "current"."""
    tar = _tar_with(tmp_path, ["plate1_A01_1_1.png"],
                    sidecar_bytes=b"{not json at all")

    dataset = io.TarImageDataset(tar)

    assert dataset.crop_format is None
    assert len(dataset) == 1, "the sidecar is still excluded from the samples"


def test_a_readable_crop_format_sidecar_is_surfaced(tmp_path):
    """The contrast that makes the unreadable case meaningful."""
    from spacr.crops import CROP_FORMAT_LEGACY_BGR

    tar = _tar_with(
        tmp_path, ["plate1_A01_1_1.png"],
        sidecar_bytes=b'{"spacr_crop_format": %d}' % CROP_FORMAT_LEGACY_BGR)

    dataset = io.TarImageDataset(tar)

    assert dataset.crop_format == CROP_FORMAT_LEGACY_BGR


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


def test_items_from_an_unreadable_folder_are_assumed_legacy(tmp_path,
                                                            monkeypatch):
    """Same rule, applied to the crops about to be copied into a dataset."""
    import spacr.crops as crops

    def unreadable(*_args, **_kwargs):
        raise OSError("the crop folder is gone")

    monkeypatch.setattr(crops, "crop_folder_format", unreadable)

    fmt = io._crop_format_of_items([str(tmp_path / "cell_png" / "a.png")])

    assert fmt == crops.CROP_FORMAT_LEGACY_BGR


def test_a_field_token_the_schema_cannot_classify_is_matched_literally(
        monkeypatch):
    """A schema that refuses a token must not take the whole run with it.

    Normalisation is what makes ``3`` and ``'F003'`` the same field; when it
    is unavailable the token stands for itself, which is still right for the
    spelling the file names use.
    """
    import spacr.schema as schema

    def refuses(_token):
        raise ValueError("unrecognised field token")

    monkeypatch.setattr(schema, "field_index", refuses)

    assert io.select_fields(_STACKS, "f1") == ["plate1_A01_f1.npy"], (
        "the literal token still names the field the file was written with")


# ---------------------------------------------------------------------------
# writing a crop tar from mixed sources
# ---------------------------------------------------------------------------

def test_a_tar_takes_png_paths_and_on_demand_crops_side_by_side(tmp_path,
                                                                capsys):
    """The two crop sources have to be interchangeable inside one archive.

    A PNG already on disk is byte-copied; an on-demand crop is cut and
    encoded. A path that has gone missing is skipped with a line naming it,
    not allowed to abort a dataset that is otherwise complete.
    """
    import tarfile

    from PIL import Image

    from spacr.crops import CROP_FORMAT_SIDECAR

    folder = tmp_path / "cell_png"
    folder.mkdir()
    on_disk = folder / "plate1_A01_1_1.png"
    Image.fromarray(np.full((4, 4, 3), 7, dtype=np.uint8)).save(on_disk)
    items = [str(on_disk),
             io.LazyCropPNG(_Source(), {"object_label": 2},
                            name="plate1_A01_1_2.png"),
             str(folder / "deleted.png")]
    tar_name = str(tmp_path / "dataset.tar")

    written, skipped = io._write_crop_tar(items, tar_name)

    assert (written, skipped) == (2, 1)
    assert "deleted.png" in capsys.readouterr().out
    with tarfile.open(tar_name) as tar:
        names = set(tar.getnames())
        assert names == {CROP_FORMAT_SIDECAR, "plate1_A01_1_1.png",
                         "plate1_A01_1_2.png"}
        assert tar.extractfile("plate1_A01_1_1.png").read() == \
            on_disk.read_bytes(), "an existing PNG is copied byte for byte"


def test_two_crops_with_the_same_basename_both_reach_the_tar(tmp_path):
    """A collision inside the archive would silently drop one of them."""
    import tarfile

    from PIL import Image

    first = tmp_path / "plate1" / "cell_png"
    second = tmp_path / "plate2" / "cell_png"
    for folder in (first, second):
        folder.mkdir(parents=True)
        Image.fromarray(
            np.full((4, 4, 3), 9, dtype=np.uint8)).save(folder / "same.png")
    tar_name = str(tmp_path / "dataset.tar")

    written, skipped = io._write_crop_tar(
        [str(first / "same.png"), str(second / "same.png")], tar_name)

    assert (written, skipped) == (2, 0)
    with tarfile.open(tar_name) as tar:
        crops_in_tar = [n for n in tar.getnames() if n.endswith(".png")]
    assert len(crops_in_tar) == 2
    assert len(set(crops_in_tar)) == 2
