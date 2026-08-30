"""``spacr.foreign``: the states a real destination is in, not the clean one.

Everything here is a path an import reaches on somebody's actual data and
that the happy-path tests do not:

* a measurement row whose image key names a *folder*, and a table whose keys
  are declared rather than guessed;
* ``verify_labels=False`` — the fast plan, which pairs masks without reading
  a single one — and the capped examples list that keeps a report readable
  when six thousand rows carry a bad label;
* :func:`~spacr.foreign.release_canonical_copy` against a database whose
  provenance tables are gone, which is exactly the database
  ``spacr.resume.importer_written_columns`` documents its fallback for: an
  older importer replaced them wholesale. The un-claim has nothing to write
  and the release must still happen;
* the column tidy-up after that release, which drops an imported column only
  when it is provably empty and never fails the release it follows;
* a conversion-map merge that cannot be applied, and the staging table it
  leaves behind for the next one to clear.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

from spacr import convert as cv
from spacr import foreign as fg

SIZE = 24

#: What the two channels of each field are filled with. They differ so that a
#: crop cut from channel 0 alone is distinguishable from one cut from both.
CHANNEL_VALUES = {1: 11, 2: 12}


# --------------------------------------------------------------------------- #
#  Their data
# --------------------------------------------------------------------------- #

def _write(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, array)
    return path


def _label_image():
    mask = np.zeros((SIZE, SIZE), np.uint16)
    mask[2:8, 2:8] = 1
    mask[12:20, 12:20] = 2
    return mask


class Theirs:
    """A collaborator's folder: two fields, two channels, one mask class."""

    def __init__(self, root):
        self.root = str(root)
        self.images = os.path.join(self.root, "their_images")
        self.cell_masks = os.path.join(self.root, "their_cell_masks")
        for field in (1, 2):
            for channel in (1, 2):
                _write(os.path.join(self.images,
                                    f"fov{field:02d}_C{channel}.tif"),
                       np.full((SIZE, SIZE), CHANNEL_VALUES[channel],
                               np.uint16))
            _write(os.path.join(self.cell_masks,
                                f"fov{field:02d}_cell_mask.tif"),
                   _label_image())

    def rows(self):
        return [{"ImageNumber": f"fov{field:02d}_C1.tif",
                 "ObjectNumber": label,
                 "AreaShape_Area": 36.0 * label,
                 "Metadata_Treatment": "wt" if field == 1 else "ko"}
                for field in (1, 2) for label in (1, 2)]

    def table(self, rows, name="results.csv"):
        path = os.path.join(self.root, name)
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def plan(self, rows=None, name="results.csv", **kwargs):
        kwargs.setdefault("um_per_px", 0.5)
        table = self.table(self.rows() if rows is None else rows, name)
        return fg.plan_import(self.images, {"cell": self.cell_masks}, table,
                              **kwargs)


@pytest.fixture
def theirs(tmp_path):
    return Theirs(tmp_path)


def _tables(db_path):
    connection = sqlite3.connect(db_path)
    try:
        return sorted(str(r[0]) for r in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"))
    finally:
        connection.close()


def _columns(db_path, table):
    connection = sqlite3.connect(db_path)
    try:
        return [str(r[1]) for r in connection.execute(
            f'PRAGMA table_info("{table}")')]
    finally:
        connection.close()


def _query(db_path, sql, *params):
    connection = sqlite3.connect(db_path)
    try:
        return connection.execute(sql, params).fetchall()
    finally:
        connection.close()


def _run(db_path, *statements):
    connection = sqlite3.connect(db_path)
    try:
        for statement in statements:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


# --------------------------------------------------------------------------- #
#  One real import, shared. It is the slowest thing in this file.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    """A finished import, with crops cut from channel 0 only.

    Module-scoped because ``run_import`` converts, stacks, merges and writes
    a database; every test below that needs a finished import copies the
    database it produced rather than making another.
    """
    root = tmp_path_factory.mktemp("their_data")
    data = Theirs(root)
    plan = data.plan()
    dst = str(root / "imported")
    result = fg.run_import(plan, dst, crops=True, crop_channels=(0,),
                           crop_size=(16, 16))
    return dataclasses.replace(result), plan, dst


@pytest.fixture
def db_copy(imported, tmp_path):
    """A private copy of the imported database, for tests that mutate it."""
    result, _plan, _dst = imported
    copy = str(tmp_path / "measurements.db")
    shutil.copy(result.db_path, copy)
    return copy


# --------------------------------------------------------------------------- #
#  The join: keys that are declared, and rows that name a folder
# --------------------------------------------------------------------------- #

def test_a_row_whose_image_key_names_a_folder_resolves_to_no_field(theirs):
    """A path with a trailing separator has no basename to match on.

    Their ``ImageNumber`` is a filename in every export this module has
    seen, but it is *their* column: a table exported per-directory puts a
    folder there. It has to come back as an unresolved field, counted and
    named, rather than matching an arbitrary stem.
    """
    rows = theirs.rows() + [{"ImageNumber": "their_images/",
                             "ObjectNumber": 1, "AreaShape_Area": 1.0,
                             "Metadata_Treatment": "wt"}]

    plan = theirs.plan(rows, image_key="ImageNumber", label_key="ObjectNumber")

    assert plan.join.rows_total == 5
    assert plan.join.rows_matched == 4, "the four real rows still match"
    assert plan.join.unresolved_fields == [("their_images/", 1)]


def test_declared_keys_are_used_instead_of_the_inferred_ones(theirs):
    """Passing the keys must switch the inference off, not run beside it.

    This table carries a pair of columns whose *names* are the two the
    inference reaches for first -- ``ImageNumber`` and ``ObjectNumber`` --
    and neither holds the join key; the real ones are called something the
    hints have never heard of. Left to guess, the import joins on the decoys
    and matches nothing.
    """
    rows = [{"their_frame": row["ImageNumber"],
             "their_object_index": row["ObjectNumber"],
             "ImageNumber": f"decoy{index}.tif", "ObjectNumber": 99,
             "AreaShape_Area": row["AreaShape_Area"]}
            for index, row in enumerate(theirs.rows())]

    guessed = theirs.plan(rows, name="guessed.csv")
    declared = theirs.plan(rows, name="declared.csv",
                           image_key="their_frame",
                           label_key="their_object_index")

    assert guessed.join.image_key == "ImageNumber", "the decoy is inferred"
    assert guessed.join.rows_matched == 0
    assert declared.join.image_key == "their_frame"
    assert declared.join.label_key == "their_object_index"
    assert declared.join.rows_matched == 4


# --------------------------------------------------------------------------- #
#  Planning without reading the masks
# --------------------------------------------------------------------------- #

def test_verification_off_pairs_the_masks_without_reading_their_labels(theirs):
    """``verify_labels=False`` is the plan you can run on 4000 fields.

    The pairing is by name and must still be complete; what goes is the
    label set of each mask, and with it every check that needs one.
    """
    verified = theirs.plan(name="verified.csv")
    quick = theirs.plan(name="quick.csv", verify_labels=False)

    assert set(quick.masks.fields) == set(verified.masks.fields)
    assert verified.masks.fields["plate1_A01_1"]["cell"].labels == (1, 2)
    assert quick.masks.fields["plate1_A01_1"]["cell"].labels == (), (
        "no mask may be read when verification is off")
    assert quick.join.rows_matched == verified.join.rows_matched


def test_an_unverified_plan_reports_unmatched_rows_but_no_unmeasured_objects(
        theirs):
    """"Which objects were never measured" cannot be answered without labels.

    The row-side half of the report survives — those rows are unmatched
    whatever the masks hold — and the object-side half is silent rather
    than wrong.
    """
    rows = theirs.rows() + [
        {"ImageNumber": "fov01_C1.tif", "ObjectNumber": f"x{index}",
         "AreaShape_Area": 1.0, "Metadata_Treatment": "wt"}
        for index in range(6)]

    quick = theirs.plan(rows, name="quick.csv", verify_labels=False,
                        image_key="ImageNumber", label_key="ObjectNumber")

    assert quick.join.rows_unmatched == 6
    assert quick.join.objects_unmeasured == []
    assert any("match no object" in warning for warning in quick.warnings)

    # Verified, the same table also knows which objects nobody measured.
    verified = theirs.plan(rows, name="verified.csv",
                           image_key="ImageNumber", label_key="ObjectNumber")
    assert verified.join.rows_unmatched == 6
    assert verified.join.objects_unmeasured == []


@pytest.mark.parametrize("bad_label, phrase", [
    ("not-a-number", "is not an integer"),
    (900, "no object with label"),
])
def test_only_five_examples_are_kept_but_the_count_is_complete(
        theirs, bad_label, phrase):
    """A report is read by a human; six thousand examples is not one.

    The cap is on the examples, never on the counts — a truncated *count*
    would understate how much of their table failed to join, which is the
    number the decision to import is made on.
    """
    rows = theirs.rows() + [
        {"ImageNumber": "fov01_C1.tif",
         "ObjectNumber": f"{bad_label}{index}" if isinstance(bad_label, str)
                         else bad_label + index,
         "AreaShape_Area": 1.0, "Metadata_Treatment": "wt"}
        for index in range(6)]

    plan = theirs.plan(rows, image_key="ImageNumber",
                       label_key="ObjectNumber")

    assert plan.join.rows_no_object == [("plate1_A01_1", 6)]
    assert len(plan.join.examples) == 5
    assert all(phrase in example for example in plan.join.examples)


# --------------------------------------------------------------------------- #
#  What the run reports
# --------------------------------------------------------------------------- #

def test_a_result_with_no_conversion_summarises_everything_else(imported):
    """``conversion`` is optional and defaults to None.

    A summary that raised on it would be unprintable for exactly the runs
    whose report matters most — the ones that stopped before their images
    were converted.
    """
    result, _plan, _dst = imported

    with_images = result.summary()
    without = dataclasses.replace(result, conversion=None).summary()

    assert "image file(s) converted" in with_images
    assert result.conversion.map_path in with_images
    assert "image file(s) converted" not in without
    # Everything that does not come from the conversion is still there.
    assert f"{result.db_path}" in without
    assert "mask file(s) written" in without
    assert without.splitlines()[0] == with_images.splitlines()[0]


def test_crops_are_cut_from_the_channels_that_were_asked_for(imported):
    """``crop_channels`` is how a 5-channel import gets a 3-channel PNG.

    Cut from channel 0 alone, every pixel of the PNG is grey: the one plane
    is repeated into R, G and B. The merged array proves the two channels
    really do differ, so a crop that had taken both could not be grey.
    """
    result, _plan, _dst = imported
    merged = np.load(result.merged[0])

    assert merged.shape[2] == 3, "two intensity channels and one mask plane"
    assert merged[..., 0].max() != merged[..., 1].max(), (
        "the channels differ, so which one was taken is visible")

    assert result.crops, "the import cut crops"
    crop = np.array(Image.open(result.crops[0]).convert("RGB"))
    assert crop.shape == (16, 16, 3)
    assert (crop[..., 0] == crop[..., 1]).all()
    assert (crop[..., 1] == crop[..., 2]).all()


# --------------------------------------------------------------------------- #
#  Releasing the canonical copy from a database with no provenance left
# --------------------------------------------------------------------------- #

def test_a_release_works_on_a_database_whose_provenance_is_gone(db_copy):
    """The database an older importer left: no ``foreign_columns``, no
    ``foreign_import``.

    ``resume.importer_written_columns`` falls back to the columns of
    ``foreign_cell`` for exactly this case, and the un-claim then has
    nothing to write. The release must still remove the copy and must
    still prove, row by row, that their numbers survive it.
    """
    _run(db_copy, "DROP TABLE foreign_columns", "DROP TABLE foreign_import")
    assert _query(db_copy, "SELECT COUNT(*) FROM cell")[0][0] == 4

    removed = fg.release_canonical_copy(db_copy, "cell")

    assert removed == 4
    assert _query(db_copy, "SELECT COUNT(*) FROM cell")[0][0] == 0
    assert _query(db_copy, "SELECT COUNT(*) FROM foreign_cell")[0][0] == 4, (
        "their numbers are what the release must not touch")
    assert "cell_with_foreign" in _tables(db_copy)
    # The emptied imported columns go with the rows that filled them.
    assert "foreign_areashape_area" not in _columns(db_copy, "cell")
    assert "foreign_areashape_area" in _columns(db_copy, "foreign_cell")


def test_an_import_record_too_old_to_carry_the_claim_is_left_alone(db_copy,
                                                                  tmp_path):
    """``foreign_import`` grew its ``canonical_table`` columns later.

    A record written before them cannot be un-claimed, and the release has
    to go ahead rather than fail on a schema it can read but not update.
    """
    thin = str(tmp_path / "thin.db")
    shutil.copy(db_copy, thin)
    for path in (db_copy, thin):
        _run(path, "DROP TABLE foreign_columns")
    _run(thin,
         "DROP TABLE foreign_import",
         'CREATE TABLE foreign_import (dst TEXT, object_types TEXT)',
         "INSERT INTO foreign_import VALUES ('/old/dst', 'cell')")

    assert fg.release_canonical_copy(thin, "cell") == 4
    assert fg.release_canonical_copy(db_copy, "cell") == 4

    assert _query(thin, "SELECT dst, object_types FROM foreign_import") == [
        ("/old/dst", "cell")], "an unclaimable record is left exactly as it was"
    # The record that *can* carry the claim has it withdrawn.
    written, note = _query(
        db_copy, "SELECT canonical_table_written, canonical_table_note "
                 "FROM foreign_import")[0]
    assert written == 0
    assert "released back to spaCR" in note


def test_a_column_an_index_names_is_kept_and_the_release_still_stands(db_copy):
    """SQLite refuses ``DROP COLUMN`` for an indexed column.

    The tidy-up runs *after* the transaction that removed the rows, and a
    tidier schema is not worth undoing a release that has already happened
    -- so the refusal stops the tidying and nothing else.
    """
    _run(db_copy, "DROP TABLE foreign_columns", "DROP TABLE foreign_import",
         'CREATE INDEX idx_keep ON cell(foreign_areashape_area)')

    removed = fg.release_canonical_copy(db_copy, "cell")

    assert removed == 4
    assert _query(db_copy, "SELECT COUNT(*) FROM cell")[0][0] == 0
    assert "foreign_areashape_area" in _columns(db_copy, "cell"), (
        "the indexed column could not be dropped, and that is not an error")
    assert _query(db_copy, "SELECT COUNT(*) FROM foreign_cell")[0][0] == 4


def test_a_column_spacr_also_writes_is_never_dropped_from_under_it(tmp_path):
    """The tidy-up drops an imported column only when it is provably empty.

    Their ``AreaShape_Area`` is mapped onto spaCR's own ``cell_area`` here
    -- which is what ``allow_spacr_targets`` is for -- and spaCR has
    measured a third field into the same table. Releasing the import's
    rows leaves ``cell_area`` holding spaCR's values, so dropping it
    because the importer once wrote it would delete measurements nobody
    asked to release.
    """
    from spacr.utils import _merge_and_save_to_database

    data = Theirs(tmp_path)
    plan = data.plan(column_maps=[
        fg.ColumnMap(source="AreaShape_Area", target="cell_area"),
        fg.ColumnMap(source="Metadata_Treatment",
                     target="foreign_metadata_treatment")],
        allow_spacr_targets=True)
    dst = str(tmp_path / "imported")
    db_path = fg.run_import(plan, dst).db_path

    # A field of spaCR's own, written by the function measure_crop uses.
    _merge_and_save_to_database(
        pd.DataFrame({"label": [1, 2], "cell_area": [11.0, 22.0]}),
        pd.DataFrame({"label": [1, 2],
                      "cell_channel_0_mean_intensity": [3.0, 4.0]}),
        "cell", dst, "plate1_A01_9", "spacr_run")
    assert _query(db_path, "SELECT COUNT(*) FROM cell")[0][0] == 6

    removed = fg.release_canonical_copy(db_path, "cell")

    assert removed == 4
    assert _query(db_path, "SELECT prcf, cell_area FROM cell") == [
        ("plate1_r1_c1_f9", 11.0), ("plate1_r1_c1_f9", 22.0)]
    assert "cell_area" in _columns(db_path, "cell"), (
        "spaCR's own values are in it")
    assert "foreign_metadata_treatment" not in _columns(db_path, "cell"), (
        "an imported column left empty by the release does go")


# --------------------------------------------------------------------------- #
#  Merging a conversion map into a destination that already has one
# --------------------------------------------------------------------------- #

def test_a_merge_that_cannot_be_applied_leaves_the_map_that_was_there(
        imported, tmp_path):
    """The destination's provenance back to its own filenames is the point.

    A conversion map the destination cannot take -- here because a unique
    index on ``source`` says one original file may appear once, and the
    incoming map converted the same originals under different target names
    -- must leave the rows that were already there. It is written under one
    transaction for that reason.

    The staging table it leaves behind is cleared by the next merge, which
    is the other half of the same recovery: the second call below is the
    same call, after the index has been removed.
    """
    result, _plan, _dst = imported
    original = str(tmp_path / "already_converted.db")
    cv.populate_db_from_map(original, result.conversion.map_path)
    held = _query(original, "SELECT target FROM conversion_map")
    assert len(held) == 4

    incoming = pd.read_csv(result.conversion.map_path)
    incoming["target"] = [str(t).replace("plate1", "plateX")
                          for t in incoming["target"]]
    incoming_path = str(tmp_path / "incoming_map.csv")
    incoming.to_csv(incoming_path, index=False)

    _run(original, "CREATE UNIQUE INDEX uq_source ON conversion_map(source)")

    with pytest.raises(sqlite3.IntegrityError):
        fg._populate_conversion_map(original, incoming_path)

    assert _query(original, "SELECT target FROM conversion_map") == held
    assert fg._CONVERSION_STAGING in _tables(original), (
        "the failed merge is what leaves a staging table behind")

    # The same merge, once the constraint that refused it is gone.
    _run(original, "DROP INDEX uq_source")
    fg._populate_conversion_map(original, incoming_path)

    targets = {row[0] for row in _query(original,
                                        "SELECT target FROM conversion_map")}
    assert len(targets) == 8, "both generations of names are joinable"
    assert any(t.startswith("plateX") for t in targets)
    assert fg._CONVERSION_STAGING not in _tables(original), (
        "the stale staging table was cleared before the merge, not left")


# --------------------------------------------------------------------------- #
#  Provenance is a merge, not a replace
# --------------------------------------------------------------------------- #

def test_provenance_named_no_table_rewrites_the_run_and_no_column(imported,
                                                                  db_copy):
    """``_write_provenance`` deletes only the tables it is told about.

    ``run_import`` always names at least one -- it records a row count for
    the foreign table before it calls this -- so the empty case is reachable
    only by calling the function, which is why it is called directly here.
    It is the case that says the write is a *merge*: the run record is
    rewritten and no other import's column provenance is disturbed, which is
    the property ``_importer_owns`` later depends on.
    """
    _result, plan, dst = imported
    columns_sql = ('SELECT "table", "column", status FROM foreign_columns '
                   'ORDER BY 1, 2')
    before = _query(db_copy, columns_sql)
    assert before, "the import recorded its columns"
    assert _query(db_copy,
                  "SELECT canonical_table_written FROM foreign_import") == [(1,)]

    fg._write_provenance(db_copy, plan, dst, (), mode="preserve")

    assert _query(db_copy, columns_sql) == before, (
        "no table was named, so no column record may change")
    written, note = _query(
        db_copy, "SELECT canonical_table_written, canonical_table_note "
                 "FROM foreign_import")[0]
    assert written == 0, "the run record itself is still rewritten"
    assert "left as it was found" in note

    # Naming the tables replaces their rows rather than adding to them.
    fg._write_provenance(db_copy, plan, dst, ("cell", "foreign_cell"))
    assert _query(db_copy, columns_sql) == before
    assert _query(db_copy,
                  "SELECT canonical_table_written FROM foreign_import") == [(1,)]


# --------------------------------------------------------------------------- #
#  Proved unreachable
# --------------------------------------------------------------------------- #

def test_a_staged_conversion_map_always_carries_a_target_column(imported,
                                                                tmp_path):
    """Why ``if 'target' in shared`` in ``_populate_conversion_map`` cannot
    be false.

    The staging table is written by
    :func:`spacr.convert.populate_db_from_map`, which reads the CSV through
    :func:`spacr.convert.read_map`; that function raises
    ``ConfigurationError`` unless every name in
    ``convert._REQUIRED_MAP_COLUMNS`` is a column of the file, and
    ``'target'`` is one of them. The staging table is that frame, so
    ``incoming`` always holds ``'target'``; the loop above adds every
    incoming column missing from ``held``, so ``shared`` is ``incoming``.

    This test pins the guarantee rather than the guard: a map file without
    a ``target`` column never reaches the merge at all.
    """
    result, _plan, _dst = imported
    frame = pd.read_csv(result.conversion.map_path).drop(columns=["target"])
    headless = str(tmp_path / "no_target.csv")
    frame.to_csv(headless, index=False)

    destination = str(tmp_path / "measurements.db")
    cv.populate_db_from_map(destination, result.conversion.map_path)

    with pytest.raises(Exception) as raised:
        fg._populate_conversion_map(destination, headless)
    assert "target" in str(raised.value)
    assert "target" in _columns(destination, cv.CONVERSION_TABLE), (
        "the destination still has the map it had")
