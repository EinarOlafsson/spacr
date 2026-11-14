"""``spacr.foreign``: the states a real destination is in, not the clean one.

Everything here is a path an import reaches on somebody's actual data and
that the happy-path tests do not:

* a measurement row whose image key names a *folder*, and a table whose keys
  are declared rather than guessed;
* ``verify_labels=False`` — the fast plan, which pairs masks without reading
  a single one — and the capped examples list that keeps a report readable
  when six thousand rows carry a bad label;
* the run summary without a conversion, crops cut from the asked channels,
  and provenance writes that merge rather than replace.
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


def _query(db_path, sql, *params):
    connection = sqlite3.connect(db_path)
    try:
        return connection.execute(sql, params).fetchall()
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
