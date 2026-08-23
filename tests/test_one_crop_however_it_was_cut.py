"""Three ways to cut one object, and they have to agree about everything.

    "with the streaming you need to make sure that all methods generate the
    same single crop images. so the coordinate table (which also must use
    the cell_id or object_number to name the png it generates), object id in
    array method and png crop during measure_crop should all generate the
    exact same files named with the same object number"

The three producers:

* ``measure_crop`` exports a PNG per object while it measures;
* the ARRAY method scans a merged field's label plane for the object;
* the COLUMN method reads the object's bounding box out of the measurement
  table and cuts that.

They must give the same pixels under the same filename, or a model trained
through one is not comparable with a model trained through another and
nobody can tell from the folder which they have.

WHAT WAS WRONG. The coordinate method takes its object id from the object
table's own column, and ``png_list`` -- the crop table, which is the one a
selection is usually built from -- spells it ``cell_id = 'o2'`` where the
measurement tables spell it ``object_label = 2``. The dataset writer parsed
it as ``int(float(row[...]))``, which raises on ``'o2'``, and the exception
counted the object as MISSING and moved on. So the coordinate method
silently produced an empty dataset from the very table it was meant to read,
while the array method -- whose labels come off a mask plane as integers --
worked. Both go through one parser now.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.crops import object_label


PLATE1_DB = os.environ.get("SPACR_PLATE1_DB", "")
STREAM_ROOT = os.environ.get("SPACR_STREAM_ROOT", "")


# ---------------------------------------------------------------------------
# one parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (2, 2), ("2", 2), ("o2", 2), ("o13", 13), (" o7 ", 7), (np.int64(4), 4),
])
def test_every_spelling_of_an_object_id_parses_the_same(value, expected):
    assert object_label(value) == expected


def test_the_writer_uses_that_parser_and_not_int_of_float():
    """`int(float('o2'))` is what dropped every object of a png_list table."""
    import inspect

    from spacr import stream_dataset

    source = inspect.getsource(stream_dataset)
    assert "object_label(row[OBJECT_KEY])" in source
    assert "int(float(row[OBJECT_KEY]))" not in source


def test_a_selection_from_the_crop_table_keeps_its_object_ids():
    from spacr.stream_dataset import selection_from_objects

    crop_table = pd.DataFrame({
        "plateID": ["plate1"] * 3, "rowID": ["r5"] * 3,
        "columnID": ["c1"] * 3, "fieldID": ["f17"] * 3,
        "cell_id": ["o2", "o13", "o27"],
    })

    selection = selection_from_objects(crop_table, object_array="cell")

    assert list(selection["objectID"]) == ["o2", "o13", "o27"]
    # And every one of them parses, which is what the writer needs.
    assert [object_label(v) for v in selection["objectID"]] == [2, 13, 27]


# ---------------------------------------------------------------------------
# one name
# ---------------------------------------------------------------------------

def test_a_streamed_crop_is_named_like_the_exported_one():
    from spacr.stream_dataset import crop_name

    assert crop_name("plate1_E01_17_1", 2) == "plate1_E01_17_1_2.png"
    assert crop_name("plate1_E01_17_1", object_label("o13")) == \
        "plate1_E01_17_1_13.png"


def test_the_name_carries_the_object_number_not_the_prcfo_spelling():
    """'o2' in a filename would not match anything measure_crop wrote."""
    from spacr.stream_dataset import crop_name

    name = crop_name("plate1_E01_17_1", object_label("o2"))
    assert name.endswith("_2.png")
    assert "o2" not in name


# ---------------------------------------------------------------------------
# real data: same pixels, same names
# ---------------------------------------------------------------------------

needs_project = pytest.mark.skipif(
    not STREAM_ROOT or not os.path.isdir(STREAM_ROOT),
    reason="set SPACR_STREAM_ROOT to a project with merged/, data/ and "
           "measurements/")


@needs_project
def test_all_three_methods_cut_the_same_pixels():
    """The load-bearing claim, on a real plate."""
    from spacr.crops import read_crop_png, resolve_crop_source

    database = os.path.join(STREAM_ROOT, "measurements", "measurements.db")
    with sqlite3.connect(database) as db:
        rows = pd.read_sql("SELECT * FROM png_list LIMIT 6", db).to_dict("records")
    assert rows, "the crop table is empty, so this proves nothing"

    source = resolve_crop_source(STREAM_ROOT, prefer="merged")
    for row in rows:
        exported_path = os.path.join(
            STREAM_ROOT, "data", *row["png_path"].split("/data/")[1].split("/"))
        exported = read_crop_png(exported_path)

        # ARRAY: no bbox on the row, so the label plane is scanned.
        by_array = source.get(dict(row))

        # COLUMN: the bounding box a measured database carries, in the
        # skimage regionprops order the reader expects.
        spec = source.spec_for(dict(row))
        plane = np.load(spec.merged_path)[..., 4]
        ys, xs = np.where(plane == object_label(row["cell_id"]))
        with_box = dict(row)
        for index, value in enumerate((ys.min(), xs.min(),
                                       ys.max() + 1, xs.max() + 1)):
            with_box[f"bbox-{index}"] = int(value)
        by_column = source.get(with_box)

        assert np.array_equal(by_array, exported), (
            f"the array method differs from the exported PNG for "
            f"{row['cell_id']}")
        assert np.array_equal(by_column, exported), (
            f"the column method differs from the exported PNG for "
            f"{row['cell_id']}")


@needs_project
def test_the_streamed_name_is_the_exported_name():
    from spacr.stream_dataset import crop_name

    database = os.path.join(STREAM_ROOT, "measurements", "measurements.db")
    with sqlite3.connect(database) as db:
        rows = pd.read_sql("SELECT * FROM png_list LIMIT 6", db).to_dict("records")

    for row in rows:
        exported = os.path.basename(row["png_path"])
        stem = os.path.splitext(exported)[0].rsplit("_", 1)[0]
        assert crop_name(stem, object_label(row["cell_id"])) == exported
