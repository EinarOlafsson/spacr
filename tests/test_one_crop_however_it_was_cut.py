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
# a measured project: same pixels, same names
# ---------------------------------------------------------------------------

def _disks(shape, n, radius=11):
    mask = np.zeros(shape, dtype=np.uint16)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    per_row = int(np.ceil(np.sqrt(n)))
    step_y, step_x = shape[0] // per_row, shape[1] // per_row
    for label in range(1, n + 1):
        i, j = divmod(label - 1, per_row)
        cy, cx = i * step_y + step_y // 2, j * step_x + step_x // 2
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = label
    return mask


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """merged/, measurements/ and data/ written by measure's own crop core.

    The run's settings go into the database as measure_crop leaves them, which
    is how the merged source learns the crop size and normalisation used.
    """
    from spacr.io import _save_settings_to_db
    from spacr.measure import _measure_crop_core
    from spacr.settings import get_measure_crop_settings

    root = str(tmp_path_factory.mktemp("one_crop"))
    merged = os.path.join(root, "merged")
    os.makedirs(merged)
    os.makedirs(os.path.join(root, "measurements"))
    rng = np.random.default_rng(0)
    cell = _disks((120, 120), 9)
    nucleus = np.where(_disks((120, 120), 9, radius=4) > 0, cell, 0)
    channels = []
    for c in range(4):
        base = rng.integers(50, 200, size=cell.shape).astype(np.uint16)
        base[cell > 0] += 900 * (c + 1)
        channels.append(base)
    name = "plate1_E01_17.npy"
    np.save(os.path.join(merged, name),
            np.stack(channels + [cell, nucleus, np.zeros_like(cell)],
                     axis=-1).astype(np.uint16))

    settings = get_measure_crop_settings(settings={})
    settings.update({
        "src": merged, "channels": [0, 1, 2, 3],
        "cell_mask_dim": 4, "nucleus_mask_dim": 5, "pathogen_mask_dim": 6,
        "png_dims": [0, 1, 2], "png_size": [40, 40],
        "save_measurements": True, "save_png": True, "save_arrays": False,
        "plot": False, "verbose": False, "timelapse": False,
        "crop_mode": ["cell"], "normalize": [1, 99], "normalize_by": "png",
        "experiment": "exp", "n_jobs": 1, "test_mode": False,
        "cytoplasm": True, "cell_min_size": 1, "nucleus_min_size": 1,
        "pathogen_min_size": 1, "cytoplasm_min_size": 1,
    })
    _measure_crop_core(0, [], name, settings)
    _save_settings_to_db(settings)
    return root


def _crop_rows(project):
    database = os.path.join(project, "measurements", "measurements.db")
    with sqlite3.connect(database) as db:
        rows = pd.read_sql("SELECT * FROM png_list LIMIT 6", db).to_dict("records")
    assert rows, "the crop table is empty, so this proves nothing"
    return rows


def test_all_three_methods_cut_the_same_pixels(project):
    """The load-bearing claim: export, array scan and bounding box agree."""
    from spacr import crops
    from spacr.crops import read_crop_png, resolve_crop_source

    crops.clear_field_cache()
    rows = _crop_rows(project)
    assert all(str(row["cell_id"]).startswith("o") for row in rows), (
        "png_list must spell its objects the prcfo way, 'o2'")

    source = resolve_crop_source(project, prefer="merged")
    for row in rows:
        exported = read_crop_png(row["png_path"])

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


def test_the_streamed_name_is_the_exported_name(project):
    from spacr.stream_dataset import crop_name

    for row in _crop_rows(project):
        exported = os.path.basename(row["png_path"])
        assert os.path.isfile(row["png_path"])
        stem = os.path.splitext(exported)[0].rsplit("_", 1)[0]
        assert crop_name(stem, object_label(row["cell_id"])) == exported
