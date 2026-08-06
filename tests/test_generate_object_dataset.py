"""Tests for measure.generate_object_dataset — cropping objects out of the
merged image+mask arrays by measurement/metadata criteria."""
from __future__ import annotations

import os
import sqlite3

import numpy as np
from PIL import Image


def _build_dataset(root):
    """A tiny experiment: one merged .npy with 3 image channels + a cell mask
    slice holding three objects of different sizes, and a measurements.db whose
    `cell` table describes them."""
    merged = os.path.join(root, "merged")
    meas = os.path.join(root, "measurements")
    os.makedirs(merged); os.makedirs(meas)

    H = W = 64
    # 3 image channels (0,1,2) then a cell mask at slice 3.
    arr = np.zeros((H, W, 4), dtype=np.float32)
    arr[..., 0] = 10.0            # ch0 constant
    arr[..., 1] = 20.0            # ch1 constant
    arr[..., 2] = 30.0            # ch2 constant
    mask = np.zeros((H, W), dtype=np.int32)
    mask[2:6, 2:6] = 1           # 16 px
    mask[10:30, 10:30] = 2       # 400 px
    mask[40:62, 40:62] = 3       # 484 px
    arr[..., 3] = mask
    npy = os.path.join(merged, "plate1_r1_c1_f1.npy")
    np.save(npy, arr)

    db = os.path.join(meas, "measurements.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE cell (object_label INT, path_name TEXT, "
                "plateID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT, "
                "cell_area REAL)")
    rows = [
        (1, npy, "plate1", "r1", "c1", "f1", 16.0),
        (2, npy, "plate1", "r1", "c1", "f1", 400.0),
        (3, npy, "plate1", "r1", "c2", "f1", 484.0),
    ]
    con.executemany("INSERT INTO cell VALUES (?,?,?,?,?,?,?)", rows)
    con.commit(); con.close()
    return root


def test_min_area_filters_objects(tmp_path):
    from spacr.measure import generate_object_dataset
    root = _build_dataset(str(tmp_path))
    man = generate_object_dataset(
        root, object_type="cell", channels=(0, 1, 2),
        min_area=100, mask_dims={"cell": 3}, verbose=False)
    # Only objects 2 (400) and 3 (484) survive.
    labels = sorted(e["object_label"] for e in man)
    assert labels == [2, 3]
    for e in man:
        assert os.path.isfile(e["png_path"])
        assert Image.open(e["png_path"]).size == (128, 128)


def test_column_filter(tmp_path):
    from spacr.measure import generate_object_dataset
    root = _build_dataset(str(tmp_path))
    man = generate_object_dataset(
        root, object_type="cell", channels=(0, 1, 2),
        columns=[2], mask_dims={"cell": 3}, verbose=False)
    assert [e["object_label"] for e in man] == [3]
    assert man[0]["columnID"] == "c2"


def test_channel_selection_builds_rgb(tmp_path):
    from spacr.measure import generate_object_dataset
    root = _build_dataset(str(tmp_path))
    man = generate_object_dataset(
        root, object_type="cell", channels=(0, 1, 2), min_area=100,
        mask_dims={"cell": 3}, mask_background=True, normalize=False,
        return_arrays=True, save_png=False, verbose=False)
    arr = man[0]["array"]
    assert arr.shape[2] == 3          # RGB
    # background is zeroed; object pixels carry the channel constants (10,20,30)
    assert arr.max() > 0


def test_combined_criteria_and_where(tmp_path):
    from spacr.measure import generate_object_dataset
    root = _build_dataset(str(tmp_path))
    man = generate_object_dataset(
        root, object_type="cell", channels=(0, 1),
        criteria={"cell_area": (">", 450)},
        where="columnID = 'c2'",
        mask_dims={"cell": 3}, verbose=False)
    assert [e["object_label"] for e in man] == [3]
    # two channels → padded to RGB PNG
    assert Image.open(man[0]["png_path"]).mode in ("RGB", "RGBA")


def test_missing_db_raises(tmp_path):
    from spacr.measure import generate_object_dataset
    import pytest
    with pytest.raises(FileNotFoundError):
        generate_object_dataset(str(tmp_path), verbose=False)


def test_max_area_rows_fields_plates_filters(tmp_path):
    """Exercise the max_area + rowID/fieldID/plateID IN clause builders."""
    from spacr.measure import generate_object_dataset
    root = _build_dataset(str(tmp_path))
    man = generate_object_dataset(
        root, object_type="cell", channels=(0, 1, 2),
        max_area=450, rows=[1], fields=[1], plates=["plate1"],
        mask_dims={"cell": 3}, verbose=True)
    # max_area=450 keeps 16 and 400 px objects (labels 1, 2).
    assert sorted(e["object_label"] for e in man) == [1, 2]


def test_criteria_in_operator_and_limit(tmp_path):
    """criteria with an 'in' operator + a LIMIT clause."""
    from spacr.measure import generate_object_dataset
    root = _build_dataset(str(tmp_path))
    man = generate_object_dataset(
        root, object_type="cell", channels=(0,),
        criteria={"columnID": ("in", ["c1", "c2"])},
        limit=1, mask_dims={"cell": 3}, verbose=True)
    assert len(man) == 1                       # LIMIT 1
    assert Image.open(man[0]["png_path"]).mode in ("RGB", "L", "RGBA")


def test_unknown_object_type_raises(tmp_path):
    from spacr.measure import generate_object_dataset
    import pytest
    root = _build_dataset(str(tmp_path))
    with pytest.raises(ValueError):
        generate_object_dataset(root, object_type="mitochondrion",
                                mask_dims={"cell": 3}, verbose=False)


# ---------------------------------------------------------------------------
# crop_objects_from_array channel/area branches
# ---------------------------------------------------------------------------

def _array_with_mask(n_img_channels):
    H = W = 40
    total = n_img_channels + 1
    data = np.zeros((H, W, total), dtype=np.float32)
    for c in range(n_img_channels):
        data[..., c] = 10.0 * (c + 1)
    mask = np.zeros((H, W), dtype=np.int32)
    mask[2:6, 2:6] = 1            # 16 px  (small)
    mask[10:34, 10:34] = 2        # 576 px (large)
    data[..., n_img_channels] = mask
    return data, n_img_channels    # mask_dim index


def test_crop_objects_single_channel_repeats_to_rgb():
    from spacr.measure import crop_objects_from_array
    data, mdim = _array_with_mask(1)
    out = crop_objects_from_array(data, mask_dim=mdim, channels=(0,))
    assert out and out[0]["crop"].shape[2] == 3     # 1→repeat


def test_crop_objects_two_channels_pad_to_rgb():
    from spacr.measure import crop_objects_from_array
    data, mdim = _array_with_mask(2)
    out = crop_objects_from_array(data, mask_dim=mdim, channels=(0, 1))
    assert out[0]["crop"].shape[2] == 3             # 2→pad blue


def test_crop_objects_more_than_three_channels_truncated():
    from spacr.measure import crop_objects_from_array
    data, mdim = _array_with_mask(4)
    out = crop_objects_from_array(data, mask_dim=mdim, channels=(0, 1, 2, 3))
    assert out[0]["crop"].shape[2] == 3             # >3→first 3


def test_crop_objects_area_filter_and_limit():
    from spacr.measure import crop_objects_from_array
    data, mdim = _array_with_mask(3)
    # min_area drops the 16px object; max_area drops nothing; limit caps.
    out = crop_objects_from_array(data, mask_dim=mdim, channels=(0, 1, 2),
                                  min_area=100, max_area=10000, limit=5)
    assert all(o["area"] >= 100 for o in out)
    # largest-first ordering
    areas = [o["area"] for o in out]
    assert areas == sorted(areas, reverse=True)


# ---------------------------------------------------------------------------
# Crop format marker
# ---------------------------------------------------------------------------

def test_the_output_folder_is_marked_so_readers_do_not_reverse_it(tmp_path):
    """The object-dataset folder must carry the crop-format sidecar.

    What this defends: `_save_object_crop` writes through PIL, which is
    already RGB, so the bytes it puts on disk are correct and there is
    nothing to reverse. But an *unmarked* folder means LEGACY to every
    spaCR reader (crops.read_crop_png's documented precedence), so
    read_crop_png reversed a correct file on load. The annotator, the crop
    grid and the training loaders showed channel 0 as blue and channel 2 as
    red while an external image viewer showed them the right way round --
    the two disagreeing about the same file is the symptom.

    Asserted end to end rather than on the sidecar alone, because the claim
    that matters is "PIL and read_crop_png agree", not "a JSON file exists".
    """
    from spacr.crops import CROP_FORMAT_SIDECAR, read_crop_png
    from spacr.measure import generate_object_dataset

    root = _build_dataset(str(tmp_path))
    man = generate_object_dataset(
        root, object_type="cell", channels=(0, 1, 2), min_area=100,
        mask_dims={"cell": 3}, normalize=False, mask_background=False,
        verbose=False)
    assert man, "fixture produced no crops"

    out_dir = os.path.dirname(man[0]["png_path"])
    assert os.path.isfile(os.path.join(out_dir, CROP_FORMAT_SIDECAR)), (
        "the object-dataset folder is unmarked, so every reader will treat "
        "these format-2 crops as legacy and reverse them")

    for entry in man:
        path = entry["png_path"]
        direct = np.array(Image.open(path).convert("RGB"))
        through = read_crop_png(path)
        assert np.array_equal(direct, through), (
            "read_crop_png disagrees with a plain PIL read of the same file")
        # ch0 (10.0) < ch1 (20.0) < ch2 (30.0) in the fixture: the ordering
        # survives, so the array was not reversed on the way out either.
        means = [float(direct[..., i].mean()) for i in range(3)]
        assert means[0] < means[1] < means[2], means
