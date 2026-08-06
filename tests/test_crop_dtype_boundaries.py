"""The crop pipeline's dtype contract.

Two rules, and every test here is one of them:

1. **The working dtype survives the pipeline.** A ``uint16`` merged array
   gives ``uint16`` crops -- in ``generate_object_dataset``'s manifest, in the
   ``.npy`` it writes for a >3-channel crop, and out of
   ``crop_objects_from_array(to_rgb=False)``. Normalisation stretches into
   that dtype's range, not into 0-255.

2. **8 bit is a declared boundary, and it rescales.** PNG export
   (``_save_object_crop``) and RGB assembly for a GUI
   (``crop_objects_from_array(to_rgb=True)``) narrow through
   ``_crop_to_uint8``, which is linear and maps 0 to 0.

What this replaces: ``np.clip(crop, 0, 255).astype(np.uint8)`` at the end of
both crop bodies. On a raw ``uint16`` field with ``normalize=False`` that
clipped every pixel brighter than 255 -- i.e. every pixel of the object -- to
exactly 255. Measured on the field built below: **1728 of 5808 crop pixels
(29.75%) came back at exactly 255**, and the PNG written to disk was 26.64%
pure white. Unnormalised 16-bit data displayed at 8 bit has to look dark; a
clip is what turned it white, and those crops were being written into training
datasets with nothing anywhere reporting it.

The measurements database here is built by the real writer -- a full
``_measure_crop_core`` run -- not by hand. That is what exposed the second bug
covered below: the ``path_name`` a real run records does not exist on disk, so
``generate_object_dataset`` skipped every object of every real dataset while
every hand-built test database passed.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest
from PIL import Image

from spacr.settings import get_measure_crop_settings


# ---------------------------------------------------------------------------
# A synthetic 16-bit field, and a real measure run over it
# ---------------------------------------------------------------------------

#: Intensity of the field background, in 16-bit counts.
BACKGROUND = (150, 400)
#: Intensity added inside every cell, per channel.
SIGNAL = 3000


def merged_uint16(size=96, n_channels=4):
    """A merged ``(H, W, C)`` field: ``n_channels`` uint16 intensity planes
    then cell / nucleus / pathogen label planes, at the dims measure_crop
    expects (cell=n, nucleus=n+1, pathogen=n+2)."""
    rng = np.random.default_rng(0)
    cell = np.zeros((size, size), np.uint16)
    nucleus = np.zeros((size, size), np.uint16)
    pathogen = np.zeros((size, size), np.uint16)
    yy, xx = np.mgrid[:size, :size]
    for i, (cy, cx) in enumerate([(28, 28), (28, 68), (68, 28)], start=1):
        cell[(yy - cy) ** 2 + (xx - cx) ** 2 <= 15 ** 2] = i
        nucleus[(yy - cy) ** 2 + (xx - cx) ** 2 <= 6 ** 2] = i
        pathogen[(yy - cy - 8) ** 2 + (xx - cx - 8) ** 2 <= 3 ** 2] = i
    planes = []
    for c in range(n_channels):
        base = rng.integers(*BACKGROUND, size=(size, size)).astype(np.uint16)
        base[cell > 0] += np.uint16(SIGNAL * (c + 1))
        planes.append(base)
    return np.stack(planes + [cell, nucleus, pathogen], axis=-1).astype(np.uint16)


def _measure_settings(merged_dir, **over):
    s = get_measure_crop_settings(settings={})
    s.update({
        "src": str(merged_dir), "channels": [0, 1, 2, 3],
        "cell_mask_dim": 4, "nucleus_mask_dim": 5, "pathogen_mask_dim": 6,
        "png_dims": [0, 1, 2], "png_size": [32, 32],
        "save_measurements": True, "save_png": True, "save_arrays": False,
        "plot": False, "verbose": False, "timelapse": False,
        "crop_mode": ["cell"], "normalize": [1, 99], "normalize_by": "png",
        "experiment": "exp", "n_jobs": 1, "test_mode": False,
        "cytoplasm": True, "cell_min_size": 1, "nucleus_min_size": 1,
        "pathogen_min_size": 1, "cytoplasm_min_size": 1,
    })
    s.update(over)
    return s


@pytest.fixture
def real_run(tmp_path):
    """Run ``_measure_crop_core`` over one synthetic field.

    Returns ``(root, merged_npy_path, data)``. The measurements.db is written
    by ``spacr.utils._merge_and_save_to_database`` through the real code path,
    so its ``path_name`` column holds whatever a real run holds.
    """
    from spacr.measure import _measure_crop_core

    root = tmp_path / "plate1"
    merged = root / "merged"
    merged.mkdir(parents=True)
    (root / "measurements").mkdir(parents=True)
    data = merged_uint16()
    name = "plate1_A01_F001.npy"
    np.save(merged / name, data)
    index, _, cells, _ = _measure_crop_core(0, [], name, _measure_settings(merged))
    assert index == 0 and np.max(cells) >= 1
    assert (root / "measurements" / "measurements.db").is_file()
    return str(root), str(merged / name), data


# ---------------------------------------------------------------------------
# The reported defect, end to end on a real database
# ---------------------------------------------------------------------------

def test_a_real_run_records_a_path_generate_object_dataset_can_find(real_run):
    """``path_name`` from a real run points one folder above the array.

    ``_merge_and_save_to_database`` writes ``os.path.join(source_folder,
    file_name + '.npy')`` and ``source_folder`` is the parent of ``merged/``,
    so ``os.path.isfile(path_name)`` is False for every row spaCR has ever
    written. ``generate_object_dataset`` used to test that path directly and
    skip: 3 objects selected, 0 crops produced, on every real dataset.
    """
    from spacr.measure import generate_object_dataset

    root, npy, _ = real_run
    con = sqlite3.connect(os.path.join(root, "measurements", "measurements.db"))
    recorded = [r[0] for r in con.execute("SELECT path_name FROM cell")]
    con.close()
    assert recorded, "the real writer produced no cell rows"
    # The premise: the recorded path really is not the array's path.
    assert not any(os.path.isfile(p) for p in recorded)
    assert all(os.path.basename(p) == os.path.basename(npy) for p in recorded)

    man = generate_object_dataset(root, object_type="cell", channels=(0, 1, 2),
                                  save_png=False, return_arrays=True,
                                  verbose=False)
    assert len(man) == len(recorded)


def test_unnormalised_16bit_crops_are_dark_not_saturated(real_run):
    """The defect itself: no pixel is clipped to 255, and the crop is dark."""
    from spacr.measure import generate_object_dataset

    root, _, data = real_run
    man = generate_object_dataset(root, object_type="cell", channels=(0, 1, 2),
                                  normalize=False, return_arrays=True,
                                  save_png=False, verbose=False)
    arr = man[0]["array"]
    assert arr.dtype == np.uint16                     # working dtype survives
    assert arr.max() > 255                            # still real 16-bit counts

    from spacr.measure import _crop_to_uint8
    eight = _crop_to_uint8(arr)
    assert eight.dtype == np.uint8
    assert int((eight == 255).sum()) == 0             # was: every object pixel
    assert eight.max() == arr.max() // 256            # linear, high byte
    # "dark, not blown out": the brightest pixel of a 3000-12000 count object
    # is well under half of the 8-bit range.
    assert eight.max() < 128


def test_the_png_written_to_disk_is_not_a_white_silhouette(real_run):
    """Not preview-only: ``save_png=True`` writes the crop, and it used to be
    a solid white blob on every object."""
    from spacr.measure import generate_object_dataset

    root, _, _ = real_run
    out = os.path.join(root, "ds_raw")
    man = generate_object_dataset(root, object_type="cell", channels=(0, 1, 2),
                                  normalize=False, save_png=True,
                                  return_arrays=True, output_dir=out,
                                  png_size=(48, 48), verbose=False)
    png = np.array(Image.open(man[0]["png_path"]))
    assert png.dtype == np.uint8
    assert int((png == 255).sum()) == 0
    assert png.max() > 0                              # not black either
    assert 0 < png.mean() < 64


def test_normalising_stretches_into_the_dtype_not_into_255(real_run):
    """``normalize=True`` fills the *working dtype's* range, so the 8-bit view
    still spans 0-255 while the array keeps its 16 bits."""
    from spacr.measure import generate_object_dataset

    root, _, _ = real_run
    out = os.path.join(root, "ds_norm")
    man = generate_object_dataset(root, object_type="cell", channels=(0, 1, 2),
                                  normalize=True, percentiles=(1, 99),
                                  save_png=True, return_arrays=True,
                                  output_dir=out, png_size=(48, 48),
                                  verbose=False)
    arr = man[0]["array"]
    assert arr.dtype == np.uint16
    assert arr.max() == 65535                         # full 16-bit range
    png = np.array(Image.open(man[0]["png_path"]))
    assert png.max() == 255                           # full 8-bit range
    # The top percentile is clipped on purpose, but only a sliver of the crop.
    assert int((png == 255).sum()) < 0.02 * png.size


def test_more_than_three_channels_keeps_full_depth_in_the_npy(real_run):
    """The ``.npy`` is data, not a picture: it is not narrowed."""
    from spacr.measure import generate_object_dataset

    root, _, _ = real_run
    out = os.path.join(root, "ds_many")
    man = generate_object_dataset(root, object_type="cell",
                                  channels=(0, 1, 2, 3), normalize=False,
                                  save_png=True, output_dir=out,
                                  png_size=(32, 32), verbose=False)
    npy_path = man[0]["png_path"]
    assert npy_path.endswith(".npy")
    stack = np.load(npy_path)
    assert stack.dtype == np.uint16 and stack.shape[2] == 4
    preview = np.array(Image.open(os.path.splitext(npy_path)[0] + ".png"))
    assert preview.dtype == np.uint8
    assert int((preview == 255).sum()) == 0


def test_verbose_reports_what_it_wrote(real_run, capsys):
    """The verbose summary lines still run."""
    from spacr.measure import generate_object_dataset

    root, _, _ = real_run
    generate_object_dataset(root, object_type="cell", channels=(0,),
                            output_dir=os.path.join(root, "ds_v"),
                            png_size=(16, 16), verbose=True)
    out = capsys.readouterr().out
    assert "objects match" in out and "wrote 3 PNGs" in out


def test_src_may_be_the_merged_folder_itself(real_run):
    """``src`` accepts either the experiment root or its ``merged/`` folder."""
    from spacr.measure import generate_object_dataset

    root, npy, _ = real_run
    man = generate_object_dataset(os.path.dirname(npy), object_type="cell",
                                  channels=(0,), save_png=False,
                                  return_arrays=True, verbose=False)
    assert len(man) == 3 and man[0]["array"].dtype == np.uint16


def test_a_volumetric_merged_array_is_refused(tmp_path):
    """A (Z, Y, X, C) array would be sliced along X without raising."""
    from spacr.measure import generate_object_dataset

    root = tmp_path / "plate1"
    merged = root / "merged"
    merged.mkdir(parents=True)
    (root / "measurements").mkdir(parents=True)
    npy = merged / "plate1_A01_F001.npy"
    np.save(npy, np.zeros((3, 8, 8, 5), np.uint16))
    db = root / "measurements" / "measurements.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE cell (object_label INT, path_name TEXT, "
                "plateID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT)")
    con.execute("INSERT INTO cell VALUES (1,?,'plate1','r1','c1','f1')",
                (str(npy),))
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="2-D merged arrays"):
        generate_object_dataset(str(root), object_type="cell", channels=(0,),
                                mask_dims={"cell": 4}, verbose=False)


def test_mask_background_off_keeps_the_surrounding_field(real_run):
    """``mask_background=False`` leaves the bounding box intact, in dtype."""
    from spacr.measure import generate_object_dataset

    root, _, _ = real_run
    man = generate_object_dataset(root, object_type="cell", channels=(0, 1),
                                  mask_background=False, normalize=True,
                                  save_png=False, return_arrays=True,
                                  verbose=False)
    arr = man[0]["array"]
    assert arr.dtype == np.uint16
    assert arr[0, 0].max() > 0        # corner is background, and it is kept


# ---------------------------------------------------------------------------
# crop_objects_from_array -- the live-preview path
# ---------------------------------------------------------------------------

def test_preview_rgb_is_uint8_and_never_clipped():
    from spacr.measure import crop_objects_from_array

    data = merged_uint16()
    out = crop_objects_from_array(data, mask_dim=4, channels=(0, 1, 2),
                                  normalize=False, to_rgb=True)
    assert out, "no objects cropped"
    crop = out[0]["crop"]
    assert crop.dtype == np.uint8 and crop.shape[2] == 3
    assert int((crop == 255).sum()) == 0
    assert crop.max() > 0


def test_preview_without_rgb_returns_the_merged_values_untouched():
    from spacr.measure import crop_objects_from_array

    data = merged_uint16()
    out = crop_objects_from_array(data, mask_dim=4, channels=(0, 1, 2),
                                  normalize=False, to_rgb=False)
    entry = out[0]
    crop = entry["crop"]
    assert crop.dtype == data.dtype
    y0, y1, x0, x1 = entry["bbox"]
    region = data[y0:y1, x0:x1, 4] == entry["label"]
    expected = np.where(region[:, :, None], data[y0:y1, x0:x1, :3], 0)
    assert np.array_equal(crop, expected)


def test_preview_normalises_into_the_working_dtype():
    from spacr.measure import crop_objects_from_array

    data = merged_uint16()
    out = crop_objects_from_array(data, mask_dim=4, channels=(0, 1, 2),
                                  normalize=True, to_rgb=False)
    crop = out[0]["crop"]
    assert crop.dtype == np.uint16
    assert crop.max() == 65535        # not 255


@pytest.mark.parametrize("channels,expected", [((0,), 3), ((0, 1), 3),
                                               ((0, 1, 2), 3),
                                               ((0, 1, 2, 3), 3)])
def test_preview_rgb_assembly_for_every_channel_count(channels, expected):
    from spacr.measure import crop_objects_from_array

    data = merged_uint16()
    out = crop_objects_from_array(data, mask_dim=4, channels=channels,
                                  normalize=False, to_rgb=True)
    crop = out[0]["crop"]
    assert crop.shape[2] == expected and crop.dtype == np.uint8
    if len(channels) == 2:
        assert crop[:, :, 2].max() == 0      # blue is the pad plane


def test_preview_area_filters_and_limit():
    from spacr.measure import crop_objects_from_array

    data = merged_uint16()
    out = crop_objects_from_array(data, mask_dim=4, channels=(0,),
                                  min_area=10, max_area=10 ** 6, limit=2)
    assert len(out) == 2
    assert [o["area"] for o in out] == sorted((o["area"] for o in out),
                                              reverse=True)
    assert not crop_objects_from_array(data, mask_dim=4, channels=(0,),
                                       min_area=10 ** 6)
    assert not crop_objects_from_array(data, mask_dim=4, channels=(0,),
                                       max_area=1)


def test_a_label_the_database_names_but_the_plane_does_not_hold_is_skipped(
        real_run):
    """``generate_object_dataset``'s "label vanished" guard is real: the label
    comes from the database and the plane from disk, so they can disagree."""
    from spacr.measure import generate_object_dataset

    root, _, _ = real_run
    db = os.path.join(root, "measurements", "measurements.db")
    con = sqlite3.connect(db)
    (path_name,) = con.execute("SELECT path_name FROM cell LIMIT 1").fetchone()
    con.execute("INSERT INTO cell (object_label, path_name, plateID, rowID, "
                "columnID, fieldID) VALUES (9999, ?, 'plate1', 'r1', 'c1', 'f1')",
                (path_name,))
    con.commit()
    con.close()
    man = generate_object_dataset(root, object_type="cell", channels=(0,),
                                  save_png=False, verbose=False)
    assert 9999 not in {e["object_label"] for e in man}
    assert len(man) == 3


# ---------------------------------------------------------------------------
# _crop_to_uint8 -- the declared boundary
# ---------------------------------------------------------------------------

def test_uint8_crosses_the_boundary_unchanged():
    from spacr.measure import _crop_to_uint8

    arr = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert _crop_to_uint8(arr) is arr


@pytest.mark.parametrize("dtype", [np.uint16, np.int32])
def test_wide_integers_take_the_high_byte(dtype):
    from spacr.measure import _crop_to_uint8

    arr = np.array([[[0, 255, 256, 65535]]], dtype=dtype)
    assert np.array_equal(_crop_to_uint8(arr),
                          np.array([[[0, 0, 1, 255]]], dtype=np.uint8))


def test_a_clip_would_have_saturated_where_the_rescale_does_not():
    """The two rules, side by side, on the same 16-bit data."""
    from spacr.measure import _crop_to_uint8

    arr = np.array([[[300, 3000, 12000, 65535]]], dtype=np.uint16)
    clipped = np.clip(arr, 0, 255).astype(np.uint8)     # what it used to do
    assert int((clipped == 255).sum()) == 4             # everything is white
    rescaled = _crop_to_uint8(arr)
    assert int((rescaled == 255).sum()) == 1            # only true full scale
    assert list(rescaled.ravel()) == [1, 11, 46, 255]


def test_normalised_floats_are_multiplied_by_255():
    from spacr.measure import _crop_to_uint8

    arr = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    assert list(_crop_to_uint8(arr).ravel()) == [0, 128, 255]


def test_unnormalised_floats_rescale_off_their_own_maximum():
    """A float array carries no dtype range, so the crop's own is used."""
    from spacr.measure import _crop_to_uint8

    arr = np.array([[[10.0, 20.0, 30.0]]], dtype=np.float32)
    assert list(_crop_to_uint8(arr).ravel()) == [85, 170, 255]


def test_negative_floats_clamp_to_background():
    from spacr.measure import _crop_to_uint8

    arr = np.array([[[-5.0, 0.0, 10.0]]], dtype=np.float32)
    assert list(_crop_to_uint8(arr).ravel()) == [0, 0, 255]


def test_an_empty_float_crop_is_an_empty_uint8_crop():
    from spacr.measure import _crop_to_uint8

    out = _crop_to_uint8(np.zeros((0, 4, 3), dtype=np.float32))
    assert out.dtype == np.uint8 and out.shape == (0, 4, 3)


@pytest.mark.parametrize("arr", [
    np.zeros((2, 2, 1), dtype=np.float32),                       # nothing there
    np.full((2, 2, 1), -1.0, dtype=np.float32),                  # all negative
    np.full((2, 2, 1), np.nan, dtype=np.float32),                # all NaN
])
def test_a_float_crop_with_no_positive_signal_is_black(arr):
    from spacr.measure import _crop_to_uint8

    out = _crop_to_uint8(arr)
    assert out.dtype == np.uint8 and int(out.max()) == 0


def test_a_boolean_region_crosses_as_a_float():
    """bool is not an integer dtype to numpy, so it takes the range-less path."""
    from spacr.measure import _crop_to_uint8

    arr = np.array([[[True, False]]])
    assert list(_crop_to_uint8(arr).ravel()) == [255, 0]


# ---------------------------------------------------------------------------
# _crop_full_scale / _normalize_crop / _crop_channels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype,expected", [
    (np.uint8, 255.0), (np.uint16, 65535.0), (np.int32, 2 ** 31 - 1),
    (np.float32, 1.0), (np.float64, 1.0),
])
def test_full_scale_per_dtype(dtype, expected):
    from spacr.measure import _crop_full_scale

    assert _crop_full_scale(dtype) == pytest.approx(expected)


def test_normalise_keeps_dtype_and_reaches_full_scale():
    from spacr.measure import _normalize_crop

    crop = np.zeros((4, 4, 2), np.uint16)
    crop[1:3, 1:3, 0] = [[1000, 2000], [3000, 4000]]
    crop[1:3, 1:3, 1] = [[10, 20], [30, 40]]
    out = _normalize_crop(crop, (0, 100), mask_background=True)
    assert out.dtype == np.uint16
    for c in range(2):
        assert out[:, :, c].max() == 65535
    # Background stays background.
    assert out[0, 0].max() == 0


def test_normalise_falls_back_to_the_maximum_when_the_percentiles_tie():
    """A flat object has ``hi == lo``; the stretch would divide by zero."""
    from spacr.measure import _normalize_crop

    crop = np.zeros((4, 4, 1), np.uint16)
    crop[1:3, 1:3, 0] = 700
    out = _normalize_crop(crop, (1, 99), mask_background=True)
    assert out[1, 1, 0] == 65535 and out[0, 0, 0] == 0


def test_normalise_leaves_an_empty_channel_alone():
    from spacr.measure import _normalize_crop

    crop = np.zeros((3, 3, 1), np.uint16)
    out = _normalize_crop(crop, (1, 99), mask_background=True)
    assert out.dtype == np.uint16 and int(out.max()) == 0


def test_normalise_without_masking_uses_every_pixel():
    """``mask_background`` decides whether the zeros count as data."""
    from spacr.measure import _normalize_crop

    crop = np.zeros((4, 4, 1), np.uint16)
    crop[1:2, :, 0] = 1000
    crop[2:, :, 0] = 4000
    masked = _normalize_crop(crop, (0, 100), mask_background=True)
    unmasked = _normalize_crop(crop, (0, 100), mask_background=False)
    assert masked.max() == 65535 and unmasked.max() == 65535
    # Masked: the range is 1000-4000, so the dim pixels go to black.
    assert masked[1, 0, 0] == 0
    # Unmasked: the range is 0-4000, so they keep their quarter of it.
    # int(), and it is not cosmetic: pytest.approx computes
    # `abs(expected - actual)`, and with a uint16 actual that subtraction
    # wraps. 16384 vs 16383 came out as 65535 and the test failed on a
    # one-count difference it was explicitly tolerating. Red on the clean
    # tree since it was written.
    assert int(unmasked[1, 0, 0]) == pytest.approx(65535 // 4, abs=2)


def test_normalise_of_a_float_crop_targets_one():
    from spacr.measure import _normalize_crop

    crop = np.zeros((4, 4, 1), np.float32)
    crop[1:3, 1:3, 0] = [[0.1, 0.2], [0.3, 0.4]]
    out = _normalize_crop(crop, (0, 100), mask_background=True)
    assert out.dtype == np.float32
    assert out.max() == pytest.approx(1.0)


def test_normalise_of_an_all_zero_float_channel_stays_zero():
    from spacr.measure import _normalize_crop

    out = _normalize_crop(np.zeros((2, 2, 1), np.float32), (1, 99), False)
    assert out.dtype == np.float32 and float(out.max()) == 0.0


def test_crop_channels_preserves_dtype_with_and_without_a_region():
    from spacr.measure import _crop_channels

    data = merged_uint16(size=32, n_channels=2)
    whole = _crop_channels(data, 0, 8, 0, 8, (0, 1))
    assert whole.dtype == np.uint16
    assert np.array_equal(whole, data[0:8, 0:8, :2])
    region = np.zeros((8, 8), bool)
    region[2:4, 2:4] = True
    masked = _crop_channels(data, 0, 8, 0, 8, (0, 1), region)
    assert masked.dtype == np.uint16
    assert masked[0, 0].max() == 0
    assert np.array_equal(masked[2:4, 2:4], data[2:4, 2:4, :2])


# ---------------------------------------------------------------------------
# _save_object_crop -- the PNG boundary
# ---------------------------------------------------------------------------

def _uint16_crop(n_chan, size=8):
    crop = np.zeros((size, size, n_chan), np.uint16)
    for c in range(n_chan):
        crop[2:6, 2:6, c] = 1000 * (c + 1) + 5000
    return crop


def test_png_export_narrows_by_rescaling(tmp_path):
    from spacr.measure import _save_object_crop

    crop = _uint16_crop(3)
    path = str(tmp_path / "rgb.png")
    assert _save_object_crop(crop, (0, 1, 2), path, (8, 8)) == path
    png = np.array(Image.open(path))
    assert png.dtype == np.uint8
    assert np.array_equal(png, (crop // 256).astype(np.uint8))
    assert int((png == 255).sum()) == 0


def test_png_export_single_channel_is_greyscale(tmp_path):
    from spacr.measure import _save_object_crop

    crop = _uint16_crop(1)
    path = str(tmp_path / "grey.png")
    _save_object_crop(crop, (0,), path, (8, 8))
    img = Image.open(path)
    assert img.mode == "L"
    assert np.array_equal(np.array(img), (crop[:, :, 0] // 256).astype(np.uint8))


def test_png_export_two_channels_pads_blue(tmp_path):
    from spacr.measure import _save_object_crop

    crop = _uint16_crop(2)
    path = str(tmp_path / "two.png")
    _save_object_crop(crop, (0, 1), path, (8, 8))
    png = np.array(Image.open(path))
    assert png.shape[2] == 3 and png[:, :, 2].max() == 0
    assert np.array_equal(png[:, :, :2], (crop // 256).astype(np.uint8))


def test_png_export_of_many_channels_keeps_the_npy_at_full_depth(tmp_path):
    from spacr.measure import _save_object_crop

    crop = _uint16_crop(5)
    path = str(tmp_path / "many.png")
    out = _save_object_crop(crop, tuple(range(5)), path, (8, 8))
    assert out.endswith(".npy")
    stack = np.load(out)
    assert stack.dtype == np.uint16 and np.array_equal(stack, crop)
    preview = np.array(Image.open(path))
    assert preview.dtype == np.uint8
    assert np.array_equal(preview, (crop[:, :, :3] // 256).astype(np.uint8))


# ---------------------------------------------------------------------------
# _resolve_merged_path
# ---------------------------------------------------------------------------

def test_resolve_prefers_the_recorded_path_when_it_exists(tmp_path):
    from spacr.measure import _resolve_merged_path

    real = tmp_path / "here.npy"
    real.write_bytes(b"x")
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "here.npy").write_bytes(b"y")
    assert _resolve_merged_path(str(real), str(merged)) == str(real)


def test_resolve_falls_back_to_the_merged_folder(tmp_path):
    from spacr.measure import _resolve_merged_path

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "field.npy").write_bytes(b"y")
    recorded = str(tmp_path / "field.npy")          # what a real run records
    assert _resolve_merged_path(recorded, str(merged)) == str(merged / "field.npy")


@pytest.mark.parametrize("recorded", [None, "", "/nowhere/field.npy"])
def test_resolve_returns_none_when_nothing_exists(tmp_path, recorded):
    from spacr.measure import _resolve_merged_path

    assert _resolve_merged_path(recorded, str(tmp_path)) is None


@pytest.mark.parametrize("verbose", [True, False])
def test_a_row_pointing_nowhere_is_reported_and_skipped(real_run, capsys,
                                                        verbose):
    from spacr.measure import generate_object_dataset

    root, npy, _ = real_run
    os.remove(npy)
    man = generate_object_dataset(root, object_type="cell", channels=(0,),
                                  save_png=False, verbose=verbose,
                                  db_path=os.path.join(root, "measurements",
                                                       "measurements.db"))
    assert man == []
    assert ("missing array" in capsys.readouterr().out) is verbose


# ---------------------------------------------------------------------------
# _promote_merged_to_uint16 -- the pipeline's own working-dtype boundary
# ---------------------------------------------------------------------------

_MASK_SETTINGS = {"cell_mask_dim": 4, "nucleus_mask_dim": None,
                  "pathogen_mask_dim": None, "organelle_mask_dim": None}


def _float_merged():
    """What ``spacr.io._normalize_img_batch`` writes: float32 signal on [0, 1]
    with integer label values in the mask plane."""
    rng = np.random.default_rng(1)
    signal = (rng.random((8, 8)) * 0.8 + 0.05).astype(np.float32)
    mask = np.zeros((8, 8), np.float32)
    mask[2:6, 2:6] = 1
    mask[0:2, 0:2] = 2
    return np.stack([signal, signal * 0.5, signal, signal, mask], axis=-1)


def test_normalised_float_planes_are_rescaled_not_truncated():
    from spacr.measure import _promote_merged_to_uint16

    data = _float_merged()
    plain = data.astype(np.uint16)               # what it used to do
    assert int((plain[..., 0] > 0).sum()) == 0   # the whole field went black

    out, factor = _promote_merged_to_uint16(data, _MASK_SETTINGS)
    assert out.dtype == np.uint16 and factor == 65535.0
    assert int((out[..., 0] > 0).sum()) == out[..., 0].size
    # One factor for every intensity plane, so channel ratios are untouched.
    assert (out[..., 1].astype(float).sum()
            / out[..., 0].astype(float).sum()) == pytest.approx(0.5, abs=1e-3)
    # A label is an identity: it is rounded, never rescaled.
    assert sorted(np.unique(out[..., 4]).tolist()) == [0, 1, 2]


def test_integer_counts_are_left_on_their_own_scale():
    """The ordinary case -- an int32 merged array from a concatenated cellpose
    label plane -- must convert exactly as it always did."""
    from spacr.measure import _promote_merged_to_uint16

    data = np.stack([np.full((4, 4), 3000, np.int32)] * 4
                    + [np.array([[1, 1, 0, 0]] * 4, np.int32)], axis=-1)
    out, factor = _promote_merged_to_uint16(data, _MASK_SETTINGS)
    assert factor == 1.0
    assert np.array_equal(out, data.astype(np.uint16))


def test_counts_above_the_16bit_ceiling_are_scaled_down_not_wrapped():
    from spacr.measure import _promote_merged_to_uint16

    data = np.zeros((2, 2, 5), np.int32)
    data[..., 0] = 131070                        # 2x the ceiling
    data[..., 1] = 65535
    data[..., 4] = 1
    out, factor = _promote_merged_to_uint16(data, _MASK_SETTINGS)
    assert factor == pytest.approx(0.5)
    assert out[..., 0].max() == 65535 and out[..., 1].max() == 32768
    assert out[..., 4].max() == 1


def test_non_finite_values_do_not_wrap():
    from spacr.measure import _promote_merged_to_uint16

    data = np.zeros((2, 2, 5), np.float32)
    data[..., 0] = [[np.nan, np.inf], [-np.inf, 0.5]]
    out, factor = _promote_merged_to_uint16(data, _MASK_SETTINGS)
    assert factor == 65535.0
    assert out[0, 0, 0] == 0 and out[1, 0, 0] == 0
    assert out[0, 1, 0] == 65535 and out[1, 1, 0] == 32768


def test_an_all_zero_field_needs_no_scaling():
    from spacr.measure import _promote_merged_to_uint16

    out, factor = _promote_merged_to_uint16(np.zeros((2, 2, 5), np.float32),
                                            _MASK_SETTINGS)
    assert factor == 1.0 and int(out.max()) == 0


def test_a_field_that_is_all_label_planes_has_no_intensity_to_scale():
    from spacr.measure import _promote_merged_to_uint16

    data = np.zeros((2, 2, 1), np.float32)
    data[..., 0] = 3
    out, factor = _promote_merged_to_uint16(data, {"cell_mask_dim": 0})
    assert factor == 1.0 and out[..., 0].max() == 3


@pytest.mark.parametrize("settings,expected", [
    ({"cell_mask_dim": 4}, {4}),
    ({"cell_mask_dim": None}, set()),
    ({"cell_mask_dim": "5"}, {5}),                 # settings CSVs hold strings
    ({"cell_mask_dim": "cell"}, set()),            # unparseable
    ({"cell_mask_dim": 99}, set()),                # out of range
    ({"cell_mask_dim": 4, "nucleus_mask_dim": 5}, {4, 5}),
])
def test_which_planes_hold_labels(settings, expected):
    from spacr.measure import _merged_mask_planes

    assert _merged_mask_planes(np.zeros((2, 2, 7)), settings) == expected


@pytest.mark.parametrize("verbose", [True, False])
def test_measure_crop_core_measures_a_float_field_instead_of_zeroing_it(
        tmp_path, capsys, verbose):
    """The caller, not just the helper: a full ``_measure_crop_core`` run over
    a float32 [0, 1] merged field now records non-zero intensities."""
    from spacr.measure import _measure_crop_core

    root = tmp_path / "plate1"
    merged = root / "merged"
    merged.mkdir(parents=True)
    (root / "measurements").mkdir(parents=True)
    data = merged_uint16().astype(np.float32)
    # Rescale only the intensity planes onto [0, 1], exactly as
    # spacr.io._normalize_img_batch does; the label planes keep their IDs.
    data[..., :4] /= float(data[..., :4].max())
    name = "plate1_A01_F001.npy"
    np.save(merged / name, data)

    index, _, cells, _ = _measure_crop_core(
        0, [], name, _measure_settings(merged, verbose=verbose))
    assert index == 0 and np.max(cells) >= 1
    reported = "Converted data from float32 to uint16 (intensity x65535)" \
        in capsys.readouterr().out
    assert reported is verbose

    con = sqlite3.connect(str(root / "measurements" / "measurements.db"))
    cols = [r[1] for r in con.execute("PRAGMA table_info(cell)")]
    mean_col = next(c for c in cols if c.endswith("_mean_intensity"))
    means = [r[0] for r in con.execute(f'SELECT "{mean_col}" FROM cell')]
    con.close()
    assert means and all(m and m > 0 for m in means), \
        "a float merged field was measured as black"
