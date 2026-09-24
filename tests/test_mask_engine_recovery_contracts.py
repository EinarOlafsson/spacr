"""Boundary inputs retain identities and do not invent detections or edits."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr.qt import mask_engine as engine


@pytest.mark.parametrize("original", ["corrupt", "wrong_shape", "missing"])
def test_bundle_falls_back_to_embedded_channel_first_image(tmp_path, original):
    queue = tmp_path / "queue"
    queue.mkdir()
    originals = tmp_path / "new_originals"
    originals.mkdir()
    if original == "corrupt":
        (originals / "original.tif").write_bytes(b"not a TIFF")
    elif original == "wrong_shape":
        tifffile.imwrite(originals / "original.tif", np.zeros((8, 9), np.uint16))
    pixels = np.arange(20, dtype=np.uint16).reshape(4, 5) + 1
    labels = np.zeros((4, 5), np.uint16)
    labels[1:3, 1:3] = 9
    path = queue / "field_seg.npy"
    np.save(path, {"source_image": "/unrelated-machine/original.tif",
                   "img": np.stack([pixels] * 3), "masks": labels})
    before = path.read_bytes()
    image, loaded = engine.load_seg_bundle(str(path))
    assert image.shape == (4, 5) and image.dtype == np.uint16
    assert image[0, 0] < image[-1, -1] and image.max() == 65535
    np.testing.assert_array_equal(loaded, labels)
    assert path.read_bytes() == before


def test_outline_dtype_widens_instead_of_truncating_saved_object_ids():
    labels = np.zeros((7, 7), np.uint16)
    labels[2:5, 2:5] = 900
    outlines = engine.seg_outlines(labels, np.zeros((7, 7), np.uint8))
    assert outlines.dtype == np.uint16 and set(np.unique(outlines)) == {0, 900}
    assert outlines[3, 3] == 0 and outlines[2, 3] == 900


@pytest.mark.parametrize("text,expected", [(" KEEP ", True), ("yes", True), ("maybe", None)])
def test_hand_edited_curation_csv_distinguishes_keep_from_unknown(tmp_path, text, expected):
    image = str(tmp_path / "field.tif")
    path = Path(engine.curation_csv_path(str(tmp_path)))
    path.parent.mkdir()
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(engine.CURATION_COLUMNS)
        writer.writerow([image, "mask.tif", "3", text])
    assert engine.curation_verdict(str(tmp_path), image) is expected


@pytest.mark.parametrize("operation", [engine.invert_normalized, engine.invert_intensity,
                                        engine.invert_for_detection, engine.normalize_for_detection])
def test_empty_preprocessing_returns_empty_without_mutating_source(operation):
    source = np.empty((0, 3), np.float32)
    result = operation(source)
    assert result.shape == source.shape and result.dtype == source.dtype
    assert source.size == 0


def test_normalization_and_inversion_handle_flat_float_and_boolean_fields():
    flat = np.full((4, 5), 3.5, np.float32)
    np.testing.assert_array_equal(engine.normalize_for_detection(flat), np.zeros_like(flat))
    np.testing.assert_array_equal(engine.invert_normalized(flat), np.ones_like(flat))
    truth = np.array([[True, False], [False, True]])
    np.testing.assert_array_equal(engine.invert_normalized(truth), ~truth)
    np.testing.assert_array_equal(flat, np.full((4, 5), 3.5))
    with pytest.raises(TypeError, match="numeric or boolean"):
        engine.invert_intensity(np.array([["unreadable"]]))


def test_float_inversion_uses_field_bounds_and_preserves_nonfinite_pixels():
    crop = np.array([[2.0, 4.0, np.nan], [np.inf, -np.inf, 8.0]], np.float32)
    result = engine.invert_for_detection(crop, bounds=(0.0, 10.0))
    np.testing.assert_allclose(result, [[8, 6, np.nan], [np.inf, -np.inf, 2]], equal_nan=True)
    assert result.dtype == crop.dtype and crop[0, 0] == 2
    invalid = np.array([[np.nan, np.inf]], np.float32)
    np.testing.assert_allclose(engine.invert_for_detection(invalid), invalid, equal_nan=True)


def test_exact_id_hole_fill_preserves_another_primary_inside_the_ring():
    labels = np.zeros((9, 9), np.uint16)
    labels[1:8, 1:8] = 900
    labels[2:7, 2:7] = 0
    labels[4, 4] = 17
    before = labels.copy()
    filled = engine.fill_holes(labels, preserve_ids=True)
    assert filled[3, 3] == 900 and filled[4, 4] == 17
    assert set(np.unique(filled)) == {0, 17, 900}
    np.testing.assert_array_equal(labels, before)


@pytest.mark.parametrize("settings", [{"algorithm": "missing"}, {"classes": 1},
                                     {"algorithm": "niblack", "classes": 3}])
def test_threshold_invalid_settings_fail_instead_of_substituting_otsu(settings):
    image = np.arange(49, dtype=np.float32).reshape(7, 7)
    with pytest.raises(ValueError):
        engine._otsu_instances(image, **settings)
    with pytest.raises(ValueError):
        engine._local_level_map(image, "missing", window=7, k=0.2)
    assert {"otsu", "niblack", "sauvola"} <= set(engine.threshold_algorithms())


def test_histogram_levels_refuse_a_single_class():
    with pytest.raises(ValueError, match="two classes"):
        engine._otsu_levels(np.arange(20).reshape(4, 5), classes=1)


def test_local_dark_and_bright_cuts_select_opposite_sides_and_flat_field_is_empty():
    ramp = np.tile(np.arange(31, dtype=np.float32), (21, 1))
    bright = engine._local_otsu_binary(ramp, window=7, bright=True, correction=1)
    dark = engine._local_otsu_binary(ramp, window=7, bright=False, correction=1)
    assert bright[:, -1].all() and dark[:, 0].all()
    assert not (bright & dark).any()
    assert not engine._local_otsu_binary(np.ones_like(ramp), window=7,
                                        bright=False, correction=1).any()


@pytest.mark.parametrize("image,settings", [(np.empty((0, 3)), {}),
                                           (np.zeros((3, 4, 2)), {}),
                                           (np.ones((5, 5)), {"stop": "missing"})])
def test_propagation_refuses_invalid_inputs(image, settings):
    with pytest.raises(ValueError):
        engine.maxima_propagate_instances(image, **settings)


def test_propagation_distinguishes_absent_seeds_from_seeds_blocked_by_threshold():
    blank = np.ones((21, 21), np.float32)
    result = engine.maxima_propagate_instances(blank, sigma=0)
    assert result.seeds == 0 and result.level is None and not result.labels.any()
    peak = np.zeros((21, 21), np.float32)
    peak[10, 10] = 5
    result = engine.maxima_propagate_instances(peak, sigma=0, seed_level=1,
                    seed_level_is_percentile=False, stop="absolute", stop_value=6)
    assert result.seeds == 1 and result.level == 6 and not result.labels.any()


@pytest.mark.parametrize("shape", [(0, 3), (1, 1), (12, 12)])
def test_classical_region_does_not_invent_objects_on_flat_or_tiny_inputs(shape):
    image = np.ones(shape, np.float32)
    result = engine._classical_region_labels(image, bright=False)
    assert result.shape == shape and not result.any()


def test_history_empty_undo_redo_and_outside_paste_create_no_edit():
    history = engine.MaskHistory()
    assert history.undo() is None and history.redo() is None
    mask = np.zeros((4, 5), np.uint16)
    result, added = engine._paste_region_objects(mask, np.ones((3, 3), np.int32), (50, -30))
    assert not added and result is not mask
    np.testing.assert_array_equal(result, mask)


def test_one_pixel_foreground_gets_a_seed_when_peak_finding_has_no_interior():
    result = engine._split_touching_objects(np.ones((1, 1), bool))
    np.testing.assert_array_equal(result, [[1]])


@pytest.mark.parametrize("algorithm", ["niblack", "sauvola"])
def test_classical_local_detection_finds_dark_centres_without_inverting_source(algorithm):
    yy, xx = np.indices((41, 61))
    image = (1 - 0.8 * np.exp(-((yy - 20)**2 + (xx - 16)**2) / 18)
               - 0.8 * np.exp(-((yy - 20)**2 + (xx - 44)**2) / 18)).astype(np.float32)
    before = image.copy()
    result = engine._classical_region_labels(image, bright=False, algorithm=algorithm,
                                             window=15, split_touching=False)
    assert result[20, 16] > 0 and result[20, 44] > 0
    assert not result[0, 0] and result.shape == image.shape
    np.testing.assert_array_equal(image, before)


def test_threshold_border_exclusion_keeps_the_complete_internal_object():
    image = np.zeros((30, 40), np.float32)
    image[0:8, 2:10] = 100
    image[16:24, 20:28] = 100
    result = engine._otsu_instances(image, correction=0.99, exclude_border=True)
    assert not result[:8].any()
    assert result[20, 24] > 0 and np.count_nonzero(result) == 64


def test_object_readout_averages_raw_rgb_channels_and_preserves_exact_ids():
    mask = np.array([[0, 900], [0, 900]], np.uint16)
    image = np.array([[[1, 2, 3], [3, 6, 9]], [[0, 0, 0], [6, 12, 18]]], np.uint16)
    lookup = engine.ObjectLookup(mask, image, preserve_ids=True)
    assert lookup.at(1, 0).intensity == 6
    assert lookup.measure(900) == (2, 9)
    np.testing.assert_array_equal(lookup.labels, mask)
