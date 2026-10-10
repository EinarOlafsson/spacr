"""Edge answers of the mask engine around N685's divide, cut and vvvv export.

Each test pins down what a guard returns, writes or raises, so the
early exits of the divide/cut stroke, the vvvv manifest, the exact-ID
fill, the bundle writer and the magnifier paste stay as they are.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr.qt import mask_engine as engine

pytestmark = pytest.mark.qt


def _two_blobs():
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[2:8, 2:8] = 4
    mask[12:18, 12:18] = 9
    return mask


def test_a_point_band_is_a_disc_around_the_point():
    band = engine._segment_band((11, 11), (5, 5), (5, 5), 3.0)
    assert band[5, 5] and band[5, 6] and band[6, 5]
    assert not band[5, 8] and not band[0, 0]
    assert int(band.sum()) == 9


def test_a_band_wholly_off_the_field_is_empty():
    band = engine._segment_band((10, 10), (-30, -30), (-20, -25), 1.5)
    assert band.shape == (10, 10) and not band.any()


def test_divide_object_off_the_field_returns_an_unchanged_copy():
    mask = _two_blobs()
    out, splits = engine.divide_object(mask, (-40, -40), (-30, -35))
    assert splits == []
    np.testing.assert_array_equal(out, mask)
    assert out is not mask


def test_divide_object_through_background_changes_nothing():
    mask = _two_blobs()
    out, splits = engine.divide_object(mask, (10, 0), (10, 9))
    assert splits == []
    np.testing.assert_array_equal(out, mask)


def test_divide_object_that_only_nicks_an_object_keeps_it_whole():
    mask = _two_blobs()
    out, splits = engine.divide_object(mask, (2, 2), (3, 2))
    assert splits == []
    np.testing.assert_array_equal(out, mask)


def test_divide_object_skips_the_nicked_object_and_splits_the_crossed_one():
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[2:8, 2:16] = 4
    mask[12:16, 9:14] = 6
    out, splits = engine.divide_object(mask, (9, 0), (9, 19))
    assert [source for source, _new in splits] == [4]
    new = splits[0][1]
    assert {int(out[4, 3]), int(out[4, 14])} == {4, new}
    assert (out[12:16, 9:14] == 6).all()
    assert int(mask[4, 9]) == 4


def test_a_cut_off_the_field_leaves_the_mask_alone():
    mask = _two_blobs()
    out, splits, cut = engine._divide_or_cut(mask, (-40, -40), (-30, -35))
    assert splits == [] and cut == []
    np.testing.assert_array_equal(out, mask)


def test_a_cut_through_background_leaves_the_mask_alone():
    mask = _two_blobs()
    out, splits, cut = engine._divide_or_cut(mask, (10, 0), (10, 9))
    assert splits == cut == []
    np.testing.assert_array_equal(out, mask)


def test_vvvv_export_refuses_a_stack_of_labels(tmp_path):
    with pytest.raises(ValueError, match="2-D"):
        engine._export_vvvv(tmp_path, "stack.tif",
                            np.zeros((2, 4, 4), dtype=np.uint16))
    assert list(tmp_path.iterdir()) == []


def test_vvvv_export_refuses_negative_ids(tmp_path):
    with pytest.raises(ValueError, match="16-bit"):
        engine._export_vvvv(tmp_path, "neg.tif",
                            np.full((3, 3), -1, dtype=np.int32))


def test_vvvv_manifest_records_an_empty_version_without_one(tmp_path,
                                                            monkeypatch):
    import spacr

    monkeypatch.delattr(spacr, "__version__", raising=False)
    labels = np.zeros((6, 6), dtype=np.uint16)
    labels[1:3, 1:3] = 2
    folder = Path(engine._export_vvvv(tmp_path, "field.png", labels,
                                      np.ones((6, 6)), pixel_size=(1, 2),
                                      unit="nm"))
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    assert manifest["spacr_version"] == ""
    assert manifest["pixel_size"] == {"x": 1.0, "y": 2.0, "unit": "nm"}
    assert manifest["channel_names"] == ["channel_1"]
    assert manifest["objects"] == 1
    assert manifest["width"] == 6 and manifest["height"] == 6


def test_canonical_labels_refuses_a_split_that_needs_an_id_past_uint16():
    width = 65535 + 2
    mask = np.zeros((1, width), dtype=np.int64)
    mask[0, :65535] = np.arange(1, 65536)
    mask[0, 65536] = 5
    with pytest.raises(ValueError, match="uint16"):
        engine.canonical_labels(mask)


def test_fill_holes_on_an_empty_integer_mask_is_empty():
    mask = np.zeros((6, 6), dtype=np.int32)
    out = engine.fill_holes(mask)
    assert out.shape == (6, 6) and not out.any()


def test_save_seg_bundle_without_outlines_or_flags_writes_only_the_masks(
        tmp_path):
    path = tmp_path / "field_seg.npy"
    old = np.zeros((8, 8), dtype=np.uint16)
    old[1:4, 1:4] = 1
    np.save(path, {"masks": old, "outlines": np.zeros((3, 3)),
                   "ismanual": np.zeros((2, 2), dtype=bool),
                   "filename": "field.tif"}, allow_pickle=True)
    edited = old.copy()
    edited[5:7, 5:7] = 2
    assert engine.save_seg_bundle(str(path), edited) == str(path)
    payload = engine.read_seg_bundle(str(path))
    np.testing.assert_array_equal(payload["masks"], edited)
    assert payload["outlines"].shape == (3, 3)
    assert payload["ismanual"].shape == (2, 2)
    assert payload["filename"] == "field.tif"
    np.save(path, {"masks": old}, allow_pickle=True)
    engine.save_seg_bundle(str(path), edited)
    payload = engine.read_seg_bundle(str(path))
    assert set(payload) == {"masks"}
    np.testing.assert_array_equal(payload["masks"], edited)


def test_apply_filters_without_an_image_judges_area_only():
    mask = np.zeros((12, 12), dtype=np.uint16)
    mask[1:3, 1:3] = 1
    mask[5:11, 5:11] = 2
    out, removals = engine.apply_filters(
        mask, None, [{"property": "area", "min": 10, "max": None}])
    assert [removal.label for removal in removals] == [1]
    assert not out[1:3, 1:3].any() and (out[5:11, 5:11] == 2).all()
    assert int(mask[1, 1]) == 1


def test_drop_border_objects_of_an_empty_array():
    out = engine._drop_border_objects(np.zeros((0, 5), dtype=np.int32))
    assert out.shape == (0, 5) and out.dtype == np.int32


def test_split_object_at_keeps_the_object_when_the_watershed_finds_one_piece(
        monkeypatch):
    import skimage.segmentation as segmentation

    mask = np.zeros((30, 50), dtype=np.uint16)
    yy, xx = np.mgrid[0:30, 0:50]
    mask[(xx - 14) ** 2 + (yy - 15) ** 2 <= 64] = 3
    mask[(xx - 34) ** 2 + (yy - 15) ** 2 <= 64] = 3
    mask[13:18, 14:35] = 3
    split, new_ids = engine.split_object_at(mask, 14, 15)
    assert len(new_ids) == 1
    monkeypatch.setattr(segmentation, "watershed",
                        lambda distance, markers, mask=None:
                        np.where(mask, 1, 0).astype(np.int32))
    out, new_ids = engine.split_object_at(mask, 14, 15)
    assert new_ids == []
    np.testing.assert_array_equal(out, mask)


def test_square_footprint_falls_back_to_square(monkeypatch):
    import skimage.morphology as morphology

    assert engine._square_footprint(3).shape == (3, 3)
    monkeypatch.delattr(morphology, "footprint_rectangle", raising=False)
    monkeypatch.setattr(morphology, "square",
                        lambda size: np.full((size, size), 7, dtype=np.uint8),
                        raising=False)
    footprint = engine._square_footprint(4)
    assert footprint.shape == (4, 4) and int(footprint[0, 0]) == 7


def test_a_level_above_every_pixel_falls_back_to_the_noise_floor(monkeypatch):
    region = np.zeros((40, 40), dtype=np.float32)
    region[15:25, 15:25] = 100.0
    found = engine._classical_region_labels(region, smoothing=0.0)
    assert found[20, 20] > 0 and found[2, 2] == 0
    monkeypatch.setattr(engine, "_global_level", lambda values, name: 2.0)
    flat = np.zeros((40, 40), dtype=np.float32)
    flat[15:25, 15:25] = 100.0
    out = engine._classical_region_labels(flat, smoothing=0.0)
    assert out.shape == (40, 40) and not out.any()


def test_merge_onto_one_object_joins_it_without_renumbering():
    mask = np.zeros((10, 10), dtype=np.uint16)
    mask[0:4, 0:4] = 6
    incoming = np.zeros((4, 4), dtype=np.int32)
    incoming[1:4, 1:4] = 1
    out, ids = engine._paste_region_objects(mask, incoming, (2, 2),
                                            overlap="merge")
    assert ids == [6]
    assert (out[3:6, 3:6] == 6).all()
    assert (out[0:4, 0:4] == 6).all()
    assert set(np.unique(out)) == {0, 6}


def test_center_puncta_instances_keeps_the_peak_when_the_centre_drifts(
        monkeypatch):
    from spacr.qt import cpu_modes

    image = np.zeros((20, 20), dtype=np.float32)
    image[11, 10] = 50.0
    parents = np.zeros((20, 20), dtype=np.int32)
    parents[2:18, 2:18] = 1
    frame = pd.DataFrame([{"y": 10, "x": 10, "parent_id": 1,
                           "included": True, "object_label": 0,
                           "mask_pixels": 0}])
    monkeypatch.setattr(engine, "_center_puncta_candidates",
                        lambda im, parent_labels, params: frame.copy())
    params = cpu_modes.CpuParams(puncta_center_pixels=1)
    labels, candidates = engine._center_puncta_instances(image, parents,
                                                         params)
    assert int(labels[10, 10]) == 1
    assert int(labels[11, 10]) == 0
    assert int(labels.sum()) == 1
    assert int(candidates.at[0, "mask_pixels"]) == 1
    assert int(candidates.at[0, "object_label"]) == 1


def test_merge_with_exact_ids_keeps_a_new_objects_own_id():
    mask = np.zeros((10, 10), dtype=np.uint16)
    mask[0:3, 0:3] = 6
    incoming = np.zeros((4, 4), dtype=np.int32)
    incoming[1:3, 1:3] = 12
    out, ids = engine._paste_region_objects(mask, incoming, (5, 5),
                                            overlap="merge",
                                            preserve_ids=True)
    assert ids == [12]
    assert (out[6:8, 6:8] == 12).all()
    assert (out[0:3, 0:3] == 6).all()
    assert set(np.unique(out)) == {0, 6, 12}
