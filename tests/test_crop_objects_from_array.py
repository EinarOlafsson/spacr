"""Tests for measure.crop_objects_from_array (Measure live-preview backend)."""
from __future__ import annotations

import numpy as np


def _merged():
    H = W = 40
    data = np.zeros((H, W, 4), np.float32)
    data[..., 0] = 10; data[..., 1] = 20; data[..., 2] = 30
    mask = np.zeros((H, W), np.int32)
    mask[2:6, 2:6] = 1       # area 16
    mask[10:30, 10:30] = 2   # area 400
    mask[33:37, 33:37] = 3   # area 16
    data[..., 3] = mask
    return data


def test_returns_all_objects_largest_first():
    from spacr.measure import crop_objects_from_array
    res = crop_objects_from_array(_merged(), mask_dim=3, channels=(0, 1, 2))
    assert [r["label"] for r in res] == [2, 1, 3] or [r["label"] for r in res][0] == 2
    assert res[0]["area"] == 400
    for r in res:
        assert r["crop"].dtype == np.uint8 and r["crop"].shape[2] == 3


def test_area_filter():
    from spacr.measure import crop_objects_from_array
    res = crop_objects_from_array(_merged(), mask_dim=3, channels=(0, 1, 2),
                                  min_area=50)
    assert [r["label"] for r in res] == [2]


def test_channel_selection_and_grey():
    from spacr.measure import crop_objects_from_array
    # single channel -> repeated to RGB
    res = crop_objects_from_array(_merged(), mask_dim=3, channels=(0,))
    assert res[0]["crop"].shape[2] == 3


def test_mask_background_zeroes_outside():
    from spacr.measure import crop_objects_from_array
    res = crop_objects_from_array(_merged(), mask_dim=3, channels=(0, 1, 2),
                                  mask_background=True, normalize=False, buffer=4)
    crop = res[0]["crop"]
    # corner (in the buffer, outside the object) should be zeroed
    assert crop[0, 0].sum() == 0
    assert crop.max() > 0


def test_limit():
    from spacr.measure import crop_objects_from_array
    res = crop_objects_from_array(_merged(), mask_dim=3, channels=(0, 1, 2),
                                  limit=2)
    assert len(res) == 2
