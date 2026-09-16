"""418: a run consumes absolute object bounds, independently of old switches."""
from __future__ import annotations

import numpy as np
import pytest

from spacr.object import merge_split_filter_masks
from tests.test_mask_mean_intensity_bounds import labelled_field


@pytest.mark.parametrize("role", ["cell", "nucleus", "pathogen", "organelle",
                                  "organelleb", "organellec", "organelled"])
def test_run_filters_each_roles_own_raw_object_means(role):
    mask, raw = labelled_field()
    before = mask.copy()
    inactive = merge_split_filter_masks([mask], [raw], {}, role)
    np.testing.assert_array_equal(inactive[0], mask)
    actual = merge_split_filter_masks(
        [mask], [raw], {f"{role}_min_intensity": 10,
                        f"{role}_max_intensity": 10}, role)
    np.testing.assert_array_equal(actual[0], (mask == 17).astype(np.uint16))
    np.testing.assert_array_equal(mask, before)


def test_area_only_run_needs_no_intensity_image():
    mask, _raw = labelled_field()
    result = merge_split_filter_masks([mask], None, {"cell_max_area": 3}, "cell")
    assert not result[0].any()
    kept = merge_split_filter_masks([mask], None, {"cell_max_area": 4}, "cell")
    np.testing.assert_array_equal(kept[0] > 0, mask > 0)


def test_enabled_bounds_refuse_missing_raw_data_instead_of_disappearing():
    mask, raw = labelled_field()
    settings = {"cell_min_intensity": 10}
    kept = merge_split_filter_masks([mask], [raw], settings, "cell")[0]
    assert kept[mask == 17].all()
    for image in (None, raw[:-1]):
        with pytest.raises(ValueError, match="intensity.*same shape"):
            merge_split_filter_masks([mask], [image], settings, "cell")


def test_raw_float64_precision_is_not_replaced_by_a_float32_image_copy():
    mask = np.zeros((6, 6), np.uint16)
    mask[1:3, 1:3] = 1
    raw = np.full(mask.shape, 2**30 + 1, np.float64)
    assert float(raw.astype(np.float32)[1, 1]) < 2**30 + 0.5
    result = merge_split_filter_masks(
        [mask], [raw], {"cell_min_intensity": 2**30 + 0.5}, "cell")[0]
    np.testing.assert_array_equal(result, mask)
    rejected = merge_split_filter_masks(
        [mask], [raw], {"cell_min_intensity": 2**30 + 1.5}, "cell")[0]
    assert not rejected.any()
