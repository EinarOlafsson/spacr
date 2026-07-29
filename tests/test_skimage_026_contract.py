"""Compatibility contracts for APIs deprecated by scikit-image 0.26."""
from __future__ import annotations

import warnings

import numpy as np


def test_object_size_threshold_keeps_equal_sized_component():
    from spacr.object import _remove_objects_smaller_than

    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True       # area 4: keep at min_size=4
    mask[5, 5:7] = True         # area 2: remove
    result = _remove_objects_smaller_than(mask, 4)
    assert result[1:3, 1:3].all()
    assert not result[5, 5:7].any()


def test_hole_threshold_keeps_equal_sized_hole():
    from spacr.object import _fill_holes_smaller_than

    mask = np.ones((10, 10), dtype=bool)
    mask[2:4, 2:4] = False      # area 4: keep at threshold=4
    mask[7, 7:9] = False        # area 2: fill
    result = _fill_holes_smaller_than(mask, 4)
    assert not result[2:4, 2:4].any()
    assert result[7, 7:9].all()


def test_migrated_measure_and_object_paths_emit_no_future_warning():
    from spacr import measure, object as object_module

    label = np.zeros((20, 20), dtype=np.int32)
    label[4:16, 4:16] = 1
    image = np.arange(400, dtype=float).reshape(20, 20)
    settings = {
        "organelle_min_size": 3,
        "organelle_tophat_radius": 1,
        "organelle_watershed_spots": False,
        "organelle_adaptive_block_size": 5,
        "organelle_adaptive_offset": 0,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        frame = measure._extended_regionprops_table(
            label, image, ["label", "mean_intensity"])
        segmented = object_module._segment_spots(image, "otsu", settings)
    assert len(frame) == 1
    assert segmented.shape == image.shape
