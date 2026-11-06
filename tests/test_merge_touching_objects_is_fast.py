"""merge_touching_objects gives the old answer without eroding the whole image per object."""
from __future__ import annotations

import time

import numpy as np
import pytest
from skimage import morphology

from spacr.utils import merge_touching_objects


def _old_merge_touching_objects(mask, threshold=0.25):
    perimeters = {}
    labels = np.unique(mask)
    for label in labels:
        if label != 0:
            edges = morphology.erosion(mask == label) ^ (mask == label)
            perimeters[label] = np.sum(edges)
    shared_perimeters = {}
    dilated = morphology.dilation(mask > 0)
    for label in labels:
        if label != 0:
            dilated_label = morphology.dilation(mask == label)
            touching_labels = np.unique(mask[dilated & (dilated_label != 0) & (mask != 0)])
            for touching_label in touching_labels:
                if touching_label != label:
                    shared_boundary = dilated_label & morphology.dilation(mask == touching_label)
                    shared_perimeters[(label, touching_label)] = np.sum(shared_boundary)
    for (label1, label2), shared_perimeter in shared_perimeters.items():
        if shared_perimeter > threshold * min(perimeters[label1], perimeters[label2]):
            mask[mask == label2] = label1
    return mask


def _blobs(shape, count, seed, radius=(3, 9)):
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=np.int32)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    for label in range(1, count + 1):
        cy, cx = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        r = rng.integers(*radius)
        mask[((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r) & (mask == 0)] = label
    return mask


@pytest.mark.parametrize("threshold", [0.05, 0.25, 0.6])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_touching_blobs_match_old_implementation(seed, threshold):
    mask = _blobs((120, 140), 40, seed)
    expected = _old_merge_touching_objects(mask.copy(), threshold)
    assert np.array_equal(merge_touching_objects(mask.copy(), threshold), expected)


def test_objects_on_the_image_border_match_old_implementation():
    mask = np.zeros((30, 30), dtype=np.int32)
    mask[0:10, 0:10] = 1
    mask[0:10, 10:20] = 2
    mask[20:30, 25:30] = 3
    mask[20:30, 20:25] = 4
    expected = _old_merge_touching_objects(mask.copy())
    assert np.array_equal(merge_touching_objects(mask.copy()), expected)


def test_touching_merge_and_separate_stay_separate():
    mask = np.zeros((40, 60), dtype=np.int32)
    mask[5:25, 5:20] = 1
    mask[5:25, 20:35] = 2
    mask[30:38, 45:55] = 3
    out = merge_touching_objects(mask.copy())
    assert len(np.unique(out)) == 3 and out[32, 50] == 3
    assert np.array_equal(out, _old_merge_touching_objects(mask.copy()))


def test_empty_mask_is_unchanged():
    mask = np.zeros((10, 10), dtype=np.int32)
    assert np.array_equal(merge_touching_objects(mask.copy()), mask)


def test_many_objects_on_a_large_field_run_in_seconds():
    mask = _blobs((2000, 2000), 500, 3, radius=(8, 25))
    start = time.perf_counter()
    merge_touching_objects(mask)
    assert time.perf_counter() - start < 30
