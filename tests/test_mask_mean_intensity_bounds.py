"""418: absolute object means, not a pixel threshold or a percentile quota."""
from __future__ import annotations

import numpy as np
import pytest

from spacr.utils import _filter_objects, _process_single_fov_in_memory


def labelled_field():
    """Three equal-area objects with sparse IDs and means 2, 10 and 30."""
    mask = np.zeros((12, 12), dtype=np.uint16)
    mask[1:3, 1:3] = 2
    mask[5:7, 5:7] = 17
    mask[9:11, 9:11] = 65535
    intensity = np.full(mask.shape, np.nan)
    intensity[mask == 2] = 2
    intensity[mask == 17] = [0, 0, 0, 40]
    intensity[mask == 65535] = 30
    return mask, intensity


@pytest.mark.parametrize("minimum,maximum,kept", [
    (0, 0, [2, 17, 65535]),
    (10, 0, [17, 65535]),
    (0, 10, [2, 17]),
    (10, 10, [17]),
    (10.01, 29.99, []),
    (1, 31, [2, 17, 65535]),
])
def test_absolute_bounds_keep_equality_and_zero_disables_each_side(
        minimum, maximum, kept):
    mask, intensity = labelled_field()
    original_intensity = intensity.copy()
    result = _filter_objects(mask.copy(), intensity,
                             min_intensity=minimum, max_intensity=maximum)
    expected = np.zeros_like(mask)
    for new_id, old_id in enumerate(kept, 1):
        expected[mask == old_id] = new_id
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(intensity, original_intensity)


def test_intensity_is_an_absolute_mean_not_an_image_relative_quota():
    mask, intensity = labelled_field()
    dim = _filter_objects(mask.copy(), intensity, min_intensity=11)
    bright = _filter_objects(mask.copy(), intensity * 10, min_intensity=11)
    assert np.count_nonzero(dim) == 4
    assert np.count_nonzero(bright) == 12
    assert np.all(dim[mask == 65535] == 1)
    assert np.all(bright[mask == 2] == 1)


@pytest.mark.parametrize("bounds", [{"min_intensity": 1}, {"max_intensity": 10}])
def test_active_bounds_require_an_intensity_plane_with_the_mask_shape(bounds):
    mask, intensity = labelled_field()
    # Positive counterpart: the field really contains objects and a valid plane.
    assert _filter_objects(mask.copy(), intensity, **bounds).any()
    for invalid in (None, intensity[:-1], intensity[..., None]):
        with pytest.raises(ValueError, match="intensity.*same shape"):
            _filter_objects(mask.copy(), invalid, **bounds)


def test_disabled_intensity_filter_does_not_read_an_intensity_image():
    mask, _ = labelled_field()
    result = _filter_objects(mask.copy(), None, min_intensity=0, max_intensity=0)
    np.testing.assert_array_equal(result > 0, mask > 0)
    assert result.max() == 3


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_object_mean_is_refused_instead_of_silently_kept(bad):
    mask, intensity = labelled_field()
    assert _filter_objects(mask.copy(), intensity, min_intensity=1).max() == 3
    intensity[mask == 17] = bad
    with pytest.raises(ValueError, match="finite object mean"):
        _filter_objects(mask.copy(), intensity, min_intensity=1)
    # Nonfinite background is allowed; it does not contribute to any object.
    intensity[mask == 17] = 10
    assert _filter_objects(mask.copy(), intensity, min_intensity=1).max() == 3


def test_mean_accumulation_does_not_overflow_uint16():
    mask = np.zeros((50, 50), np.uint16)
    mask[1:-1, 1:-1] = 65535
    raw = np.full(mask.shape, 60000, np.uint16)
    assert _filter_objects(mask.copy(), raw, min_intensity=60000).max() == 1
    assert not _filter_objects(mask.copy(), raw, min_intensity=60001).any()


def test_intensity_area_and_border_filters_all_apply():
    mask, intensity = labelled_field()
    mask[0, :3] = 300
    intensity[0, :3] = 10
    mask[4, 10] = 400
    intensity[4, 10] = 10
    # All five survive when the constraints are off.
    assert _filter_objects(mask.copy(), intensity).max() == 5
    result = _filter_objects(mask.copy(), intensity, min_area=2, max_area=4,
                             remove_border=True, min_intensity=5,
                             max_intensity=20)
    np.testing.assert_array_equal(result, (mask == 17).astype(np.uint16))


def test_area_and_intensity_can_reject_the_same_border_object():
    mask, intensity = labelled_field()
    mask[0, :5] = 400
    intensity[0, :5] = 100
    result = _filter_objects(mask.copy(), intensity, max_area=4,
                             remove_border=True, max_intensity=30)
    expected = np.zeros_like(mask)
    for new_id, old_id in enumerate((2, 17, 65535), 1):
        expected[mask == old_id] = new_id
    np.testing.assert_array_equal(result, expected)


def test_empty_mask_needs_no_intensity_data():
    empty = np.zeros((6, 6), np.uint16)
    result = _filter_objects(empty, min_intensity=1, max_intensity=100)
    assert result is empty
    assert not result.any()


def test_random_fields_agree_with_a_per_object_float64_reference():
    rng = np.random.default_rng(418)
    for _ in range(100):
        mask = rng.choice([0, 2, 17, 65535], size=(12, 15)).astype(np.uint16)
        intensity = rng.integers(0, 65536, mask.shape, dtype=np.uint16)
        lower, upper = sorted(rng.uniform(20000, 45000, 2))
        expected = np.zeros_like(mask)
        next_label = 1
        for label in np.unique(mask[mask > 0]):
            mean = np.mean(intensity[mask == label], dtype=np.float64)
            if lower <= mean <= upper:
                expected[mask == label] = next_label
                next_label += 1
        result = _filter_objects(mask.copy(), intensity,
                                 min_intensity=lower, max_intensity=upper)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, -np.inf, "bad", []])
@pytest.mark.parametrize("bound", ["min_intensity", "max_intensity"])
def test_invalid_bounds_are_not_silently_disabled_even_on_an_empty_field(bad, bound):
    mask, intensity = labelled_field()
    assert _filter_objects(mask.copy(), intensity, **{bound: "10"}).any()
    for field in (mask, np.zeros_like(mask)):
        with pytest.raises(ValueError, match="finite nonnegative"):
            _filter_objects(field.copy(), intensity, **{bound: bad})
    # Saved nulls retain the historical disabled meaning.
    np.testing.assert_array_equal(
        _filter_objects(mask.copy(), None, **{bound: None}) > 0, mask > 0)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("face", [0, -1])
def test_volumetric_border_filter_checks_every_face(axis, face):
    mask = np.zeros((5, 6, 7), np.uint16)
    mask[2, 2, 2] = 11
    position = [2, 2, 2]
    position[axis] = face
    mask[tuple(position)] = 22
    raw = np.full(mask.shape, 10.0)
    unfiltered = _filter_objects(mask.copy(), raw, min_intensity=10)
    assert unfiltered[2, 2, 2] == 1
    assert unfiltered[tuple(position)] == 2
    result = _filter_objects(mask.copy(), raw, min_intensity=10, remove_border=True)
    np.testing.assert_array_equal(result, (mask == 11).astype(np.uint16))


def test_full_field_object_is_not_mistaken_for_an_empty_mask(capsys):
    mask = np.full((3, 4), 17, np.uint16)
    raw = np.full(mask.shape, 10.0)
    kept = _process_single_fov_in_memory(mask, raw, min_intensity=10)
    np.testing.assert_array_equal(kept, np.ones_like(mask))
    assert "Filter summary: 1 objects → 1 objects (0 removed)" in capsys.readouterr().out
    removed = _process_single_fov_in_memory(mask, raw, min_intensity=11)
    np.testing.assert_array_equal(removed, np.zeros_like(mask))
    np.testing.assert_array_equal(mask, np.full((3, 4), 17, np.uint16))
