"""Item 0ab, second part: the compiled per-object kernels of the shared measurement core.

Each kernel in :mod:`spacr.measure` that replaced a per-object library call
or a chain of whole-field array passes is checked here against a plain
reference: the per-object NumPy and SciPy calls it replaced, the bincounts
it fuses (to the bit, where it adds in the same order), or a brute-force
definition. Values that are summed in another order must agree within
float32 rounding (the maintainer's rule for development changes), with
NaN in the same places and the dtype each column had.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import measure as M


def _blobs(seed, shape=(90, 100), n=14, dtype=np.float32):
    from skimage.segmentation import expand_labels

    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, np.int32)
    for k in range(1, n + 1):
        labels[rng.integers(4, shape[0] - 4), rng.integers(4, shape[1] - 4)] = k
    labels = expand_labels(labels, 6)
    image = (rng.random(shape + (3,)) * 100).astype(dtype)
    return labels, image


def _close(got, want, *, rel=1e-6, sd=1e-7):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    keep = ~np.isnan(want)
    if keep.any():
        spread = np.std(want[keep])
        assert np.all(np.abs(got[keep] - want[keep])
                      <= rel * np.abs(want[keep]) + sd * spread + 1e-300)


def _reference_statistics(labels, image, field_p90, field_p10):
    """The per-object loop :func:`_label_intensity_statistics` replaced."""
    from skimage.measure import regionprops

    rows = {name: [] for name in (
        'integrated_intensity', 'std_intensity', 'median_intensity',
        'skew_intensity', 'kurtosis_intensity', 'mode_intensity',
        'range_intensity', 'iqr_intensity', 'cv_intensity', 'gini_intensity',
        'frac_high90', 'frac_low10', 'entropy_intensity')}
    for p in (5, 10, 25, 75, 85, 95):
        rows[f'percentile_{p}'] = []
    for region in regionprops(labels, intensity_image=image):
        whole = region.image_intensity[region.image]
        intens = whole[~np.isnan(whole)] if whole.dtype.kind == 'f' else whole
        varying = not np.all(intens == intens[0])
        spread, level = np.std(intens), np.mean(intens)
        wide = intens.astype(np.float64)
        rows['integrated_intensity'].append(np.sum(intens))
        rows['std_intensity'].append(spread)
        rows['median_intensity'].append(np.median(intens))
        rows['skew_intensity'].append(
            M._SCIPY_SKEW(wide) if intens.size > 2 and varying else np.nan)
        rows['kurtosis_intensity'].append(
            M._SCIPY_KURTOSIS(wide) if intens.size > 3 and varying else np.nan)
        values, counts = np.unique(intens, return_counts=True)
        rows['mode_intensity'].append(float(values[np.argmax(counts)]))
        rows['range_intensity'].append(np.ptp(intens))
        upper, lower = M._percentiles_of(intens, [75, 25])
        rows['iqr_intensity'].append(upper - lower)
        rows['cv_intensity'].append(spread / level if level != 0 else np.nan)
        magnitude = np.sort(np.abs(intens))
        n = magnitude.size
        rows['gini_intensity'].append(
            np.sum((2 * np.arange(1, n + 1) - n - 1) * magnitude)
            / (n * np.sum(magnitude)) if np.sum(magnitude) else np.nan)
        rows['frac_high90'].append(np.mean(intens > field_p90))
        rows['frac_low10'].append(np.mean(intens < field_p10))
        rows['entropy_intensity'].append(
            M._SCIPY_ENTROPY(counts, base=2) if intens.size > 1 else 0.0)
        for p, value in zip((5, 10, 25, 75, 85, 95),
                            M._percentiles_of(whole, [5, 10, 25, 75, 85, 95])):
            rows[f'percentile_{p}'].append(value)
    import pandas as pd

    return pd.DataFrame(rows)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16, np.uint8])
@pytest.mark.parametrize("seed", range(2))
def test_the_compiled_intensity_statistics_match_the_per_object_calls(dtype, seed):
    labels, image = _blobs(seed, dtype=dtype)
    plane = image[..., 0]
    plane[10:20, 10:20] = plane[10, 10]
    if np.dtype(dtype).kind == 'f':
        plane[30, 30] = np.nan
        if seed:
            plane -= 50
    p90, p10 = M._field_reference_percentiles(plane)
    want = _reference_statistics(labels, plane, p90, p10)
    rows = np.unique(labels[labels > 0])
    got = M._label_intensity_statistics(labels, plane, rows, p90, p10)
    assert list(got) == list(want.columns)
    for column, values in got.items():
        assert values.dtype == want[column].to_numpy().dtype, column
        _close(values, want[column].to_numpy())


def test_the_compiled_intensity_statistics_measure_3d_objects():
    labels, image = _blobs(5)
    stack = np.stack([labels, labels])
    planes = np.stack([image[..., 0], image[..., 1]]).astype(np.float64)
    p90, p10 = M._field_reference_percentiles(planes)
    want = _reference_statistics(stack, planes, p90, p10)
    got = M._label_intensity_statistics(stack, planes, np.unique(stack[stack > 0]),
                                        p90, p10)
    for column, values in got.items():
        _close(values, want[column].to_numpy())


def test_the_texture_kernels_give_the_same_table_twice_and_on_any_thread_count(monkeypatch):
    labels, image = _blobs(7)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    one = M.measure_texture(labels, image)
    again = M.measure_texture(labels, image)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    three = M.measure_texture(labels, image)
    for other in (again, three):
        assert np.array_equal(one.to_numpy(), other.to_numpy(), equal_nan=True)
