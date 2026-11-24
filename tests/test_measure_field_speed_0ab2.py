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


def _brute_otsu(values):
    """First observed value maximising between-class variance, equal values in the lower class."""
    values = np.sort(values)
    best, winner = -np.inf, 0.0
    for t in np.unique(values)[:-1]:
        low, high = values[values <= t], values[values > t]
        score = low.size * high.size * (low.mean() - high.mean()) ** 2
        if score > best * (1 + 1e-12):
            best, winner = score, t
    return winner


@pytest.mark.parametrize("seed", range(4))
def test_the_compiled_otsu_picks_the_brute_force_threshold(seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 12, (40, 50)).astype(np.int64)
    values = np.round(rng.gamma(2.0, 30.0, labels.shape), 1)
    labels[0, :5] = 11
    values[0, :5] = 7.0
    flat = labels.ravel()[labels.ravel() > 0]
    v = values.ravel()[labels.ravel() > 0]
    got = M._otsu_by_label(flat, v)
    for label in np.unique(flat):
        run = v[flat == label]
        want = 0.0 if run.min() == run.max() else _brute_otsu(run)
        assert got[label] == want


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


@pytest.mark.parametrize("scale", [0.25, 0.5, 0.37])
def test_the_granularity_weights_give_the_resized_planes_means(scale):
    from scipy import ndimage
    from skimage.transform import resize

    labels, image = _blobs(2, shape=(83, 97))
    index = M._mask_index([labels.astype(np.int64)])
    small = ndimage.zoom(image[..., 0].astype(np.float64), scale, order=1)
    weights = M._granularity_weights(
        [labels.astype(np.int64)], index, tuple(int(round(n * scale)) for n in labels.shape), labels.shape)
    full = resize(small, labels.shape, order=1, mode='edge',
                  anti_aliasing=False, preserve_range=True)
    flat, inside, size, _present, count = index[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        want = np.bincount(flat, full.ravel()[inside], size) / count
        got = M._granularity_means(small, index, weights)[0]
    _close(got, want, rel=1e-12, sd=0)


def _foreground_values(seed):
    labels, image = _blobs(seed, dtype=np.float64)
    labels64 = labels.astype(np.int64)
    flat, rows, cols, inside, size = M._foreground(labels64)
    x = image[..., 0].ravel()[inside]
    y = np.round(image[..., 1].ravel()[inside] / 5)
    return flat, rows, cols, size, x, y


@pytest.mark.parametrize("seed", range(3))
def test_the_fused_pair_sums_are_the_bincounts_to_the_bit(seed):
    flat, _rows, _cols, size, x, y = _foreground_values(seed)
    count = np.bincount(flat, None, size)
    with np.errstate(invalid='ignore', divide='ignore'):
        dx = x - (np.bincount(flat, x, size) / count)[flat]
        dy = y - (np.bincount(flat, y, size) / count)[flat]
    mx, sxx = M._label_centring(flat, x, size)
    my, syy = M._label_centring(flat, y, size)
    sxy = M._cross_sums(flat, x, y, mx, my, size)
    with np.errstate(invalid='ignore', divide='ignore'):
        assert np.array_equal(mx, np.bincount(flat, x, size) / count,
                              equal_nan=True)
    for value, want in zip((sxy, sxx, syy), (dx * dy, dx * dx, dy * dy)):
        assert np.array_equal(value, np.bincount(flat, want, size))


@pytest.mark.parametrize("seed", range(3))
def test_the_overlap_and_otsu_pair_sums_are_the_bincounts_to_the_bit(seed):
    flat, _rows, _cols, size, x, y = _foreground_values(seed)
    tx, ty = M._robust_threshold(flat, x), M._robust_threshold(flat, y)
    rx, ry = M._within_label_ranks(flat, x), M._within_label_ranks(flat, y)
    mx, my = M._label_centring(flat, x, size)[0], M._label_centring(flat, y, size)[0]
    dx, dy = x - mx[flat], y - my[flat]
    top = np.maximum(rx[2], ry[2])
    sums = np.zeros((8, size))
    M._compiled(M._overlap_sums_kernel)(flat, x, y, tx, ty, mx, my, rx[1],
                                        ry[1], top, sums)
    a, b = np.clip(x - tx, 0, None), np.clip(y - ty, 0, None)
    both = (x > tx) & (y > ty)
    weight = (top[flat] - np.abs(rx[1] - ry[1])) / top[flat]
    want = (
        (dx * dy > 0).astype(float), a * b, a * a, b * b, a, b,
        np.where(both, a * weight, 0.0), np.where(both, b * weight, 0.0))
    for row, values in zip(sums, want):
        assert np.array_equal(row, np.bincount(flat, values, size))
    above_x, above_y = x > M._otsu_by_label(flat, x)[flat], y > M._otsu_by_label(flat, y)[flat]
    pair = np.zeros((5, size))
    M._compiled(M._otsu_pair_sums_kernel)(flat, x, y, above_x, above_y, rx[1],
                                          ry[1], top, pair)
    want = (x * y, x * above_y, y * above_x, x * weight * above_y,
            y * weight * above_x)
    for row, values in zip(pair, want):
        assert np.array_equal(row, np.bincount(flat, values, size))


def _vectorised_ranks(flat, values):
    """The array-pass ranks the compiled kernel replaced."""
    order, ordered, starts, counts, _present = M._sorted_by_label(flat, values)
    labels_sorted = flat[order]
    n = ordered.size
    position = np.arange(n)
    first = np.zeros(n, dtype=bool)
    first[starts] = True
    new_value = first.copy()
    new_value[1:] |= ordered[1:] != ordered[:-1]
    run = np.cumsum(new_value) - 1
    run_first = position[new_value]
    run_size = np.bincount(run)
    start_of_label = np.repeat(starts, counts)
    average = (run_first[run] - start_of_label) + (run_size[run] + 1) / 2
    dense_all = np.cumsum(new_value)
    dense = dense_all - dense_all[start_of_label] + 1
    avg_out, dense_out = np.empty(n), np.empty(n)
    avg_out[order] = average
    dense_out[order] = dense
    max_dense = np.zeros(int(flat.max()) + 1)
    np.maximum.at(max_dense, labels_sorted, dense)
    return avg_out, dense_out, max_dense


@pytest.mark.parametrize("seed", range(3))
def test_the_compiled_ranks_and_rank_bins_are_the_array_passes_to_the_bit(seed):
    flat, _rows, _cols, size, x, y = _foreground_values(seed)
    for values in (x, y):
        for got, want in zip(M._within_label_ranks(flat, values),
                             _vectorised_ranks(flat, values)):
            assert np.array_equal(got, want)
    count = np.bincount(flat, None, size).astype(float)
    rx, ry = M._within_label_ranks(flat, x)[0], M._within_label_ranks(flat, y)[0]
    qx, qy = M._rank_bins(flat, rx, count, 4), M._rank_bins(flat, ry, count, 4)
    joint = np.zeros((size, 4, 4), dtype=np.int64)
    M._compiled(M._joint_bins_kernel)(flat, qx, qy, joint)
    want = np.bincount(flat * 16 + qx * 4 + qy, minlength=size * 16)
    assert np.array_equal(joint.ravel(), want)


@pytest.mark.parametrize("seed", range(3))
def test_the_fused_central_moments_match_the_power_arrays(seed):
    flat, rows, cols, size, x, _y = _foreground_values(seed)
    centre_r, centre_c, _mu00 = M._weighted_centre(flat, rows, cols, x, size)
    sums = np.zeros((7, size))
    M._compiled(M._central_moments_kernel)(flat, rows, cols, x, centre_r,
                                           centre_c, sums)
    dr, dc = rows - centre_r[flat], cols - centre_c[flat]
    present = np.unique(flat)
    for k, (p, q) in enumerate(M._CENTRAL_MOMENT_ORDERS):
        want = np.bincount(flat, x * dr ** p * dc ** q, size)
        _close(sums[k][present], want[present], rel=1e-12, sd=1e-12)


def test_the_texture_kernels_give_the_same_table_twice_and_on_any_thread_count(monkeypatch):
    labels, image = _blobs(7)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    one = M.measure_texture(labels, image)
    again = M.measure_texture(labels, image)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    three = M.measure_texture(labels, image)
    for other in (again, three):
        assert np.array_equal(one.to_numpy(), other.to_numpy(), equal_nan=True)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("radius", [1, 2])
def test_the_compiled_lbp_codes_are_skimages(seed, radius):
    import warnings

    from skimage.feature import local_binary_pattern

    rng = np.random.default_rng(seed)
    image = rng.random((47, 53)) * 100
    if seed == 1:
        image = np.round(image / 10)
    needed = rng.random(image.shape) < 0.7
    points = 8 * radius
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = local_binary_pattern(image, points, float(radius),
                                    method="uniform").astype(np.int64)
    rows, cols = M._lbp_offsets(points, float(radius))
    got = np.full(image.shape, -1, dtype=np.int64)
    M._compiled(M._lbp_kernel)(image, needed, rows, cols, got)
    assert np.array_equal(got[needed], want[needed])
    assert (got[~needed] == -1).all()


def test_the_foreground_coordinates_are_the_index_grids():
    labels, _image = _blobs(4)
    labels = labels.astype(np.int64)
    flat, rows, cols, inside, size = M._foreground(labels)
    grid_r, grid_c = np.indices(labels.shape, dtype=np.float64)
    assert np.array_equal(rows, grid_r.ravel()[inside])
    assert np.array_equal(cols, grid_c.ravel()[inside])
    assert rows.dtype == cols.dtype == np.float64
    assert size == labels.max() + 1 and np.array_equal(flat, labels.ravel()[inside])


def test_one_mask_makes_each_pair_of_pearson_sums_once_for_both_colocalisation_families(monkeypatch):
    import pandas as pd

    labels, image = _blobs(11)
    wide = image.astype(np.float64)
    labels = labels.astype(np.int64)
    families = (M.measure_colocalisation, M.measure_colocalisation_otsu)
    alone = [family(labels, wide) for family in families]
    calls = []
    cross = M._cross_sums
    monkeypatch.setattr(M, "_cross_sums",
                        lambda *args: calls.append(1) or cross(*args))
    with M._shared_foregrounds():
        shared = [family(labels, wide) for family in families]
    assert len(calls) == 2 * 3
    for a, b in zip(alone, shared):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_the_channel_pair_pearson_is_the_bincount_definition_to_the_bit():
    labels, image = _blobs(12)
    labels = labels.astype(np.int64)
    flat, _rows, _cols, inside, size = M._foreground(labels)
    values = M._channel_values(image.astype(np.float64), inside)
    present = np.unique(flat)
    got = M._pearson_pair_columns(flat, size, present, values, ["a", "b", "c"])
    count = np.bincount(flat, None, size)
    with np.errstate(invalid='ignore', divide='ignore'):
        centred = [v - (np.bincount(flat, v, size) / count)[flat] for v in values]
        for (i, j), name in zip(((0, 1), (0, 2), (1, 2)), ("a_b", "a_c", "b_c")):
            sxy = np.bincount(flat, centred[i] * centred[j], size)
            sxx = np.bincount(flat, centred[i] ** 2, size)
            syy = np.bincount(flat, centred[j] ** 2, size)
            want = np.where((sxx > 0) & (syy > 0),
                            np.clip(sxy / np.sqrt(sxx * syy), -1, 1), np.nan)
            assert np.array_equal(got[f"correlation_{name}"], want[present],
                                  equal_nan=True)


def test_texture_reads_each_channel_in_place_with_the_values_of_a_contiguous_copy():
    labels, image = _blobs(13)
    wide = image.astype(np.float64)
    together = M.measure_texture(labels, wide)
    apart = [M.measure_texture(labels, np.ascontiguousarray(wide[..., c]))
             for c in range(wide.shape[-1])]
    for c, frame in enumerate(apart):
        for column in frame.columns.drop("label"):
            assert np.array_equal(together[column.replace("channel_0", f"channel_{c}")],
                                  frame[column], equal_nan=True)
