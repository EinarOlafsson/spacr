"""Item 0ab: the faster per-field measurement core gives the values it gave before.

Every speed-up in :mod:`spacr.measure` for item 0ab replaces a computation by
one that returns the same floats, or (by the maintainer's rule for
development changes, from the second part of the item on) floats equal
within float32 rounding with each column's ranks and NaN unchanged. Each
test below pits a
new path against the code it replaced (kept here as the reference, or
called from the library it bypasses) on data built to hit ties, empty
objects and edge cases. The last test measures the four Funk et al. 2022
plate A well A1 check fields end to end and compares every column with the
table the code before item 0ab wrote; it runs where those fields are staged
and is skipped elsewhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import measure as M


def _labels_and_values(seed, shape=(60, 70), n=25, ties=True):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n, shape).astype(np.int64)
    values = rng.random(shape)
    if ties:
        values = np.round(values * 7) / 7
    return labels, values


def _old_sorted_by_label(flat, values):
    order = np.lexsort((values, flat))
    labels_sorted = flat[order]
    counts = np.bincount(labels_sorted)
    present = np.nonzero(counts)[0]
    counts = counts[present]
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return order, values[order], starts, counts, present


def _old_robust_threshold(flat, values):
    _o, ordered, starts, counts, present = _old_sorted_by_label(flat, values)
    median = np.zeros(int(flat.max()) + 1)
    median[present] = M._grouped_median(ordered, starts, counts)
    deviation = np.abs(values - median[flat])
    _o, ordered, starts, counts, present = _old_sorted_by_label(flat, deviation)
    mad = np.zeros_like(median)
    mad[present] = M._grouped_median(ordered, starts, counts)
    return (median + 3.0 * 1.4826 * mad)[flat]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("big_labels", [False, True])
def test_the_two_stable_sorts_give_lexsorts_order(seed, big_labels):
    labels, values = _labels_and_values(seed)
    flat = labels.ravel()[labels.ravel() > 0]
    if big_labels:
        flat = flat * 5000
    picked = values.ravel()[labels.ravel() > 0]
    picked[::17] = np.nan
    picked[::13] = -0.0
    new = M._sorted_by_label(flat, picked)
    old = _old_sorted_by_label(flat, picked)
    for a, b in zip(new, old):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("seed", range(6))
def test_the_partitioned_mad_gives_the_sorted_mads_threshold(seed):
    labels, values = _labels_and_values(seed, ties=seed % 2 == 0)
    flat = labels.ravel()[labels.ravel() > 0]
    picked = values.ravel()[labels.ravel() > 0]
    np.testing.assert_array_equal(M._robust_threshold(flat, picked),
                                  _old_robust_threshold(flat, picked))


@pytest.mark.parametrize("seed", range(4))
def test_the_compiled_reconstruction_is_skimages_to_the_bit(seed):
    from scipy import ndimage
    from skimage.morphology import reconstruction

    rng = np.random.default_rng(seed)
    cross = ndimage.generate_binary_structure(2, 1)
    for _ in range(60):
        h, w = rng.integers(1, 45, 2)
        mask = rng.random((h, w))
        if rng.random() < 0.5:
            mask = np.round(mask * 4) / 4
        seed_plane = mask
        for _step in range(int(rng.integers(1, 6))):
            seed_plane = ndimage.grey_erosion(seed_plane, footprint=cross)
        expected = reconstruction(seed_plane, mask, method='dilation')
        got = M._reconstruct_by_dilation(seed_plane, mask)
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)
    assert np.array_equal(
        M._dilation_reconstruction(seed_plane, mask),
        reconstruction(seed_plane, mask, method='dilation'))


def test_the_glcm_homogeneity_is_skimages_to_the_bit():
    from skimage.feature import graycomatrix, graycoprops

    rng = np.random.default_rng(3)
    for _ in range(400):
        h, w = rng.integers(1, 40, 2)
        image = (rng.random((h, w)) ** rng.uniform(0.2, 3) * 256).astype(np.uint8)
        if rng.random() < 0.3:
            image[rng.random((h, w)) < 0.5] = 0
        for offset in (1, 2, 4, 8):
            if offset >= w:
                continue
            glcm = graycomatrix(image, [offset], [0], symmetric=True, normed=True)
            expected = (np.nan if not np.any(glcm)
                        else graycoprops(glcm, 'homogeneity')[0, 0])
            got = M._glcm_homogeneity(image, offset)
            assert got == expected or (np.isnan(got) and np.isnan(expected))


def test_the_unwrapped_scipy_statistics_match_the_wrapped_ones():
    from scipy.stats import entropy, kurtosis, mode, skew
    from skimage.measure import shannon_entropy

    rng = np.random.default_rng(4)
    for size in (3, 4, 9, 50, 400):
        for dtype in (np.float32, np.float64, np.uint16):
            values = (rng.random(size) * 50).astype(dtype)
            wide = values.astype(np.float64)
            assert M._SCIPY_SKEW(wide) == skew(wide)
            assert M._SCIPY_KURTOSIS(wide) == kurtosis(wide)
            uniques, counts = np.unique(values, return_counts=True)
            assert float(uniques[np.argmax(counts)]) == float(
                np.atleast_1d(mode(values, nan_policy='omit').mode)[0])
            assert M._SCIPY_ENTROPY(counts, base=2) == shannon_entropy(values)
    assert entropy is not None


def _blobs(seed, shape=(90, 100), n=14):
    from skimage.segmentation import expand_labels

    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, np.int32)
    for k in range(1, n + 1):
        labels[rng.integers(4, shape[0] - 4), rng.integers(4, shape[1] - 4)] = k
    labels = expand_labels(labels, 6)
    image = rng.random(shape + (3,)).astype(np.float32) * 100
    return labels, image


@pytest.mark.parametrize("threads", ["1", "3"])
def test_the_whole_field_families_do_not_depend_on_the_thread_count(monkeypatch, threads):
    labels, image = _blobs(5)
    nuclei = np.where(labels % 2 == 0, labels, 0)
    masks = [labels.astype(np.int64), nuclei.astype(np.int64)]
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    serial = (M._lbp_frames(masks, image.astype(np.float64)),
              M._granularity_frames(masks, image.astype(np.float64),
                                    subsample=0.5))
    monkeypatch.setenv("OMP_NUM_THREADS", threads)
    threaded = (M._lbp_frames(masks, image.astype(np.float64)),
                M._granularity_frames(masks, image.astype(np.float64),
                                      subsample=0.5))
    for old, new in zip(serial, threaded):
        for a, b in zip(old, new):
            pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_the_lbp_tally_is_the_per_code_share():
    from skimage.feature import local_binary_pattern

    labels, image = _blobs(6)
    frame = M._lbp_frames([labels.astype(np.int64)], image.astype(np.float64))[0]
    flat = labels.ravel()[labels.ravel() > 0]
    present = np.unique(flat)
    count = np.bincount(flat)
    for c in range(image.shape[-1]):
        codes = local_binary_pattern(image[..., c].astype(np.float64), 8, 1.0,
                                     method='uniform').astype(np.int64)
        code = codes.ravel()[labels.ravel() > 0]
        for k in range(10):
            with np.errstate(invalid='ignore', divide='ignore'):
                share = np.bincount(flat, (code == k).astype(float),
                                    count.size) / count
            np.testing.assert_array_equal(
                frame[f'channel_{c}_lbp_r1_{k}'].to_numpy(), share[present])


@pytest.mark.parametrize("threads", ["1", "4"])
def test_the_shared_colocalisation_pixels_change_nothing(monkeypatch, threads):
    monkeypatch.setenv("OMP_NUM_THREADS", threads)
    labels, image = _blobs(7)
    wide = image.astype(np.float64)
    labels = labels.astype(np.int64)
    families = (M.measure_colocalisation, M.measure_colocalisation_extra,
                M.measure_colocalisation_otsu)
    alone = [family(labels, wide) for family in families]
    with M._shared_foregrounds():
        shared = [family(labels, wide) for family in families]
        assert M._label_pixels(labels, wide) is M._label_pixels(labels, wide)
    for a, b in zip(alone, shared):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


@pytest.mark.parametrize("threads", ["1", "2", "4"])
def test_the_profile_frames_do_not_depend_on_the_thread_count(monkeypatch, threads):
    labels, image = _blobs(11)
    nuclei = np.where(labels % 3 == 0, labels, 0).astype(np.int32)
    cytoplasm = np.where(nuclei > 0, 0, labels).astype(np.int32)
    masks = [labels.astype(np.int32), nuclei, cytoplasm]
    roles = ['cell', 'nucleus', 'cytoplasm']
    families = ('edge_intensity', 'intensity_moments', 'texture_lbp',
                'colocalisation', 'colocalisation_extra',
                'colocalisation_otsu', 'granularity')

    def frames():
        dfs = [[] for _ in masks]
        M._add_profile_frames(dfs, roles, masks, image, families,
                              granularity_subsample=0.5)
        return [pd.concat(d, axis=1) for d in dfs]

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    serial = frames()
    monkeypatch.setenv("OMP_NUM_THREADS", threads)
    for a, b in zip(serial, frames()):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_the_correlation_cache_changes_nothing():
    labels, image = _blobs(8)
    cache = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        plain = M._calculate_correlation_object_level(
            image[..., i], image[..., j], labels, {})
        shared = M._calculate_correlation_object_level(
            image[..., i], image[..., j], labels, {}, cache=cache,
            channels=(i, j))
        pd.testing.assert_frame_equal(plain, shared, check_exact=True)


def test_the_shared_ring_pixels_change_nothing():
    labels, image = _blobs(9)
    for c in range(image.shape[-1]):
        channel = image[..., c]
        periphery = M._periphery_selections(labels)
        outside = M._outside_selections(labels, exclude_foreground=True)
        assert (M._periphery_intensity(labels, channel)
                == M._periphery_intensity(labels, channel, selections=periphery))
        assert (M._outside_intensity(labels, channel, exclude_foreground=True)
                == M._outside_intensity(labels, channel, exclude_foreground=True,
                                        selections=outside))


def test_the_skeleton_length_is_the_sum_of_its_piece_areas():
    from skimage import measure

    skeleton = np.zeros((9, 9), bool)
    skeleton[2, 1:6] = True
    skeleton[6, 6] = True
    pieces = measure.regionprops(measure.label(skeleton))
    summed = sum(p.area for p in pieces)
    assert summed == np.float64(np.count_nonzero(skeleton))
    assert type(summed) is np.float64
    assert sum(p.area for p in measure.regionprops(
        measure.label(np.zeros((4, 4), bool)))) == 0
    labels, image = _blobs(10)
    lengths = [M._analyze_cytoskeleton(image, labels == label, 0)[
        'skeleton_length'].iloc[0] for label in np.unique(labels)[1:6]]
    assert all(np.isfinite(lengths))


#: Where the A1 check fields and the tables the code before item 0ab wrote
#: for them are staged (``fields/site-<n>.npz`` and
#: ``bench/base/site-<n>.parquet``).
_STAGE = Path("/mnt/firecuda2/codex/scratch-docs/claude-0ab")
_SITES = (357, 575, 713, 1078)


def assert_tables_equivalent(table, expected):
    """Every column as before within float32 rounding, ranks kept, NaN in place.

    The maintainer's rule for development changes: per column the same
    dtype, NaN in the same places, every finite difference within ``1e-6 *
    |value| + 1e-7 * SD`` of the column, and a Spearman correlation with
    the reference of at least 0.99999 for a column that is not identical.
    """
    from scipy.stats import spearmanr

    assert list(table.columns) == list(expected.columns)
    for column in expected.columns:
        got = table[column].to_numpy()
        want = expected[column].to_numpy()
        assert got.dtype == want.dtype, column
        got = got.astype(float)
        want = want.astype(float)
        missing = np.isnan(want)
        assert np.array_equal(np.isnan(got), missing), column
        got, want = got[~missing], want[~missing]
        if not want.size or np.array_equal(got, want):
            continue
        tolerance = 1e-6 * np.abs(want) + 1e-7 * np.std(want)
        assert np.all(np.abs(got - want) <= tolerance), column
        if np.ptp(want) > 0:
            assert spearmanr(got, want)[0] >= 0.99999, column


@pytest.mark.parametrize("site", _SITES)
def test_the_a1_check_fields_measure_to_the_same_table(site):
    field = _STAGE / "fields" / f"site-{site}.npz"
    reference = _STAGE / "bench" / "base" / f"site-{site}.parquet"
    settings_path = Path("/mnt/wd4tb/final_funk/settings/20200202_6W-LaC024A.json")
    if not (field.is_file() and reference.is_file() and settings_path.is_file()):
        pytest.skip("the A1 check fields are staged on the maintainer's machine only")
    from spacr import ops_engine
    from spacr.ops_sample import measure_phenotype_field

    settings = json.loads(settings_path.read_text())
    staged = np.load(field)
    table = measure_phenotype_field(
        staged["nuclei"], staged["cells"], staged["stack"],
        [c.strip() for c in settings["ops_ph_channels"].split(",")],
        texture=True, funk_features=True, zernike=True, gpu=False,
        profile_features=ops_engine._ph_profile_features(settings),
        granularity_subsample=float(
            ops_engine._ph_granularity_subsample(settings) or 1.0),
        cytoskeleton_channels=settings.get("ops_ph_cytoskeleton_channels"))
    expected = pd.read_parquet(reference)
    assert len(table.columns) >= 2799
    assert_tables_equivalent(table.reset_index(drop=True),
                             expected.reset_index(drop=True))
