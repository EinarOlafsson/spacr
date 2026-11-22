"""Item 1008 P3/P4: raw-pixel Brieflow parity and observed-value Otsu.

The two reference routines are vendored from Brieflow
``workflow/lib/external/cp_emulator.py`` on the WS (file SHA256
b2ddc5d2cc027320ed1bf93fdd95ce2b058741ff9647c1581f73d399f18ff10f).
Only their names and documentation are adapted; the Otsu branch and the
multichannel least-squares computation are preserved below. The reference
threshold uses mahotas when available and an independent scalar calculation
otherwise, so only the direct mahotas threshold contract needs that package.
"""

from itertools import combinations, permutations
from types import SimpleNamespace
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from spacr import measure as M


STATS = ('lstsq_slope_ij', 'lstsq_slope_ji', 'raw_overlap', 'raw_k1',
         'raw_k2', 'otsu_m1', 'otsu_m2', 'otsu_rwc1', 'otsu_rwc2')


def _reference_otsu(values):
    """Independent per-object Otsu with optional mahotas integer reference."""
    try:
        from mahotas.thresholding import otsu
    except ImportError:
        levels = np.unique(values)
        if levels.size == 1:
            return 0.0
        scores = []
        for threshold in levels[:-1]:
            low, high = values[values <= threshold], values[values > threshold]
            scores.append(low.size * high.size * (low.mean() - high.mean()) ** 2)
        return levels[np.argmax(scores)]
    return otsu(values)


def _cp_lstsq_slope_all_multichannel(r):
    """Vendored cp_emulator multichannel slopes, in permutations order."""
    V = r.intensity_image[r.image]

    slopes = []
    for ch in range(r.intensity_image.shape[-1]):
        slopes.extend(
            np.linalg.lstsq(
                np.vstack([V[..., ch], np.ones(V.shape[0])]).T,
                np.delete(V, ch, axis=1),
                rcond=None,
            )[0][0]
        )

    return slopes


def _cp_measure_colocalization(A, B):
    """Vendored Otsu branch of cp_emulator measure_colocalization."""
    if (A.sum() == 0) | (B.sum() == 0):
        return (np.nan,) * 7

    results = []
    A_thresh, B_thresh = _reference_otsu(A), _reference_otsu(B)
    A, B = A.astype(float), B.astype(float)
    overlap = (A * B).sum() / np.sqrt((A**2).sum() * (B**2).sum())
    results.append(overlap)
    K1 = (A * B).sum() / (A**2).sum()
    K2 = (A * B).sum() / (B**2).sum()
    results.extend([K1, K2])
    M1 = A[B > B_thresh].sum() / A.sum()
    M2 = B[A > A_thresh].sum() / B.sum()
    results.extend([M1, M2])
    A_ranks = rankdata(A, method="dense")
    B_ranks = rankdata(B, method="dense")
    R = max([A_ranks.max(), B_ranks.max()])
    weight = (R - abs(A_ranks - B_ranks)) / R
    RWC1 = ((A * weight)[B > B_thresh]).sum() / A.sum()
    RWC2 = ((B * weight)[A > A_thresh]).sum() / B.sum()
    results.extend([RWC1, RWC2])
    return results


def _compartment_field(seed):
    """Four-channel uint16 cell, nucleus and cytoplasm masks with tied values."""
    rng = np.random.default_rng(seed)
    cell = np.zeros((37, 43), dtype=np.int64)
    cell[1:17, 2:20] = 2
    cell[20:36, 3:22] = 9
    cell[3:33, 27:42] = 15
    nucleus = np.zeros_like(cell)
    nucleus[5:12, 6:15] = 2
    nucleus[24:32, 8:17] = 9
    nucleus[11:20, 31:38] = 15
    cytoplasm = np.where(nucleus == 0, cell, 0)
    image = rng.integers(0, 16384, size=cell.shape + (4,), dtype=np.uint16)
    image[..., 1] = image[..., 1] // 257 * 257
    image[..., 2] = np.clip(
        image[..., 0].astype(np.int64) // 3 + image[..., 2], 0, 65535)
    return (cell, nucleus, cytoplasm), image


@pytest.mark.parametrize('seed', [0, 1008, 2026])
def test_raw_values_equal_vendored_cp_emulator_in_three_compartments(seed):
    """Every pair and direction agrees to 1e-12 in all three compartments."""
    compartments, image = _compartment_field(seed)
    total_columns = 0
    for labels in compartments:
        frame = M.measure_colocalisation_otsu(labels, image)
        total_columns += len(frame.columns) - 1
        assert list(frame['label']) == [2, 9, 15]
        assert list(frame.columns) == ['label'] + [
            f'channel_{i}_channel_{j}_{stat}'
            for i, j in combinations(range(4), 2) for stat in STATS]
        for row, label in enumerate(frame['label']):
            mask = labels == label
            region = SimpleNamespace(intensity_image=image, image=mask)
            slopes = dict(zip(permutations(range(4), 2),
                              _cp_lstsq_slope_all_multichannel(region)))
            pixels = image[mask]
            for i, j in combinations(range(4), 2):
                expected = [slopes[i, j], slopes[j, i]] + list(
                    _cp_measure_colocalization(pixels[:, i], pixels[:, j]))
                columns = [f'channel_{i}_channel_{j}_{stat}' for stat in STATS]
                np.testing.assert_allclose(
                    frame.loc[row, columns].to_numpy(float),
                    np.nan_to_num(expected), atol=1e-12, rtol=0)
    assert total_columns == 162


def test_otsu_matches_mahotas_for_200_uint16_objects():
    """The vectorised threshold matches 200 independent mahotas calls."""
    mahotas = pytest.importorskip('mahotas')
    rng = np.random.default_rng(1008)
    sizes = rng.integers(2, 500, size=200)
    flat = np.repeat(np.arange(1, 201), sizes)
    values = rng.integers(0, 65536, size=flat.size, dtype=np.uint16)
    thresholds = M._otsu_by_label(flat, values.astype(float))
    for label in range(1, 201):
        assert thresholds[label] == mahotas.thresholding.otsu(values[flat == label])


def test_otsu_constant_ties_real_values_and_presorted_input():
    """Constants return zero and equal maxima choose the first observed split."""
    flat = np.array([4, 4, 4, 4, 8, 8, 8, 12, 12, 12, 12])
    values = np.array([0, 1, 2, 3, 17, 17, 17, -3.5, -2.5, -1.5, 4.25])
    got = M._otsu_by_label(flat, values)
    expected = np.zeros(13)
    for label in (4, 8, 12):
        pixels = values[flat == label]
        levels = np.unique(pixels)
        if levels.size == 1:
            continue
        scores = []
        for threshold in levels[:-1]:
            lower, upper = pixels[pixels <= threshold], pixels[pixels > threshold]
            scores.append(lower.size * upper.size * (lower.mean() - upper.mean()) ** 2)
        expected[label] = levels[np.argmax(scores)]
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(
        got, M._otsu_by_label(flat, values, M._sorted_by_label(flat, values)))
    assert M._otsu_by_label(np.ones(3, dtype=int), np.array([0., 1., 2.]))[1] == 0


@pytest.mark.parametrize('scale', [1.0, 1e-200, 1e200])
def test_otsu_is_independent_of_other_label_intensity_and_pixel_count(scale):
    """Earlier bright objects cannot erase later object cumulative sums."""
    earlier = np.array([1e16, 1e16 + 2, 1e16 + 4, 1e16 + 16])
    later = np.array([-3.5, -2.5, -1.5, 4.25]) * scale
    flat = np.repeat([1, 3], 4)
    together = M._otsu_by_label(flat, np.concatenate((earlier, later)))[3]
    alone = M._otsu_by_label(np.ones(4, dtype=int), later)[1]
    assert together == alone == later[2]
    plain = np.array([0., 1., 2., 3.])
    together = M._otsu_by_label(flat, np.concatenate((earlier, plain)))[3]
    assert together == 1.0
    many = np.repeat(earlier, 2000)
    flat = np.concatenate((np.ones(many.size, dtype=int), np.full(4, 3)))
    assert M._otsu_by_label(flat, np.concatenate((many, later)))[3] == later[2]


def test_flat_and_zero_channels_have_finite_declared_degenerate_values():
    """Zero-sum colocalisation and flat-predictor slopes become 0.0."""
    labels = np.array([[2, 2, 2, 2], [7, 7, 7, 7], [19, 19, 19, 19]])
    image = np.zeros(labels.shape + (4,), dtype=np.uint16)
    image[..., 1] = 5
    image[..., 2] = np.array([1, 2, 2, 4])
    image[..., 3] = np.array([4, 2, 2, 1])
    frame = M.measure_colocalisation_otsu(labels, image)
    assert np.isfinite(frame.to_numpy()).all()
    for j in (1, 2, 3):
        assert (frame[[f'channel_0_channel_{j}_{stat}' for stat in STATS]] == 0).all().all()
    for j in (2, 3):
        assert (frame[f'channel_1_channel_{j}_lstsq_slope_ij'] == 0).all()
        np.testing.assert_allclose(frame[f'channel_1_channel_{j}_lstsq_slope_ji'], 0, atol=1e-15)
    for stat in STATS[2:]:
        expected = _cp_measure_colocalization(image[0, :, 1], image[0, :, 2])[
            STATS[2:].index(stat)]
        np.testing.assert_allclose(frame[f'channel_1_channel_2_{stat}'], expected, atol=1e-12, rtol=0)


def test_single_pixel_empty_and_single_channel_objects_keep_their_schema():
    """Empty and one-channel fields retain predictable columns and label order."""
    empty = M.measure_colocalisation_otsu(np.zeros((0, 0), int), np.empty((0, 0, 4)))
    assert empty.shape == (0, 55)
    empty = M.measure_colocalisation_otsu(np.zeros((3, 4), int), np.ones((3, 4, 2)))
    assert empty.shape == (0, 10)
    single = M.measure_colocalisation_otsu(np.array([[21]]), np.array([[[3., 9.]]]))
    assert single.loc[0, 'channel_0_channel_1_lstsq_slope_ij'] == 0
    assert single.loc[0, 'channel_0_channel_1_lstsq_slope_ji'] == 0
    np.testing.assert_allclose(single.loc[0, ['channel_0_channel_1_raw_k1',
                                             'channel_0_channel_1_raw_k2']], [3, 1 / 3])
    frame = M.measure_colocalisation_otsu(np.array([[11, 0, 3]]), np.ones((1, 3)))
    pd.testing.assert_frame_equal(frame, pd.DataFrame({'label': [3, 11]}))


def test_large_constant_predictors_have_exactly_zero_slopes():
    """Dense-rank constants remain flat despite rounded intensity means."""
    labels = np.ones((1, 19), dtype=int)
    image = np.stack((np.full(19, 1e20), np.arange(19.)), axis=-1)[None]
    row = M.measure_colocalisation_otsu(labels, image).iloc[0]
    assert row['channel_0_channel_1_lstsq_slope_ij'] == 0.0
    assert row['channel_0_channel_1_lstsq_slope_ji'] == 0.0


def test_dense_ties_directionality_and_raw_values_are_preserved():
    """Ties use dense ranks and slopes fit raw values with an intercept."""
    labels = np.ones((1, 6), dtype=int)
    x = np.array([1, 1, 1, 3, 7, 7], dtype=np.uint16)
    y = np.array([9, 9, 3, 3, 1, 1], dtype=np.uint16)
    frame = M.measure_colocalisation_otsu(labels, np.stack((x, y), axis=-1)[None])
    expected = _cp_measure_colocalization(x, y)
    np.testing.assert_allclose(frame.iloc[0, 3:].to_numpy(float), expected, atol=1e-12, rtol=0)
    linear = np.stack((x, 5 * x + 11), axis=-1)[None]
    row = M.measure_colocalisation_otsu(labels, linear).iloc[0]
    assert row['channel_0_channel_1_lstsq_slope_ij'] == pytest.approx(5, abs=1e-12)
    assert row['channel_0_channel_1_lstsq_slope_ji'] == pytest.approx(0.2, abs=1e-12)


def test_public_docstring_has_an_exact_arrival_contract():
    """The new callable ships its exact reviewed-source admission fixture."""
    root = Path(__file__).resolve().parents[1]
    contract = root / 'tests/data/release_contracts/1008b_colocalisation_otsu_api_arrivals_2026-11-21.json'
    entries = json.loads(contract.read_text(encoding='utf-8'))
    assert set(entries) == {'spacr.measure.measure_colocalisation_otsu'}
    source = ast.parse((root / 'spacr/measure.py').read_text(encoding='utf-8'))
    node = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                and node.name == 'measure_colocalisation_otsu')
    assert ast.get_docstring(node) == entries['spacr.measure.measure_colocalisation_otsu']
    assert [arg.arg for arg in node.args.args] == ['labels', 'image']
