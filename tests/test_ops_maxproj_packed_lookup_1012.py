"""Exact packed lookup acceleration preserves legacy histogram and calibrated-score values."""
import numpy as np
import pytest

from spacr import ops_maxproj as om
from tests.test_ops_maxproj_external_1012 import legacy


@pytest.mark.parametrize('grid', [om.SIMILARITY_GRID, om.SCORE_GRID])
def test_packed_grid_indices_match_searchsorted_at_every_boundary_and_neighbour(grid):
    """Match the original bins at all grid boundaries, adjacent float32 values and random inputs."""
    rng = np.random.default_rng(101220261122)
    boundary = grid.astype(np.float32)
    values = np.concatenate((boundary, np.nextafter(boundary, np.float32(-np.inf)),
                             np.nextafter(boundary, np.float32(np.inf)),
                             rng.uniform(grid[0] - 2, grid[-1] + 2, 200000).astype(np.float32),
                             np.array([-np.inf, np.inf, np.nan], np.float32)))
    expected = np.clip(np.searchsorted(grid, values, side='right') - 1, 0, len(grid) - 2)
    observed = om._packed_grid_indices(values, grid)
    np.testing.assert_array_equal(observed, expected)


def _square_values():
    """Build an exact symmetric similarity/pair-bin fixture with boundary and nonfinite values."""
    rng = np.random.default_rng(1012)
    n = 48
    pair = np.triu_indices(n, 1)
    values = rng.uniform(-1.1, 1.1, len(pair[0])).astype(np.float32)
    values[:5] = [-np.inf, np.inf, np.nan, -1, 1]
    square = np.zeros((n, n), np.float32)
    square[pair] = values
    square[(pair[1], pair[0])] = values
    bins = np.zeros((n, n), np.uint8)
    bins[pair] = rng.integers(0, 9, len(values), dtype=np.uint8)
    bins[(pair[1], pair[0])] = bins[pair]
    return square, bins, pair


def test_packed_histogram_counts_are_exactly_the_frozen_legacy_counts(legacy):
    """Preserve every histogram integer while using packed boundary-corrected lookup."""
    square, bins, pair = _square_values()
    expected = legacy[0]._counts(square, bins, 9, om.SIMILARITY_GRID)
    observed = om._counts(square[pair], bins[pair], 9, om.SIMILARITY_GRID)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_packed_calibration_scores_match_frozen_legacy_bytes(legacy, dtype):
    """Reuse normal quantiles without changing score bits, signed zero or infinite endpoints."""
    square, bins, pair = _square_values()
    square = square.astype(dtype)
    rng = np.random.default_rng(606)
    counts = rng.integers(0, 20, (9, len(om.SIMILARITY_GRID) - 1), dtype=np.int64)
    counts[0] = 0
    counts[1] = 0
    counts[1, 0], counts[1, -1] = 2, 1
    expected = legacy[0].calibrate_view(square, counts, bins)[pair].astype(dtype)
    observed = om.calibrate_view(square[pair], counts, bins[pair])
    assert observed.dtype == expected.dtype
    assert observed.tobytes() == expected.tobytes()
