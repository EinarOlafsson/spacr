"""Shared KMeans assignment arithmetic on actual CPU Torch tensors."""

import numpy as np
import pytest


def test_cpu_assignment_matches_numpy_across_chunk_boundary_and_first_index_ties():
    """All 1025 rows retain their nearest-center distance and earliest tied label."""
    torch = pytest.importorskip('torch')
    from spacr.mps_reduce import MetalKMeans

    index = np.arange(1025)
    points = np.column_stack((index % 17 - 8, index // 17 % 7 - 3,
                              index % 5 - 2)).astype(np.float32)
    centers = np.array([[-2, 0, 1], [2, 0, 1], [-2, 0, 1]], dtype=np.float32)
    points[0] = points[1023] = [0, 0, 1]
    points[1024] = [-2, 3, 1]
    delta = points.astype(np.float64)[:, None, :] - centers.astype(np.float64)[None, :, :]
    squared = np.einsum('nkd,nkd->nk', delta, delta)
    expected_labels = np.argmin(squared, axis=1)
    expected_distances = np.take_along_axis(squared, expected_labels[:, None], axis=1)[:, 0]
    np.testing.assert_array_equal(squared[0], [4, 4, 4])
    np.testing.assert_array_equal(squared[1024], [9, 25, 9])
    assert expected_labels[0] == expected_labels[1023] == expected_labels[1024] == 0
    assert set(expected_labels) == {0, 1}
    x = torch.from_numpy(points.copy())
    c = torch.from_numpy(centers.copy())
    original_x, original_c = x.clone(), c.clone()

    labels, distances = MetalKMeans._assign(x, c)

    assert x.device.type == c.device.type == labels.device.type == distances.device.type == 'cpu'
    assert labels.shape == distances.shape == (1025,)
    assert labels.dtype == torch.int64 and distances.dtype == torch.float32
    np.testing.assert_array_equal(labels.numpy(), expected_labels)
    np.testing.assert_array_equal(distances.numpy(), expected_distances)
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(c, original_c, rtol=0, atol=0)
