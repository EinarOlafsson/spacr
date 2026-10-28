"""Shared KMeans assignment arithmetic on actual CPU Torch tensors."""

import numpy as np
import pytest


def test_cpu_assignment_matches_numpy_across_chunk_boundary_and_first_index_ties(monkeypatch):
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
    monkeypatch.setattr(MetalKMeans, "_CHUNK_ELEMENTS", 1024 * centers.size)

    labels, distances = MetalKMeans._assign(x, c)

    assert x.device.type == c.device.type == labels.device.type == distances.device.type == 'cpu'
    assert labels.shape == distances.shape == (1025,)
    assert labels.dtype == torch.int64 and distances.dtype == torch.float32
    np.testing.assert_array_equal(labels.numpy(), expected_labels)
    np.testing.assert_array_equal(distances.numpy(), expected_distances)
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(c, original_c, rtol=0, atol=0)


def _cpu_matrix(data):
    """Supply actual CPU tensors at the explicit backend boundary, without simulating Metal."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))


def _unexpected_tensor_allocation(*args, **kwargs):
    pytest.fail("invalid input reached tensor allocation")


def test_cpu_update_matches_numpy_means_and_seeded_empty_cluster_rows(monkeypatch):
    """Shared centroid arithmetic must preserve means, chunks and seeded empty-cluster replacement."""
    import torch
    from spacr.mps_reduce import MetalKMeans

    points = np.arange(27, dtype=np.float32).reshape(9, 3) - 13
    membership = np.array([0, 0, 2, 2, 0, 2, 0, 2, 0], dtype=np.int64)
    expected = np.empty((4, 3), dtype=np.float64)
    reference_rng = np.random.default_rng(27)
    for cluster in range(4):
        members = points[membership == cluster].astype(np.float64)
        expected[cluster] = members.mean(axis=0) if len(members) else points[reference_rng.integers(9)]
    x, labels = torch.from_numpy(points.copy()), torch.from_numpy(membership.copy())
    before_x, before_labels = x.clone(), labels.clone()
    rng = np.random.default_rng(27)
    monkeypatch.setattr(MetalKMeans, "_CHUNK_ELEMENTS", 4 * 4)
    result = MetalKMeans(n_clusters=4)._update(x, labels, rng)
    assert result.device.type == "cpu" and result.dtype == torch.float32
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-7, atol=1e-7)
    assert rng.bit_generator.state == reference_rng.bit_generator.state
    torch.testing.assert_close(x, before_x, rtol=0, atol=0)
    torch.testing.assert_close(labels, before_labels, rtol=0, atol=0)


@pytest.mark.parametrize("initialization,runs,max_iter", [
    ("k-means++", "auto", 20), ("random", "auto", 20), ("k-means++", 3, 1),
])
def test_cpu_lloyd_boundary_matches_independent_clusters_and_sklearn(
        monkeypatch, initialization, runs, max_iter):
    """Only _matrix is injected with real CPU tensors; this checks shared Lloyd math, not MPS execution."""
    from sklearn.cluster import KMeans
    from spacr import mps_reduce

    points = np.array([[-6, -1], [-6, 1], [-4, -1], [-4, 1],
                       [4, -1], [4, 1], [6, -1], [6, 1]], dtype=np.float64)
    points += [2**20, -(2**19)]
    before = points.copy()
    expected_centers = np.array([[-5, 0], [5, 0]], dtype=np.float64) + [2**20, -(2**19)]
    differences = points[:, None, :] - expected_centers[None, :, :]
    expected_labels = np.argmin(np.einsum("nkd,nkd->nk", differences, differences), axis=1)
    reference = KMeans(n_clusters=2, init=expected_centers, n_init=1,
                       max_iter=20, tol=0, random_state=7).fit(points)
    monkeypatch.setattr(mps_reduce, "_matrix", _cpu_matrix)
    monkeypatch.setattr(mps_reduce.MetalKMeans, "_CHUNK_ELEMENTS", 16)
    model = mps_reduce.MetalKMeans(n_clusters=2, init=initialization, n_init=runs,
                                  max_iter=max_iter, tol=0, random_state=7)
    labels = model.fit_predict(points)
    order = np.argsort(model.cluster_centers_[:, 0])
    np.testing.assert_array_equal(model.cluster_centers_[order], expected_centers)
    np.testing.assert_array_equal(np.argsort(order)[labels], expected_labels)
    np.testing.assert_array_equal(reference.labels_, expected_labels)
    np.testing.assert_array_equal(reference.cluster_centers_, expected_centers)
    assert model.inertia_ == reference.inertia_ == 16
    assert 1 <= model.n_iter_ <= max_iter
    if max_iter == 1:
        assert model.n_iter_ == 1
    assert model.n_features_in_ == 2
    np.testing.assert_array_equal(labels, model.labels_)
    np.testing.assert_array_equal(points, before)


def test_cpu_lloyd_identical_rows_have_zero_inertia_and_stable_ties(monkeypatch):
    """A zero-distance initialization still gives real CPU means and first-center labels."""
    from spacr import mps_reduce

    points = np.tile([19., -23.], (9, 1))
    monkeypatch.setattr(mps_reduce, "_matrix", _cpu_matrix)
    model = mps_reduce.MetalKMeans(n_clusters=2, n_init=2, random_state=3).fit(points)
    np.testing.assert_array_equal(model.cluster_centers_, [[19., -23.], [19., -23.]])
    np.testing.assert_array_equal(model.labels_, np.zeros(9, dtype=np.int64))
    assert model.inertia_ == 0 and model.n_iter_ == 1


def test_reducer_input_errors_precede_native_allocation(monkeypatch):
    """Invalid shape/data/options and mismatched prediction features must not allocate an MPS tensor."""
    import torch
    from spacr import mps_reduce

    monkeypatch.setattr(torch, "as_tensor", _unexpected_tensor_allocation)
    for data in (np.empty((0, 2)), np.ones(3), [[np.nan, 0]], [[0, np.inf]]):
        with pytest.raises(ValueError, match="nonempty finite two-dimensional"):
            mps_reduce._matrix(data)
        with pytest.raises(ValueError, match="nonempty finite two-dimensional"):
            mps_reduce.MetalKMeans(n_clusters=1).fit(data)
    for options in ({"algorithm": "elkan"}, {"init": "unsupported"}, {"init": np.ones((2, 2))}):
        with pytest.raises(ValueError, match="initialization/algorithm"):
            mps_reduce.MetalKMeans(**options)
    model = mps_reduce.MetalKMeans()
    model._offset = np.zeros(3)
    for data in (np.ones((2, 2)), np.ones(3)):
        with pytest.raises(ValueError, match="Feature count differs"):
            model.predict(data)


def test_cpu_lloyd_rejects_invalid_cluster_iteration_and_restart_counts(monkeypatch):
    """Real CPU boundary injection reaches configuration errors without impersonating an MPS device."""
    from spacr import mps_reduce

    monkeypatch.setattr(mps_reduce, "_matrix", _cpu_matrix)
    for options in ({"n_clusters": 0}, {"n_clusters": 4}, {"max_iter": 0}):
        with pytest.raises(ValueError, match="Invalid cluster or iteration count"):
            mps_reduce.MetalKMeans(**{"n_clusters": 2, **options}).fit([[0., 0.], [1., 1.], [2., 2.]])
    with pytest.raises(ValueError, match="n_init must be positive"):
        mps_reduce.MetalKMeans(n_clusters=2, n_init=0).fit([[0., 0.], [1., 1.], [2., 2.]])
