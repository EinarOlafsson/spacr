"""Clustering a fixed UMAP and walking HDBSCAN scales (``spacr.umap_search``).

``cluster_embedding`` refuses maps and thresholds that cannot describe a
partition and a backend that returns the wrong label count;
``walk_clusters`` names a size that is not a number, records an all-noise
scale as a row, and keeps a silhouette sklearn refuses as NaN (score -inf)
rather than zero.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.umap_search import cluster_embedding, walk_clusters


def _blobs(seed: int = 0) -> np.ndarray:
    """Two well-separated clouds of 40 points each."""
    rng = np.random.default_rng(seed)
    return np.vstack([
        rng.normal(loc=(0.0, 0.0), scale=0.15, size=(40, 2)),
        rng.normal(loc=(6.0, 6.0), scale=0.15, size=(40, 2)),
    ])


def test_cluster_embedding_refuses_meaningless_requests(subtests, monkeypatch):
    """Each invalid map or threshold raises its own message; valid ones label."""
    refusals = [
        ("two points", np.array([[0.0, 0.0], [1.0, 1.0]]), {},
         ValueError, "at least 3 points"),
        ("one-dimensional coordinates", np.zeros((5, 1)), {},
         ValueError, r"shape \(rows, 2\) or \(rows, 3\)"),
        ("non-finite coordinates", np.array([[0.0, 0.0], [1.0, np.nan],
                                             [2.0, 2.0]]), {},
         ValueError, "NaN or infinite"),
        ("cluster size one", _blobs(), {"min_cluster_size": 1},
         ValueError, "min_cluster_size must be at least 2"),
        ("cluster size as large as the map", _blobs(),
         {"min_cluster_size": 80}, ValueError, "smaller than the 80 points"),
        ("negative min_samples", _blobs(),
         {"min_cluster_size": 5, "min_samples": -1},
         ValueError, "min_samples must be at least 1"),
    ]
    for case_id, embedding, kwargs, error, message in refusals:
        with subtests.test(msg=case_id):
            with pytest.raises(error, match=message):
                cluster_embedding(embedding, **kwargs)

    expected = np.repeat([0, 1], 40)
    for case_id, samples in [("min_samples zero means unset", 0),
                             ("min_samples None", None),
                             ("min_samples three", 3)]:
        with subtests.test(msg=case_id):
            labels = cluster_embedding(_blobs(), min_cluster_size=5,
                                       min_samples=samples)
            assert labels.dtype == int
            assert labels.tolist() == expected.tolist()

    import sklearn.cluster

    class ShortHDBSCAN:
        def __init__(self, **kwargs):
            pass

        def fit_predict(self, values):
            return np.zeros(len(values) - 1, dtype=int)

    monkeypatch.setattr(sklearn.cluster, "HDBSCAN", ShortHDBSCAN)
    with subtests.test(msg="a short label array is refused"):
        with pytest.raises(RuntimeError, match="does not match the map"):
            cluster_embedding(_blobs(), min_cluster_size=5)


def test_walk_clusters_scores_every_valid_scale_honestly(subtests, monkeypatch):
    """Exact rows per scale; bad sizes name themselves; failed silhouettes are NaN."""
    for case_id, sizes, message in [
            ("text size", (5, "twelve"), r"got 'twelve'"),
            ("missing size", (None,), r"got None"),
            ("no size in range", (1, 80, 500), "between 2 and 79")]:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError, match=message):
                walk_clusters(_blobs(), min_cluster_sizes=sizes)

    with subtests.test(msg="two blobs score at every scale, duplicates dropped"):
        rows = walk_clusters(_blobs(), min_cluster_sizes=(10, 5, 10, 1))
        assert [(r.min_cluster_size, r.n_clusters, r.noise_fraction)
                for r in rows] == [(5, 2, 0.0), (10, 2, 0.0)]
        assert [r.silhouette for r in rows] == pytest.approx(
            [0.9692963235666671] * 2)
        assert [r.score for r in rows] == pytest.approx(
            [0.9692963235666671] * 2)
        assert rows[0].labels.tolist() == np.repeat([0, 1], 40).tolist()

    with subtests.test(msg="structureless data is still one recorded row"):
        (row,) = walk_clusters(np.zeros((8, 2)), min_cluster_sizes=(5,))
        assert (row.min_cluster_size, row.n_clusters, row.noise_fraction) == (
            5, 0, 1.0)
        assert np.isnan(row.silhouette)
        assert row.score == float("-inf")
        assert row.labels.tolist() == [-1] * 8

    import sklearn.metrics

    def refuse(*args, **kwargs):
        raise ValueError("Number of labels is 1")

    monkeypatch.setattr(sklearn.metrics, "silhouette_score", refuse)
    with subtests.test(msg="a refused silhouette is NaN, not zero"):
        rows = walk_clusters(_blobs(), min_cluster_sizes=(5, 10))
        assert [(r.min_cluster_size, r.n_clusters) for r in rows] == [
            (5, 2), (10, 2)]
        assert all(np.isnan(r.silhouette) for r in rows)
        assert [r.score for r in rows] == [float("-inf")] * 2
