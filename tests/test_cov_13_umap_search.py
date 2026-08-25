"""A UMAP search must refuse a nonsense scale and never fake a score.

The search table drives which embedding a user keeps, so every number in it
has to be either real or plainly absent. An unscored row must not sort as if
it were the worst one, a clustering that could not be scored must read as NaN
rather than as zero separation, and a clustering request that cannot mean
anything -- fewer than two points to a cluster, a size that is not a number --
has to fail at the call rather than return a partition.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.umap_search import (ClusterWalkRow, SearchRow, SearchTable,
                               UmapRecipe, cluster_embedding, walk_clusters)


def _blobs(seed: int = 0) -> np.ndarray:
    """Two well-separated clouds, so HDBSCAN finds a real partition."""
    rng = np.random.default_rng(seed)
    return np.vstack([
        rng.normal(loc=(0.0, 0.0), scale=0.15, size=(40, 2)),
        rng.normal(loc=(6.0, 6.0), scale=0.15, size=(40, 2)),
    ])


def _row(**scores) -> SearchRow:
    return SearchRow(recipe=UmapRecipe(), scores=dict(scores))


def test_a_row_nobody_scored_reports_nan_rather_than_a_number():
    """An unscored trial must be absent from the ranking, not last in it.

    ``score`` is what the table sorts on. Returning 0.0 for a row that was
    never scored would place it among the real results and make "the best
    recipe" depend on where zero happens to fall.
    """
    assert np.isnan(_row().score)
    assert np.isnan(_row(clusters=3.0).score)
    assert _row(silhouette=0.4).score == 0.4
    assert _row(trustworthiness=0.9, silhouette=0.4).score == 0.9
    assert _row(score=0.1, trustworthiness=0.9).score == 0.1


def test_the_best_row_ignores_the_unscored_ones_entirely():
    """An unscored row must not win, and must not stop a scored row winning."""
    table = SearchTable()
    table.add(_row())
    winner = table.add(_row(score=0.7))
    table.add(_row(score=0.2))

    assert table.best() is winner
    assert SearchTable().best() is None


def test_the_table_reports_its_length_and_hands_back_a_detachable_row_list():
    """``rows`` must be a copy, so a caller cannot edit the search in place.

    Screens sort and filter what they get back. Handing out the live list
    would let a display-side sort silently reorder the record of the search.
    """
    table = SearchTable()
    first = table.add(_row(score=0.1))
    second = table.add(_row(score=0.9))

    assert len(table) == 2
    listed = table.rows
    assert listed == [first, second]

    listed.reverse()
    assert table.rows == [first, second]
    assert list(table) == [first, second]
    assert table[1] is second


def test_a_clustering_that_could_not_be_scored_ranks_below_every_real_one():
    """A NaN silhouette must sort last, not compare as a mid-range number.

    ``score`` multiplies separation by the assigned fraction; feeding NaN
    through that arithmetic would make the walk's ordering depend on how NaN
    compares, which is neither stable nor meaningful.
    """
    unscored = ClusterWalkRow(min_cluster_size=5, labels=np.zeros(4, dtype=int),
                              silhouette=float("nan"), n_clusters=1,
                              noise_fraction=0.0)
    scored = ClusterWalkRow(min_cluster_size=10, labels=np.zeros(4, dtype=int),
                            silhouette=-0.9, n_clusters=2, noise_fraction=0.5)

    assert unscored.score == float("-inf")
    assert scored.score < 0
    assert unscored.score < scored.score


def test_clustering_refuses_a_map_too_small_to_have_structure():
    """Two points cannot be partitioned, and must not be reported as one cluster."""
    with pytest.raises(ValueError, match="at least 3 points"):
        cluster_embedding(np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_clustering_refuses_a_cluster_size_below_two():
    """A cluster of one is every point, which is not a clustering."""
    with pytest.raises(ValueError, match="min_cluster_size must be at least 2"):
        cluster_embedding(_blobs(), min_cluster_size=1)


def test_clustering_refuses_a_min_samples_below_one():
    """Zero means "let HDBSCAN choose"; a negative is a caller mistake.

    ``None`` and ``0`` are the documented way to leave it unset, so a value
    below one is not a shorthand for that -- it is a number that would reach
    the estimator and change the partition unpredictably.
    """
    with pytest.raises(ValueError, match="min_samples must be at least 1"):
        cluster_embedding(_blobs(), min_cluster_size=5, min_samples=-1)

    labels = cluster_embedding(_blobs(), min_cluster_size=5, min_samples=0)
    assert labels.shape == (80,)


def test_a_backend_that_returns_the_wrong_number_of_labels_is_refused(monkeypatch):
    """A label array shorter than the map would colour the wrong points.

    The labels are zipped against the coordinates to draw the map. A silent
    length mismatch there is not an exception downstream, it is a picture in
    which every point after the first is attributed to its neighbour's cluster.
    """
    import sklearn.cluster

    class ShortHDBSCAN:
        def __init__(self, **kwargs):
            pass

        def fit_predict(self, values):
            return np.zeros(len(values) - 1, dtype=int)

    monkeypatch.setattr(sklearn.cluster, "HDBSCAN", ShortHDBSCAN)

    with pytest.raises(RuntimeError, match="does not match the map"):
        cluster_embedding(_blobs(), min_cluster_size=5)


def test_a_cluster_walk_size_that_is_not_a_number_names_itself(monkeypatch):
    """A size read from a settings field can arrive as text.

    The message has to echo the offending value, because the walk takes a
    whole sequence and "one of these is not a number" is not actionable.
    """
    with pytest.raises(ValueError, match=r"got 'twelve'"):
        walk_clusters(_blobs(), min_cluster_sizes=(5, "twelve"))

    with pytest.raises(ValueError, match=r"got None"):
        walk_clusters(_blobs(), min_cluster_sizes=(None,))


def test_a_silhouette_sklearn_refuses_to_compute_becomes_nan_not_zero(monkeypatch):
    """A scale sklearn cannot score must drop out of the ranking, not top it.

    Zero silhouette is a real answer meaning "no separation"; substituting it
    for a failed computation would put an unmeasurable scale ahead of every
    genuinely poor one.
    """
    import sklearn.metrics

    def refuse(*args, **kwargs):
        raise ValueError("Number of labels is 1")

    monkeypatch.setattr(sklearn.metrics, "silhouette_score", refuse)

    rows = walk_clusters(_blobs(), min_cluster_sizes=(5, 10))

    assert rows, "the walk found no scale to try"
    assert all(np.isnan(row.silhouette) for row in rows)
    assert all(row.score == float("-inf") for row in rows)


def test_a_walk_with_a_working_silhouette_still_scores_its_scales():
    """The failure path must not be the only one exercised.

    Two clean blobs are separable at every scale tried, so a real silhouette
    has to come back or the guard above is hiding a genuine break.
    """
    rows = walk_clusters(_blobs(), min_cluster_sizes=(5, 10))

    assert rows
    assert any(np.isfinite(row.silhouette) for row in rows)
    assert rows[0].score >= rows[-1].score
