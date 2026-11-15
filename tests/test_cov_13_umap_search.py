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

from spacr.umap_search import (ClusterWalkRow, SearchRow, SearchTable,
                               UmapRecipe)


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
    """``rows`` must copy the outer list, so a caller cannot reorder the table.

    The retained row objects remain shared. Screens sort and filter the outer
    list; handing out that live list would let a display-side sort silently
    reorder the record of the search.
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
