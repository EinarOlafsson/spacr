"""Four edges in the metrics and the UMAP search, each an empty answer.

An empty confusion matrix, an attribution table with nothing in it, a
neighbour graph that could not be built, and a cluster walk that found no
clustering. None of the four is a failure -- each is a run that produced
no result, and the difference matters because a NaN written into a
metrics CSV is indistinguishable from "not computed".
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from spacr import deep_spacr as D
from spacr import hyperparam as H


# ---------------------------------------------------------------------------
# _multiclass_metrics -- no labels at all
# ---------------------------------------------------------------------------

class TestTheMulticlassMetrics:

    def test_the_one_hot_fill_needs_no_second_guard(self):
        """THE PIN for the unconditional one-hot fill.

        The function's first check already returned: an empty ``y_true`` never
        reaches the fill. Indexing with a FLOAT-typed empty array (what an
        empty pandas column gives you) is not legal, but the early return makes
        that case moot.
        """
        source = inspect.getsource(D._multiclass_metrics)
        early = source.index("if len(y_true) == 0:")
        one_hot = source.index("y_true_oh[np.arange(len(y_true)), y_true] = 1")
        assert early < one_hot, (
            "the empty case is no longer answered before the one-hot fill")
        assert "if len(y_true):" not in source

        empty = pd.Series([], dtype=float).to_numpy()
        with pytest.raises((IndexError, TypeError)):
            np.zeros((0, 3), dtype=int)[np.arange(0), empty] = 1


# ---------------------------------------------------------------------------
# analyze_activation_maps -- a table with nothing to sort
# ---------------------------------------------------------------------------

class TestSortingTheAttributionTable:

    def test_an_empty_table_explains_the_public_input_guard(self):
        """An empty internal table would have no columns to sort by.

        ``sort_values(['image', 'deletion_auc'])`` on an empty frame
        raises KeyError for the columns that were never created -- and
        this runs after every image has been attributed, so losing it
        there loses the whole analysis rather than the ordering.
        """
        table = pd.DataFrame([])

        assert table.empty
        with pytest.raises(KeyError):
            table.sort_values(["image", "deletion_auc"])

        source = inspect.getsource(D.analyze_activation_maps)
        assert "if not table.empty and 'deletion_auc' in table.columns:" \
            not in source
        assert "table = table.sort_values(['image', 'deletion_auc']" in source

    def test_every_internal_row_must_supply_the_metric(self):
        """A row without the metric would also make sorting invalid.

        ``analyze_activation_maps`` supplies ``deletion_auc`` on both its
        successful and failed row paths, which is why this malformed example
        cannot arise from the public function.
        """
        table = pd.DataFrame({"image": ["a.png", "b.png"],
                              "method": ["saliency", "saliency"]})

        assert not table.empty
        assert "deletion_auc" not in table.columns
        with pytest.raises(KeyError):
            table.sort_values(["image", "deletion_auc"])

    def test_a_complete_table_sorts_by_image_then_metric(self):
        table = pd.DataFrame({
            "image": ["b.png", "a.png", "a.png"],
            "deletion_auc": [0.5, 0.9, 0.1],
        })

        ordered = table.sort_values(["image", "deletion_auc"],
                                    na_position="last").reset_index(drop=True)

        assert ordered["image"].tolist() == ["a.png", "a.png", "b.png"]
        assert ordered["deletion_auc"].tolist() == [0.1, 0.9, 0.5]


# ---------------------------------------------------------------------------
# hyperparam -- a neighbour graph that came out the wrong shape
# ---------------------------------------------------------------------------

class TestTheStabilityNeighbourGraph:

    def test_a_complete_graph_gives_one_row_per_point(self):
        """Each point's k nearest neighbours, itself excluded."""
        raw = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]])
        k = 2
        cleaned = np.asarray([
            [int(value) for value in row if int(value) != int(index)][:k]
            for index, row in enumerate(raw)
        ], dtype=int)

        assert cleaned.shape == (3, k)


    def test_the_impossible_shape_refusal_is_removed(self):
        source = inspect.getsource(H.embedding_stability)
        assert "n_neighbors=k + 1" in source
        assert "[:k]" in source
        assert "if cleaned.shape != (shape[0], k):" not in source
        assert "Could not construct a complete nearest-neighbour graph" \
            not in source


# ---------------------------------------------------------------------------
# hyperparam -- every valid cluster walk returns at least one row
# ---------------------------------------------------------------------------

class TestTheClusterWalkContract:

    def test_every_valid_walk_returns_a_row_even_for_structureless_data(self):
        """A valid candidate yields a row; all-noise is recorded, not omitted."""
        from spacr.umap_search import walk_clusters

        rows = walk_clusters(np.zeros((8, 2)), min_cluster_sizes=(5,))

        assert len(rows) == 1
        assert np.isnan(rows[0].silhouette)

        source = inspect.getsource(H.umap_search)
        assert "if cluster_walk:" not in source
        assert "chosen = cluster_walk[0]" in source

    def test_the_walk_is_wrapped_so_a_failure_keeps_the_map(self):
        source = inspect.getsource(H)
        block = source[source.index("from .umap_search import walk_clusters"):]
        assert "except Exception as exc:" in block[:1600], (
            "a clustering failure now takes the embedding with it")
        assert block.index("except Exception as exc:") > block.index(
            "walk_clusters("), "the guard no longer wraps the walk itself"
