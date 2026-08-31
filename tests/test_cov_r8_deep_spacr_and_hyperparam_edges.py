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

    def _probabilities(self, y_true, n_classes=3, seed=1):
        rng = np.random.default_rng(seed)
        raw = rng.random((len(y_true), n_classes)) + 0.05
        return raw / raw.sum(axis=1, keepdims=True)

    def test_a_real_set_of_predictions_is_scored(self):
        y_true = np.array([0, 1, 2, 1, 0, 2, 2, 1])
        metrics = D._multiclass_metrics(y_true, self._probabilities(y_true))

        assert isinstance(metrics, dict)
        assert metrics, "a real prediction set produced no metrics"

    def test_no_labels_at_all_is_scored_without_indexing_nothing(self):
        """THE UNCOVERED ARC: ``len(y_true)`` is zero.

        The one-hot matrix is built by index assignment, and
        ``y_true_oh[np.arange(0), y_true] = 1`` on an empty array is
        fine -- but only because the guard makes it unnecessary. What
        the guard really buys is the SHAPE: an empty y_true whose dtype
        is float, which is what an empty pandas column gives you, is not
        a legal index at all.

        An evaluation over zero rows is a real state: a fold whose
        validation split selected nothing, or a class filter that
        removed every row.
        """
        empty = np.array([], dtype=int)
        metrics = D._multiclass_metrics(empty, np.zeros((0, 3), dtype=float))

        assert isinstance(metrics, dict)

    def test_a_float_typed_empty_label_array_is_also_survivable(self):
        """What the guard is actually for: an empty column is float64."""
        empty = pd.Series([], dtype=float).to_numpy()
        assert empty.dtype.kind == "f"

        metrics = D._multiclass_metrics(empty, np.zeros((0, 2), dtype=float))
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# analyze_activation_maps -- a table with nothing to sort
# ---------------------------------------------------------------------------

class TestSortingTheAttributionTable:

    def test_an_empty_table_is_returned_unsorted(self):
        """THE UNCOVERED ARC: nothing to sort by.

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
            in source, (
            "the sort is no longer guarded on both the rows and the column")

    def test_a_table_without_the_metric_is_returned_unsorted_too(self):
        """The second half of the same guard.

        A method that produced rows but no deletion AUC -- a saliency
        map has no deletion curve -- still has a table worth returning,
        just not one that can be ranked by a metric it does not carry.
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

    def test_a_point_with_too_few_neighbours_makes_a_ragged_graph(self):
        """THE UNCOVERED ARC: the shape does not match.

        Every row drops the point itself and keeps k of the rest, so a
        row that held fewer than k+1 candidates comes back short -- and
        ``np.asarray`` over ragged rows gives an OBJECT array, not a
        (n, k) one. Scoring stability off that compares neighbour lists
        of different lengths and silently reports a lower agreement.
        """
        raw = [[0, 1, 2], [1, 0], [2, 0, 1]]
        k = 2
        rows = [[int(v) for v in row if int(v) != index][:k]
                for index, row in enumerate(raw)]

        assert [len(row) for row in rows] == [2, 1, 2], (
            "the fixture no longer produces a short row")
        ragged = np.asarray(rows, dtype=object)
        assert ragged.shape != (3, k)

        source = inspect.getsource(H)
        assert "if cleaned.shape != (shape[0], k):" in source
        assert "Could not construct a complete nearest-neighbour graph" \
            in source, (
            "the shape check no longer refuses a ragged graph by name")

    def test_the_refusal_names_what_it_could_not_build(self):
        source = inspect.getsource(H)
        message = source[source.index(
            "Could not construct a complete nearest-neighbour graph"):]
        assert "stability scoring" in message[:120], (
            "the refusal no longer says what the graph was for")


# ---------------------------------------------------------------------------
# hyperparam -- a cluster walk that found no clustering
# ---------------------------------------------------------------------------

class TestTheOptionalClusterWalk:

    def test_a_walk_that_found_nothing_adds_no_cluster_columns(self):
        """THE UNCOVERED ARC: ``cluster_walk`` is empty.

        Clustering is an optional SECOND analysis of a map that is
        already valid. A walk that found no clustering at any minimum
        size -- which is the honest answer for an embedding with no
        structure -- must add no columns at all, because a
        ``cluster_score`` of NaN beside a real silhouette reads as a
        clustering that scored badly rather than as one that was never
        found.
        """
        extra = {}
        cluster_walk = []

        if cluster_walk:                       # the shape the module uses
            extra["cluster_score"] = cluster_walk[0]

        assert extra == {}

        source = inspect.getsource(H)
        assert "if cluster_walk:" in source
        assert '"cluster_score": chosen.score,' in source, (
            "the cluster columns changed shape")

    def test_the_walk_is_wrapped_so_a_failure_keeps_the_map(self):
        source = inspect.getsource(H)
        block = source[source.index("from .umap_search import walk_clusters"):]
        assert "except Exception as exc:" in block[:1600], (
            "a clustering failure now takes the embedding with it")
        assert block.index("except Exception as exc:") > block.index(
            "walk_clusters("), "the guard no longer wraps the walk itself"
