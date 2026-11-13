"""hyperparam: the cuML embedding path, and a shape check that cannot fail.

The GPU embedder exists to make a checked GPU run honest -- there is no
silent downgrade to CPU. That posture is the point: a table row saying
"gpu" that was actually computed on the processor is false provenance.

`embedding_stability` measures repeat-to-repeat neighbour preservation.
Its shape check is unreachable, and this file pins it to the arithmetic
that makes it so.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from spacr import hyperparam as H


class TestTheNeighbourGraphShapeContract:
    """The removed postcondition check could never see a wrong shape.

    The graph is queried with `n_neighbors=k + 1`, because a row is its
    own nearest point. Each row of the result therefore has exactly
    k + 1 entries, the row's own index is removed -- and `kneighbors`
    returns distinct indices, so at most ONE entry can match -- leaving
    k or k + 1, which `[:k]` truncates to exactly k.

    So every row yields exactly k, and the array has one row per input
    row. The shape can only be (n, k).
    """


    def test_the_query_still_asks_for_one_more_than_it_keeps(self):
        source = inspect.getsource(H.embedding_stability)
        assert "n_neighbors=k + 1" in source, (
            "the graph no longer over-queries by one, so a row can yield "
            "fewer than k neighbours and needs an explicit refusal again")
        assert "[:k]" in source, (
            "the rows are no longer truncated to k")
        assert "if cleaned.shape != (shape[0], k):" not in source
        assert "Could not construct a complete nearest-neighbour graph" \
            not in source

    @pytest.mark.parametrize("k", [1, 2, 5, 9])
    def test_every_row_yields_exactly_k_neighbours(self, k):
        """The argument, checked rather than asserted in prose."""
        from sklearn.neighbors import NearestNeighbors

        rng = np.random.default_rng(2)
        array = rng.normal(size=(20, 3))
        raw = NearestNeighbors(n_neighbors=k + 1).fit(array).kneighbors(
            array, return_distance=False)
        cleaned = [[int(v) for v in row if int(v) != int(i)][:k]
                   for i, row in enumerate(raw)]
        assert all(len(row) == k for row in cleaned), (
            "a row yielded fewer than k neighbours; the removed postcondition "
            "is no longer guaranteed")
        assert np.asarray(cleaned, dtype=int).shape == (20, k)
