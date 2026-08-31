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


class TestTheGpuEmbedder:
    """Reached only when cuML says it is there and builds what was asked."""

    @staticmethod
    def _features(rows=40, cols=6, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(rows, cols))

    def test_a_cuml_array_is_brought_back_to_the_host(self, monkeypatch):
        """THE UNCOVERED LINE.

        cuML returns a device array. `.get()` copies it to host memory;
        without that the value goes into a checkpoint as a handle to
        memory the next process cannot read.
        """
        from spacr import gpu_reduce

        brought_back = []

        class _DeviceArray:
            def __init__(self, host):
                self._host = host

            def get(self):
                brought_back.append(True)
                return self._host

        class _Reducer:
            def fit_transform(self, feats):
                return _DeviceArray(np.zeros((len(feats), 2)))

        monkeypatch.setattr(gpu_reduce, "rapids_available", lambda: True)
        monkeypatch.setattr(gpu_reduce, "make_reducer",
                            lambda *a, **k: (_Reducer(), "cuml"))

        result = H.umap_search(
            self._features(), space=H.SearchSpace({"n_neighbors": [5]}), backend="cuml",
            n_trials=1, seed=0)
        assert brought_back, "the device array was never copied to the host"
        assert result is not None

    def test_a_host_array_is_passed_through_unchanged(self, monkeypatch):
        """The other side of `hasattr(value, "get")`.

        A reducer that already returns a numpy array has no `.get`, and
        calling one would be an AttributeError rather than a copy.
        """
        from spacr import gpu_reduce

        class _Reducer:
            def fit_transform(self, feats):
                return np.zeros((len(feats), 2))

        monkeypatch.setattr(gpu_reduce, "rapids_available", lambda: True)
        monkeypatch.setattr(gpu_reduce, "make_reducer",
                            lambda *a, **k: (_Reducer(), "cuml"))

        assert H.umap_search(self._features(), space=H.SearchSpace({"n_neighbors": [5]}),
                             backend="cuml", n_trials=1, seed=0) is not None

    def test_no_cuml_is_refused_rather_than_downgraded(self, monkeypatch):
        """A GPU run that quietly used the CPU would be false provenance."""
        from spacr import gpu_reduce

        monkeypatch.setattr(gpu_reduce, "rapids_available", lambda: False)
        with pytest.raises(RuntimeError, match="GPU UMAP was requested"):
            H.umap_search(self._features(), space=H.SearchSpace({"n_neighbors": [5]}),
                          backend="cuml", n_trials=1, seed=0)

    def test_a_reducer_that_is_not_cuml_fails_the_trial(self, monkeypatch):
        """A checked GPU run must not silently produce CPU rows.

        The refusal is raised inside the embedder, and the search
        RECORDS it against the trial rather than letting it escape --
        which is right: one failed configuration is a result about that
        configuration, not a reason to lose the whole sweep. What must
        not happen is a trial that succeeded on the wrong backend.
        """
        from spacr import gpu_reduce

        class _Reducer:
            def fit_transform(self, feats):
                return np.zeros((len(feats), 2))

        monkeypatch.setattr(gpu_reduce, "rapids_available", lambda: True)
        monkeypatch.setattr(gpu_reduce, "make_reducer",
                            lambda *a, **k: (_Reducer(), "umap-learn"))

        result = H.umap_search(
            self._features(), space=H.SearchSpace({"n_neighbors": [5]}),
            backend="cuml", n_trials=1, seed=0)

        errors = [getattr(t, "error", None) for t in result.trials]
        assert any(e and "cuML failed to construct" in str(e)
                   for e in errors), (
            "a non-cuML reducer produced a trial with no recorded refusal")


class TestTheNeighbourGraphShapeCheck:
    """`if cleaned.shape != (shape[0], k): raise RuntimeError(...)` is dead.

    The graph is queried with `n_neighbors=k + 1`, because a row is its
    own nearest point. Each row of the result therefore has exactly
    k + 1 entries, the row's own index is removed -- and `kneighbors`
    returns distinct indices, so at most ONE entry can match -- leaving
    k or k + 1, which `[:k]` truncates to exactly k.

    So every row yields exactly k, and the array has one row per input
    row. The shape can only be (n, k).
    """

    def test_the_stability_of_identical_embeddings_is_one(self):
        rng = np.random.default_rng(0)
        embedding = rng.normal(size=(30, 2))
        assert H.embedding_stability([embedding, embedding.copy()],
                                     neighbourhood_k=5) == pytest.approx(1.0)

    def test_unrelated_embeddings_score_below_identical_ones(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(40, 2))
        b = rng.normal(size=(40, 2))
        assert H.embedding_stability([a, b], neighbourhood_k=5) < 1.0

    def test_the_query_still_asks_for_one_more_than_it_keeps(self):
        source = inspect.getsource(H.embedding_stability)
        assert "n_neighbors=k + 1" in source, (
            "the graph no longer over-queries by one, so a row can yield "
            "fewer than k neighbours and the shape check becomes reachable")
        assert "[:k]" in source, (
            "the rows are no longer truncated to k")

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
            "a row yielded fewer than k neighbours; the shape check in "
            "embedding_stability is now reachable")
        assert np.asarray(cleaned, dtype=int).shape == (20, k)
