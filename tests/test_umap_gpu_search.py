"""GPU UMAP wiring and per-row backend/clustering provenance."""
from __future__ import annotations

import numpy as np
import pytest

from spacr.hyperparam import SearchSpace, umap_search


def _features(rows=60, columns=6):
    return np.random.default_rng(12).normal(size=(rows, columns))


def test_injected_embedder_is_named_custom_not_cpu():
    result = umap_search(
        _features(), SearchSpace({"n_neighbors": [8]}),
        embed_fn=lambda values, _params: values[:, :2],
    )
    assert result.best.extra_metrics["backend"] == "custom"


def test_unknown_backend_is_refused_before_a_trial_runs():
    with pytest.raises(ValueError, match="'cpu' or 'cuml'"):
        umap_search(
            _features(), SearchSpace({"n_neighbors": [8]}),
            backend="magic")


def test_requested_gpu_runs_the_cuml_reducer_and_records_it(monkeypatch):
    from spacr import gpu_reduce

    built = []

    class FakeReducer:
        def fit_transform(self, values):
            return values[:, :3]

    def make_reducer(method, *, prefer_gpu=False, **kwargs):
        built.append((method, prefer_gpu, dict(kwargs)))
        return FakeReducer(), "cuml"

    monkeypatch.setattr(gpu_reduce, "rapids_available", lambda: True)
    monkeypatch.setattr(gpu_reduce, "make_reducer", make_reducer)
    result = umap_search(
        _features(), SearchSpace({"n_neighbors": [8]}),
        backend="cuml", n_components=3, seed=19)

    assert result.ok
    assert result.best.extra_metrics["backend"] == "cuml"
    assert result.best.extra_metrics["n_components"] == 3
    assert result.best.extra_metrics["embedding"].shape == (60, 3)
    assert built == [("umap", True, {
        "n_neighbors": 8, "n_components": 3, "random_state": 19,
    })]


def test_gpu_constructor_failure_stops_instead_of_mixing_backends(monkeypatch):
    from spacr import gpu_reduce

    monkeypatch.setattr(gpu_reduce, "rapids_available", lambda: True)
    monkeypatch.setattr(
        gpu_reduce, "make_reducer",
        lambda *_a, **_k: (object(), "cpu"))
    result = umap_search(
        _features(), SearchSpace({"n_neighbors": [8]}), backend="cuml")
    assert not result.ok
    assert result.n_failed == 1
    assert "instead of mixing CPU and GPU rows" in result.failed[0].error


def test_clustering_walk_is_retained_on_the_same_trial():
    rng = np.random.default_rng(3)
    embedding = np.vstack((rng.normal(-3, 0.15, (30, 2)),
                           rng.normal(3, 0.15, (30, 2))))
    result = umap_search(
        _features(), SearchSpace({"n_neighbors": [8]}),
        embed_fn=lambda _values, _params: embedding,
        cluster_during_search=True,
        cluster_sizes=(5, 8, 12),
    )
    extra = result.best.extra_metrics
    assert extra["embedding"] is embedding
    assert extra["cluster_labels"].shape == (60,)
    assert extra["n_clusters"] == 2
    assert {row["min_cluster_size"] for row in extra["cluster_walk"]} == {
        5, 8, 12,
    }
    assert "HDBSCAN clustering walk" in " ".join(result.notes)


def test_missing_gpu_is_an_actionable_error_not_a_cpu_fallback(monkeypatch):
    from spacr import gpu_reduce

    monkeypatch.setattr(gpu_reduce, "rapids_available", lambda: False)
    with pytest.raises(RuntimeError, match="Restart spaCR"):
        umap_search(
            _features(), SearchSpace({"n_neighbors": [8]}), backend="cuml")
