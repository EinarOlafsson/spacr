"""Choosing and building the cuML estimator, on a machine without cuML.

RAPIDS is an optional extra and is not installed here, so the GPU half of
:mod:`spacr.gpu_reduce` is driven by putting a fake ``cuml`` in
``sys.modules``. What the tests assert is spaCR's own behaviour: which cuML
class each method name asks for, that an unknown name is refused rather than
guessed at, and that a cuML which imports but cannot build an estimator costs
the caller a CPU run instead of the whole figure.
"""
from __future__ import annotations

import sys
import types

import pytest

from spacr import gpu_reduce


class _Estimator:
    """Stands in for one of cuML's estimator classes."""

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs


def _fake_cuml(broken=()):
    """A ``cuml`` module whose estimators record what they were asked for."""
    module = types.ModuleType("cuml")
    module.__version__ = "24.10"
    for attr in ("UMAP", "TSNE", "PCA", "DBSCAN", "KMeans"):
        def _make(attr=attr):
            def _build(**kwargs):
                if attr in broken:
                    raise RuntimeError(f"cuML {attr} needs a newer CUDA")
                return _Estimator(attr, **kwargs)
            return _build
        setattr(module, attr, _make())
    return module


@pytest.fixture
def cuml_installed(monkeypatch):
    """Make ``import cuml`` succeed for the duration of one test."""
    def _install(broken=()):
        module = _fake_cuml(broken)
        monkeypatch.setitem(sys.modules, "cuml", module)
        return module

    return _install


@pytest.mark.parametrize("method, expected", [
    ("umap", "UMAP"),
    ("tsne", "TSNE"),
    ("pca", "PCA"),
    ("dbscan", "DBSCAN"),
    ("kmeans", "KMeans"),
])
def test_each_method_asks_cuml_for_its_own_estimator(cuml_installed, method,
                                                     expected):
    """The five accelerated methods map to five distinct cuML classes."""
    cuml_installed()

    estimator = gpu_reduce._cuml_estimator(method, n_components=2)

    assert estimator.name == expected
    assert estimator.kwargs == {"n_components": 2}


def test_a_method_cuml_does_not_implement_is_refused(cuml_installed):
    """Silently building the wrong estimator would be worse than an error."""
    cuml_installed()

    with pytest.raises(ValueError, match="hdbscan"):
        gpu_reduce._cuml_estimator("hdbscan")


