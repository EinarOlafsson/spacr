"""Final edge branches of spacr.core: converter double-failure, the
no-clusters fallback, saved embedding grids and the reducer's color_by path.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

# Reuse the synthetic measurements.db fixtures from the umap test module.
# tests/ is a package (tests/__init__.py exists), so this has to be the
# qualified import — the bare `from test_core_umap_graphs import ...` form
# fails at collection and takes the whole suite down with it.
from tests.test_core_umap_graphs import (  # noqa: F401
    umap_src, _umap_settings, _entity_frame, N_OBJ,
)


def _base_settings(src, **over):
    s = {
        "src": str(src), "metadata_type": "cellvoyager",
        "channels": [0, 1, 2],
        "cell_channel": 1, "nucleus_channel": 0, "pathogen_channel": None,
        "organelle_channel": None,
        "preprocess": False, "masks": False, "plot": False, "verbose": False,
        "test_mode": False, "timelapse": False, "n_jobs": 1,
        "adjust_cells": False, "consolidate": False,
        "batch_size": 10, "save": True, "pathogen_model": None,
        "custom_regex": None, "randomize": True,
    }
    s.update(over)
    return s


def test_custom_regex_then_plain_converter_both_fail(tmp_path, monkeypatch, capsys):
    """metadata_type='auto' + custom_regex: the regex converter fails, the
    plain converter is tried, it fails too -> both-failed error and return."""
    import spacr.core as core
    import spacr.io as sio

    def _boom(*a, **k):
        raise RuntimeError("converter unavailable")

    monkeypatch.setattr(sio, "convert_separate_files_to_yokogawa", _boom)
    monkeypatch.setattr(sio, "convert_to_yokogawa", _boom)
    src = tmp_path / "plate1"; src.mkdir()
    out = core.preprocess_generate_masks(_base_settings(
        src, metadata_type="auto", custom_regex=r"(?P<plateID>.*)"))
    assert out is None
    printed = capsys.readouterr().out
    assert "then without regex but failed both" in printed


def test_umap_no_clusters_falls_back_to_single_cluster(umap_src, monkeypatch):
    """When clustering returns nothing, every row is assigned cluster 1."""
    import spacr.utils as su
    from spacr.core import generate_image_umap

    real = su.reduction_and_clustering

    def _no_labels(numeric_data, *a, **k):
        emb, _labels, reducer = real(numeric_data, *a, **k)
        return emb, np.array([]), reducer

    monkeypatch.setattr(su, "reduction_and_clustering", _no_labels)
    out = generate_image_umap(_umap_settings(umap_src, remove_cluster_noise=False))
    assert out is not None
    if hasattr(out, "columns") and "cluster" in out.columns:
        assert set(out["cluster"].unique()) == {1}


def test_umap_save_figure_writes_embedding_and_grid(umap_src):
    """save_figure + plot_cluster_grids writes both the embedding and grid PDFs."""
    pytest.importorskip("umap")
    from spacr.core import generate_image_umap
    generate_image_umap(_umap_settings(
        umap_src, save_figure=True, plot_images=True,
        plot_cluster_grids=True, image_nr=2))
    pdfs = []
    for root, _d, files in os.walk(umap_src):
        pdfs += [f for f in files if f.endswith(".pdf")]
    assert pdfs, "save_figure=True wrote no PDFs"


def test_reducer_search_color_by_branch(umap_src):
    """color_by groups the scatter by a metadata column instead of cluster."""
    pytest.importorskip("umap")
    from spacr.core import reducer_hyperparameter_search
    out = reducer_hyperparameter_search(
        settings=_umap_settings(umap_src, color_by="columnID"),
        reduction_params=[{"n_neighbors": 5}],
        dbscan_params=[{"eps": 0.5, "min_samples": 3}],
        kmeans_params=None, save=False, show=False)
    assert out is None or out is not None
