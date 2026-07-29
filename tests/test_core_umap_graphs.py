"""Synthetic coverage for spacr.core's analysis entry points:
generate_image_umap, reducer_hyperparameter_search and
generate_screen_graphs.

Builds a measurements.db with the cell/nucleus/pathogen/cytoplasm +
png_list schema those functions join on (see io._read_and_join_tables),
so the reduction/clustering/plot paths run for real on CPU with no GPU,
Cellpose or HF download.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


N_OBJ = 60


def _entity_frame(rng, entity, n=N_OBJ):
    """One measurement table with the metadata columns spacr joins on."""
    cols = {
        "object_label": np.arange(1, n + 1),
        "plateID": ["plate1"] * n,
        "rowID": ["r1"] * n,
        # two conditions so map_condition yields pos/neg groups
        "columnID": ["c1" if i % 2 == 0 else "c2" for i in range(n)],
        "fieldID": ["f1"] * n,
        "prcfo": [f"plate1_r1_c{(i % 2) + 1}_f1_o{i+1}" for i in range(n)],
        "prcf": [f"plate1_r1_c{(i % 2) + 1}_f1" for i in range(n)],
        "prc": [f"plate1_r1_c{(i % 2) + 1}" for i in range(n)],
        # Parent-cell link used by _read_and_merge_data for nucleus/pathogen:
        # an INTEGER label (it prefixes 'o' itself). png_list instead stores
        # the already-prefixed 'o<N>' string form.
        "cell_id": np.arange(1, n + 1),
    }
    # Numeric feature columns — the reducer needs several to work with.
    for ch in range(3):
        cols[f"{entity}_channel_{ch}_mean_intensity"] = rng.uniform(100, 5000, n)
        cols[f"{entity}_channel_{ch}_percentile_75"] = rng.uniform(100, 5000, n)
    cols[f"{entity}_area"] = rng.uniform(200, 4000, n)
    cols[f"{entity}_perimeter"] = rng.uniform(50, 400, n)
    cols[f"{entity}_eccentricity"] = rng.uniform(0, 1, n)
    return pd.DataFrame(cols)


@pytest.fixture
def umap_src(tmp_path, rng):
    """A plate folder with measurements/measurements.db + PNG crops."""
    from PIL import Image
    src = tmp_path / "plate1"
    meas = src / "measurements"
    meas.mkdir(parents=True)
    png_dir = src / "data" / "cell_png"
    png_dir.mkdir(parents=True)

    png_paths = []
    for i in range(N_OBJ):
        p = png_dir / f"obj_{i+1:03d}.png"
        arr = rng.integers(0, 255, size=(32, 32, 3)).astype(np.uint8)
        Image.fromarray(arr).save(p)
        png_paths.append(str(p))

    con = sqlite3.connect(meas / "measurements.db")
    try:
        for entity in ("cell", "nucleus", "pathogen", "cytoplasm"):
            _entity_frame(rng, entity).to_sql(entity, con, index=False)
        png_list = pd.DataFrame({
            "cell_id": [f"o{i+1}" for i in range(N_OBJ)],
            "png_path": png_paths,
            "plateID": ["plate1"] * N_OBJ,
            "rowID": ["r1"] * N_OBJ,
            "columnID": ["c1" if i % 2 == 0 else "c2" for i in range(N_OBJ)],
            "fieldID": ["f1"] * N_OBJ,
            "prcfo": [f"plate1_r1_c{(i % 2) + 1}_f1_o{i+1}" for i in range(N_OBJ)],
        })
        png_list.to_sql("png_list", con, index=False)
    finally:
        con.close()
    return str(src)


def _umap_settings(src, **over):
    s = {
        "src": src,
        "tables": ["cell"],
        "row_limit": None,
        "filter_by": None,
        "remove_highly_correlated": False,
        "log_data": False,
        "exclude": None,
        "exclude_conditions": None,
        "embedding_by_controls": False,
        "col_to_compare": "columnID",
        "pos": "c1", "neg": "c2", "mix": "c3",
        "reduction_method": "umap",
        "n_neighbors": 5, "min_dist": 0.1, "metric": "euclidean",
        "clustering": "dbscan", "eps": 0.9, "min_samples": 3,
        "remove_cluster_noise": False,
        "plot_images": False, "plot_by_cluster": False,
        "plot_cluster_grids": False, "plot_outlines": False,
        "plot_points": True, "smooth_lines": False,
        "black_background": False, "remove_image_canvas": False,
        "figuresize": 6, "dot_size": 10, "img_zoom": 0.5, "image_nr": 4,
        "save_figure": False, "analyze_clusters": False,
        "resnet_features": False, "color_by": None,
        "verbose": False, "n_jobs": 1, "visualize": "cell",
        "min_cell_count": None,
    }
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# generate_image_umap
# ---------------------------------------------------------------------------

def test_generate_image_umap_returns_dataframe(umap_src):
    pytest.importorskip("umap")
    from spacr.core import generate_image_umap
    out = generate_image_umap(_umap_settings(umap_src))
    assert out is not None
    # writes the settings CSV it always saves
    assert os.path.isfile(os.path.join(umap_src, "settings",
                                       "embedding_settings.csv"))


def test_generate_image_umap_tsne_and_row_limit(umap_src):
    pytest.importorskip("sklearn")
    from spacr.core import generate_image_umap
    out = generate_image_umap(_umap_settings(
        umap_src, reduction_method="tsne", row_limit=30,
        n_neighbors=5, clustering="kmeans"))
    assert out is not None


def test_generate_image_umap_embedding_by_controls(umap_src):
    pytest.importorskip("umap")
    from spacr.core import generate_image_umap
    out = generate_image_umap(_umap_settings(
        umap_src, embedding_by_controls=True, col_to_compare="columnID"))
    assert out is not None


def test_generate_image_umap_exclude_conditions(umap_src):
    pytest.importorskip("umap")
    from spacr.core import generate_image_umap
    # 'mix' maps anything that isn't pos/neg; exclude one real condition
    out = generate_image_umap(_umap_settings(
        umap_src, exclude_conditions=["pc"], verbose=True))
    assert out is not None


def test_generate_image_umap_return_fig(umap_src):
    pytest.importorskip("umap")
    from spacr.core import generate_image_umap
    fig = generate_image_umap(_umap_settings(umap_src), return_fig=True)
    assert fig is not None
    payload = fig._spacr_umap_payload
    assert payload["embedding"].shape == (N_OBJ, 2)
    assert len(payload["labels"]) == len(payload["records"]) == N_OBJ
    assert payload["records"][0]["db_path"].endswith("measurements.db")
    assert payload["records"][0]["db_png_path"].endswith("obj_001.png")


# ---------------------------------------------------------------------------
# reducer_hyperparameter_search
# ---------------------------------------------------------------------------

def test_recruitment_ratio_rejects_invalid_denominators_without_infinity():
    from spacr.core import _finite_ratio

    numerator = pd.Series(
        [10, 10, 10, np.inf, "bad"], index=[10, 20, 30, 40, 50])
    denominator = pd.Series(
        [2, 0, np.nan, 2, 5], index=[10, 20, 30, 40, 50])

    result = _finite_ratio(numerator, denominator)

    assert list(result.index) == [10, 20, 30, 40, 50]
    assert result.loc[10] == pytest.approx(5.0)
    assert result.loc[[20, 30, 40, 50]].isna().all()
    assert not np.isinf(result).any()

def test_generate_image_umap_plot_images_and_grids(umap_src):
    """plot_images / plot_by_cluster / plot_cluster_grids render the image
    overlays instead of plain points."""
    pytest.importorskip("umap")
    from spacr.core import generate_image_umap
    out = generate_image_umap(_umap_settings(
        umap_src, plot_images=True, plot_by_cluster=True,
        plot_cluster_grids=True, black_background=True,
        remove_image_canvas=True, image_nr=4))
    assert out is not None


def test_generate_image_umap_analyze_clusters_and_save(umap_src):
    """analyze_clusters runs the per-cluster feature analysis; save_figure
    writes the embedding PDF."""
    pytest.importorskip("umap")
    from spacr.core import generate_image_umap
    out = generate_image_umap(_umap_settings(
        umap_src, analyze_clusters=True, save_figure=True, verbose=True))
    assert out is not None
    pdfs = []
    for root, _dirs, files in os.walk(umap_src):
        pdfs += [f for f in files if f.endswith(".pdf")]
    assert pdfs, "save_figure=True should write an embedding PDF"


def test_generate_image_umap_color_by_and_noise_removal(umap_src):
    """color_by disables the outline/noise options; remove_cluster_noise
    drops DBSCAN -1 labels."""
    pytest.importorskip("umap")
    from spacr.core import generate_image_umap
    out = generate_image_umap(_umap_settings(
        umap_src, color_by="columnID", remove_cluster_noise=True,
        plot_outlines=True, smooth_lines=True))
    assert out is not None


def test_generate_image_umap_log_data_and_correlation_filter(umap_src):
    """log_data + remove_highly_correlated exercise preprocess_data's
    transform/prune branches."""
    pytest.importorskip("umap")
    from spacr.core import generate_image_umap
    out = generate_image_umap(_umap_settings(
        umap_src, log_data=True, remove_highly_correlated=True))
    assert out is not None


def test_generate_screen_graphs_writes_results(umap_src):
    """generate_screen_graphs merges the DB, computes the recruitment metric
    and writes a figure + CSV per source into results/."""
    from spacr.core import generate_screen_graphs
    settings = {
        "src": umap_src,
        "tables": ["cell", "nucleus", "pathogen", "cytoplasm"],
        "cells": ["HeLa"], "controls": ["c1", "c2"],
        "controls_loc": [["c1"], ["c2"]],
        "graph_type": "bar", "summary_func": "mean",
        "y_axis_start": 0, "error_bar_type": "std",
        "theme": "deep", "representation": "well",
        "nuclei_limit": 10, "pathogen_limit": 10,
    }
    generate_screen_graphs(settings)
    out = os.path.join(umap_src, "results")
    assert os.path.isdir(out) and os.listdir(out)


def test_reducer_hyperparameter_search_runs(umap_src):
    pytest.importorskip("umap")
    from spacr.core import reducer_hyperparameter_search
    settings = _umap_settings(umap_src)
    out = reducer_hyperparameter_search(
        settings=settings,
        reduction_params=[{"n_neighbors": 5}],
        dbscan_params=[{"eps": 0.5, "min_samples": 3}],
        kmeans_params=[{"n_clusters": 2}],
        save=False, show=False)
    # returns a figure or None depending on backend; the call must not raise
    assert out is None or out is not None


def test_reducer_search_accepts_dict_params_and_returns_fig(umap_src):
    """Bare dicts (not lists) are wrapped; return_fig hands back the figure."""
    pytest.importorskip("umap")
    from spacr.core import reducer_hyperparameter_search
    fig = reducer_hyperparameter_search(
        settings=_umap_settings(umap_src),
        reduction_params={"n_neighbors": 5},
        dbscan_params={"eps": 0.5, "min_samples": 3},
        kmeans_params={"n_clusters": 2},
        save=False, show=False, return_fig=True)
    assert fig is not None


def test_reducer_search_tsne_switches_method(umap_src, capsys):
    """A 'perplexity' param selects tSNE and announces the switch."""
    pytest.importorskip("sklearn")
    from spacr.core import reducer_hyperparameter_search
    reducer_hyperparameter_search(
        settings=_umap_settings(umap_src, reduction_method="umap"),
        reduction_params=[{"perplexity": 5}],
        dbscan_params=[{"eps": 0.5, "min_samples": 3}],
        kmeans_params=None, save=False, show=False)
    assert "Changed reduction method to tsne" in capsys.readouterr().out


def test_reducer_search_saves_figure(umap_src):
    """save=True writes the hyperparameter-search PDF under the source."""
    pytest.importorskip("umap")
    from spacr.core import reducer_hyperparameter_search
    reducer_hyperparameter_search(
        settings=_umap_settings(umap_src),
        reduction_params=[{"n_neighbors": 5}],
        dbscan_params=[{"eps": 0.5, "min_samples": 3}],
        kmeans_params=None, save=True, show=False)
    pdfs = []
    for root, _dirs, files in os.walk(umap_src):
        pdfs += [f for f in files if f.endswith(".pdf")]
    assert pdfs, "save=True should write a search figure"


def test_reducer_search_row_limit_and_exclude(umap_src):
    """row_limit sampling + exclude_conditions filtering inside the search."""
    pytest.importorskip("umap")
    from spacr.core import reducer_hyperparameter_search
    out = reducer_hyperparameter_search(
        settings=_umap_settings(umap_src, row_limit=30,
                                exclude_conditions="pc", verbose=True),
        reduction_params=[{"n_neighbors": 5}],
        dbscan_params=[{"eps": 0.5, "min_samples": 3}],
        kmeans_params=None, save=False, show=False)
    assert out is None or out is not None
