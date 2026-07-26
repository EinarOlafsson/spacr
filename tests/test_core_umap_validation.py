"""Argument validation and error paths of spacr.core's analysis entry points.

The theme is that a misconfigured run must say what is wrong. Before these
tests (and the fixes they pin) the four common ways of pointing
``generate_image_umap`` at an unusable measurements database produced a bare
``KeyError("['cell_id'] not in index")``, a ``KeyError('png_path')``, an
``UnboundLocalError`` from ``correct_paths`` or a raw sqlite
``OperationalError`` — none of which name the database at fault.

Everything here is CPU-only and deterministic: where the embedding itself is
not the subject, the reducer (a several-second UMAP/tSNE fit) is replaced by
a fixed one so the row bookkeeping around it can be asserted exactly.
"""
from __future__ import annotations

import os
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

# The synthetic measurements.db fixture (tests/ is a package, so the import
# has to be qualified — see tests/test_core_edge_branches.py).
from tests.test_core_umap_graphs import (  # noqa: F401
    umap_src, _umap_settings, _entity_frame, N_OBJ,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _db(tmp_path, name, tables):
    """Write a measurements.db under <tmp_path>/<name>/measurements/."""
    root = tmp_path / name
    meas = root / "measurements"
    meas.mkdir(parents=True)
    con = sqlite3.connect(meas / "measurements.db")
    try:
        for table, frame in tables.items():
            frame.to_sql(table, con, index=False)
    finally:
        con.close()
    return str(root)


def _png_list(with_cell_id=True, with_png_path=True, n=6):
    cols = {}
    if with_cell_id:
        cols["cell_id"] = [f"o{i+1}" for i in range(n)]
    if with_png_path:
        cols["png_path"] = [f"/x/data/cell_png/o{i+1}.png" for i in range(n)]
    cols.update({
        "plateID": ["plate1"] * n, "rowID": ["r1"] * n,
        "columnID": ["c1"] * n, "fieldID": ["f1"] * n,
    })
    return pd.DataFrame(cols)


def _cells(n=6):
    rng = np.random.default_rng(0)
    return _entity_frame(rng, "cell", n=n)


class FixedReducer:
    """Deterministic stand-in for utils.reduction_and_clustering.

    Returns a fixed 2-D embedding and a labels vector in which every
    ``noise_every``-th object is DBSCAN noise (-1).
    """

    def __init__(self, noise_every=None):
        self.noise_every = noise_every
        self.calls = []

    def __call__(self, numeric_data, *args, **kwargs):
        n = len(numeric_data)
        self.calls.append((args, kwargs))
        rng = np.random.default_rng(7)
        embedding = rng.normal(size=(n, 2))
        labels = np.zeros(n, dtype=int)
        if self.noise_every:
            labels[::self.noise_every] = -1
        return embedding, labels, None


# ---------------------------------------------------------------------------
# generate_image_umap: the measurements database has to be usable
# ---------------------------------------------------------------------------

def test_missing_database_is_named(tmp_path):
    src = tmp_path / "never_measured"
    src.mkdir()
    from spacr.core import generate_image_umap
    with pytest.raises(ValueError) as ei:
        generate_image_umap(_umap_settings(str(src)))
    msg = str(ei.value)
    assert "measurements.db" in msg and str(src) in msg
    assert "Measure" in msg


def test_missing_png_list_table_is_named(tmp_path):
    """A db measured with save_png=False has no png_list; the old failure was
    KeyError('png_path') raised 40 lines later."""
    src = _db(tmp_path, "no_png", {"cell": _cells()})
    from spacr.core import generate_image_umap
    with pytest.raises(ValueError) as ei:
        generate_image_umap(_umap_settings(src))
    msg = str(ei.value)
    assert "png_list" in msg
    assert "Tables present: cell" in msg
    assert "save_png" in msg


def test_png_list_without_cell_id_is_named(tmp_path):
    """The known sharp edge: png_list.cell_id is the join key, and losing it
    used to surface as KeyError("['cell_id'] not in index") from spacr.io."""
    src = _db(tmp_path, "no_cell_id", {
        "cell": _cells(), "png_list": _png_list(with_cell_id=False)})
    from spacr.core import generate_image_umap
    with pytest.raises(ValueError) as ei:
        generate_image_umap(_umap_settings(src))
    msg = str(ei.value)
    assert "cell_id" in msg
    assert "png_list" in msg and src in msg
    # it also reports what the table *does* have, so the user can tell
    # whether it is the wrong table or an old schema
    assert "columnID" in msg and "png_path" in msg


def test_png_list_without_png_path_is_named(tmp_path):
    src = _db(tmp_path, "no_path", {
        "cell": _cells(), "png_list": _png_list(with_png_path=False)})
    from spacr.core import generate_image_umap
    with pytest.raises(ValueError) as ei:
        generate_image_umap(_umap_settings(src))
    assert "png_path" in str(ei.value)


def test_empty_database_is_named(tmp_path):
    """An empty (or never-written) db used to die with UnboundLocalError:
    local variable 'image_paths' referenced before assignment."""
    src = _db(tmp_path, "empty", {})
    from spacr.core import generate_image_umap
    with pytest.raises(ValueError) as ei:
        generate_image_umap(_umap_settings(src))
    assert "png_list" in str(ei.value)


def test_requested_feature_tables_all_absent_is_named(tmp_path):
    src = _db(tmp_path, "png_only", {"png_list": _png_list()})
    from spacr.core import generate_image_umap
    with pytest.raises(ValueError) as ei:
        generate_image_umap(_umap_settings(src, tables=["cell", "nucleus"]))
    msg = str(ei.value)
    assert "cell, nucleus" in msg
    assert "Tables present: png_list" in msg


def test_missing_cell_table_is_named_even_when_others_exist(tmp_path):
    """The join is anchored on cell objects: nucleus features alone cannot be
    embedded, and io._read_and_join_tables returns bare png metadata for it."""
    rng = np.random.default_rng(1)
    src = _db(tmp_path, "no_cell", {
        "nucleus": _entity_frame(rng, "nucleus", n=6),
        "png_list": _png_list()})
    from spacr.core import generate_image_umap
    with pytest.raises(ValueError) as ei:
        generate_image_umap(_umap_settings(src, tables=["cell", "nucleus"]))
    assert "'cell' table" in str(ei.value)


def test_optional_feature_tables_may_be_absent(umap_src, monkeypatch):
    """A screen without a pathogen channel has no pathogen table; asking for
    it must NOT be fatal — io just skips it."""
    import spacr.utils as su
    from spacr.core import generate_image_umap
    monkeypatch.setattr(su, "reduction_and_clustering", FixedReducer())

    out = generate_image_umap(_umap_settings(
        umap_src, tables=["cell", "pathogen_that_does_not_exist"]))
    assert len(out) == N_OBJ


def test_default_settings_point_at_a_placeholder_source(tmp_path, monkeypatch):
    """generate_image_umap() with no settings falls back to src='path'; that
    has to be reported as a missing database, not crash on it."""
    monkeypatch.chdir(tmp_path)
    from spacr.core import generate_image_umap
    with pytest.raises(ValueError) as ei:
        generate_image_umap()
    assert os.path.join("path", "measurements", "measurements.db") in str(ei.value)


# ---------------------------------------------------------------------------
# generate_image_umap: row bookkeeping
# ---------------------------------------------------------------------------

def test_remove_cluster_noise_keeps_frame_and_embedding_aligned(umap_src, monkeypatch):
    """BUG (fixed): remove_cluster_noise dropped DBSCAN's -1 points from the
    embedding but not from all_df, so assigning the cluster column blew up
    with "Length of values (40) does not match length of index (60)" — the
    setting was unusable whenever some, but not all, points were noise."""
    import spacr.utils as su
    from spacr.core import generate_image_umap

    monkeypatch.setattr(su, "reduction_and_clustering", FixedReducer(noise_every=3))

    seen = {}
    real_plot = su.plot_embedding

    def _spy_plot(embedding, image_paths, labels, *a, **k):
        seen["embedding"] = len(embedding)
        seen["paths"] = len(image_paths)
        seen["labels"] = len(labels)
        return real_plot(embedding, image_paths, labels, *a, **k)

    monkeypatch.setattr(su, "plot_embedding", _spy_plot)

    out = generate_image_umap(_umap_settings(
        umap_src, remove_cluster_noise=True))

    kept = N_OBJ - len(range(0, N_OBJ, 3))          # 60 - 20
    assert len(out) == kept
    assert (out["cluster"] == 0).all()
    assert -1 not in set(out["cluster"])
    # the points, the crops behind them and the labels all describe the
    # same 40 objects
    assert seen == {"embedding": kept, "paths": kept, "labels": kept}
    # and the surviving rows are real rows, not a reindexed shell
    assert out["png_path"].notna().all()
    assert out["png_path"].is_unique


def test_all_noise_keeps_every_row_in_one_cluster(umap_src, monkeypatch, capsys):
    """When every point is noise there is nothing to keep, so the frame is
    left intact and flagged as a single cluster rather than emptied."""
    import spacr.utils as su
    from spacr.core import generate_image_umap
    monkeypatch.setattr(su, "reduction_and_clustering", FixedReducer(noise_every=1))

    out = generate_image_umap(_umap_settings(umap_src, remove_cluster_noise=True))
    assert len(out) == N_OBJ
    assert set(out["cluster"]) == {1}
    assert "No clusters found" in capsys.readouterr().out


def test_row_limit_above_the_row_count_is_a_cap_not_an_error(umap_src, monkeypatch, capsys):
    """row_limit=1000 on a 60-object screen used to abort with pandas'
    "Cannot take a larger sample than population"."""
    import spacr.utils as su
    from spacr.core import generate_image_umap
    monkeypatch.setattr(su, "reduction_and_clustering", FixedReducer())

    out = generate_image_umap(_umap_settings(umap_src, row_limit=1000))
    assert len(out) == N_OBJ
    assert "exceeds the 60 rows available" in capsys.readouterr().out


def test_row_limit_below_the_row_count_samples(umap_src, monkeypatch):
    import spacr.utils as su
    from spacr.core import generate_image_umap
    monkeypatch.setattr(su, "reduction_and_clustering", FixedReducer())

    out = generate_image_umap(_umap_settings(umap_src, row_limit=25))
    assert len(out) == 25


def test_exclude_conditions_accepts_a_bare_string(umap_src, monkeypatch, capsys):
    """A single condition may be given unwrapped; it must filter rows, not be
    iterated character by character."""
    import spacr.utils as su
    from spacr.core import generate_image_umap
    monkeypatch.setattr(su, "reduction_and_clustering", FixedReducer())

    # _umap_settings sets pos='c1' / neg='c2', so map_condition labels the
    # 30 c1 rows 'pos' and the 30 c2 rows 'neg'.
    out = generate_image_umap(_umap_settings(
        umap_src, exclude_conditions="pos", verbose=True))
    assert set(out["columnID"]) == {"c2"}
    assert len(out) == N_OBJ // 2
    assert "Excluded 30 rows" in capsys.readouterr().out


def test_resnet_features_reports_that_it_is_unimplemented(umap_src):
    """BUG (fixed): the resnet_features branch was an empty `pass`, so the
    run continued and died on an unbound 'embedding' inside plot_embedding."""
    from spacr.core import generate_image_umap
    with pytest.raises(NotImplementedError) as ei:
        generate_image_umap(_umap_settings(umap_src, resnet_features=True))
    assert "resnet_features" in str(ei.value)


def test_color_by_with_embedding_by_controls_labels_by_metadata(umap_src, monkeypatch):
    """color_by replaces the cluster labels with a metadata column, whichever
    way the embedding was fitted."""
    import spacr.utils as su
    from spacr.core import generate_image_umap
    monkeypatch.setattr(su, "reduction_and_clustering", FixedReducer())

    out = generate_image_umap(_umap_settings(
        umap_src, embedding_by_controls=True, color_by="columnID",
        col_to_compare="columnID"))
    assert (out["cluster"] == out["columnID"]).all()
    assert set(out["cluster"]) == {"c1", "c2"}


def test_umap_writes_the_embedding_results_csv(umap_src, monkeypatch):
    """The returned frame and the CSV on disk are the same rows."""
    import spacr.utils as su
    from spacr.core import generate_image_umap
    monkeypatch.setattr(su, "reduction_and_clustering", FixedReducer())

    out = generate_image_umap(_umap_settings(umap_src))
    csv = os.path.join(umap_src, "results", "embedding_results.csv")
    on_disk = pd.read_csv(csv)
    assert len(on_disk) == len(out) == N_OBJ
    assert "cluster" in on_disk.columns
    assert list(on_disk["png_path"]) == list(out["png_path"])


# ---------------------------------------------------------------------------
# reducer_hyperparameter_search: parameter validation
# ---------------------------------------------------------------------------

def test_reduction_params_is_required():
    """Omitting it used to raise TypeError: 'NoneType' object is not
    iterable, from a generator expression that never names the argument."""
    from spacr.core import reducer_hyperparameter_search
    with pytest.raises(ValueError) as ei:
        reducer_hyperparameter_search(settings=None, reduction_params=None)
    assert "reduction_params" in str(ei.value)


def test_reduction_params_without_a_recognised_key_is_reported(umap_src):
    """BUG (fixed): neither key left reduction_method unbound and the run
    died with UnboundLocalError on the next line."""
    from spacr.core import reducer_hyperparameter_search
    with pytest.raises(ValueError) as ei:
        reducer_hyperparameter_search(
            settings=_umap_settings(umap_src),
            reduction_params=[{"min_dist": 0.2}])
    msg = str(ei.value)
    assert "n_neighbors" in msg and "perplexity" in msg


def test_umap_and_tsne_params_together_are_rejected(umap_src):
    """BUG (fixed): this check was a third `elif` after the UMAP branch, so it
    could never fire — a mixed sweep silently ran as UMAP and every
    perplexity value in it was ignored."""
    from spacr.core import reducer_hyperparameter_search
    with pytest.raises(ValueError) as ei:
        reducer_hyperparameter_search(
            settings=_umap_settings(umap_src),
            reduction_params=[{"n_neighbors": 5}, {"perplexity": 5}],
            dbscan_params={"eps": 0.5, "min_samples": 3})
    assert "not both" in str(ei.value)


def test_unsupported_reduction_method_is_rejected_before_reading_data(umap_src, monkeypatch):
    """BUG (fixed): the "Unsupported reduction method" guard sat inside the
    per-cell plotting loop, below the line that had already overwritten
    reduction_method with 'umap'/'tsne' — so it could never fire and e.g.
    reduction_method='pca' was silently swapped for UMAP. It now fails up
    front, before the database is touched."""
    import spacr.io as sio
    from spacr.core import reducer_hyperparameter_search

    def _never(*a, **k):
        raise AssertionError("the database must not be read")

    monkeypatch.setattr(sio, "_read_and_join_tables", _never)
    with pytest.raises(ValueError) as ei:
        reducer_hyperparameter_search(
            settings=_umap_settings(umap_src, reduction_method="pca"),
            reduction_params=[{"n_neighbors": 5}],
            dbscan_params={"eps": 0.4, "min_samples": 3})
    assert "Unsupported reduction method: pca" in str(ei.value)


def test_fractional_n_neighbors_is_scaled_by_the_row_count(umap_src, monkeypatch):
    """A float n_neighbors means "this fraction of the objects"."""
    import spacr.utils as su
    from spacr.core import reducer_hyperparameter_search

    seen = []

    def _spy(numeric_data, n_neighbors, min_dist, metric, eps, min_samples,
             clustering, reduction_method, verbose, reduction_param=None,
             n_jobs=-1, **k):
        seen.append((len(numeric_data), n_neighbors, min_dist, eps,
                     min_samples, clustering, reduction_method))
        rng = np.random.default_rng(3)
        return rng.normal(size=(len(numeric_data), 2)), np.zeros(len(numeric_data), int)

    monkeypatch.setattr(su, "search_reduction_and_clustering", _spy)
    reducer_hyperparameter_search(
        settings=_umap_settings(umap_src),
        reduction_params=[{"n_neighbors": 0.1, "min_dist": 0.4}],
        dbscan_params={"eps": 0.7, "min_samples": 4},
        save=False, show=False)

    assert len(seen) == 1
    n_rows, n_neighbors, min_dist, eps, min_samples, clustering, method = seen[0]
    assert n_rows == N_OBJ
    assert n_neighbors == int(0.1 * N_OBJ) == 6
    assert min_dist == 0.4
    assert (eps, min_samples) == (0.7, 4)
    assert clustering == "dbscan"
    assert method == "umap"


def test_fractional_perplexity_is_scaled_by_the_row_count(umap_src, monkeypatch):
    import spacr.utils as su
    from spacr.core import reducer_hyperparameter_search

    seen = []

    def _spy(numeric_data, n_neighbors, *a, **k):
        seen.append(n_neighbors)
        rng = np.random.default_rng(3)
        return rng.normal(size=(len(numeric_data), 2)), np.zeros(len(numeric_data), int)

    monkeypatch.setattr(su, "search_reduction_and_clustering", _spy)
    reducer_hyperparameter_search(
        settings=_umap_settings(umap_src, reduction_method="tsne"),
        reduction_params=[{"perplexity": 0.25}],
        kmeans_params={"n_clusters": 2},
        save=False, show=False)

    assert seen == [int(0.25 * N_OBJ)] == [15]


def test_search_grid_covers_every_parameter_pair(umap_src, monkeypatch):
    """2 reduction x 2 clustering parameter sets = a 2x2 grid of embeddings,
    one axes each."""
    import spacr.utils as su
    from spacr.core import reducer_hyperparameter_search

    seen = []

    def _spy(numeric_data, n_neighbors, min_dist, metric, eps, min_samples,
             clustering, *a, **k):
        seen.append((n_neighbors, eps, min_samples, clustering))
        rng = np.random.default_rng(3)
        return rng.normal(size=(len(numeric_data), 2)), np.zeros(len(numeric_data), int)

    monkeypatch.setattr(su, "search_reduction_and_clustering", _spy)
    fig = reducer_hyperparameter_search(
        settings=_umap_settings(umap_src),
        reduction_params=[{"n_neighbors": 5}, {"n_neighbors": 9}],
        dbscan_params=[{"eps": 0.4, "min_samples": 3}],
        kmeans_params=[{"n_clusters": 2}],
        save=False, show=False, return_fig=True)

    assert len(seen) == 4
    assert {s[0] for s in seen} == {5, 9}
    assert {s[3] for s in seen} == {"dbscan", "kmeans"}
    assert len(fig.axes) == 4


def test_search_row_limit_is_a_cap(umap_src, monkeypatch, capsys):
    import spacr.utils as su
    from spacr.core import reducer_hyperparameter_search

    seen = []

    def _spy(numeric_data, *a, **k):
        seen.append(len(numeric_data))
        rng = np.random.default_rng(3)
        return rng.normal(size=(len(numeric_data), 2)), np.zeros(len(numeric_data), int)

    monkeypatch.setattr(su, "search_reduction_and_clustering", _spy)
    reducer_hyperparameter_search(
        settings=_umap_settings(umap_src, row_limit=10 ** 6),
        reduction_params=[{"n_neighbors": 5}],
        dbscan_params=[{"eps": 0.4, "min_samples": 3}],
        save=False, show=False)

    assert seen == [N_OBJ]
    assert "exceeds the 60 rows available" in capsys.readouterr().out


def test_search_shows_the_figure_when_not_saving(umap_src, monkeypatch):
    """show=True + save=False is the interactive path: it calls plt.show once."""
    import matplotlib.pyplot as plt
    import spacr.utils as su
    from spacr.core import reducer_hyperparameter_search

    def _spy(numeric_data, *a, **k):
        rng = np.random.default_rng(3)
        return rng.normal(size=(len(numeric_data), 2)), np.zeros(len(numeric_data), int)

    shown = []
    monkeypatch.setattr(su, "search_reduction_and_clustering", _spy)
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    reducer_hyperparameter_search(
        settings=_umap_settings(umap_src),
        reduction_params=[{"n_neighbors": 5}],
        dbscan_params=[{"eps": 0.4, "min_samples": 3}],
        save=False, show=True)
    assert shown == [True]


# ---------------------------------------------------------------------------
# generate_screen_graphs
# ---------------------------------------------------------------------------

def _screen_settings(src):
    return {
        "src": src,
        "tables": ["cell", "nucleus", "pathogen", "cytoplasm"],
        "cells": ["HeLa"], "controls": ["c1", "c2"],
        "controls_loc": [["c1"], ["c2"]],
        "graph_type": "bar", "summary_func": "mean",
        "y_axis_start": 0, "error_bar_type": "std",
        "theme": "deep", "representation": "well",
        "nuclei_limit": 10, "pathogen_limit": 10,
    }


def test_screen_graphs_over_two_sources_writes_three_result_sets(umap_src, tmp_path):
    """A list src produces one figure+CSV per source plus a combined one, and
    the combined pair is filed under the first source."""
    from spacr.core import generate_screen_graphs

    second = str(tmp_path / "plate2")
    shutil.copytree(umap_src, second)
    generate_screen_graphs(_screen_settings([umap_src, second]))

    first_results = sorted(os.listdir(os.path.join(umap_src, "results")))
    second_results = sorted(os.listdir(os.path.join(second, "results")))
    suffix = "well_mean_bar"
    assert first_results == [
        f"figure_controls_0_{suffix}.pdf", f"figure_controls_2_{suffix}.pdf",
        f"results_controls_0_{suffix}.csv", f"results_controls_2_{suffix}.csv",
    ]
    assert second_results == [
        f"figure_controls_1_{suffix}.pdf", f"results_controls_1_{suffix}.csv",
    ]
    # the combined result summarises both plates, so it cannot be empty
    combined = pd.read_csv(os.path.join(
        umap_src, "results", f"results_controls_2_{suffix}.csv"))
    assert len(combined) > 0


# ---------------------------------------------------------------------------
# dry_run
# ---------------------------------------------------------------------------

def test_dry_run_validates_and_returns_problems_without_writing(tmp_path):
    """dry_run returns the preflight problem list and touches nothing —
    importantly it returns *before* the cellpose/model imports."""
    from spacr.core import preprocess_generate_masks
    from spacr.validate import Problem

    src = tmp_path / "plate1"
    src.mkdir()
    problems = preprocess_generate_masks({
        "src": str(src), "dry_run": True,
        "cell_channel": "not-an-int", "nucleus_channel": None,
    })
    assert isinstance(problems, list)
    assert all(isinstance(p, Problem) for p in problems)
    assert any(p.setting == "cell_channel" and p.is_error for p in problems)
    assert os.listdir(src) == [], "dry_run must not write anything"
